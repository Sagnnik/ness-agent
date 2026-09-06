"""Headless one-shot query runner (``ness -p "query"``).

Print mode runs a single turn outside the TUI: the final assistant text goes
to stdout, diagnostics go to stderr, and the process exits with a status code
(0 success, 1 turn error, 130 interrupted).

Approvals are deny-by-default: with no interactive approval handler wired,
the graph's approval gate auto-denies any gated tool call the permission
rules don't already allow (the denial is fed back to the model), while
``--yolo`` bypasses the gate entirely — mirroring ``claude -p`` semantics.
Questions from the ``question`` tool are auto-answered with the recommended
option so the turn never blocks on a human.
"""

from __future__ import annotations

import sys
import uuid
from typing import TYPE_CHECKING

from ness_agent.session_context import SessionContext, set_session_context
from ness_agent.tools import is_git_repo

from ness_cli.chat_model import provider_key_missing
from ness_cli.config import settings
from ness_cli.factory import build_coding_session, prepare_paths
from ness_cli.mcp_trust import is_mcp_trusted
from ness_cli.mcp_oauth import MCPOAuthService
from ness_cli.mcp_manager import ProjectMCPManager

if TYPE_CHECKING:
    from ness_cli.coding_session import CodingSession


async def auto_answer_questions(questions: list[dict]) -> list[dict]:
    """Headless question handler: pick the recommended option (else the first)."""
    answers: list[dict] = []
    for index, question in enumerate(questions, 1):
        options = list(question.get("options") or [])
        selected = next((o for o in options if o.get("recommended")), None)
        if selected is None and options:
            selected = options[0]
        if selected is None:
            selected = {"id": "0", "label": "proceed"}
        answers.append(
            {
                "id": question.get("id", str(index)),
                "selected": {"id": selected.get("id"), "label": selected.get("label")},
                "note": "auto-answered (headless mode)",
            }
        )
    return answers


def merge_prompt_parts(
    prompt_parts: list[str] | None, stdin_text: str | None
) -> str | None:
    """Merge positional prompt args and piped stdin into one query string.

    Piped stdin (if any) is prepended as context, matching ``claude -p``.
    Returns ``None`` when both are empty (caller reports a usage error).
    """
    query = " ".join(part for part in (prompt_parts or []) if part).strip()
    piped = (stdin_text or "").strip()
    if piped and query:
        return f"{piped}\n\n{query}"
    return piped or query or None


async def run_headless_turn(coding: CodingSession, prompt: str) -> tuple[str, int]:
    """Run one turn on ``coding`` and return ``(final_text, exit_code)``.

    The final text is the last non-empty ``assistant_final`` content (what
    ``claude -p`` prints). Errors/warnings are reported on stderr; an
    ``error`` event maps to exit code 1, an ``interrupted`` event to 130.
    """
    final_text = ""
    exit_code = 0
    async for ev in coding.run_turn(prompt):
        if ev.kind == "assistant_final":
            text = str(ev.data.get("content") or "").strip()
            if text:
                final_text = text
        elif ev.kind == "error":
            print(f"error: {ev.data.get('message') or ev.data}", file=sys.stderr)
            exit_code = 1
        elif ev.kind == "warning":
            print(f"warning: {ev.data.get('message') or ev.data}", file=sys.stderr)
        elif ev.kind == "interrupted":
            partial = str(ev.data.get("partial_text") or "").strip()
            if partial:
                final_text = partial
            if exit_code == 0:
                exit_code = 130
    return final_text, exit_code


async def run_headless(
    prompt: str,
    *,
    resume_thread_id: str | None = None,
    yolo: bool = False,
) -> int:
    """Run a one-shot query, print the final response to stdout, return the exit code.

    Mirrors the TUI's ``_main`` setup (paths, MCP manager, session context,
    deferred MCP tool catalog) minus the TuiApp, so a headless turn has
    durable-event, checkpoint, and rollback parity with an interactive one.
    The thread is archived on exit so ``--resume`` keeps working; the resume
    hint goes to stderr to keep stdout clean for scripting.
    """
    git_available = is_git_repo()
    paths = prepare_paths()

    if provider_key_missing():
        print(
            f"error: no provider API key configured or {settings.model_provider} is not "
            "authenticated — run interactive Ness and use /login",
            file=sys.stderr,
        )
        return 1

    mcp_oauth = MCPOAuthService(
        project_root=paths.project_root,
        config_dir=paths.config_dir,
    )
    mcp = ProjectMCPManager(
        mcp_file=paths.ness_dir / "mcp.json",
        project_root=paths.project_root,
        http_auth_factory=mcp_oauth.startup_auth,
    )
    if is_mcp_trusted(mcp, config_dir=paths.config_dir):
        await mcp.start()
    else:
        mcp.mark_untrusted()
        print(
            "warning: MCP configuration is not trusted; run interactive Ness once "
            "to review and approve it",
            file=sys.stderr,
        )

    thread_id = f"session-{uuid.uuid4().hex[:8]}"
    coding = build_coding_session(
        thread_id=thread_id,
        yolo=yolo,
        vision=settings.supports_vision,
        git_available=git_available,
        approval_handler=None,  # deny-by-default; allow rules still apply
        question_handler=auto_answer_questions,
        paths=paths,
    )
    coding.permission_store.clear_session_rules()

    # Register MCP tools as known but deferred, same as the TUI path.
    mcp_tools = list(mcp.tools.values())
    coding.tool_registry.register_dynamic(mcp_tools)
    coding.tool_registry.set_mcp_catalog(mcp.catalog())

    set_session_context(
        SessionContext(
            permissions=coding.permission_store,
            options=coding.cfg.options,
            thread_store=coding.thread_store,
            ness_dir=coding.ness_dir,
            project_root=coding.project_root,
            agent_config=coding.cfg,
            all_skills=coding.skill_loader.load() if coding.skill_loader else None,
            vision=settings.supports_vision,
        )
    )

    message, level = mcp.startup_summary()
    if level == "warn":
        print(f"warning: {message}", file=sys.stderr)
    for oauth_warning in mcp_oauth.warnings:
        print(f"warning: {oauth_warning}", file=sys.stderr)

    try:
        if resume_thread_id:
            resumed = await coding.resume(resume_thread_id)
            if not resumed:
                print(f"error: no saved thread '{resume_thread_id}'", file=sys.stderr)
                return 1
        text, exit_code = await run_headless_turn(coding, prompt)
        if text:
            sys.stdout.write(text + "\n")
        return exit_code
    finally:
        saved_id: str | None = None
        try:
            await coding.finalize_reflection()
            saved_id = (
                coding.thread_id
                if (settings.auto_save_threads and coding.turn_count > 0)
                else None
            )
            coding.save_thread()
        except Exception as exc:
            print(f"warning: archive skipped: {exc}", file=sys.stderr)
        await mcp.stop()
        from ness_cli.provider.registry import close_providers

        await close_providers()
        if saved_id:
            print(f"Resume: ness --resume {saved_id}", file=sys.stderr)
