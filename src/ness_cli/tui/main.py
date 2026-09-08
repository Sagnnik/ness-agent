"""Ness Agent CLI entrypoint (Ness).

A clean, scrollable Rich + prompt_toolkit CLI. Run with:

    uv run ness
    # or
    uv run python -m ness_cli.tui.main

Headless one-shot queries (final response to stdout, then exit):

    uv run ness -p "explain this project"
    cat log.txt | uv run ness -p "find the root cause"

The TUI is wired directly to the SDK stack: a
:class:`~ness_cli.CodingSession` built via
:func:`ness_cli.factory.build_coding_session`, an SDK
:class:`~ness_cli.mcp_manager.ProjectMCPManager`, and the SDK tool registry. The old
root-level monolith modules are no longer imported here.
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from pathlib import Path

# Allow running as `python -m ness_cli.tui.main` (or a direct file path)
# by making sure ``src/`` is on sys.path so ``ness_cli`` is importable.
_SRC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


def _bootstrap_worktree() -> None:
    argv = sys.argv[1:]
    name = None
    for i, arg in enumerate(argv):
        if arg in ("--worktree", "-w") and i + 1 < len(argv):
            name = argv[i + 1]
        elif arg.startswith("--worktree="):
            name = arg.split("=", 1)[1]
    if not name:
        return
    from ness_cli.worktree import WorktreeError, ensure_worktree

    try:
        path = ensure_worktree(name)
    except WorktreeError as exc:
        print(f"worktree error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    os.chdir(path)
    os.environ["NESS_AGENT_WORKTREE"] = name
    os.environ["NESS_AGENT_WORKTREE_PATH"] = str(path)


_bootstrap_worktree()

import typer

from ness_agent.context.budget import resolve_token_count
from ness_agent.session_context import SessionContext, set_session_context
from ness_agent.tools import is_git_repo
from ness_cli.chat_model import (
    ModelOverrides,
    configure_model,
    provider_key_missing,
    validate_effort,
)
from ness_cli.config import settings
from ness_cli.factory import build_coding_session, prepare_paths
from ness_cli.headless import merge_prompt_parts, run_headless
from ness_cli.mcp_trust import authorize_mcp_interactively
from ness_cli.mcp_oauth import MCPOAuthService
from ness_cli.mcp_manager import ProjectMCPManager

from ness_cli.tui import render
from ness_cli.tui.app import TuiApp
from ness_cli.tui.theme import build_console

_MCP_HELP = """MCP management commands:

ness mcp status [SERVER] — Show configured servers and authentication state.

ness mcp login SERVER — Authenticate an HTTP MCP server.

ness mcp logout SERVER — Remove stored OAuth credentials.

ness mcp import PATH — Import a Cursor- or Claude-compatible MCP config.

Run 'ness mcp --help' for command options."""

app = typer.Typer(add_completion=False, help="Ness Agent agent CLI")


def _version_callback(value: bool | None) -> None:
    if not value:
        return
    from importlib.metadata import PackageNotFoundError, version

    try:
        print(f"ness {version('ness-agent')}")
    except PackageNotFoundError:
        print("ness (version unknown — package not installed)")
    raise typer.Exit()


def _overrides(
    model: str | None,
    reflection_model: str | None,
    api_key: str | None,
    base_url: str | None,
    session_id: str | None,
    reasoning_effort: str | None,
) -> ModelOverrides | None:
    fields = {
        "model_name": model,
        "reflection_model_name": reflection_model,
        "openai_api_key": api_key,
        "openai_base_url": base_url,
        "openrouter_session_id": session_id,
        "reasoning_effort": reasoning_effort,
    }
    if reasoning_effort is not None:
        target_model = model or settings.model_name
        try:
            validate_effort(target_model, reasoning_effort)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
    active = {key: value for key, value in fields.items() if value is not None}
    return ModelOverrides(**active) if active else None


@app.command(epilog=_MCP_HELP)
def run(
    prompt: list[str] = typer.Argument(
        None, help="One-shot query text (requires --print)"
    ),
    model: str = typer.Option(
        None, "--model", help="Chat model name (overrides MODEL_NAME)"
    ),
    reflection_model: str = typer.Option(
        None, "--reflection-model", help="Reflection model name"
    ),
    api_key: str = typer.Option(None, "--api-key", help="OpenAI-compatible API key"),
    base_url: str = typer.Option(None, "--base-url", help="OpenAI-compatible base URL"),
    session_id: str = typer.Option(
        None, "--openrouter-session-id", help="Stable prompt-cache session id"
    ),
    reasoning_effort: str = typer.Option(
        None,
        "--reasoning-effort",
        help="Provider-literal OpenRouter reasoning effort",
    ),
    worktree: str = typer.Option(
        None, "--worktree", "-w", help="Run inside an isolated git worktree"
    ),
    resume: str = typer.Option(
        None,
        "--resume",
        help="Resume a saved thread id at startup (loads prior conversation into the transcript)",
    ),
    yolo: bool = typer.Option(
        False,
        "--yolo",
        help="Approve all act-mode tool calls and ignore permission deny rules",
    ),
    print_mode: bool = typer.Option(
        False,
        "--print",
        "-p",
        help="Run the query non-interactively, print the final response, and exit",
    ),
    version: bool | None = typer.Option(
        None,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the installed Ness version and exit",
    ),
) -> None:
    """Start an interactive Ness Agent session (or a one-shot query with -p)."""
    configure_model(
        _overrides(
            model, reflection_model, api_key, base_url, session_id, reasoning_effort
        )
    )
    if print_mode:
        stdin_text = ""
        if not sys.stdin.isatty():
            try:
                stdin_text = sys.stdin.read()
            except OSError:
                pass
        query = merge_prompt_parts(prompt, stdin_text)
        if query is None:
            print(
                "error: --print requires a prompt argument or piped stdin",
                file=sys.stderr,
            )
            raise SystemExit(2)
        try:
            exit_code = asyncio.run(
                run_headless(query, resume_thread_id=resume or None, yolo=yolo)
            )
        except KeyboardInterrupt:
            exit_code = 130
        raise SystemExit(exit_code)
    if prompt:
        print(
            "error: a positional prompt requires --print/-p",
            file=sys.stderr,
        )
        raise SystemExit(2)
    asyncio.run(_main(resume_thread_id=resume or None, yolo=yolo))


def _render_mcp_startup(mcp: ProjectMCPManager) -> None:
    message, level = mcp.startup_summary()
    if level != "warn":
        return
    render.render_warning(message + "  (/mcp for details)")


_STATIC_PREFIX_TOKEN_TARGET = 7000


def _check_prompt_budget(coding, git_available: bool) -> str | None:
    """Soft-warn when the cached static prefix (L0+L1+L2) exceeds the token target."""
    from langchain_core.messages import SystemMessage

    cfg = coding.agent.config
    prefix = cfg.prompts.build_stable_prefix(
        cfg.tool_registry.active_tools,
        user_memory=cfg.memory_store.load_user(),
        project_memory=cfg.memory_store.load_project(),
        skill_catalog=cfg.skill_loader.render_catalog(cfg.skill_loader.load()),
        git_available=git_available,
        metadata={},
        tool_catalog_groups=cfg.tool_registry.tool_catalog_groups(),
        deferred_mcp=cfg.tool_registry.deferred_mcp_summary(),
    ).strip()
    tokens = resolve_token_count(
        [SystemMessage(content=prefix)], known_input_tokens=None
    )
    if tokens > _STATIC_PREFIX_TOKEN_TARGET:
        return (
            f"Static prompt prefix (L0+L1+L2) is ~{tokens:,} tokens "
            f"(> {_STATIC_PREFIX_TOKEN_TARGET:,} target). Consider trimming NESS.md, "
            "its @includes, or USER.md."
        )
    return None


async def _main(*, resume_thread_id: str | None = None, yolo: bool = False) -> None:
    git_available = is_git_repo()

    paths = prepare_paths()

    mcp_oauth = MCPOAuthService(
        project_root=paths.project_root,
        config_dir=paths.config_dir,
    )
    mcp = ProjectMCPManager(
        mcp_file=paths.ness_dir / "mcp.json",
        project_root=paths.project_root,
        http_auth_factory=mcp_oauth.startup_auth,
    )
    if authorize_mcp_interactively(mcp, config_dir=paths.config_dir):
        await mcp.start()

    thread_id = f"session-{uuid.uuid4().hex[:8]}"
    coding = build_coding_session(
        thread_id=thread_id,
        yolo=yolo,
        vision=settings.supports_vision,
        git_available=git_available,
        approval_handler=render.render_approval_handler,
        question_handler=render.ask_questions,
        paths=paths,
    )
    coding.permission_store.clear_session_rules()

    # Register MCP tools as known but deferred: nothing is bound until
    # /mcp or the model-facing add_tools activates them on this registry.
    mcp_tools = list(mcp.tools.values())
    coding.tool_registry.register_dynamic(mcp_tools)
    coding.tool_registry.set_mcp_catalog(mcp.catalog())

    # Install session context early so idle-path helpers (e.g. MCP arg
    # previews in tool_display) and discover tools share one ToolRegistry.
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

    ui = TuiApp(
        coding,
        mcp=mcp,
        history_path=paths.cli_history,
    )
    render.set_sink(ui)
    # The startup header is rendered by ``TuiApp`` once the transcript pane's
    # real width is known (see ``TuiApp._start_initial_header_task``), so the
    # pre-wrapped TranscriptLines aren't built against a stale fallback width
    # and re-wrap into a half-screen artifact on first render.

    worktree_name = os.environ.get("NESS_AGENT_WORKTREE")
    if worktree_name:
        worktree_path = os.environ.get("NESS_AGENT_WORKTREE_PATH", os.getcwd())
        render.render_notice(
            f"worktree: {worktree_name} @ {worktree_path}",
            title="worktree",
        )

    warning = coding.memory_store.check_health()
    if warning:
        render.render_warning(warning)
    if provider_key_missing():
        render.render_warning(
            f"{settings.model_provider} is not authenticated — use /login to connect."
        )
    budget_warning = _check_prompt_budget(coding, git_available)
    if budget_warning:
        render.render_warning(budget_warning)
    _render_mcp_startup(mcp)
    for oauth_warning in mcp_oauth.warnings:
        render.render_warning(oauth_warning)

    try:
        await ui.run_async(resume_thread_id=resume_thread_id)
    finally:
        resume_thread_id = None
        try:
            await coding.finalize_reflection()
            resume_thread_id = (
                coding.thread_id
                if (settings.auto_save_threads and coding.turn_count > 0)
                else None
            )
            coding.save_thread()
        except Exception as exc:
            render.render_warning(f"Archive skipped: {exc}")
        await mcp.stop()
        from ness_cli.provider.registry import close_providers

        await close_providers()
        render.set_sink(None)
        # The fullscreen Textual TUI has exited, so the transcript sink is no
        # longer on screen. Print the session summary straight to stdout via
        # a standalone Rich console rather than routing through the (dead)
        # in-app transcript widget, otherwise it is silently dropped.
        from rich.panel import Panel

        report = coding.cost_tracker.report()
        if resume_thread_id:
            report += f"\nResume:  ness --resume {resume_thread_id}"
        build_console(file=sys.stdout).print(
            Panel(
                report,
                title="session summary",
                style="usage.value",
                border_style="usage.value",
            )
        )


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "mcp":
        from ness_cli.mcp_cli import app as mcp_app

        mcp_app(args=sys.argv[2:], prog_name="ness mcp")
        return
    app()


if __name__ == "__main__":
    main()
