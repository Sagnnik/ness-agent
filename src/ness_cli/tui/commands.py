"""Slash-command registry and handlers for the kept command set.

Each handler takes the TuiApp and the raw argument string. Session-level
operations delegate to ``app.coding`` (the CodingSession) or the TuiApp
facade; the dispatcher also resolves project-local disk commands
(.ness/commands/*.md).
"""

from __future__ import annotations

import asyncio
import re
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Awaitable, Callable, Literal

import yaml
from langchain_core.messages import HumanMessage

from ness_agent.tools.discover import TOOL_COUNT_WARN_THRESHOLD
from ness_agent.workspace import setup_ness_structure
from ness_agent.workspace.project_context import get_project_context
from ness_cli.chat_model import (
    activate_provider,
    active_provider_id,
    openrouter_session,
)
from ness_cli.provider.profile import provider_profile
from ness_cli.provider.registry import get_provider, provider_ids
from ness_cli.config import settings
from ness_cli.export import ExportError, export_thread_html, resolve_export_path
from ness_cli.prompts import build_init_memory_prompt

from ness_cli.tui import render
from ness_cli.tui.config_flow import run_config
from ness_cli.tui.command_catalog import COMMAND_CATALOG
from ness_cli.tui.models import MenuItem

if TYPE_CHECKING:
    from ness_cli.provider.base import LoginResult, ProviderAdapter
    from ness_cli.tui.app import TuiApp

CommandHandler = Callable[["TuiApp", str], Awaitable[None]]


def _add_text(args: str) -> str | None:
    """Return the text after ``add `` if present, else ``None``."""
    args = args.strip()
    return args[4:].strip() if args.startswith("add ") else None


def _display_skill_source(source: str, project_root: Path) -> str:
    """Return a compact, useful source path for the skill catalog."""
    if not source:
        return ""
    path = Path(source)
    try:
        return str(path.relative_to(project_root))
    except (ValueError, OSError):
        pass
    try:
        return str(Path("~") / path.relative_to(Path.home()))
    except (ValueError, OSError):
        return str(path)


# --- handlers ---------------------------------------------------------------
async def cmd_exit(app: "TuiApp", args: str) -> None:
    app.should_exit = True


async def cmd_help(app: "TuiApp", args: str) -> None:
    rows: list[list[str]] = []
    for spec in COMMAND_CATALOG:
        rows.append([spec.usage or f"/{spec.name}", spec.summary])
    render.render_table(title="commands", columns=["command", "description"], rows=rows)
    render.render_notice("Shift+Tab toggles plan/act mode.")


async def cmd_config(app: "TuiApp", args: str) -> None:
    result = await run_config()
    for message in result.messages:
        render.render_notice(message)
    if result.rebuild:
        app.rebuild_graph()
        await app.refresh_context_snapshot()
    if result.session_update:
        options = app.coding.cfg.options
        defaults = app.coding.agent.config.options
        for target in (options, defaults):
            target.enable_approval = (
                settings.enable_approval and not getattr(target, "yolo_mode", False)
            )
            target.auto_save_threads = settings.auto_save_threads
            target.session_end_reflection = settings.session_end_reflection
        app.coding.thread_store.auto_save = settings.auto_save_threads
        app.render_header()


LoginFlowResult = Literal["complete", "back", "stop"]


def _provider_picker_items() -> list[MenuItem]:
    active_id = active_provider_id()
    items: list[MenuItem] = []
    for provider_id in provider_ids():
        provider = get_provider(provider_id)
        connected = provider.is_authenticated()
        if provider_id == active_id:
            suffix = "active · connected" if connected else "active · not connected"
        else:
            suffix = "connected" if connected else "not connected"
        items.append(
            MenuItem(
                provider_id,
                provider.display_name,
                description=f"{provider.login_description} · {suffix}",
            )
        )
    return items


async def _activate_login_provider(
    app: "TuiApp", provider_id: str, *, force_rebuild: bool = False
) -> str:
    provider = get_provider(provider_id)
    models = await provider.models(refresh=False)
    profile = provider_profile(provider_id)
    preferred_id = str(profile.get("model_name") or "")
    selected = next((model for model in models if model.id == preferred_id), None)
    selected = selected or next(
        (model for model in models if model.is_default),
        models[0] if models else None,
    )
    if selected is None:
        # API-key providers may have a valid persisted profile while their
        # optional catalog cache is still cold/offline. Preserve that known
        # model rather than making authentication depend on catalog refresh.
        if not preferred_id:
            raise RuntimeError(f"{provider.display_name} did not return any models.")
        selected_id = preferred_id
        reasoning_effort = str(profile.get("reasoning_effort") or "") or None
    else:
        selected_id = selected.id
        preferred_effort = str(profile.get("reasoning_effort") or "")
        allowed_efforts = selected.reasoning_efforts
        if allowed_efforts and preferred_effort not in allowed_efforts:
            preferred_effort = (
                selected.default_reasoning_effort
                or ("medium" if "medium" in allowed_efforts else allowed_efforts[0])
            )
        reasoning_effort = preferred_effort or selected.default_reasoning_effort

    changed = active_provider_id() != provider_id
    activate_provider(
        provider_id,
        model_name=selected_id,
        reasoning_effort=reasoning_effort,
    )
    if changed or force_rebuild:
        app.rebuild_graph()
        await app.refresh_context_snapshot()
        app.render_header()
    return selected_id


async def _wait_for_provider_login(
    app: "TuiApp", provider: "ProviderAdapter", login_id: str
) -> "LoginResult | None":
    """Wait for provider completion while keeping an immediate TUI cancel path open."""
    login_task = asyncio.create_task(provider.wait_for_login(login_id))
    cancel_task = asyncio.create_task(
        app.ask_picker(
            f"/login > {provider.display_name} — waiting for sign-in",
            [
                MenuItem(
                    "cancel",
                    "Cancel pending login",
                    description="Return to the current session",
                )
            ],
            initial_key="cancel",
            hint="Enter cancel · Esc cancel",
        )
    )
    try:
        done, _ = await asyncio.wait(
            {login_task, cancel_task}, return_when=asyncio.FIRST_COMPLETED
        )
        if login_task in done:
            # The picker owns a prompt Future, so resolve it before joining the
            # task. Cancelling _ask_question directly would strand the picker.
            if app._cancel_open_prompt():
                await cancel_task
            else:
                cancel_task.cancel()
                with suppress(asyncio.CancelledError):
                    await cancel_task
            return login_task.result()

        login_task.cancel()
        with suppress(asyncio.CancelledError):
            await login_task
        with suppress(Exception):
            await asyncio.wait_for(provider.cancel_login(login_id), timeout=5)
        return None
    finally:
        for task in (login_task, cancel_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(login_task, cancel_task, return_exceptions=True)


async def _authenticate_provider(
    app: "TuiApp", provider: "ProviderAdapter"
) -> LoginFlowResult:
    methods = provider.login_methods()
    if not methods:
        render.render_error(f"{provider.display_name} does not support interactive login.")
        return "stop"

    selected_method = methods[0]
    if len(methods) > 1:
        method_id = await app.ask_picker(
            f"/login > {provider.display_name} — sign in",
            [
                MenuItem(
                    method.id,
                    method.label,
                    description=method.description,
                )
                for method in methods
            ],
            initial_key=next(
                (method.id for method in methods if method.default),
                methods[0].id,
            ),
        )
        if method_id is None:
            return "back"
        selected_method = next(method for method in methods if method.id == method_id)

    secret: str | None = None
    if selected_method.input_kind == "secret":
        secret = (
            await app.ask_secret(
                selected_method.input_label or selected_method.label,
                example=selected_method.input_example,
            )
        ).strip()
        if not secret:
            return "back"

    if selected_method.guidance:
        render.render_warning(selected_method.guidance)
    started = await provider.login(method=selected_method.id, secret=secret)
    if started.status == "cancelled":
        render.render_warning(started.message)
        return "stop"
    if started.status == "error":
        render.render_error(started.message)
        return "stop"
    if started.status == "complete":
        render.render_notice(started.message, title="login")
        return "complete"

    url = started.auth_url or started.verification_url
    details = [started.message]
    if url:
        details.append(url)
    if started.user_code:
        details.append(f"Code: {started.user_code}")
    render.render_notice("\n".join(details), title=f"{provider.display_name} login")
    if url:
        opened = await provider.open_login_url(url)
        if not opened:
            render.render_warning(
                "Automatic browser launch is unavailable. Copy the login URL above "
                "and open it manually. You can cancel the pending login below."
            )
    if not started.login_id:
        render.render_error(started.message)
        return "stop"
    try:
        completed = await _wait_for_provider_login(app, provider, started.login_id)
    except (asyncio.CancelledError, TimeoutError):
        with suppress(Exception):
            await provider.cancel_login(started.login_id)
        if asyncio.current_task() is not None and asyncio.current_task().cancelling():
            raise
        render.render_error(
            f"{provider.display_name} sign-in timed out; the pending login was cancelled."
        )
        return "stop"
    if completed is None:
        render.render_warning(f"{provider.display_name} sign-in was cancelled.")
        return "stop"
    if completed.status != "complete":
        render.render_error(completed.message)
        return "stop"
    return "complete"


async def _logout_with_fallback(
    app: "TuiApp", provider_id: str, previous_active_id: str
) -> None:
    provider = get_provider(provider_id)
    message = await provider.logout()
    render.render_notice(message, title="login")

    candidates: list[str] = []
    if previous_active_id != provider_id:
        candidates.append(previous_active_id)
    candidates.extend(
        candidate
        for candidate in provider_ids()
        if candidate != provider_id and candidate not in candidates
    )
    for candidate in candidates:
        fallback = get_provider(candidate)
        if not fallback.is_authenticated():
            continue
        try:
            model_name = await _activate_login_provider(
                app, candidate, force_rebuild=True
            )
        except Exception as exc:
            render.render_warning(
                f"Could not switch to {fallback.display_name}: {exc}"
            )
            continue
        render.render_notice(
            f"Switched to {fallback.display_name} with {model_name}.",
            title="login",
        )
        return

    # Rebuild so a model object cannot retain credentials after logout. The
    # thread remains intact and future calls surface the normal login-required error.
    app.rebuild_graph()
    await app.refresh_context_snapshot()
    app.render_header()
    render.render_warning(
        "No connected model provider remains. The current thread was preserved; "
        "run /login before sending another message."
    )


async def _manage_connected_provider(
    app: "TuiApp", provider_id: str, previous_active_id: str
) -> LoginFlowResult:
    provider = get_provider(provider_id)
    while True:
        action = await app.ask_picker(
            f"/login > {provider.display_name} — connected",
            [
                MenuItem(
                    "reconnect",
                    "Reconnect",
                    description="Replace the current credentials",
                ),
                MenuItem(
                    "logout",
                    "Log out",
                    description="Remove the saved credentials",
                ),
            ],
            initial_key="reconnect",
        )
        if action is None:
            return "back"
        if action == "logout":
            await _logout_with_fallback(app, provider_id, previous_active_id)
            return "complete"
        result = await _authenticate_provider(app, provider)
        if result == "back":
            continue
        if result == "complete":
            model_name = await _activate_login_provider(
                app, provider_id, force_rebuild=True
            )
            render.render_notice(
                f"{provider.display_name} reconnected with {model_name}. "
                "The current thread was preserved.",
                title="login",
            )
        return result


async def cmd_login(app: "TuiApp", args: str) -> None:
    del args
    while True:
        provider_id = await app.ask_picker(
            "/login — model providers",
            _provider_picker_items(),
            initial_key=active_provider_id(),
            hint="↑/↓ select · Enter open · Esc close",
        )
        if provider_id is None:
            return
        provider = get_provider(provider_id)
        previous_active_id = active_provider_id()

        if provider.is_authenticated():
            if provider_id != previous_active_id:
                try:
                    await _activate_login_provider(app, provider_id)
                except Exception as exc:
                    render.render_error(
                        f"Could not activate {provider.display_name}: {exc}"
                    )
                    continue
            result = await _manage_connected_provider(
                app, provider_id, previous_active_id
            )
            if result == "back":
                continue
            return

        result = await _authenticate_provider(app, provider)
        if result == "back":
            continue
        if result == "complete":
            try:
                model_name = await _activate_login_provider(
                    app, provider_id, force_rebuild=True
                )
            except Exception as exc:
                render.render_error(
                    f"Signed in to {provider.display_name}, but could not activate it: {exc}"
                )
                return
            render.render_notice(
                f"{provider.display_name} is active with {model_name}. "
                "The current thread was preserved.",
                title="login",
            )
        return


def _window_label(minutes: int | None) -> str:
    if minutes == 300:
        return "5-hour"
    if minutes == 10_080:
        return "weekly"
    if minutes == 43_200:
        return "monthly"
    if minutes is None:
        return "unknown window"
    if minutes % 1_440 == 0:
        return f"{minutes // 1_440}-day"
    if minutes % 60 == 0:
        return f"{minutes // 60}-hour"
    return f"{minutes}-minute"


def _reset_label(timestamp: int | None) -> str:
    if timestamp is None:
        return "reset unknown"
    reset = datetime.fromtimestamp(timestamp, tz=timezone.utc).astimezone()
    seconds = max(0, int((reset - datetime.now().astimezone()).total_seconds()))
    if seconds < 60:
        relative = "in under a minute"
    elif seconds < 3_600:
        relative = f"in {seconds // 60}m"
    elif seconds < 86_400:
        relative = f"in {seconds // 3_600}h {seconds % 3_600 // 60}m"
    else:
        relative = f"in {seconds // 86_400}d {seconds % 86_400 // 3_600}h"
    return f"{reset.strftime('%Y-%m-%d %H:%M %Z')} ({relative})"


async def cmd_status(app: "TuiApp", args: str) -> None:
    provider = get_provider(app.coding.provider_id)
    try:
        provider_status = await provider.status(refresh=False)
    except Exception as exc:
        provider_status = None
        render.render_warning(f"Provider status unavailable: {exc}")
    tracker = app.coding.cost_tracker
    input_tokens = int(tracker.input_tokens or 0)
    cached = int(tracker.cached_input_tokens or 0)
    cache_write = int(getattr(tracker, "cache_write_input_tokens", 0) or 0)
    cache_hit = cached / input_tokens if input_tokens else None
    lines = [
        f"provider       {provider.display_name}",
        f"session id     {app.thread_id}",
        f"model          {app.coding.model_name}",
        f"reasoning      {app.coding.reasoning_effort}",
        f"input tokens   {input_tokens:,}",
        f"output tokens  {int(tracker.output_tokens or 0):,}",
        (
            "cost           subscription"
            if provider.billing_label == "subscription"
            else (f"cost           ${tracker.cost_usd:.4f}" if tracker.cost_usd > 0 else "cost           unknown")
        ),
        f"turns          {int(app.turn_count or 0)}",
        f"cache read     {cached:,}",
        f"cache write    {cache_write:,}",
        f"cache hit      {cache_hit:.0%}" if cache_hit is not None else "cache hit      n/a",
    ]
    if app.coding.provider_id == "openrouter":
        lines.append(f"openrouter id  {openrouter_session(app.thread_id) or 'not set'}")
    if provider_status is not None:
        lines.extend(
            [
                f"auth           {'connected' if provider_status.auth.authenticated else 'signed out'} ({provider_status.auth.method})",
                f"email          {provider_status.account.email or 'unavailable'}",
                f"subscription   {provider_status.account.tier or 'unavailable'}",
            ]
        )
        has_weekly = False
        for bucket in provider_status.limits:
            window = _window_label(bucket.window_minutes)
            has_weekly = has_weekly or bucket.window_minutes == 10_080
            state = "LIMIT REACHED" if bucket.reached else "available"
            lines.append(
                f"limit          {bucket.name} · {window} · {bucket.used_percent or 0:.0f}% used / {bucket.remaining_percent or 0:.0f}% remaining · {state}"
            )
            lines.append(f"reset          {_reset_label(bucket.resets_at)}")
        if active_provider_id() == "codex" and not has_weekly:
            lines.append("weekly limit   unavailable")
        if provider_status.credits is not None:
            lines.append(f"reset credits  {provider_status.credits}")
        if provider_status.warning:
            lines.append(f"provider note  {provider_status.warning}")
    render.render_panel_text("\n".join(lines), title="session status", style="usage.value")


async def cmd_skill(app: "TuiApp", args: str) -> None:
    name = args.strip()
    loader = app.coding.skill_loader
    skills = loader.load()
    if not name:
        if not skills:
            render.render_notice("No skills found.")
        else:
            rows = [
                [
                    s.get("name", ""),
                    _display_skill_source(
                        str(s.get("source", "")), app.coding.project_root
                    ),
                    s.get("description", ""),
                ]
                for s in sorted(
                    skills.values(), key=lambda item: str(item.get("name", ""))
                )
            ]
            render.render_table(
                title=f"skills ({len(rows)})",
                columns=["skill", "source", "description"],
                rows=rows,
            )
        if loader.errors:
            render.render_warning("Skill load warnings:\n" + "\n".join(loader.errors))
        render.render_notice("Load a skill with /skill <name>.")
        return
    if name not in skills:
        render.render_error(f"Unknown skill: {name}  (/skill to list)")
        return
    app.coding.stage_skills([name])
    render.render_notice(f"Skill '{name}' will load on your next message.", title="skill")


async def cmd_init(app: "TuiApp", args: str) -> None:
    from ness_cli.paths import ensure_global_config, resolve_paths

    paths = resolve_paths(
        project_root=app.coding.project_root,
        ness_dir=app.coding.ness_dir,
    )
    created = setup_ness_structure(app.coding.ness_dir)
    created.extend(ensure_global_config(paths))
    if created:
        render.render_notice(
            f"Initialized .ness/ + global config ({', '.join(created)})",
            title="init",
        )
    else:
        render.render_notice(".ness/ structure already present", title="init")


async def cmd_memory(app: "TuiApp", args: str) -> None:
    memory = app.coding.memory_store
    raw = args.strip()
    if raw.startswith("create"):
        rest = raw[6:].strip()
        force = rest in ("force", "--force")
        if rest and not force:
            render.render_error("Usage: /memory create [force]")
            return
        with render.thinking("generating NESS.md"):
            response = await app.model.ainvoke(
                [HumanMessage(content=build_init_memory_prompt(
                    get_project_context(),
                    instructions_dir=app.coding.instructions_dir,
                ))]
            )
        result = memory.write_project(str(response.content), overwrite=force)
        if result.startswith("Error:"):
            render.render_error(result)
        else:
            render.render_notice(result, title="memory")
        return

    text = _add_text(args)
    if text is None:
        if not raw:
            render.render_panel_text(memory.load_project() or "(empty)", title=str(memory.ness_file), style="usage.value")
            return
        render.render_error("Usage: /memory or /memory add <note> or /memory create [force]")
        return
    render.render_notice(memory.append_project(text))


async def cmd_user(app: "TuiApp", args: str) -> None:
    memory = app.coding.memory_store
    text = _add_text(args)
    if text is None:
        if not args.strip():
            render.render_panel_text(memory.load_user() or "(empty)", title=str(memory.user_file), style="usage.value")
            return
        render.render_error("Usage: /user or /user add <preference>")
        return
    render.render_notice(memory.append_user(text))


async def cmd_permissions(app: "TuiApp", args: str) -> None:
    permission_store = app.coding.permission_store
    parts = args.split()
    if not parts or parts[0] == "list":
        render.render_panel_text(permission_store.list_rules(), title="permissions", style="usage.value")
        return
    if len(parts) >= 2 and parts[0] in {"allow", "deny"}:
        permission_store.persist_rule(" ".join(parts[1:]), parts[0])
        render.render_notice(f"Added {parts[0]} rule.")
        return
    if len(parts) == 3 and parts[0] == "remove" and parts[1] in {"allow", "deny"}:
        try:
            removed = permission_store.remove_rule(parts[1], int(parts[2]))
            render.render_notice(f"Removed {removed}")
        except ValueError as exc:
            render.render_error(str(exc))
        return
    render.render_error("Usage: /permissions [list | allow <pattern> | deny <pattern> | remove <allow|deny> <index>]")


async def cmd_hooks(app: "TuiApp", args: str) -> None:
    render.render_panel_text(app.coding.hook_runner.describe(), title="hooks", style="usage.value")


async def cmd_mcp(app: "TuiApp", args: str) -> None:
    if app.mcp is None:
        render.render_error("/mcp is unavailable: no MCP manager configured.")
        return
    parts = args.split()
    if not parts:
        render.render_panel_text(app.mcp.status(), title="mcp", style="usage.value")
        return

    server = parts[0]
    catalog = app.mcp.catalog()
    server_info = catalog.get(server)
    if server_info is None:
        render.render_error(f"Unknown MCP server: {server}  (/mcp for status)")
        return

    entries = server_info.get("tools", [])
    if len(parts) >= 2:
        tool_short = parts[1]
        wanted = [e["name"] for e in entries if e.get("tool") == tool_short]
        if not wanted:
            render.render_error(f"Unknown tool '{tool_short}' on server '{server}'.")
            return
    else:
        wanted = [e["name"] for e in entries]

    added, unknown = app.coding.tool_registry.activate_mcp(wanted)
    if added:
        render.render_notice(
            f"Loaded {len(added)} tool(s) from {server}: {', '.join(sorted(added))}",
            title="mcp",
        )
    else:
        render.render_notice(f"No new tools loaded from {server}.", title="mcp")
    if unknown:
        render.render_warning(f"Skipped unknown: {', '.join(sorted(set(unknown)))}")

    total = len(app.coding.tool_registry.tool_names())
    if total > TOOL_COUNT_WARN_THRESHOLD:
        render.render_warning(
            f"{total} tools now loaded (> {TOOL_COUNT_WARN_THRESHOLD}); "
            "tool-selection accuracy may degrade."
        )


def _thread_rows(threads: list[dict], store) -> list[list[str]]:
    rows: list[list[str]] = []
    for item in threads:
        input_tokens = int(item.get("input_tokens", 0) or 0)
        cached = int(item.get("cached_input_tokens", 0) or 0)
        cache_hit = cached / input_tokens if input_tokens else 0.0
        label = (
            item.get("name")
            or item.get("summary")
            or store.first_user_message(item.get("thread_id", ""))
            or "(no messages)"
        )
        if "archived_at" not in item:
            label = f"{label} (active)"
        rows.append(
            [
                item.get("thread_id", ""),
                label,
                str(item.get("turn_count", 0)),
                f"${float(item.get('total_cost_usd', 0.0)):.4f}",
                f"{cache_hit:.0%}",
            ]
        )
    return rows


async def cmd_threads(app: "TuiApp", args: str) -> None:
    store = app.coding.thread_store
    threads = store.list_threads(100)
    live_statuses = app.active_thread_statuses()
    known_ids = {str(item.get("thread_id") or "") for item in threads}
    for thread_id in live_statuses:
        if thread_id not in known_ids:
            threads.append({"thread_id": thread_id, "updated_at": ""})
    if not threads:
        render.render_notice("No saved sessions.")
        return
    sink = render.get_sink()
    if sink is None:
        render.render_error("/threads requires the interactive TUI.")
        return
    for item in threads:
        item["live_status"] = live_statuses.get(
            str(item.get("thread_id") or "")
        )
        item["label"] = (
            item.get("name")
            or item.get("summary")
            or store.first_user_message(item.get("thread_id", ""))
            or "(no messages)"
        )
    target = await sink.request_threads_picker(
        threads,
        current_thread_id=app.thread_id,
    )
    if target and target != app.thread_id:
        await app.resume_thread(target)


async def cmd_rename(app: "TuiApp", args: str) -> None:
    name = args.strip()
    if not name:
        render.render_error("Usage: /rename <name>")
        return
    try:
        saved = app.coding.set_name(name)
    except ValueError as exc:
        render.render_error(str(exc))
        return
    if not saved:
        render.render_warning("Thread autosave is disabled; session name was not saved.")
        return
    normalized = " ".join(name.split())
    render.render_notice(f"Session renamed to {normalized}", title="rename")


async def cmd_save(app: "TuiApp", args: str) -> None:
    render.render_notice(app.save_thread(), title="save")


async def cmd_new(app: "TuiApp", args: str) -> None:
    await app.reset_thread()
    render.render_notice("Started a fresh thread.")


async def cmd_compact(app: "TuiApp", args: str) -> None:
    app.request_compact()
    render.render_notice("Compaction will run on the next model turn.")


async def cmd_reflection(app: "TuiApp", args: str) -> None:
    if args.strip():
        render.render_error("Usage: /reflection")
        return
    result = await app.run_reflection()
    if result.bullets:
        render.render_notice(
            "\n".join(f"- {bullet}" for bullet in result.bullets),
            title="reflection",
        )
    elif not result.error:
        message = (
            "Reflection completed; no new memory bullets."
            if result.message_index is not None
            else "Nothing new to reflect on."
        )
        render.render_notice(message, title="reflection")
    if result.error:
        render.render_error(result.error)


async def cmd_export(app: "TuiApp", args: str) -> None:
    try:
        destination = resolve_export_path(args, app.coding.project_root)
        result = await asyncio.to_thread(
            export_thread_html,
            thread_store=app.coding.thread_store,
            thread_id=app.thread_id,
            project_root=app.coding.project_root,
            destination=destination,
        )
    except ExportError as exc:
        render.render_error(str(exc))
        return
    render.render_notice(
        f"Exported {result.event_count} entries to {result.path}",
        title="export",
    )


async def cmd_clear(app: "TuiApp", args: str) -> None:
    """Clear only the rendered transcript; preserve the live conversation."""
    app.clear_transcript()


async def cmd_copy(app: "TuiApp", args: str) -> None:
    history = app.assistant_history
    if not history:
        render.render_warning("No assistant message to copy.")
        return
    text = history[-1]
    args = args.strip()
    if args == "code":
        blocks = re.findall(r"```(?:\w+)?\n(.*?)```", text, re.DOTALL)
        if blocks:
            text = blocks[-1]
    elif args.isdigit():
        idx = int(args)
        if 1 <= idx <= len(history):
            text = history[-idx]
    try:
        import pyperclip

        pyperclip.copy(text)
        render.render_notice("Copied to clipboard.")
    except Exception:
        render.render_panel_text(text, title="clipboard unavailable", style="usage.value")


async def cmd_rollback(app: "TuiApp", args: str) -> None:
    """Roll the current thread back to a prior user turn.

    Usage:
      /rollback                open a picker of every user message in this thread
      /rollback <seq>          roll back directly to the user message at seq N
                               (use /status or the picker to find the seq)

    Restores agent-modified files (git snapshot), the per-thread session
    memory file, and truncates the durable events tail at the chosen user
    message. The in-process cost tracker is intentionally preserved.
    """
    arg = args.strip()
    if arg.isdigit():
        await app.rollback_to(int(arg))
        return

    sink = render.get_sink()
    if sink is None:
        render.render_error("/rollback picker requires the interactive TUI; use /rollback <seq>.")
        return

    turns = app.coding.thread_store.list_user_turns(app.thread_id)
    if not turns:
        render.render_notice("No user turns in this thread to roll back to.")
        return

    seq_str = await sink.request_rollback_picker(turns)
    if not seq_str:
        return  # user cancelled the picker
    try:
        seq = int(seq_str)
    except ValueError:
        render.render_error(f"Invalid rollback seq: {seq_str!r}")
        return
    await app.rollback_to(seq)


async def cmd_fork(app: "TuiApp", args: str) -> None:
    turns = app.coding.thread_store.list_user_turns(app.thread_id)
    if not turns:
        render.render_notice("No user turns in this thread to fork from.")
        return
    sink = render.get_sink()
    if sink is None:
        render.render_error("/fork requires the interactive TUI.")
        return
    seq_str = await sink.request_fork_picker(turns)
    if not seq_str:
        return
    try:
        await app.fork_thread(int(seq_str))
    except ValueError as exc:
        render.render_error(str(exc))


async def cmd_goal(app: "TuiApp", args: str) -> None:
    goal = args.strip()
    if not goal:
        render.render_error("Usage: /goal <objective>")
        return
    await app.run_goal(goal)


HANDLERS: dict[str, CommandHandler] = {
    "exit": cmd_exit,
    "quit": cmd_exit,
    "help": cmd_help,
    "login": cmd_login,
    "config": cmd_config,
    "status": cmd_status,
    "skill": cmd_skill,
    "init": cmd_init,
    "memory": cmd_memory,
    "user": cmd_user,
    "permissions": cmd_permissions,
    "hooks": cmd_hooks,
    "mcp": cmd_mcp,
    "threads": cmd_threads,
    "rename": cmd_rename,
    "fork": cmd_fork,
    "goal": cmd_goal,
    "save": cmd_save,
    "new": cmd_new,
    "compact": cmd_compact,
    "reflection": cmd_reflection,
    "export": cmd_export,
    "clear": cmd_clear,
    "copy": cmd_copy,
    "rollback": cmd_rollback,
}

# Slash commands safe to run while a task is streaming: read-only or file-write
# side effects that do not touch the live graph or thread state.
# Exception: /memory create invokes the chat model and is refused when busy
# (see ``dispatch``); /memory read and /memory add remain allowed.
BUSY_SAFE_COMMANDS: frozenset[str] = frozenset(
    {
        "help",
        "status",
        "permissions",
        "hooks",
        "mcp",
        "copy",
        "memory",
        "user",
        "skill",
        "rename",
        "threads",
        "new",
    }
)


async def dispatch(app: "TuiApp", command_line: str, *, busy: bool = False) -> None:
    raw = command_line[1:].strip()
    if not raw:
        return
    name, _, args = raw.partition(" ")
    name = name.lower()

    handler = HANDLERS.get(name)
    if handler is not None:
        if busy and name not in BUSY_SAFE_COMMANDS:
            render.render_warning(f"/{name} is not available while a task is running")
            return
        if busy and name == "memory" and args.strip().startswith("create"):
            render.render_warning("/memory create is not available while a task is running")
            return
        await handler(app, args)
        return

    disk_command = _load_disk_commands().get(name)
    if disk_command is not None:
        app.queued_prompt = disk_command.replace("{{args}}", args)
        return

    if busy:
        render.render_warning(f"/{name} is not available while a task is running")
        return
    render.render_error(f"Unknown command: /{name}  (try /help)")


def _load_disk_commands() -> dict[str, str]:
    commands_dir = Path(settings.ness_dir) / "commands"
    if not commands_dir.exists():
        return {}
    commands: dict[str, str] = {}
    for path in commands_dir.glob("*.md"):
        text = path.read_text(encoding="utf-8")
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) == 3:
                yaml.safe_load(parts[1]) or {}
                text = parts[2]
        commands[path.stem] = text.strip()
    return commands
