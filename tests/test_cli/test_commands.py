from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from ness_cli.config import settings
from ness_cli.provider.base import LoginMethod, LoginResult

from ness_cli.tui import render
from ness_cli.tui.commands import (
    _logout_with_fallback,
    _provider_picker_items,
    _wait_for_provider_login,
    cmd_login,
    dispatch,
)
from ness_cli.tui.config_flow import ConfigResult
from ness_cli.tui.models import MenuItem


async def _dispatch_with_sink(app, command: str) -> None:
    render.set_sink(app)
    try:
        await dispatch(app, command)
    finally:
        render.set_sink(None)


def test_help_command_lists_supported_commands(make_app):
    app = make_app()
    asyncio.run(_dispatch_with_sink(app, "/help"))
    text = "\n".join(line.text for line in app._lines)
    assert "commands" in text
    assert "/config" in text
    assert "/status" in text
    assert "/reflection" in text
    assert "/clear" in text
    assert "/menu" not in text
    assert "/cost" not in text
    assert "/cache" not in text
    assert "/skills" not in text
    assert "/image" not in text


def test_reflection_command_runs_and_displays_new_bullets(make_app):
    app = make_app()

    asyncio.run(_dispatch_with_sink(app, "/reflection"))

    assert app.coding.reflection_runs == 1
    text = "\n".join(line.text for line in app._lines)
    assert "Captured the latest work" in text


def test_reflection_command_is_not_available_while_busy(make_app):
    app = make_app()

    render.set_sink(app)
    try:
        asyncio.run(dispatch(app, "/reflection", busy=True))
    finally:
        render.set_sink(None)

    assert app.coding.reflection_runs == 0
    assert "not available while a task is running" in "\n".join(
        line.text for line in app._lines
    )


def test_skill_catalog_uses_compact_sources_and_wrapped_table(
    make_app, tmp_path
):
    app = make_app()
    app._on_transcript_render_size(72, 20)
    app.coding.project_root = tmp_path
    app.coding.skill_loader = SimpleNamespace(
        errors=[],
        load=lambda: {
            "long-skill-name": {
                "name": "long-skill-name",
                "source": str(
                    tmp_path / ".agents" / "skills" / "long-skill-name" / "SKILL.md"
                ),
                "description": "A long skill description that should wrap cleanly inside its column.",
            }
        },
    )

    asyncio.run(_dispatch_with_sink(app, "/skill"))

    lines = [line.text for line in app._lines]
    text = "\n".join(lines)
    assert "skills (1)" in text
    assert ".agents/skills" in text
    assert "SKILL.md" in text
    assert str(tmp_path) not in text
    assert max(map(len, lines)) <= 72


def test_dispatch_exit_sets_session_flag(make_app):
    app = make_app()
    asyncio.run(_dispatch_with_sink(app, "/exit"))
    assert app.should_exit is True


def test_clear_command_only_clears_rendered_transcript(make_app):
    app = make_app()
    app.coding.turn_count = 2
    app.coding.thread_store.events[app.thread_id] = [
        {"kind": "user", "content": "remember this"}
    ]
    app.assistant_history.append("remembered answer")
    app.append_user("visible question")
    app.append_assistant("visible answer")

    asyncio.run(_dispatch_with_sink(app, "/clear"))

    assert app._lines == []
    assert app.coding.turn_count == 2
    assert app.coding.thread_store.events[app.thread_id] == [
        {"kind": "user", "content": "remember this"}
    ]
    assert app.assistant_history == ["remembered answer"]


def test_export_command_writes_html_and_refuses_overwrite(make_app, tmp_path):
    app = make_app()
    app.coding.project_root = tmp_path
    app.coding.thread_store.events[app.thread_id] = [
        {"kind": "user", "content": "export this", "t": "2026-01-01T00:00:00+00:00"},
        {"kind": "assistant", "content": "exported", "t": "2026-01-01T00:00:01+00:00"},
    ]
    destination = tmp_path / "nested" / "My export.html"

    asyncio.run(_dispatch_with_sink(app, f'/export "{destination}"'))

    assert destination.is_file()
    transcript = destination.read_text(encoding="utf-8")
    assert "export this" in transcript
    text = "\n".join(line.text for line in app._lines)
    assert "Exported 2 entries" in text

    asyncio.run(_dispatch_with_sink(app, f'/export "{destination}"'))
    text = "\n".join(line.text for line in app._lines)
    assert "Refusing to overwrite" in text


def test_export_command_validates_persistence_and_usage(make_app, tmp_path):
    app = make_app()
    app.coding.project_root = tmp_path

    asyncio.run(_dispatch_with_sink(app, "/export"))
    assert "Usage: /export <path.html>" in "\n".join(line.text for line in app._lines)

    app._lines.clear()
    asyncio.run(_dispatch_with_sink(app, "/export transcript.html"))
    assert "no durable events" in "\n".join(line.text for line in app._lines)

    app._lines.clear()
    app.coding.thread_store.auto_save = False
    asyncio.run(_dispatch_with_sink(app, "/export transcript.html"))
    assert "autosave is disabled" in "\n".join(line.text for line in app._lines)


def test_export_command_is_not_available_while_busy(make_app, tmp_path):
    app = make_app()
    app.coding.project_root = tmp_path
    app.coding.thread_store.events[app.thread_id] = [
        {"kind": "user", "content": "still running"}
    ]

    asyncio.run(_dispatch_busy(app, "/export busy.html"))

    assert not (tmp_path / "busy.html").exists()
    assert "/export is not available while a task is running" in "\n".join(
        line.text for line in app._lines
    )


def test_status_command_shows_session_summary(make_app):
    app = make_app()
    asyncio.run(_dispatch_with_sink(app, "/status"))
    text = "\n".join(line.text for line in app._lines)
    assert "session status" in text
    assert "cache read" in text
    assert "cache write" in text


def test_rename_command_sets_normalized_session_name(make_app):
    app = make_app()
    asyncio.run(_dispatch_with_sink(app, "/rename   Release   prep  "))

    assert app.coding.thread_store.names[app.thread_id] == "Release prep"
    assert "Session renamed to Release prep" in "\n".join(line.text for line in app._lines)


def test_rename_command_requires_name_and_is_busy_safe(make_app):
    app = make_app()
    asyncio.run(_dispatch_with_sink(app, "/rename"))
    assert "Usage: /rename <name>" in "\n".join(line.text for line in app._lines)

    asyncio.run(_dispatch_busy(app, "/rename Busy name"))
    assert app.coding.thread_store.names[app.thread_id] == "Busy name"


def test_login_uses_native_picker_without_question_chrome(make_app):
    app = make_app()

    async def exercise():
        task = asyncio.create_task(
            app.ask_picker(
                "/login — model providers",
                [MenuItem("codex", "Codex")],
                initial_key="codex",
            )
        )
        await asyncio.sleep(0)
        assert app._prompt_kind == "picker"
        assert app._menu_kind == "picker"
        assert app._form_visible() is False
        assert app._prompt_title == "/login — model providers"
        assert app._prompt_hint == "↑/↓ select · Enter confirm · Esc back"
        assert app._buffer.text == "/login"
        prefix_text = "".join(text for _, text in app._input_prefix_fragments())
        assert prefix_text == "act > "
        assert prefix_text + app._buffer.text == "act > /login"
        assert "question" not in app._prompt_title
        assert "note" not in app._prompt_hint.lower()
        app._apply_picker_selection()
        return await task

    assert asyncio.run(exercise()) == "codex"
    assert app._prompt_kind is None


def test_login_provider_picker_keeps_full_provider_name(make_app):
    app = make_app()
    provider = SimpleNamespace(
        id="codex",
        display_name="Codex subscription",
        login_description="ChatGPT subscription",
        is_authenticated=lambda: True,
    )

    with (
        patch("ness_cli.tui.commands.provider_ids", return_value=("codex",)),
        patch("ness_cli.tui.commands.get_provider", return_value=provider),
        patch("ness_cli.tui.commands.active_provider_id", return_value="codex"),
    ):
        item = _provider_picker_items()[0]

    assert item.label == "Codex subscription"
    assert item.suffix == ""
    assert item.description == "ChatGPT subscription · active · connected"
    row_text = "".join(text for _, text in app._menu_row_fragments(item, selected=True))
    assert "Codex subscription" in row_text
    assert "Code…" not in row_text


def test_signed_out_codex_skips_connect_and_goes_to_methods(make_app):
    app = make_app()
    provider = SimpleNamespace(
        id="codex",
        display_name="Codex subscription",
        login_description="ChatGPT subscription",
        is_authenticated=lambda: False,
        login_methods=lambda: (
            LoginMethod("browser", "Browser sign-in", default=True),
            LoginMethod("device", "Device code"),
        ),
        login=AsyncMock(return_value=LoginResult("complete", "signed in")),
    )
    picker = AsyncMock(side_effect=["codex", "browser"])
    app.ask_picker = picker

    with (
        patch("ness_cli.tui.commands.provider_ids", return_value=("codex",)),
        patch("ness_cli.tui.commands.get_provider", return_value=provider),
        patch("ness_cli.tui.commands.active_provider_id", return_value="openrouter"),
        patch(
            "ness_cli.tui.commands._activate_login_provider",
            new_callable=AsyncMock,
            return_value="gpt-test",
        ),
    ):
        asyncio.run(cmd_login(app, ""))

    assert picker.await_count == 2
    method_items = picker.await_args_list[1].args[1]
    assert [item.key for item in method_items] == ["browser", "device"]
    assert all(item.key != "connect" for item in method_items)
    provider.login.assert_awaited_once_with(method="browser", secret=None)


def test_signed_out_openrouter_goes_directly_to_masked_key(make_app):
    app = make_app()
    provider = SimpleNamespace(
        id="openrouter",
        display_name="OpenRouter",
        login_description="API key",
        is_authenticated=lambda: False,
        login_methods=lambda: (
            LoginMethod(
                "api_key",
                "API key",
                input_kind="secret",
                input_label="OpenRouter API key",
                input_example="sk-or-v1-...",
            ),
        ),
        login=AsyncMock(return_value=LoginResult("complete", "saved")),
    )
    app.ask_picker = AsyncMock(return_value="openrouter")
    app.ask_secret = AsyncMock(return_value="sk-or-test")

    with (
        patch("ness_cli.tui.commands.provider_ids", return_value=("openrouter",)),
        patch("ness_cli.tui.commands.get_provider", return_value=provider),
        patch("ness_cli.tui.commands.active_provider_id", return_value="codex"),
        patch(
            "ness_cli.tui.commands._activate_login_provider",
            new_callable=AsyncMock,
            return_value="openai/test",
        ),
    ):
        asyncio.run(cmd_login(app, ""))

    app.ask_picker.assert_awaited_once()
    app.ask_secret.assert_awaited_once_with(
        "OpenRouter API key", example="sk-or-v1-..."
    )
    provider.login.assert_awaited_once_with(
        method="api_key", secret="sk-or-test"
    )


def test_connected_provider_only_offers_reconnect_and_logout(make_app):
    app = make_app()
    provider = SimpleNamespace(
        id="codex",
        display_name="Codex subscription",
        login_description="ChatGPT subscription",
        is_authenticated=lambda: True,
    )
    picker = AsyncMock(side_effect=["codex", None, None])
    app.ask_picker = picker

    with (
        patch("ness_cli.tui.commands.provider_ids", return_value=("codex",)),
        patch("ness_cli.tui.commands.get_provider", return_value=provider),
        patch("ness_cli.tui.commands.active_provider_id", return_value="codex"),
    ):
        asyncio.run(cmd_login(app, ""))

    action_items = picker.await_args_list[1].args[1]
    assert [item.key for item in action_items] == ["reconnect", "logout"]
    assert all(item.key not in {"connect", "use"} for item in action_items)


def test_active_logout_prefers_previously_active_connected_provider(make_app):
    app = make_app()
    selected = SimpleNamespace(
        display_name="Codex subscription",
        logout=AsyncMock(return_value="signed out"),
    )
    previous = SimpleNamespace(
        display_name="OpenRouter",
        is_authenticated=lambda: True,
    )
    activate = AsyncMock(return_value="openai/test")

    with (
        patch(
            "ness_cli.tui.commands.get_provider",
            side_effect=lambda provider_id: {
                "codex": selected,
                "openrouter": previous,
            }[provider_id],
        ),
        patch(
            "ness_cli.tui.commands.provider_ids",
            return_value=("codex", "openrouter"),
        ),
        patch(
            "ness_cli.tui.commands._activate_login_provider",
            activate,
        ),
    ):
        asyncio.run(_logout_with_fallback(app, "codex", "openrouter"))

    selected.logout.assert_awaited_once()
    activate.assert_awaited_once_with(app, "openrouter", force_rebuild=True)


def test_active_logout_without_fallback_preserves_thread_but_rebuilds(make_app):
    app = make_app()
    selected = SimpleNamespace(
        display_name="Codex subscription",
        logout=AsyncMock(return_value="signed out"),
    )
    app.rebuild_graph = Mock()
    app.refresh_context_snapshot = AsyncMock()
    app.render_header = Mock()

    with (
        patch("ness_cli.tui.commands.get_provider", return_value=selected),
        patch("ness_cli.tui.commands.provider_ids", return_value=("codex",)),
    ):
        asyncio.run(_logout_with_fallback(app, "codex", "codex"))

    app.rebuild_graph.assert_called_once_with()
    app.refresh_context_snapshot.assert_awaited_once_with()
    app.render_header.assert_called_once_with()


def test_pending_provider_login_can_be_cancelled_immediately(make_app):
    app = make_app()

    class PendingProvider:
        display_name = "Test subscription"

        def __init__(self):
            self.cancelled = False
            self.wait_cancelled = False

        async def wait_for_login(self, login_id):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.wait_cancelled = True
                raise

        async def cancel_login(self, login_id):
            self.cancelled = True

    async def exercise():
        provider = PendingProvider()
        task = asyncio.create_task(
            _wait_for_provider_login(app, provider, "login-1")  # type: ignore[arg-type]
        )
        for _ in range(10):
            if app._prompt_future is not None:
                break
            await asyncio.sleep(0)
        assert app._cancel_open_prompt() is True
        result = await task
        return provider, result

    provider, result = asyncio.run(exercise())

    assert result is None
    assert provider.cancelled is True
    assert provider.wait_cancelled is True
    assert app._prompt_kind is None


def test_completed_provider_login_dismisses_cancel_prompt(make_app):
    app = make_app()

    class CompletedProvider:
        display_name = "Test subscription"

        async def wait_for_login(self, login_id):
            await asyncio.sleep(0)
            return LoginResult("error", "sign-in was rejected")

        async def cancel_login(self, login_id):
            raise AssertionError("completed login must not be cancelled")

    result = asyncio.run(
        _wait_for_provider_login(app, CompletedProvider(), "login-1")  # type: ignore[arg-type]
    )

    assert result == LoginResult("error", "sign-in was rejected")
    assert app._prompt_kind is None


def test_config_session_toggles_update_active_runtime(make_app):
    app = make_app()
    options = SimpleNamespace(
        enable_approval=True,
        yolo_mode=False,
        auto_save_threads=True,
        session_end_reflection=False,
    )
    app.coding.cfg = SimpleNamespace(options=options)
    app.coding.thread_store.auto_save = True
    old = (
        settings.enable_approval,
        settings.auto_save_threads,
        settings.session_end_reflection,
    )
    settings.enable_approval = False
    settings.auto_save_threads = False
    settings.session_end_reflection = True
    try:
        with patch(
            "ness_cli.tui.commands.run_config",
            new_callable=AsyncMock,
            return_value=ConfigResult(session_update=True),
        ):
            asyncio.run(_dispatch_with_sink(app, "/config"))
    finally:
        (
            settings.enable_approval,
            settings.auto_save_threads,
            settings.session_end_reflection,
        ) = old

    assert options.enable_approval is False
    assert app.coding.agent.config.options.enable_approval is False
    assert options.auto_save_threads is False
    assert options.session_end_reflection is True
    assert app.coding.thread_store.auto_save is False


def test_config_action_can_update_persisted_setting(make_app, tmp_path, monkeypatch):
    monkeypatch.setenv("NESS_AGENT_CONFIG_DIR", str(tmp_path / "cfg"))
    from ness_cli.config_store import load_configs

    app = make_app()
    previous = settings.enable_approval
    settings.enable_approval = True
    try:
        app._config_section = "behavior"
        app._open_picker("config_section", "/config", index=0)
        items = app._visible_menu_items()
        app._menu_index = next(i for i, item in enumerate(items) if item.key == "enable_approval")
        app._apply_picker_selection()
        assert settings.enable_approval is False
        # Bool toggles stay in the section menu and persist to configs.json.
        assert app._menu_kind == "config_section"
        assert load_configs()["enable_approval"] is False
    finally:
        settings.enable_approval = previous


def test_rollback_command_with_numeric_arg_calls_rollback_to(make_app):
    app = make_app()
    asyncio.run(_dispatch_with_sink(app, "/rollback 5"))
    assert app.coding.rolled_back_seq == 5


def test_rollback_command_no_turns_warns(make_app):
    app = make_app()
    with patch.object(app.coding.thread_store, "list_user_turns", return_value=[]):
        asyncio.run(_dispatch_with_sink(app, "/rollback"))
    assert app.coding.rolled_back_seq is None


async def _dispatch_busy(app, command: str) -> None:
    render.set_sink(app)
    try:
        await dispatch(app, command, busy=True)
    finally:
        render.set_sink(None)


def test_memory_create_refused_while_busy(make_app):
    app = make_app()
    invoke = AsyncMock(return_value=SimpleNamespace(content="# Project"))
    app.coding.agent.config.model = SimpleNamespace(ainvoke=invoke)
    asyncio.run(_dispatch_busy(app, "/memory create"))
    text = "\n".join(line.text for line in app._lines)
    assert "/memory create is not available while a task is running" in text
    invoke.assert_not_called()


def test_memory_read_and_add_allowed_while_busy(make_app):
    app = make_app()
    with patch.object(app.coding.memory_store, "load_project", return_value="existing notes"):
        asyncio.run(_dispatch_busy(app, "/memory"))
    text = "\n".join(line.text for line in app._lines)
    assert "existing notes" in text

    with patch.object(
        app.coding.memory_store, "append_project", return_value="Updated .ness/NESS.md"
    ) as append:
        asyncio.run(_dispatch_busy(app, "/memory add remember this"))
        append.assert_called_once_with("remember this")
