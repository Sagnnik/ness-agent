from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from ness_agent import NessAgent, NessAgentOptions, NoOverlay, PromptLayersConfig
from ness_cli.config import ConfigApplyError, ConfigManager, ConfigPatch
from ness_cli.runtime import InteractiveRuntime, SessionModels, _SessionFactory
from ness_cli.session.export import ExportError
from ness_cli.tui.input.config import SPECS, _display_value, _edit


class Model(FakeListChatModel):
    def bind_tools(self, _tools, **_kwargs):
        return self


@pytest.fixture
def autosave_env(isolated_cli_env, monkeypatch):
    paths = isolated_cli_env.paths
    manager = ConfigManager.load(paths, environment={})
    manager.apply(ConfigPatch({"reflection_token_ratio": 0}))
    model = Model(responses=["saved answer"])
    agent = NessAgent(
        model=model,
        tools=[],
        prompt=PromptLayersConfig(l0="Test"),
        overlay=NoOverlay(),
        skills_dirs=[],
        options=NessAgentOptions(
            project_root=paths.project_root,
            ness_dir=paths.ness_dir,
        ),
    )
    monkeypatch.setattr(
        "ness_cli.runtime._session_models",
        lambda _providers, config, **_kwargs: SessionModels(
            model, None, 128_000, False, config
        ),
    )
    factory = _SessionFactory(paths=paths, config=manager, providers=None, agent=agent)
    runtime = InteractiveRuntime(SimpleNamespace(config=manager), resume_thread_id=None)
    return SimpleNamespace(
        paths=paths, manager=manager, agent=agent, factory=factory, runtime=runtime
    )


@pytest.mark.parametrize("initial", [True, False])
def test_autosave_toggle_only_changes_selected_session(autosave_env, initial):
    env = autosave_env
    env.manager.apply(ConfigPatch({"auto_save_threads": initial}))

    async def exercise():
        first = await env.factory.new(thread_id="session-first")
        second = await env.factory.new(thread_id="session-second")
        await env.runtime.apply_config(
            ConfigPatch({"auto_save_threads": not initial}), selected=first
        )

        first_seq = first._repository.append_user(first.thread_id, "selected")
        second_seq = second._repository.append_user(second.thread_id, "sibling")
        assert (first_seq is not None) is (not initial)
        assert (second_seq is not None) is initial
        assert first._repository.auto_save is (not initial)
        assert second._repository.auto_save is initial
        assert first._session.config.options.auto_save_threads is (not initial)
        assert second._session.config.options.auto_save_threads is initial
        assert second.runtime_config.settings.auto_save_threads is initial
        assert env.agent.config.thread_store.auto_save is True

    asyncio.run(exercise())


def test_model_reload_keeps_autosave_snapshot_and_pinned_behavior(autosave_env):
    env = autosave_env

    async def exercise():
        selected = await env.factory.new(thread_id="session-selected")
        sibling = await env.factory.new(thread_id="session-sibling")
        await env.runtime.apply_config(
            ConfigPatch({"auto_save_threads": False, "format_on_write": False}),
            selected=selected,
        )

        await sibling.reload_model()

        assert sibling.runtime_config.settings.auto_save_threads is True
        assert sibling.runtime_config.settings.format_on_write is True
        assert sibling._repository.auto_save is True
        assert sibling._session.config.options.auto_save_threads is True
        assert sibling._repository.append_user(sibling.thread_id, "saved") == 0

    asyncio.run(exercise())


def test_unrelated_unchanged_patch_keeps_selected_autosave_policy(autosave_env):
    env = autosave_env

    async def exercise():
        selected = await env.factory.new(thread_id="session-selected")
        sibling = await env.factory.new(thread_id="session-sibling")
        await env.runtime.apply_config(
            ConfigPatch({"auto_save_threads": False}), selected=selected
        )
        update = await env.runtime.apply_config(
            ConfigPatch({"format_on_write": True}), selected=sibling
        )
        assert not update.options_changed
        assert sibling._repository.auto_save is True
        assert sibling.runtime_config.settings.auto_save_threads is True

    asyncio.run(exercise())


def test_existing_saved_autosave_default_applies_after_partial_failure(
    autosave_env, monkeypatch
):
    env = autosave_env

    async def exercise():
        first = await env.factory.new(thread_id="session-first")
        sibling = await env.factory.new(thread_id="session-sibling")
        await env.runtime.apply_config(
            ConfigPatch({"auto_save_threads": False, "format_on_write": False}),
            selected=first,
        )
        original = os.replace

        def replace(source, destination):
            if destination == env.manager._store.secrets_file:
                raise OSError("secret write failed")
            return original(source, destination)

        monkeypatch.setattr(os, "replace", replace)
        with pytest.raises(ConfigApplyError) as error:
            await env.runtime.apply_config(
                ConfigPatch({"auto_save_threads": False, "api_key": "test-only-key"}),
                selected=sibling,
            )
        assert error.value.saved_keys == ("auto_save_threads",)
        assert not error.value.update.options_changed
        assert sibling._repository.auto_save is False
        assert sibling.runtime_config.settings.auto_save_threads is False
        assert sibling.runtime_config.settings.format_on_write is True

    asyncio.run(exercise())


def test_unsaved_autosave_change_preserves_selected_policy(autosave_env, monkeypatch):
    env = autosave_env
    env.manager.apply(ConfigPatch({"auto_save_threads": False}))

    async def exercise():
        selected = await env.factory.new(thread_id="session-selected")
        await env.runtime.apply_config(ConfigPatch({"auto_save_threads": True}))
        original = os.replace

        def replace(source, destination):
            if destination == env.manager._store.configs_file:
                raise OSError("config write failed")
            return original(source, destination)

        monkeypatch.setattr(os, "replace", replace)
        with pytest.raises(ConfigApplyError) as error:
            await env.runtime.apply_config(
                ConfigPatch({"auto_save_threads": False}), selected=selected
            )
        assert error.value.saved_keys == ()
        assert env.manager.current.settings.auto_save_threads is True
        assert selected._repository.auto_save is False
        assert selected.runtime_config.settings.auto_save_threads is False

    asyncio.run(exercise())


@pytest.mark.parametrize("update_selected", [True, False])
def test_new_session_inherits_saved_default_without_changing_open_sibling(
    autosave_env, update_selected
):
    env = autosave_env

    async def exercise():
        first = await env.factory.new(thread_id="session-first")
        sibling = await env.factory.new(thread_id="session-sibling")
        await env.runtime.apply_config(
            ConfigPatch({"auto_save_threads": False}),
            selected=first if update_selected else None,
        )
        future = await env.factory.new(thread_id="session-future")

        assert future._repository.append_user(future.thread_id, "unsaved") is None
        assert sibling._repository.append_user(sibling.thread_id, "saved") == 0
        assert sibling._repository.auto_save is True
        assert future._repository.auto_save is False
        assert first._repository.auto_save is (not update_selected)
        assert await env.factory.new(thread_id=sibling.thread_id) is sibling
        assert sibling.runtime_config.settings.auto_save_threads is True
        assert env.agent.config.thread_store.auto_save is True

    asyncio.run(exercise())


def test_actual_turns_persist_user_and_sdk_events_for_enabled_sibling(autosave_env):
    env = autosave_env

    async def exercise():
        disabled = await env.factory.new(thread_id="session-disabled")
        enabled = await env.factory.new(thread_id="session-enabled")
        await env.runtime.apply_config(
            ConfigPatch({"auto_save_threads": False}), selected=disabled
        )

        async def turn(session):
            events = [event async for event in session.run_turn("hello")]
            assert not any(event.kind == "error" for event in events), events
            assert any(event.kind == "assistant_final" for event in events), events

        await asyncio.gather(turn(disabled), turn(enabled))
        assert disabled._repository.raw_events(disabled.thread_id) == []
        rows = enabled._repository.raw_events(enabled.thread_id)
        assert {row["kind"] for row in rows} >= {"user", "assistant"}
        user_seq = enabled.user_turns()[0].seq
        assert enabled._repository.checkpoint(enabled.thread_id, user_seq) is not None
        assert (
            env.agent.config.thread_store.load_thread_events(enabled.thread_id) == rows
        )

    asyncio.run(exercise())


def test_inflight_sibling_keeps_saving_after_selected_config_changes(
    autosave_env, monkeypatch
):
    env = autosave_env

    async def exercise():
        selected = await env.factory.new(thread_id="session-selected")
        sibling = await env.factory.new(thread_id="session-inflight")
        entered = asyncio.Event()
        release = asyncio.Event()
        original = sibling._session.stream

        async def paused_stream(*args, **kwargs):
            entered.set()
            await release.wait()
            async for event in original(*args, **kwargs):
                yield event

        monkeypatch.setattr(sibling._session, "stream", paused_stream)

        async def turn():
            return [event async for event in sibling.run_turn("in progress")]

        task = asyncio.create_task(turn())
        try:
            await asyncio.wait_for(entered.wait(), timeout=5)
            await env.runtime.apply_config(
                ConfigPatch({"auto_save_threads": False}), selected=selected
            )
        finally:
            release.set()
            events = await task

        assert not any(event.kind == "error" for event in events), events
        assert {
            row["kind"] for row in sibling._repository.raw_events(sibling.thread_id)
        } >= {"user", "assistant"}
        assert sibling._repository.auto_save is True
        assert sibling.runtime_config.settings.auto_save_threads is True

    asyncio.run(exercise())


def test_disabling_and_reenabling_preserves_history_and_skips_unsaved_turns(
    autosave_env,
):
    env = autosave_env

    async def exercise():
        selected = await env.factory.new(thread_id="session-selected")
        sibling = await env.factory.new(thread_id="session-sibling")
        assert selected._repository.append_user(selected.thread_id, "before") == 0
        await env.runtime.apply_config(
            ConfigPatch({"auto_save_threads": False}), selected=selected
        )
        assert selected._repository.append_user(selected.thread_id, "not saved") is None
        assert sibling._repository.append_user(sibling.thread_id, "still saved") == 0
        assert [turn.content for turn in selected.user_turns()] == ["before"]
        assert await selected.resume()

        await env.runtime.apply_config(
            ConfigPatch({"auto_save_threads": True}), selected=selected
        )
        assert selected._repository.append_user(selected.thread_id, "after") == 1
        assert sibling._repository.append_user(sibling.thread_id, "also after") == 1
        assert [turn.content for turn in selected.user_turns()] == ["before", "after"]

    asyncio.run(exercise())


def test_archive_rename_export_fork_and_rollback_use_selected_autosave_policy(
    autosave_env,
):
    env = autosave_env

    async def exercise():
        disabled = await env.factory.new(thread_id="session-disabled")
        enabled = await env.factory.new(thread_id="session-enabled")
        for session in (disabled, enabled):
            repo = session._repository
            repo.append_user(session.thread_id, "first")
            seq = repo.append_user(session.thread_id, "second")
            repo.save_checkpoint(
                session.thread_id, seq, git_hash=None, memory_snapshot="memory"
            )

        await env.runtime.apply_config(
            ConfigPatch({"auto_save_threads": False}), selected=disabled
        )
        assert disabled.rename("unsaved name") is False
        assert enabled.rename("saved name") is True
        assert (await disabled.save()).resume_thread_id is None
        assert (await enabled.save()).resume_thread_id == enabled.thread_id
        with pytest.raises(ExportError, match="autosave is disabled"):
            await disabled.export_html(env.paths.project_root / "disabled.html")
        await enabled.export_html(env.paths.project_root / "enabled.html")
        assert (env.paths.project_root / "enabled.html").is_file()
        assert not (env.paths.project_root / "disabled.html").exists()

        with pytest.raises(ValueError, match="autosave must be enabled"):
            await disabled.fork_before(1)
        fork = await enabled.fork_before(1)
        assert [
            row["content"] for row in enabled._repository.raw_events(fork.thread_id)
        ] == ["first"]
        result = await disabled.rollback_to(1)
        assert not result.ok and "Rollback requires thread autosave" in result.message
        assert len(disabled.user_turns()) == 2
        assert len(enabled.user_turns()) == 2

    asyncio.run(exercise())


def test_partially_saved_config_applies_autosave_only_to_selected_session(
    autosave_env, monkeypatch
):
    env = autosave_env

    async def exercise():
        selected = await env.factory.new(thread_id="session-selected")
        sibling = await env.factory.new(thread_id="session-sibling")
        original = os.replace

        def replace(source, destination):
            if destination == env.manager._store.secrets_file:
                raise OSError("secret write failed")
            return original(source, destination)

        monkeypatch.setattr(os, "replace", replace)
        with pytest.raises(ConfigApplyError) as error:
            await env.runtime.apply_config(
                ConfigPatch({"auto_save_threads": False, "api_key": "test-only-key"}),
                selected=selected,
            )
        assert error.value.saved_keys == ("auto_save_threads",)
        assert selected._repository.append_user(selected.thread_id, "not saved") is None
        assert sibling._repository.append_user(sibling.thread_id, "saved") == 0
        assert selected._session.config.options.auto_save_threads is False
        assert sibling._session.config.options.auto_save_threads is True

    asyncio.run(exercise())


def test_sdk_compaction_and_subagent_writes_follow_session_autosave_policy(
    autosave_env,
):
    env = autosave_env

    async def exercise():
        disabled = await env.factory.new(thread_id="session-disabled")
        enabled = await env.factory.new(thread_id="session-enabled")
        for session in (disabled, enabled):
            session._repository.append_user(session.thread_id, "seed")
        await env.runtime.apply_config(
            ConfigPatch({"auto_save_threads": False}), selected=disabled
        )

        for session, saves in ((disabled, False), (enabled, True)):
            store = session._session.config.thread_store
            seq = store.append_compaction_checkpoint(
                session.thread_id, {"response": "summary"}, active_turn=False
            )
            assert (seq is not None) is saves
            child = f"subagent-{session.thread_id}"
            store.register_subagent(session.thread_id, child, agent_name="test")
            store.complete_subagent(child, status="completed", output="done")
            store.append_event(child, {"kind": "usage", "cost_usd": 0.5})
            rows = store.list_subagents(session.thread_id)
            assert len(rows) == int(saves)
            if saves:
                assert rows[0]["status"] == "completed"
                assert rows[0]["output"] == "done"

        metadata = {
            row["thread_id"]: row
            for row in env.agent.config.thread_store.list_threads()
        }
        assert metadata[disabled.thread_id]["total_cost_usd"] == 0
        assert metadata[enabled.thread_id]["total_cost_usd"] == 0.5
        assert [event.kind for event in disabled.history()] == ["user"]
        assert "compaction_llm" in {event.kind for event in enabled.history()}

    asyncio.run(exercise())


@pytest.mark.parametrize("initial", [True, False])
def test_config_menu_shows_selected_policy_and_can_apply_an_existing_default(
    autosave_env, initial
):
    env = autosave_env
    env.manager.apply(ConfigPatch({"auto_save_threads": initial}))
    spec = next(spec for spec in SPECS if spec.key == "auto_save_threads")

    async def exercise():
        first = await env.factory.new(thread_id="session-first")
        sibling = await env.factory.new(thread_id="session-sibling")
        await env.runtime.apply_config(
            ConfigPatch({"auto_save_threads": not initial, "format_on_write": False}),
            selected=first,
        )
        notices = []
        errors = []
        context = SimpleNamespace(
            runtime=env.runtime,
            session=sibling,
            renderer=SimpleNamespace(
                notice=lambda *values: notices.append(values), error=errors.append
            ),
        )
        assert _display_value(context, spec) == ("on" if initial else "off")
        await _edit(context, spec)
        assert sibling._repository.auto_save is (not initial)
        assert sibling._session.config.options.auto_save_threads is (not initial)
        assert sibling.runtime_config.settings.auto_save_threads is (not initial)
        assert sibling.runtime_config.settings.format_on_write is True
        assert env.manager.current.settings.auto_save_threads is (not initial)
        assert _display_value(context, spec) == ("off" if initial else "on")
        assert notices and errors == []
        assert "Thread autosave" in notices[-1][1]

    asyncio.run(exercise())
