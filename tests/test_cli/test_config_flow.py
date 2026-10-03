from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from ness_cli.config import ConfigManager, ConfigPatch
from ness_cli.config import store as store_module
from ness_cli.providers import ModelInfo
from ness_cli.runtime import InteractiveRuntime
from ness_cli.tui.input.config import (
    SECTIONS,
    SPECS,
    _config_lines,
    _edit,
    _visible_specs,
)


class FakeUI:
    def __init__(self, *, choices=(), inputs=()) -> None:
        self.choices = list(choices)
        self.inputs = list(inputs)
        self.choose_calls: list[dict] = []
        self.input_calls: list[dict] = []

    async def choose(self, title, items, *, initial_key=None, hint="", **options):
        self.choose_calls.append(
            {
                "title": title,
                "items": items,
                "initial_key": initial_key,
                "hint": hint,
                **options,
            }
        )
        return self.choices.pop(0) if self.choices else None

    async def request_input(self, label, *, default="", secret=False):
        self.input_calls.append({"label": label, "default": default, "secret": secret})
        return self.inputs.pop(0) if self.inputs else None


class FakeRenderer:
    def __init__(self) -> None:
        self.notices: list[tuple] = []
        self.errors: list[str] = []

    def notice(self, *values) -> None:
        self.notices.append(values)

    def error(self, text) -> None:
        self.errors.append(text)


class FakeProvider:
    display_name = "Provider"

    def __init__(self) -> None:
        self.models_calls: list[bool] = []
        self.model_info_calls: list[str] = []

    async def models(self, *, refresh=False):
        self.models_calls.append(refresh)
        return (
            ModelInfo(
                id="model-a",
                name="Model A",
                context_window=128_000,
                reasoning_efforts=("low", "high"),
            ),
            ModelInfo(id="model-b", name="Model B"),
        )

    def model_info(self, model_name):
        self.model_info_calls.append(model_name)
        return ModelInfo(
            id=model_name,
            name=model_name,
            reasoning_efforts=("low", "high"),
        )

    def is_authenticated(self):
        return True


class FakeProviders:
    def __init__(self, provider: FakeProvider) -> None:
        self.provider = provider

    def active(self):
        return self.provider

    async def models(self, *, provider_id=None, refresh=False):
        del provider_id
        return await self.provider.models(refresh=refresh)

    def provider_ids(self):
        return ("openrouter", "codex")

    def get(self, provider_id):
        return SimpleNamespace(display_name=provider_id.title())


@dataclass
class FakeContext:
    runtime: object
    ui: FakeUI
    renderer: FakeRenderer
    session: object = None


class FakeRuntime:
    def __init__(self, manager: ConfigManager, provider: FakeProvider) -> None:
        self.manager = manager
        self.providers = FakeProviders(provider)
        self.applied: list[ConfigPatch] = []

    @property
    def config(self):
        return self.manager.current

    @property
    def settings(self):
        return self.manager.settings

    async def apply_config(self, patch, *, selected=None):
        del selected
        self.applied.append(patch)
        return self.manager.apply(patch)


def _context(isolated_cli_env, *, choices=(), inputs=()):
    manager = ConfigManager.load(isolated_cli_env.config, environment={})
    provider = FakeProvider()
    runtime = FakeRuntime(manager, provider)
    return (
        FakeContext(runtime, FakeUI(choices=choices, inputs=inputs), FakeRenderer()),
        provider,
    )


def _spec(key: str):
    return next(spec for spec in SPECS if spec.key == key)


def test_config_controls_have_unique_keys() -> None:
    assert len({spec.key for spec in SPECS}) == len(SPECS)


def test_openrouter_only_controls_are_hidden_for_other_providers(
    isolated_cli_env,
) -> None:
    context, _provider = _context(isolated_cli_env)
    context.runtime.manager.apply(ConfigPatch({"provider_id": "codex"}))

    visible = {
        spec.key
        for section, _title in SECTIONS
        for spec in _visible_specs(context, section)
    }

    assert "openrouter_session_id" not in visible
    assert "openrouter_cache_ttl" not in visible
    assert "openrouter_anthropic_messages" not in visible
    assert "api_key" not in visible
    assert "openai_base_url" not in visible
    assert "model_name" in visible

    context.runtime.manager.apply(ConfigPatch({"provider_id": "opencode"}))
    opencode_visible = {
        spec.key
        for section, _title in SECTIONS
        for spec in _visible_specs(context, section)
    }
    assert "api_key" in opencode_visible
    assert "openai_base_url" not in opencode_visible


def test_model_choices_come_from_active_provider(isolated_cli_env) -> None:
    context, provider = _context(isolated_cli_env, choices=("model-b",))

    asyncio.run(_edit(context, _spec("model_name")))

    items = context.ui.choose_calls[0]["items"]
    assert [item.key for item in items] == ["model-a", "model-b"]
    assert context.ui.choose_calls[0]["filterable"] is True
    assert callable(context.ui.choose_calls[0]["refresh"])
    assert provider.models_calls == [False]
    assert context.runtime.applied[0].values == {"model_name": "model-b"}


def test_reasoning_choices_come_from_selected_model(isolated_cli_env) -> None:
    context, provider = _context(isolated_cli_env, choices=("low",))

    asyncio.run(_edit(context, _spec("reasoning_effort")))

    assert [item.key for item in context.ui.choose_calls[0]["items"]] == [
        "low",
        "high",
    ]
    assert provider.model_info_calls == [context.runtime.config.model.model_name]
    assert context.runtime.applied[0].values == {"reasoning_effort": "low"}


def test_optional_models_can_return_to_provider_default(isolated_cli_env) -> None:
    context, _provider = _context(isolated_cli_env, choices=("", ""))

    asyncio.run(_edit(context, _spec("reflection_model_name")))
    asyncio.run(_edit(context, _spec("goal_judge_model")))

    assert context.ui.choose_calls[0]["items"][0].label == "Use provider default"
    assert context.ui.choose_calls[1]["items"][0].label == "Use provider default"
    assert [patch.values for patch in context.runtime.applied] == [
        {"reflection_model_name": None},
        {"goal_judge_model": None},
    ]


def test_secret_input_is_masked_and_never_rendered(isolated_cli_env) -> None:
    secret = "sk-never-render-this"
    context, _provider = _context(isolated_cli_env, inputs=(secret,))

    asyncio.run(_edit(context, _spec("api_key")))

    assert context.ui.input_calls == [
        {"label": "Provider API key", "default": "", "secret": True}
    ]
    rendered = repr(context.renderer.notices) + repr(context.renderer.errors)
    assert secret not in rendered


def test_config_lines_cover_every_visible_section(isolated_cli_env) -> None:
    context, _provider = _context(isolated_cli_env)
    lines = _config_lines(context)

    for _section, title in SECTIONS:
        assert title in lines
    assert any("Chat model" in line for line in lines)


def test_model_change_reloads_selected_session_once(isolated_cli_env) -> None:
    manager = ConfigManager.load(isolated_cli_env.config, environment={})
    runtime = InteractiveRuntime(SimpleNamespace(config=manager), resume_thread_id=None)

    class Session:
        reloads = 0
        applied = []

        async def reload_model(self):
            self.reloads += 1

        def apply_runtime_config(self, config):
            self.applied.append(config)

    session = Session()

    update = asyncio.run(
        runtime.apply_config(
            ConfigPatch({"model_name": "openai/gpt-4o"}), selected=session
        )
    )

    assert update.reload_selected is True
    assert session.reloads == 1
    assert session.applied == []


def test_deferred_setting_updates_session_options_without_model_reload(
    isolated_cli_env,
) -> None:
    manager = ConfigManager.load(isolated_cli_env.config, environment={})
    runtime = InteractiveRuntime(SimpleNamespace(config=manager), resume_thread_id=None)

    class Session:
        reloads = 0

        def __init__(self):
            self.applied = []

        async def reload_model(self):
            self.reloads += 1

        def apply_runtime_config(self, config):
            self.applied.append(config)

    session = Session()

    update = asyncio.run(
        runtime.apply_config(ConfigPatch({"enable_approval": False}), selected=session)
    )

    assert update.future_sessions_changed is True
    assert update.reload_selected is False
    assert session.reloads == 0
    assert len(session.applied) == 1


@pytest.mark.parametrize("after", [False, True])
def test_config_editor_reports_save_failure_without_exposing_credentials(
    isolated_cli_env, monkeypatch, after
) -> None:
    context, _provider = _context(isolated_cli_env, inputs=["private-test-key"])
    store = context.runtime.manager._store
    original_write = store_module._atomic_write

    def fail_write(path, value, *, secret):
        if path != store.secrets_file:
            return original_write(path, value, secret=secret)
        if after:
            original_write(path, value, secret=secret)
        raise OSError("private-test-key in a write diagnostic")

    monkeypatch.setattr(store_module, "_atomic_write", fail_write)

    asyncio.run(_edit(context, _spec("api_key")))

    assert context.renderer.notices == []
    assert len(context.renderer.errors) == 1
    message = context.renderer.errors[0]
    assert "Error while saving secrets.json" in message
    assert f"Saved requested values: {'api_key' if after else 'none'}." in message
    assert f"Not saved: {'none' if after else 'api_key'}." in message
    assert "private-test-key" not in message
    assert context.runtime.manager.provider_runtime().api_key == (
        "private-test-key" if after else None
    )
