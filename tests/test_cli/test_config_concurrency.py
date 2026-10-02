from __future__ import annotations

from contextlib import contextmanager
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from ness_cli.config import CliOverrides, ConfigApplyError, ConfigManager, ConfigPatch
from ness_cli.config import store as store_module
from ness_cli.config.store import ConfigStore
from ness_cli.runtime import InteractiveRuntime


def edit_before_save_lock(monkeypatch, manager, edit):
    original = store_module.locked_path
    other = ConfigStore(manager._store.config_dir)
    applied = False

    @contextmanager
    def locked(path, *, secret=False):
        nonlocal applied
        if not applied:
            applied = True
            edit(other)
        with original(path, secret=secret):
            yield

    monkeypatch.setattr(store_module, "locked_path", locked)


@pytest.mark.parametrize(
    "patch,document,key,value",
    [
        ({"format_on_write": False}, "config", "enable_approval", False),
        ({"format_on_write": False}, "secrets", "openai_api_key", "external-test-key"),
        ({"api_key": "local-test-key"}, "config", "model_name", "external/model"),
        ({"api_key": "local-test-key"}, "secrets", "exa_api_key", "external-exa-key"),
    ],
)
def test_successful_save_adopts_concurrent_edits_in_either_document(
    isolated_cli_env, monkeypatch, patch, document, key, value
):
    manager = ConfigManager.load(isolated_cli_env.config, environment={})
    previous = manager.current
    previous_provider = manager.provider_runtime()

    def edit(other):
        write = other.write_config if document == "config" else other.write_secret
        write(key, value)

    edit_before_save_lock(monkeypatch, manager, edit)
    update = manager.apply(ConfigPatch(patch))
    actual = ConfigManager.load(isolated_cli_env.config, environment={})

    assert manager.current == actual.current
    assert manager.settings == actual.settings
    assert manager.provider_runtime() == actual.provider_runtime()
    assert update.previous == previous
    assert update.current == actual.current
    assert update.model_changed == (previous.model != actual.current.model)
    assert update.options_changed == (previous.settings != actual.current.settings)
    assert update.provider_runtime_changed == (
        previous_provider != actual.provider_runtime()
    )
    if "format_on_write" in patch:
        assert actual.current.settings.format_on_write is False
    if "api_key" in patch:
        assert actual.provider_runtime().api_key == "local-test-key"


@pytest.mark.parametrize("document", ["config", "secrets"])
def test_invalid_concurrent_values_are_revalidated_before_either_write(
    isolated_cli_env, monkeypatch, document
):
    manager = ConfigManager.load(isolated_cli_env.config, environment={})
    previous = manager.current

    def edit(other):
        if document == "config":
            other.write_config("reflection_token_ratio", 2)
        else:
            other.write_secret("exa_api_key", {"invalid": "private-test-value"})

    edit_before_save_lock(monkeypatch, manager, edit)
    with pytest.raises(ConfigApplyError) as caught:
        manager.apply(
            ConfigPatch({"format_on_write": False, "api_key": "local-test-key"})
        )

    error = caught.value
    assert isinstance(error.__cause__, ValueError)
    assert error.update is None
    assert error.saved_keys is None
    assert error.unsaved_keys is None
    assert "saved values are unknown" in str(error)
    assert "private-test-value" not in str(error)
    assert "format_on_write" not in manager._store.load_config()
    assert "openai_api_key" not in manager._store.load_secrets()
    assert manager.current == previous


def test_requested_value_can_correct_an_invalid_concurrent_edit(
    isolated_cli_env, monkeypatch
):
    manager = ConfigManager.load(isolated_cli_env.config, environment={})
    edit_before_save_lock(
        monkeypatch,
        manager,
        lambda other: other.write_config("reflection_token_ratio", 2),
    )

    update = manager.apply(ConfigPatch({"reflection_token_ratio": 0.8}))

    assert update.current.settings.reflection_token_ratio == 0.8
    assert manager._store.load_config()["reflection_token_ratio"] == 0.8


def test_mixed_save_preserves_concurrent_provider_profiles_and_secrets(
    isolated_cli_env, monkeypatch
):
    manager = ConfigManager.load(isolated_cli_env.config, environment={})

    def edit(other):
        other.update_provider_profile(
            "openrouter",
            {
                "reasoning_effort": "low",
                "reflection_model_name": "external/reflection",
            },
        )
        other.update_provider_profile("codex", {"model_name": "external/codex"})
        other.write_secret("opencode_api_key", "external-opencode-key")

    edit_before_save_lock(monkeypatch, manager, edit)
    manager.apply(
        ConfigPatch({"model_name": "local/model", "api_key": "local-test-key"})
    )
    actual = ConfigManager.load(isolated_cli_env.config, environment={})

    assert manager.current == actual.current
    assert manager.current.model.model_name == "local/model"
    assert manager.current.model.reflection_model_name == "external/reflection"
    assert manager.current.model.reasoning_effort == "low"
    assert manager.provider_profile("codex") == {"model_name": "external/codex"}
    assert manager.provider_runtime("opencode").api_key == "external-opencode-key"
    assert manager.provider_runtime().api_key == "local-test-key"


def test_profile_deletion_uses_concurrent_legacy_fallback(
    isolated_cli_env, monkeypatch
):
    store = ConfigStore(isolated_cli_env.config)
    store.update_provider_profile("openrouter", {"model_name": "old/profile"})
    manager = ConfigManager.load(isolated_cli_env.config, environment={})

    def edit(other):
        other.write_config("model_name", "external/fallback")
        other.update_provider_profile("openrouter", {"reasoning_effort": "low"})

    edit_before_save_lock(monkeypatch, manager, edit)
    update = manager.apply(ConfigPatch({"model_name": None}))

    assert update.current.model.model_name == "external/fallback"
    assert update.current.model.reasoning_effort == "low"
    assert manager.provider_profile("openrouter") == {"reasoning_effort": "low"}


@pytest.mark.parametrize("use_cli", [False, True])
def test_committed_snapshot_preserves_environment_and_cli_precedence(
    isolated_cli_env, monkeypatch, use_cli
):
    environment = {
        "MODEL_NAME": "environment/model",
        "OPENAI_API_KEY": "environment-key",
    }
    overrides = (
        CliOverrides(model_name="cli/model", api_key="cli-key") if use_cli else None
    )
    manager = ConfigManager.load(
        isolated_cli_env.config, overrides, environment=environment
    )

    def edit(other):
        other.update_provider_profile("openrouter", {"model_name": "external/model"})
        other.write_secret("openai_api_key", "external-test-key")

    edit_before_save_lock(monkeypatch, manager, edit)
    update = manager.apply(ConfigPatch({"format_on_write": False}))
    actual = ConfigManager.load(
        isolated_cli_env.config, overrides, environment=environment
    )

    assert manager.current == actual.current
    assert manager.settings == actual.settings
    assert manager.provider_runtime() == actual.provider_runtime()
    assert manager.current.model.model_name == (
        "cli/model" if use_cli else "environment/model"
    )
    assert manager.provider_runtime().api_key == (
        "cli-key" if use_cli else "environment-key"
    )
    assert update.reload_selected is False
    assert manager.provider_profile("openrouter")["model_name"] == "external/model"
    assert manager._store.load_secrets()["openai_api_key"] == "external-test-key"


@pytest.mark.parametrize(
    "patch", [{"goal_max_attempts": 4}, {"api_key": "local-test-key"}]
)
def test_selected_session_receives_effects_from_committed_concurrent_edits(
    isolated_cli_env, monkeypatch, patch
):
    manager = ConfigManager.load(isolated_cli_env.config, environment={})
    runtime = InteractiveRuntime(SimpleNamespace(config=manager), resume_thread_id=None)
    selected = SimpleNamespace(apply_runtime_config=Mock(), reload_model=AsyncMock())

    def edit(other):
        other.write_config("enable_approval", False)
        other.update_provider_profile("openrouter", {"model_name": "external/model"})

    edit_before_save_lock(monkeypatch, manager, edit)
    update = asyncio.run(runtime.apply_config(ConfigPatch(patch), selected=selected))

    assert update.current.settings.enable_approval is False
    assert update.current.model.model_name == "external/model"
    selected.apply_runtime_config.assert_called_once_with(update.current)
    selected.reload_model.assert_awaited_once()
