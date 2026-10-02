from __future__ import annotations

import os
from pathlib import Path
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from ness_cli.config import CliOverrides, ConfigApplyError, ConfigManager, ConfigPatch
from ness_cli.config import store as store_module
from ness_cli.config.store import ConfigStore
from ness_cli.runtime import InteractiveRuntime


@pytest.fixture
def config_env(isolated_cli_env):
    store = ConfigStore(isolated_cli_env.config)
    store.write_config("format_on_write", True)
    store.write_secret("openai_api_key", "old-test-key")
    manager = ConfigManager.load(isolated_cli_env.config, environment={})
    return manager._store, manager


def fail_replace(monkeypatch, path, *, after=False):
    original = os.replace

    def replace(source, destination):
        if Path(destination) == path:
            if after:
                original(source, destination)
            raise OSError("injected write failure")
        return original(source, destination)

    monkeypatch.setattr(os, "replace", replace)


@pytest.mark.parametrize("document", ["config", "secrets"])
@pytest.mark.parametrize("after", [False, True])
def test_failed_mixed_save_keeps_manager_consistent_with_disk(
    config_env, monkeypatch, document, after
):
    store, manager = config_env
    previous = manager.current
    path = store.configs_file if document == "config" else store.secrets_file
    fail_replace(monkeypatch, path, after=after)

    with pytest.raises(ConfigApplyError) as caught:
        manager.apply(
            ConfigPatch({"format_on_write": False, "api_key": "new-test-key"})
        )

    error = caught.value
    saved = []
    if document == "secrets" or after:
        saved.append("format_on_write")
    if document == "secrets" and after:
        saved.append("api_key")
    assert error.saved_keys == tuple(sorted(saved))
    assert error.unsaved_keys == tuple(
        sorted({"format_on_write", "api_key"} - set(saved))
    )
    assert error.failed_document == path.name
    assert isinstance(error.__cause__, OSError)
    assert error.update.previous == previous
    assert error.update.current == manager.current
    assert error.update.changed_keys == error.saved_keys
    assert error.update.options_changed == ("format_on_write" in saved)
    assert error.update.reload_selected == ("api_key" in saved)
    assert "new-test-key" not in str(error)
    assert "old-test-key" not in str(error)
    actual = ConfigManager.load(store.config_dir, environment={})
    assert manager.current == actual.current
    assert manager.settings == actual.settings
    assert manager.provider_runtime() == actual.provider_runtime()
    assert not list(store.config_dir.glob("*.tmp"))
    assert store.secrets_file.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("after", [False, True])
@pytest.mark.parametrize(
    "key,value,document",
    [
        ("format_on_write", False, "config"),
        ("api_key", "new-test-key", "secrets"),
    ],
)
def test_single_document_failure_reports_actual_persisted_value(
    config_env, monkeypatch, after, key, value, document
):
    store, manager = config_env
    path = store.configs_file if document == "config" else store.secrets_file
    fail_replace(monkeypatch, path, after=after)

    with pytest.raises(ConfigApplyError) as caught:
        manager.apply(ConfigPatch({key: value}))

    assert caught.value.saved_keys == ((key,) if after else ())
    assert caught.value.unsaved_keys == (() if after else (key,))
    actual = ConfigManager.load(store.config_dir, environment={})
    assert manager.current == actual.current
    assert manager.provider_runtime() == actual.provider_runtime()


def test_missing_documents_are_valid_during_failure_recovery(
    isolated_cli_env, monkeypatch
):
    manager = ConfigManager.load(isolated_cli_env.config, environment={})
    store = manager._store
    fail_replace(monkeypatch, store.configs_file)

    with pytest.raises(ConfigApplyError) as caught:
        manager.apply(
            ConfigPatch({"format_on_write": False, "api_key": "new-test-key"})
        )

    assert caught.value.saved_keys == ()
    assert caught.value.unsaved_keys == ("api_key", "format_on_write")
    assert caught.value.update.current == manager.current
    assert manager.current.settings.format_on_write is True
    assert manager.provider_runtime().api_key is None
    assert not store.configs_file.exists()
    assert not store.secrets_file.exists()


@pytest.mark.parametrize("provider_id", ["openrouter", "opencode"])
def test_partial_save_tracks_aliases_profiles_and_provider_credentials(
    config_env, monkeypatch, provider_id
):
    store, manager = config_env
    store.write_config("unrelated_setting", "leave-me")
    store.write_secret("exa_api_key", "unrelated-test-key")
    fail_replace(monkeypatch, store.secrets_file)

    with pytest.raises(ConfigApplyError) as caught:
        manager.apply(
            ConfigPatch(
                {
                    "provider_id": provider_id,
                    "model_name": "new/model",
                    "base_url": "https://example.invalid/v1",
                    "session_id": "test-session",
                    "api_key": "new-test-key",
                }
            )
        )

    assert caught.value.saved_keys == (
        "base_url",
        "model_name",
        "provider_id",
        "session_id",
    )
    assert caught.value.unsaved_keys == ("api_key",)
    assert manager.current.model.provider_id == provider_id
    assert manager.current.model.model_name == "new/model"
    assert manager.provider_profile(provider_id) == {"model_name": "new/model"}
    assert manager.provider_runtime().api_key == (
        "old-test-key" if provider_id == "openrouter" else None
    )
    assert store.load_config()["unrelated_setting"] == "leave-me"
    assert store.load_secrets()["exa_api_key"] == "unrelated-test-key"
    assert "new-test-key" not in str(caught.value)


@pytest.mark.parametrize("after", [False, True])
def test_partial_deletions_report_removed_keys_and_resolve_legacy_fallbacks(
    config_env, monkeypatch, after
):
    store, manager = config_env
    store.write_config("model_name", "legacy/model")
    store.update_provider_profile("openrouter", {"model_name": "profile/model"})
    manager.reload()
    fail_replace(monkeypatch, store.secrets_file, after=after)

    with pytest.raises(ConfigApplyError) as caught:
        manager.apply(ConfigPatch({"model_name": None, "api_key": None}))

    assert caught.value.saved_keys == (
        ("api_key", "model_name") if after else ("model_name",)
    )
    assert caught.value.unsaved_keys == (() if after else ("api_key",))
    assert manager.current.model.model_name == "legacy/model"
    assert manager.provider_profile("openrouter") == {}
    assert manager.provider_runtime().api_key == (None if after else "old-test-key")


@pytest.mark.parametrize("use_cli", [False, True])
def test_recovery_respects_overrides_when_reporting_disk_values(
    config_env, monkeypatch, use_cli
):
    store, _manager = config_env
    environment = {
        "MODEL_NAME": "environment/model",
        "OPENAI_API_KEY": "environment-key",
    }
    overrides = (
        CliOverrides(model_name="cli/model", api_key="cli-key") if use_cli else None
    )
    manager = ConfigManager.load(store.config_dir, overrides, environment=environment)
    fail_replace(monkeypatch, store.secrets_file, after=True)

    with pytest.raises(ConfigApplyError) as caught:
        manager.apply(
            ConfigPatch({"model_name": "stored/model", "api_key": "stored-key"})
        )

    assert caught.value.saved_keys == ("api_key", "model_name")
    assert caught.value.unsaved_keys == ()
    assert manager.current.model.model_name == (
        "cli/model" if use_cli else "environment/model"
    )
    assert manager.provider_runtime().api_key == (
        "cli-key" if use_cli else "environment-key"
    )
    assert caught.value.update.reload_selected is False
    assert (
        store.load_config()["provider_profiles"]["openrouter"]["model_name"]
        == "stored/model"
    )
    assert store.load_secrets()["openai_api_key"] == "stored-key"


def test_previously_saved_requested_value_is_reported_when_write_fails(
    config_env, monkeypatch
):
    store, manager = config_env
    fail_replace(monkeypatch, store.secrets_file)

    with pytest.raises(ConfigApplyError) as caught:
        manager.apply(ConfigPatch({"format_on_write": True, "api_key": "new-test-key"}))

    assert caught.value.saved_keys == ("format_on_write",)
    assert caught.value.unsaved_keys == ("api_key",)
    assert caught.value.update.future_sessions_changed is False


def test_retry_completes_unsaved_changes_without_losing_saved_settings(
    config_env, monkeypatch
):
    store, manager = config_env
    with monkeypatch.context() as patcher:
        fail_replace(patcher, store.secrets_file)
        with pytest.raises(ConfigApplyError):
            manager.apply(
                ConfigPatch({"format_on_write": False, "api_key": "new-test-key"})
            )

    update = manager.apply(ConfigPatch({"api_key": "new-test-key"}))

    assert update.previous.settings.format_on_write is False
    assert update.current.settings.format_on_write is False
    assert update.options_changed is False
    assert update.provider_runtime_changed is True
    assert store.load_secrets()["openai_api_key"] == "new-test-key"


@pytest.mark.parametrize(
    "problem", ["invalid_json", "non_object", "invalid_settings", "unreadable"]
)
@pytest.mark.parametrize("document", ["config", "secrets"])
def test_failed_recovery_reports_unknown_state_and_keeps_last_valid_snapshot(
    config_env, monkeypatch, problem, document
):
    store, manager = config_env
    runtime = InteractiveRuntime(SimpleNamespace(config=manager), resume_thread_id=None)
    selected = SimpleNamespace(apply_runtime_config=Mock(), reload_model=AsyncMock())
    unreadable_path = store.configs_file if document == "config" else store.secrets_file
    previous, previous_provider, previous_settings = (
        manager.current,
        manager.provider_runtime(),
        manager.settings,
    )
    original_read = Path.read_text
    original_write = store_module._atomic_write

    def fail_secret_write(path, value, *, secret):
        if path != store.secrets_file:
            return original_write(path, value, secret=secret)
        if problem == "unreadable":

            def read(path, *args, **kwargs):
                if path == unreadable_path:
                    raise PermissionError("private diagnostic")
                return original_read(path, *args, **kwargs)

            monkeypatch.setattr(Path, "read_text", read)
        else:
            payload = {
                "invalid_json": "{",
                "non_object": "[]",
                "invalid_settings": '{"reflection_token_ratio": 2}',
            }[problem]
            unreadable_path.write_text(payload, encoding="utf-8")
        raise OSError("private write diagnostic")

    monkeypatch.setattr(store_module, "_atomic_write", fail_secret_write)
    with pytest.raises(ConfigApplyError) as caught:
        asyncio.run(
            runtime.apply_config(
                ConfigPatch({"format_on_write": False, "api_key": "new-test-key"}),
                selected=selected,
            )
        )

    error = caught.value
    assert error.update is None
    assert error.saved_keys is None
    assert error.unsaved_keys is None
    assert "saved values are unknown" in str(error)
    assert "private" not in str(error)
    assert "new-test-key" not in str(error)
    assert isinstance(error.__cause__, OSError)
    assert manager.current == previous
    assert manager.provider_runtime() == previous_provider
    assert manager.settings == previous_settings
    selected.apply_runtime_config.assert_not_called()
    selected.reload_model.assert_not_awaited()


def test_secret_permission_failure_after_replacement_reports_saved_values(
    config_env, monkeypatch
):
    store, manager = config_env
    original_chmod = os.chmod

    def chmod(path, mode):
        if Path(path) == store.secrets_file:
            raise PermissionError("injected permission failure")
        return original_chmod(path, mode)

    monkeypatch.setattr(os, "chmod", chmod)
    with pytest.raises(ConfigApplyError) as caught:
        manager.apply(
            ConfigPatch({"format_on_write": False, "api_key": "new-test-key"})
        )

    assert isinstance(caught.value.__cause__, PermissionError)
    assert caught.value.saved_keys == ("api_key", "format_on_write")
    assert caught.value.unsaved_keys == ()
    assert manager.current.settings.format_on_write is False
    assert manager.provider_runtime().api_key == "new-test-key"
    assert store.secrets_file.stat().st_mode & 0o777 == 0o600
    assert not list(store.config_dir.glob("*.tmp"))


@pytest.mark.parametrize(
    "values,document,after,applies,reloads",
    [
        ({"format_on_write": False, "api_key": "new-test-key"}, "secrets", False, 1, 0),
        (
            {"model_name": "next/model", "api_key": "new-test-key"},
            "secrets",
            False,
            0,
            1,
        ),
        (
            {
                "format_on_write": False,
                "model_name": "next/model",
                "api_key": "new-test-key",
            },
            "secrets",
            False,
            1,
            1,
        ),
        ({"api_key": "new-test-key"}, "secrets", True, 0, 1),
        ({"format_on_write": False, "api_key": "new-test-key"}, "config", False, 0, 0),
    ],
)
def test_partial_save_updates_selected_session_from_verified_effects(
    config_env, monkeypatch, values, document, after, applies, reloads
):
    store, manager = config_env
    runtime = InteractiveRuntime(SimpleNamespace(config=manager), resume_thread_id=None)
    selected = SimpleNamespace(apply_runtime_config=Mock(), reload_model=AsyncMock())
    path = store.configs_file if document == "config" else store.secrets_file
    fail_replace(monkeypatch, path, after=after)

    with pytest.raises(ConfigApplyError):
        asyncio.run(runtime.apply_config(ConfigPatch(values), selected=selected))

    assert selected.apply_runtime_config.call_count == applies
    if applies:
        selected.apply_runtime_config.assert_called_once_with(manager.current)
    assert selected.reload_model.await_count == reloads


@pytest.mark.parametrize("step", ["options", "model"])
def test_selected_session_failure_preserves_original_save_failure(
    config_env, monkeypatch, step
):
    store, manager = config_env
    runtime = InteractiveRuntime(SimpleNamespace(config=manager), resume_thread_id=None)
    selected = SimpleNamespace(apply_runtime_config=Mock(), reload_model=AsyncMock())
    failure = RuntimeError("secret-from-session")
    if step == "options":
        selected.apply_runtime_config.side_effect = failure
    else:
        selected.reload_model.side_effect = failure
    fail_replace(monkeypatch, store.secrets_file)

    with pytest.raises(ConfigApplyError) as caught:
        asyncio.run(
            runtime.apply_config(
                ConfigPatch(
                    {
                        "format_on_write": False,
                        "model_name": "next/model",
                        "api_key": "new-test-key",
                    }
                ),
                selected=selected,
            )
        )

    error = caught.value
    assert isinstance(error.__cause__, OSError)
    assert error.saved_keys == ("format_on_write", "model_name")
    assert error.unsaved_keys == ("api_key",)
    assert "Could not apply saved configuration to the selected session" in str(error)
    assert "secret-from-session" not in str(error)
    assert "new-test-key" not in str(error)
