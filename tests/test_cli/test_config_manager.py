from __future__ import annotations

import json

import pytest

from ness_cli.config.manager import ConfigManager
from ness_cli.config.store import ConfigStore
from ness_cli.config.types import CliOverrides, ConfigPatch


def test_persisted_config_and_secrets_load_without_environment(
    isolated_cli_env,
) -> None:
    store = ConfigStore(isolated_cli_env.config)
    store.write_config("model_name", "persisted/model")
    store.write_secret("openai_api_key", "sk-persisted")

    manager = ConfigManager.load(isolated_cli_env.config, environment={})

    assert manager.current.model.model_name == "persisted/model"
    assert manager.provider_runtime().api_key == "sk-persisted"


@pytest.mark.parametrize("name", ["OPENCODE_GO_API_KEY", "OPENCODE_API_KEY"])
def test_opencode_credential_environment_aliases(
    isolated_cli_env, name
) -> None:
    store = ConfigStore(isolated_cli_env.config)
    store.write_secret("opencode_api_key", "stored-key")

    stored = ConfigManager.load(
        isolated_cli_env.config,
        CliOverrides(provider_id="opencode"),
        environment={},
    )
    assert stored.provider_runtime().api_key == "stored-key"

    manager = ConfigManager.load(
        isolated_cli_env.config,
        CliOverrides(provider_id="opencode"),
        environment={name: "environment-key"},
    )

    assert manager.provider_runtime().api_key == "environment-key"


def test_project_dotenv_is_not_loaded_or_migrated(isolated_cli_env) -> None:
    dotenv = isolated_cli_env.project / ".env"
    dotenv.write_text(
        "MODEL_NAME=dotenv/model\nOPENAI_API_KEY=dotenv-secret\n",
        encoding="utf-8",
    )

    manager = ConfigManager.load(isolated_cli_env.config, environment={})

    assert manager.current.model.model_name == "deepseek/deepseek-v4-flash"
    assert manager.provider_runtime().api_key is None
    assert not (isolated_cli_env.config / "configs.json").exists()
    assert not (isolated_cli_env.config / "secrets.json").exists()
    assert "dotenv-secret" in dotenv.read_text(encoding="utf-8")


def test_environment_beats_persisted_values(isolated_cli_env) -> None:
    store = ConfigStore(isolated_cli_env.config)
    store.write_config("model_name", "persisted/model")
    store.update_provider_profile(
        "openrouter", {"model_name": "profile/model", "reasoning_effort": "high"}
    )

    manager = ConfigManager.load(
        isolated_cli_env.config,
        environment={"MODEL_NAME": "environment/model", "REASONING_EFFORT": "low"},
    )

    assert manager.current.model.model_name == "environment/model"
    assert manager.current.model.reasoning_effort == "low"


def test_cli_overrides_beat_environment_and_persisted_values(
    isolated_cli_env,
) -> None:
    store = ConfigStore(isolated_cli_env.config)
    store.write_config("model_name", "persisted/model")
    store.update_provider_profile("openrouter", {"model_name": "profile/model"})

    manager = ConfigManager.load(
        isolated_cli_env.config,
        CliOverrides(model_name="cli/model", reasoning_effort="xhigh"),
        environment={"MODEL_NAME": "environment/model"},
    )

    assert manager.current.model.model_name == "cli/model"
    assert manager.current.model.reasoning_effort == "xhigh"


def test_changing_provider_selects_its_profile(isolated_cli_env) -> None:
    store = ConfigStore(isolated_cli_env.config)
    store.update_provider_profile(
        "codex", {"model_name": "gpt-profile", "reasoning_effort": "high"}
    )
    manager = ConfigManager.load(isolated_cli_env.config, environment={})

    update = manager.apply(ConfigPatch({"provider_id": "codex"}))

    assert update.current.model.provider_id == "codex"
    assert update.current.model.model_name == "gpt-profile"
    assert update.current.model.reflection_model_name == "gpt-profile"
    assert update.current.model.reasoning_effort == "high"
    assert update.reload_selected is True


def test_provider_profile_returns_a_detached_copy(isolated_cli_env) -> None:
    store = ConfigStore(isolated_cli_env.config)
    store.update_provider_profile(
        "codex",
        {
            "model_name": "gpt-profile",
            "reasoning_effort": "high",
        },
    )
    manager = ConfigManager.load(isolated_cli_env.config, environment={})

    profile = manager.provider_profile("codex")
    profile["model_name"] = "mutated"

    assert manager.provider_profile("codex")["model_name"] == "gpt-profile"
    assert manager.provider_profile("missing") == {}


def test_apply_reports_reload_future_session_and_restart_effects(
    isolated_cli_env,
) -> None:
    manager = ConfigManager.load(isolated_cli_env.config, environment={})

    model_update = manager.apply(ConfigPatch({"model_name": "next/model"}))
    option_update = manager.apply(ConfigPatch({"enable_approval": False}))
    restart_update = manager.apply(ConfigPatch({"ness_dir": "state"}))

    assert model_update.model_changed is True
    assert model_update.reload_selected is True
    assert model_update.future_sessions_changed is True
    assert option_update.model_changed is False
    assert option_update.reload_selected is False
    assert option_update.future_sessions_changed is True
    assert restart_update.restart_required is True


def test_failed_validation_writes_neither_document(isolated_cli_env) -> None:
    manager = ConfigManager.load(isolated_cli_env.config, environment={})

    with pytest.raises(ValueError):
        manager.apply(
            ConfigPatch(
                {
                    "reflection_token_ratio": 2,
                    "api_key": "must-not-be-written",
                }
            )
        )

    assert not (isolated_cli_env.config / "configs.json").exists()
    assert not (isolated_cli_env.config / "secrets.json").exists()


def test_secret_and_ordinary_values_use_their_own_documents(
    isolated_cli_env,
) -> None:
    manager = ConfigManager.load(isolated_cli_env.config, environment={})

    manager.apply(
        ConfigPatch(
            {
                "api_key": "sk-secret",
                "openrouter_session_id": "session-7",
            }
        )
    )

    configs = json.loads(
        (isolated_cli_env.config / "configs.json").read_text(encoding="utf-8")
    )
    secrets = json.loads(
        (isolated_cli_env.config / "secrets.json").read_text(encoding="utf-8")
    )
    assert configs["openrouter_session_id"] == "session-7"
    assert "openai_api_key" not in configs
    assert secrets == {"openai_api_key": "sk-secret"}


def test_goal_judge_and_session_id_round_trip(isolated_cli_env) -> None:
    manager = ConfigManager.load(isolated_cli_env.config, environment={})
    manager.apply(
        ConfigPatch(
            {
                "goal_judge_model": "judge/model",
                "openrouter_session_id": "stable-session",
            }
        )
    )

    reloaded = ConfigManager.load(isolated_cli_env.config, environment={})

    assert reloaded.current.settings.goal_judge_model == "judge/model"
    assert reloaded.provider_runtime().session_id == "stable-session"


def test_returned_settings_cannot_mutate_manager_state(isolated_cli_env) -> None:
    store = ConfigStore(isolated_cli_env.config)
    store.update_provider_profile("openrouter", {"model_name": "stored/model"})
    manager = ConfigManager.load(isolated_cli_env.config, environment={})

    snapshot = manager.settings
    snapshot.provider_profiles["openrouter"]["model_name"] = "mutated/model"

    assert manager.current.model.model_name == "stored/model"
    assert manager.settings.provider_profiles["openrouter"]["model_name"] == (
        "stored/model"
    )


def test_runtime_snapshot_is_immutable(isolated_cli_env) -> None:
    manager = ConfigManager.load(isolated_cli_env.config, environment={})

    with pytest.raises(AttributeError):
        manager.current.model.model_name = "mutated/model"
