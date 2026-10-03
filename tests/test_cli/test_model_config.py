from __future__ import annotations

import pytest

from ness_cli.config import ConfigManager
from ness_cli.config.model import resolve_model_selection
from ness_cli.config.settings import Settings
from ness_cli.config.store import ConfigStore
from ness_cli.config.types import CliOverrides


def test_provider_profiles_do_not_leak_between_providers() -> None:
    settings = Settings(
        model_provider="codex",
        provider_profiles={
            "codex": {"model_name": "codex/model"},
            "openrouter": {"model_name": "openrouter/model"},
        },
    )

    codex = resolve_model_selection(settings, environment={})
    openrouter = resolve_model_selection(
        settings,
        CliOverrides(provider_id="openrouter"),
        environment={},
    )

    assert codex.model_name == "codex/model"
    assert openrouter.model_name == "openrouter/model"


def test_model_selection_rejects_empty_provider_and_model() -> None:
    with pytest.raises(ValueError, match="provider cannot be empty"):
        resolve_model_selection(Settings(model_provider=" "), environment={})

    with pytest.raises(ValueError, match="model name cannot be empty"):
        resolve_model_selection(Settings(model_name=" "), environment={})


def test_opencode_secret_never_becomes_openrouter_secret(isolated_cli_env) -> None:
    store = ConfigStore(isolated_cli_env.config)
    store.write_secret("openai_api_key", "openrouter-key")
    store.write_secret("opencode_api_key", "opencode-key")
    manager = ConfigManager.load(isolated_cli_env.config, environment={})

    assert manager.provider_runtime("openrouter").api_key == "openrouter-key"
    assert manager.provider_runtime("opencode").api_key == "opencode-key"
