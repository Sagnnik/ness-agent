from __future__ import annotations

import asyncio

import pytest

from ness_cli.config import ConfigManager, ConfigPatch
from ness_cli.providers.registry import ModelRequest, ProviderRegistry
from tests.test_cli.fakes.providers import FakeProvider


def _registry(isolated_cli_env) -> tuple[ProviderRegistry, ConfigManager]:
    manager = ConfigManager.load(isolated_cli_env.config, environment={})
    return ProviderRegistry(paths=isolated_cli_env.paths, config=manager), manager


def test_builtin_provider_ids_are_stable(isolated_cli_env) -> None:
    manager = ConfigManager.load(isolated_cli_env.config, environment={})
    registry = ProviderRegistry.load(paths=isolated_cli_env.paths, config=manager)

    assert registry.provider_ids() == ("codex", "openrouter", "opencode")


def test_registries_do_not_share_adapter_instances(isolated_cli_env) -> None:
    manager_a = ConfigManager.load(isolated_cli_env.config, environment={})
    manager_b = ConfigManager.load(isolated_cli_env.config, environment={})
    first = ProviderRegistry.load(paths=isolated_cli_env.paths, config=manager_a)
    second = ProviderRegistry.load(paths=isolated_cli_env.paths, config=manager_b)

    assert first.get("openrouter") is not second.get("openrouter")
    assert first.get("codex") is not second.get("codex")


def test_active_provider_tracks_immutable_runtime_config(isolated_cli_env) -> None:
    registry, manager = _registry(isolated_cli_env)
    adapters: dict[str, FakeProvider] = {}
    for provider_id in ("openrouter", "codex"):
        registry.register(
            provider_id,
            lambda config, provider_id=provider_id: adapters.setdefault(
                provider_id, FakeProvider(provider_id, config)
            ),
        )

    assert registry.active_provider_id == "openrouter"
    manager.apply(ConfigPatch({"provider_id": "codex", "model_name": "test/model"}))

    assert registry.active_provider_id == "codex"
    assert registry.active().id == "codex"


def test_model_creation_uses_adapter_contract(isolated_cli_env) -> None:
    registry, _manager = _registry(isolated_cli_env)
    adapter: FakeProvider | None = None

    def factory(config):
        nonlocal adapter
        adapter = FakeProvider("openrouter", config)
        return adapter

    registry.register("openrouter", factory)
    result = registry.create_model(
        ModelRequest("thread-7", "reflection", "test/model", "high")
    )

    assert result == {
        "thread_id": "thread-7",
        "model_name": "test/model",
        "reasoning_effort": "high",
        "session_suffix": "reflection",
    }
    assert adapter is not None


def test_unknown_provider_errors_name_the_value(isolated_cli_env) -> None:
    registry, _manager = _registry(isolated_cli_env)
    registry.register(
        "openrouter", lambda config: FakeProvider("openrouter", config)
    )

    with pytest.raises(ValueError, match="unknown model provider: missing"):
        registry.get("missing")
    with pytest.raises(ValueError, match="unknown model provider: missing"):
        registry.create_model(
            ModelRequest("thread", "main", "model", provider_id="missing")
        )


def test_cold_openrouter_catalog_accepts_glm_5_3_flash(isolated_cli_env) -> None:
    manager = ConfigManager.load(isolated_cli_env.config, environment={})
    registry = ProviderRegistry.load(paths=isolated_cli_env.paths, config=manager)

    model = registry.create_model(
        ModelRequest(
            "thread",
            "main",
            "z-ai/glm-5.3-flash",
            "max",
            provider_id="openrouter",
        )
    )
    info = registry.model_info(
        "z-ai/glm-5.3-flash",
        provider_id="openrouter",
    )

    assert model is not None
    assert info is not None
    assert info.context_window == 1_310_720
    assert info.default_reasoning_effort == "max"


def test_close_closes_each_constructed_provider_once(isolated_cli_env) -> None:
    registry, _manager = _registry(isolated_cli_env)
    adapters: dict[str, FakeProvider] = {}
    for provider_id in ("one", "two"):
        registry.register(
            provider_id,
            lambda config, provider_id=provider_id: adapters.setdefault(
                provider_id, FakeProvider(provider_id, config)
            ),
        )
        registry.get(provider_id)

    asyncio.run(registry.close())
    asyncio.run(registry.close())

    assert {key: value.close_calls for key, value in adapters.items()} == {
        "one": 1,
        "two": 1,
    }


@pytest.mark.parametrize("base_url", [None, "http://localhost:1234/v1"])
def test_runtime_constructs_uncatalogued_models(isolated_cli_env, base_url):
    from ness_cli.config import CliOverrides
    from ness_cli.runtime import _session_models

    manager = ConfigManager.load(
        isolated_cli_env.paths,
        CliOverrides(
            model_name="my-local-model",
            reflection_model_name="my-reflection-model",
            api_key="test-only-key",
            base_url=base_url,
        ),
        environment={},
    )
    registry = ProviderRegistry.load(paths=isolated_cli_env.paths, config=manager)
    assert registry.model_info("my-local-model") is None
    assert registry.model_info("my-reflection-model") is None

    models = _session_models(registry, manager.current, thread_id="custom-endpoint")
    assert models.model.model_name == "my-local-model"
    assert models.reflection_model.model_name == "my-reflection-model"
    if base_url is not None:
        assert models.model.openrouter_api_base == base_url
        assert models.reflection_model.openrouter_api_base == base_url
    assert models.context_window is None
    assert models.vision is None
    asyncio.run(registry.close())
