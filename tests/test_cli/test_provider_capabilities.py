from __future__ import annotations

import asyncio

import pytest
from typer.testing import CliRunner

from ness_cli import cli
from ness_cli.config import ConfigManager, ConfigPatch
from ness_cli.providers.codex.catalog import load_models
from ness_cli.providers.model_metadata import (
    context_window_for,
    fallback_metadata_for,
    matching_model_family,
)
from ness_cli.providers.openrouter.catalog import OpenRouterCatalog, parse_catalog
from ness_cli.providers.registry import ProviderRegistry


def test_cli_rejects_opencode_minimal_before_runtime(monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "opencode")

    async def run(*_args, **_kwargs):
        pytest.fail("invalid effort must not reach the runtime")

    monkeypatch.setattr("ness_cli.headless.run_headless", run)
    result = CliRunner().invoke(
        cli.app,
        [
            "--print",
            "--model",
            "gpt-5.6-luna",
            "--reasoning-effort",
            "minimal",
            "hello",
        ],
    )
    assert result.exit_code == 2, result.output
    assert "high, medium, low, none" in result.stderr
    assert "opencode" in result.stderr


@pytest.mark.parametrize("name", ["gpt-5.60", "gpt-5.9", "not-gpt-5.4", "o30"])
def test_unknown_versions_do_not_inherit_family_context(isolated_cli_env, name):
    assert fallback_metadata_for(name) is None
    assert context_window_for(name) is None
    manager = ConfigManager.load(isolated_cli_env.paths, environment={})
    registry = ProviderRegistry.load(paths=isolated_cli_env.paths, config=manager)
    try:
        assert registry.model_info(f"openai/{name}") is None
    finally:
        asyncio.run(registry.close())


def test_luna_context_is_not_the_original_gpt5_window(isolated_cli_env):
    manager = ConfigManager.load(isolated_cli_env.paths, environment={})
    registry = ProviderRegistry.load(paths=isolated_cli_env.paths, config=manager)
    try:
        assert (
            registry.model_info("gpt-5.6-luna", provider_id="opencode").context_window
            == 1_050_000
        )
        assert (
            registry.model_info("gpt-5.6-luna", provider_id="codex").context_window
            == 272_000
        )
    finally:
        asyncio.run(registry.close())


@pytest.mark.parametrize(
    "provider,model,effort",
    [
        ("openrouter", "openai/gpt-5.4", "none"),
        ("openrouter", "openai/gpt-5.4", "xhigh"),
        ("codex", "gpt-5.6-luna", "max"),
        ("opencode", "gpt-5.6-luna", "none"),
        ("opencode", "glm-5.2", "max"),
        ("opencode", "deepseek-v4-flash", "xhigh"),
    ],
)
def test_cli_accepts_selected_provider_capabilities(
    monkeypatch, provider, model, effort
):
    monkeypatch.setenv("MODEL_PROVIDER", provider)

    async def run(_prompt, *, options):
        assert options.overrides.reasoning_effort == effort
        return 0

    monkeypatch.setattr("ness_cli.headless.run_headless", run)
    result = CliRunner().invoke(
        cli.app, ["--print", "--model", model, "--reasoning-effort", effort, "hello"]
    )
    assert result.exit_code == 0, result.output


@pytest.mark.parametrize(
    "provider,model,effort",
    [
        ("openrouter", "openai/gpt-5.4", "minimal"),
        ("openrouter", "openai/gpt-5", "none"),
        ("codex", "gpt-5.6-luna", "none"),
        ("codex", "gpt-5.6-luna", "minimal"),
        ("codex", "gpt-5.6-luna", "ultra"),
        ("opencode", "gpt-5.6-luna", "xhigh"),
        ("opencode", "deepseek-v4-flash", "low"),
    ],
)
def test_cli_rejects_unsupported_efforts(monkeypatch, provider, model, effort):
    monkeypatch.setenv("MODEL_PROVIDER", provider)

    async def run(*_args, **_kwargs):
        pytest.fail("invalid effort must not reach the runtime")

    monkeypatch.setattr("ness_cli.headless.run_headless", run)
    result = CliRunner().invoke(
        cli.app, ["--print", "--model", model, "--reasoning-effort", effort, "hello"]
    )
    assert result.exit_code == 2, result.output
    assert f"provider '{provider}'" in result.stderr
    allowed = result.stderr.split("reasoning effort must be one of:", 1)[1]
    assert effort not in allowed


def test_cli_uses_saved_provider_and_model_profile(isolated_cli_env, monkeypatch):
    manager = ConfigManager.load(isolated_cli_env.paths, environment={})
    manager.apply(
        ConfigPatch({"provider_id": "opencode", "model_name": "gpt-5.6-luna"})
    )

    async def run(_prompt, *, options):
        assert options.overrides.reasoning_effort == "none"
        assert options.overrides.model_name is None
        return 0

    monkeypatch.setattr("ness_cli.headless.run_headless", run)
    result = CliRunner().invoke(
        cli.app, ["--print", "--reasoning-effort", "none", "hello"]
    )
    assert result.exit_code == 0, result.output


def test_cli_environment_provider_overrides_saved_provider(
    isolated_cli_env, monkeypatch
):
    manager = ConfigManager.load(isolated_cli_env.paths, environment={})
    manager.apply(ConfigPatch({"provider_id": "codex", "model_name": "gpt-5.6-luna"}))
    monkeypatch.setenv("MODEL_PROVIDER", "opencode")

    async def run(_prompt, *, options):
        return 0

    monkeypatch.setattr("ness_cli.headless.run_headless", run)
    result = CliRunner().invoke(
        cli.app,
        ["--print", "--model", "gpt-5.6-luna", "--reasoning-effort", "none", "hello"],
    )
    assert result.exit_code == 0, result.output


@pytest.mark.parametrize("effort,exit_code", [("adaptive", 0), ("high", 2)])
def test_cli_uses_cached_provider_efforts(
    isolated_cli_env, monkeypatch, effort, exit_code
):
    model_id = "openai/gpt-5.4"
    records = parse_catalog(
        {
            "data": [
                {
                    "id": model_id,
                    "context_length": 123_456,
                    "architecture": {
                        "input_modalities": ["text"],
                        "output_modalities": ["text"],
                    },
                    "supported_parameters": ["tools"],
                    "reasoning": {
                        "supported_efforts": ["adaptive"],
                        "default_effort": "adaptive",
                    },
                }
            ]
        }
    )
    monkeypatch.setattr(
        "ness_cli.providers.openrouter.catalog.fetch_catalog", lambda: records
    )
    catalog = OpenRouterCatalog.from_cache_dir(isolated_cli_env.paths.cache_dir.parent)
    asyncio.run(catalog.refresh(force=True))

    async def run(_prompt, *, options):
        assert exit_code == 0
        return 0

    monkeypatch.setattr("ness_cli.headless.run_headless", run)
    result = CliRunner().invoke(
        cli.app, ["--print", "--model", model_id, "--reasoning-effort", effort, "hello"]
    )
    assert result.exit_code == exit_code, result.output
    manager = ConfigManager.load(isolated_cli_env.paths, environment={})
    registry = ProviderRegistry.load(paths=isolated_cli_env.paths, config=manager)
    try:
        assert registry.model_info(model_id).context_window == 123_456
    finally:
        asyncio.run(registry.close())


@pytest.mark.parametrize("effort", ["none", "minimal"])
def test_cli_defers_uncatalogued_efforts_to_endpoint(monkeypatch, effort):
    async def run(_prompt, *, options):
        return 0

    monkeypatch.setattr("ness_cli.headless.run_headless", run)
    result = CliRunner().invoke(
        cli.app,
        ["--print", "--model", "local-custom", "--reasoning-effort", effort, "hello"],
    )
    assert result.exit_code == 0, result.output


@pytest.mark.parametrize("effort,exit_code", [("none", 0), ("minimal", 2)])
def test_cli_closes_validation_registry_on_success_and_failure(
    monkeypatch, effort, exit_code
):
    monkeypatch.setenv("MODEL_PROVIDER", "opencode")
    original = ProviderRegistry.close
    closed = []

    async def close(self):
        closed.append(self)
        await original(self)

    async def run(_prompt, *, options):
        return 0

    monkeypatch.setattr(ProviderRegistry, "close", close)
    monkeypatch.setattr("ness_cli.headless.run_headless", run)
    result = CliRunner().invoke(
        cli.app,
        ["--print", "--model", "gpt-5.6-luna", "--reasoning-effort", effort, "hello"],
    )
    assert result.exit_code == exit_code, result.output
    assert len(closed) == 1
    assert closed[0]._closed


def test_family_matching_prefers_specific_names_and_accepts_snapshots():
    assert (
        matching_model_family("OPENAI/GPT-5.4-2026-03-05", ["gpt-5", "gpt-5.4"])
        == "gpt-5.4"
    )
    assert context_window_for("openai/gpt-5.4-2026-03-05") == 1_050_000
    assert context_window_for("z-ai/glm-5.3-flash:free") == 1_310_720


@pytest.mark.parametrize("model", ["gpt-5.4", "gpt-5.6-codex", "gpt-5.9"])
def test_unknown_codex_capabilities_do_not_use_api_defaults(isolated_cli_env, model):
    manager = ConfigManager.load(isolated_cli_env.paths, environment={})
    registry = ProviderRegistry.load(paths=isolated_cli_env.paths, config=manager)
    try:
        info = registry.model_info(model, provider_id="codex")
        assert info.context_window is None
        assert info.reasoning_efforts == ()
        registry.validate_reasoning_effort(model, "none", provider_id="codex")
    finally:
        asyncio.run(registry.close())


@pytest.mark.parametrize(
    "window_key",
    ["contextWindow", "contextWindowTokens", "contextLength", "context_length"],
)
def test_codex_catalog_overrides_subscription_fallback(isolated_cli_env, window_key):
    class Server:
        async def request(self, method, params):
            return {
                "data": [
                    {
                        "model": "gpt-5.6-luna",
                        window_key: 777_000,
                        "supportedReasoningEfforts": [{"reasoningEffort": "none"}],
                    }
                ]
            }

    records = asyncio.run(load_models(Server()))
    assert records[0].context_window == 777_000
    manager = ConfigManager.load(isolated_cli_env.paths, environment={})
    registry = ProviderRegistry.load(paths=isolated_cli_env.paths, config=manager)
    adapter = registry.get("codex")
    adapter._models = records
    try:
        registry.validate_reasoning_effort("gpt-5.6-luna", "none", provider_id="codex")
        with pytest.raises(ValueError):
            registry.validate_reasoning_effort(
                "gpt-5.6-luna", "high", provider_id="codex"
            )
        assert (
            registry.model_info("gpt-5.6-luna", provider_id="codex").context_window
            == 777_000
        )
    finally:
        asyncio.run(registry.close())
