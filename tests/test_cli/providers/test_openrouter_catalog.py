from __future__ import annotations

import asyncio
import json
from dataclasses import asdict

import pytest

from ness_cli.config import ProviderRuntimeConfig
from ness_cli.providers.openrouter.adapter import OpenRouterProviderAdapter
from ness_cli.providers.openrouter.catalog import (
    CATALOG_VERSION,
    ModelRecord,
    OpenRouterCatalog,
    OFFLINE_MODEL_IDS,
    parse_catalog,
)
from ness_cli.providers.openrouter.messages import OpenRouterAnthropicMessages


def _record(model_id: str = "vendor/model") -> ModelRecord:
    return ModelRecord(
        id=model_id,
        name="Model",
        input_modalities=("text", "image"),
        output_modalities=("text",),
        supported_parameters=("tools",),
        reasoning_efforts=("high", "max"),
        default_reasoning_effort="high",
    )


def _runtime(**overrides) -> ProviderRuntimeConfig:
    values = {
        "provider_id": "openrouter",
        "api_key": "explicit-key",
        "base_url": "https://openrouter.ai/api/v1",
        "max_retries": 2,
        "cache_ttl": "1h",
        "anthropic_messages": True,
        "session_id": "explicit-session",
    }
    values.update(overrides)
    return ProviderRuntimeConfig(**values)


def test_catalog_filters_models_and_preserves_provider_metadata() -> None:
    records = parse_catalog(
        {
            "data": [
                {
                    "id": "z-ai/glm-5.2",
                    "name": "GLM 5.2",
                    "context_length": 1_048_576,
                    "architecture": {
                        "input_modalities": ["text", "image"],
                        "output_modalities": ["text"],
                    },
                    "supported_parameters": ["tools", "reasoning_effort"],
                    "pricing": {
                        "prompt": "0.000001",
                        "completion": "0.000003",
                        "input_cache_read": "0.0000002",
                    },
                },
                {
                    "id": "image/only",
                    "architecture": {
                        "input_modalities": ["text"],
                        "output_modalities": ["image"],
                    },
                    "supported_parameters": ["tools"],
                },
            ]
        },
        {
            "zhipuai": {
                "models": {
                    "glm-5.2": {
                        "reasoning_options": [
                            {"type": "effort", "values": ["high", "max"]}
                        ]
                    }
                }
            }
        },
    )

    assert [item.id for item in records] == ["z-ai/glm-5.2"]
    assert records[0].reasoning_efforts == ("high", "max")
    assert records[0].supports_vision is True
    assert records[0].input_price == 1.0
    assert records[0].cache_read_ratio == pytest.approx(0.2)


def test_catalog_prefers_openrouter_native_reasoning_metadata() -> None:
    record = parse_catalog(
        {
            "data": [
                {
                    "id": "z-ai/glm-5.3-flash",
                    "name": "GLM 5.3 Flash",
                    "context_length": 1_310_720,
                    "architecture": {
                        "input_modalities": ["text", "image", "video"],
                        "output_modalities": ["text"],
                    },
                    "supported_parameters": ["tools", "reasoning_effort"],
                    "pricing": {
                        "prompt": "0.000000075",
                        "completion": "0.00000025",
                        "input_cache_read": "0.000000015",
                    },
                    "reasoning": {
                        "mandatory": True,
                        "supported_efforts": ["max", "high", "low"],
                        "default_effort": "max",
                    },
                }
            ]
        },
        {
            "zhipuai": {
                "models": {
                    "glm-5.3-flash": {
                        "reasoning_options": [{"type": "effort", "values": ["high"]}]
                    }
                }
            }
        },
    )[0]

    assert record.context_length == 1_310_720
    assert record.input_price == pytest.approx(0.075)
    assert record.output_price == pytest.approx(0.25)
    assert record.cache_read_ratio == pytest.approx(0.2)
    assert record.supports_vision is True
    assert record.reasoning_efforts == ("max", "high", "low")
    assert record.default_reasoning_effort == "max"


def test_anthropic_catalog_records_select_messages_capability() -> None:
    record = parse_catalog(
        {
            "data": [
                {
                    "id": "anthropic/claude-sonnet-5",
                    "architecture": {
                        "input_modalities": ["text", "image"],
                        "output_modalities": ["text"],
                    },
                    "supported_parameters": ["tools"],
                }
            ]
        }
    )[0]

    assert record.supports_vision is True
    assert record.supports_anthropic_messages is True


def test_cold_and_corrupt_caches_use_offline_models(isolated_cli_env) -> None:
    path = isolated_cli_env.cache / "catalog.json"
    cold = OpenRouterCatalog(path)
    assert tuple(item.id for item in cold.models()) == OFFLINE_MODEL_IDS

    path.parent.mkdir(parents=True)
    path.write_text("{bad json", encoding="utf-8")
    corrupt = OpenRouterCatalog(path)
    assert tuple(item.id for item in corrupt.models()) == OFFLINE_MODEL_IDS

    flash = cold.model_record("z-ai/glm-5.3-flash")
    assert flash is not None
    assert flash.context_length == 1_310_720
    assert flash.reasoning_efforts == ("max", "high", "low")
    assert flash.default_reasoning_effort == "max"
    assert flash.supports_vision is True


def test_v2_catalog_reads_the_existing_v1_cache(isolated_cli_env) -> None:
    legacy_path = isolated_cli_env.cache / "openrouter-models-v1.json"
    legacy_path.parent.mkdir(parents=True)
    legacy_record = asdict(_record("z-ai/glm-5.3-flash"))
    legacy_record.pop("default_reasoning_effort")
    legacy_path.write_text(
        json.dumps(
            {
                "version": 1,
                "fetched_at": 123,
                "models": [legacy_record],
            }
        ),
        encoding="utf-8",
    )

    catalog = OpenRouterCatalog.from_cache_dir(isolated_cli_env.cache)

    assert [item.id for item in catalog.cached_models()] == ["z-ai/glm-5.3-flash"]
    assert catalog.model_record("z-ai/glm-5.3-flash") is not None
    assert catalog.is_stale(now=124) is False


def test_cache_expiration_uses_fetch_time(isolated_cli_env) -> None:
    path = isolated_cli_env.cache / "catalog.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "version": CATALOG_VERSION,
                "fetched_at": 100,
                "models": [asdict(_record())],
            }
        ),
        encoding="utf-8",
    )
    catalog = OpenRouterCatalog(path, ttl_seconds=50)

    assert catalog.is_stale(now=149) is False
    assert catalog.is_stale(now=150) is True


def test_refresh_writes_cache_and_recovers_offline(
    monkeypatch, isolated_cli_env
) -> None:
    path = isolated_cli_env.cache / "catalog.json"
    catalog = OpenRouterCatalog(path)
    monkeypatch.setattr(
        "ness_cli.providers.openrouter.catalog.fetch_catalog",
        lambda: [_record("fresh/model")],
    )

    refreshed = asyncio.run(catalog.refresh(force=True))

    assert refreshed.refreshed is True
    assert [item.id for item in catalog.models()] == ["fresh/model"]
    assert json.loads(path.read_text(encoding="utf-8"))["version"] == CATALOG_VERSION

    catalog.reset()
    monkeypatch.setattr(
        "ness_cli.providers.openrouter.catalog.fetch_catalog",
        lambda: (_ for _ in ()).throw(OSError("offline")),
    )
    failed = asyncio.run(catalog.refresh(force=True))
    assert failed.refreshed is False
    assert failed.error == "offline"
    assert [item.id for item in catalog.models()] == ["fresh/model"]


def test_adapter_builds_requests_only_from_supplied_runtime_config(
    isolated_cli_env,
) -> None:
    catalog = OpenRouterCatalog(isolated_cli_env.cache / "catalog.json")
    adapter = OpenRouterProviderAdapter(_runtime(), catalog=catalog)

    model = adapter.build_chat_model(
        "thread-ignored",
        model_name="anthropic/claude-sonnet-5",
        reasoning_effort="max",
        session_suffix="reflection",
    )

    assert isinstance(model, OpenRouterAnthropicMessages)
    assert model.api_key == "explicit-key"
    assert model.session_id == "explicit-session:reflection"
    assert model.cache_ttl == "1h"
    assert model.max_retries == 2
    assert model.reasoning == {"effort": "max"}


def test_adapter_coerces_unsupported_reasoning_to_model_default(
    isolated_cli_env,
    monkeypatch,
) -> None:
    path = isolated_cli_env.cache / "catalog.json"
    catalog = OpenRouterCatalog(path)
    monkeypatch.setattr(
        catalog,
        "model_record",
        lambda _model_id: _record("anthropic/claude-sonnet-5"),
    )
    adapter = OpenRouterProviderAdapter(_runtime(), catalog=catalog)

    model = adapter.build_chat_model(
        "thread",
        model_name="anthropic/claude-sonnet-5",
        reasoning_effort="extreme",
    )

    assert isinstance(model, OpenRouterAnthropicMessages)
    assert model.reasoning == {"effort": "high"}


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        ("openai/gpt-4o-mini", (128_000, (), None, True)),
        ("openai/gpt-5", (400_000, ("high", "medium", "low", "minimal"), "medium", True)),
        (
            "anthropic/claude-opus-4.8",
            (1_000_000, ("max", "xhigh", "high", "medium", "low"), "medium", True),
        ),
        ("google/gemini-2.5-pro", (1_048_576, (), None, True)),
        ("moonshotai/kimi-k2.7-code", (262_144, (), None, True)),
        ("z-ai/glm-5.2", (1_048_576, ("high", "max"), "high", False)),
    ],
)
def test_offline_catalog_preserves_expected_family_capabilities(tmp_path, model_id, expected):
    record = OpenRouterCatalog(tmp_path / "catalog.json").model_record(model_id)
    assert record is not None
    assert (
        record.context_length,
        record.reasoning_efforts,
        record.default_reasoning_effort,
        record.supports_vision,
    ) == expected


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        ("z-ai/glm-5.3-flash:free", (1_310_720, ("max", "high", "low"), "max", True)),
        (
            "openai/gpt-5.4-2026-03-05",
            (1_050_000, ("xhigh", "high", "medium", "low", "none"), "medium", True),
        ),
        ("deepseek/deepseek-v4-flash", (1_048_576, ("xhigh", "high"), "high", False)),
    ],
)
def test_known_metadata_and_snapshot_families_keep_capabilities(
    tmp_path, model_id, expected
):
    record = OpenRouterCatalog(tmp_path / "catalog.json").model_record(model_id)
    assert (
        record.context_length,
        record.reasoning_efforts,
        record.default_reasoning_effort,
        record.supports_vision,
    ) == expected


def test_native_catalog_metadata_overrides_shared_fallbacks():
    record = parse_catalog(
        {
            "data": [
                {
                    "id": "z-ai/glm-5.3-flash",
                    "context_length": 123_456,
                    "architecture": {
                        "input_modalities": ["text"],
                        "output_modalities": ["text"],
                    },
                    "supported_parameters": ["tools"],
                    "reasoning": {
                        "supported_efforts": ["custom"],
                        "default_effort": "custom",
                    },
                    "pricing": {"prompt": "0.000002", "completion": "0.000003"},
                }
            ]
        }
    )[0]
    assert record.context_length == 123_456
    assert not record.supports_vision
    assert record.reasoning_efforts == ("custom",)
    assert record.default_reasoning_effort == "custom"
    assert (record.input_price, record.output_price) == (2.0, 3.0)


def test_models_dev_reasoning_overrides_shared_fallbacks():
    record = parse_catalog(
        {
            "data": [
                {
                    "id": "z-ai/glm-5.2",
                    "architecture": {
                        "input_modalities": ["text"],
                        "output_modalities": ["text"],
                    },
                    "supported_parameters": ["tools"],
                }
            ]
        },
        {
            "zhipuai": {
                "models": {
                    "glm-5.2": {
                        "reasoning_options": [{"type": "effort", "values": ["custom"]}],
                    }
                }
            }
        },
    )[0]
    assert record.reasoning_efforts == ("custom",)
    assert record.default_reasoning_effort == "custom"
    assert record.context_length is None


@pytest.mark.parametrize("state", ["cold", "fresh", "stale"])
@pytest.mark.parametrize("refresh", [False, True])
def test_adapter_model_listing_requires_explicit_refresh(
    tmp_path, monkeypatch, state, refresh
):
    monkeypatch.setattr(
        "ness_cli.providers.openrouter.catalog.time.time", lambda: 1_000
    )
    path = tmp_path / "catalog.json"
    if state != "cold":
        path.write_text(
            json.dumps(
                {
                    "version": CATALOG_VERSION,
                    "fetched_at": 995 if state == "fresh" else 900,
                    "models": [asdict(_record("cached/model"))],
                }
            )
        )
    previous = path.read_bytes() if path.exists() else None
    catalog = OpenRouterCatalog(path, ttl_seconds=10)
    assert catalog.is_stale() == (state != "fresh")
    calls = []

    def fetch():
        calls.append("fetch")
        return [_record("fresh/model")]

    monkeypatch.setattr("ness_cli.providers.openrouter.catalog.fetch_catalog", fetch)
    adapter = OpenRouterProviderAdapter(_runtime(), catalog=catalog)
    models = asyncio.run(adapter.models(refresh=refresh))
    if refresh:
        assert calls == ["fetch"]
        assert [model.id for model in models] == ["fresh/model"]
        assert not catalog.is_stale()
    else:
        assert calls == []
        assert tuple(model.id for model in models) == (
            OFFLINE_MODEL_IDS if state == "cold" else ("cached/model",)
        )
        assert (path.read_bytes() if path.exists() else None) == previous


def test_direct_refresh_respects_ttl_and_fetches_only_when_stale(tmp_path, monkeypatch):
    now = [1_000]
    monkeypatch.setattr(
        "ness_cli.providers.openrouter.catalog.time.time", lambda: now[0]
    )
    catalog = OpenRouterCatalog(tmp_path / "catalog.json", ttl_seconds=10)
    calls = []

    def fetch():
        calls.append("fetch")
        return [_record("fresh/model")]

    monkeypatch.setattr("ness_cli.providers.openrouter.catalog.fetch_catalog", fetch)
    assert asyncio.run(catalog.refresh()).refreshed
    now[0] = 1_009
    assert not asyncio.run(catalog.refresh()).refreshed
    assert calls == ["fetch"]
    now[0] = 1_010
    assert asyncio.run(catalog.refresh()).refreshed
    assert calls == ["fetch", "fetch"]


@pytest.mark.parametrize("cached", [False, True])
def test_explicit_refresh_failure_keeps_cached_or_offline_models(
    tmp_path, monkeypatch, cached
):
    path = tmp_path / "catalog.json"
    if cached:
        path.write_text(
            json.dumps(
                {
                    "version": CATALOG_VERSION,
                    "fetched_at": 1,
                    "models": [asdict(_record("cached/model"))],
                }
            )
        )
    previous = path.read_bytes() if cached else None

    def fetch():
        raise OSError("offline")

    monkeypatch.setattr("ness_cli.providers.openrouter.catalog.fetch_catalog", fetch)
    adapter = OpenRouterProviderAdapter(_runtime(), catalog=OpenRouterCatalog(path))
    models = asyncio.run(adapter.models(refresh=True))
    assert tuple(model.id for model in models) == (
        ("cached/model",) if cached else OFFLINE_MODEL_IDS
    )
    assert (path.read_bytes() if path.exists() else None) == previous
