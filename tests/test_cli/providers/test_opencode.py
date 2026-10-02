from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import httpx
import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from ness_cli.config import ConfigManager, ProviderRuntimeConfig
from ness_cli.config.store import ConfigStore
from ness_cli.providers.opencode.adapter import OpenCodeProviderAdapter, _usage_buckets
from ness_cli.providers.opencode.catalog import (
    FALLBACK_MODEL_IDS,
    model_infos,
    parse_models,
)
from ness_cli.providers.opencode.openai import OpenCodeChatOpenAI
from ness_cli.providers.openrouter.messages import OpenRouterAnthropicMessages
from ness_cli.providers.registry import ProviderRegistry


def _runtime(api_key: str | None = "sk-test") -> ProviderRuntimeConfig:
    return ProviderRuntimeConfig(
        provider_id="opencode",
        api_key=api_key,
        base_url=None,
        max_retries=2,
        cache_ttl=None,
        anthropic_messages=False,
        session_id=None,
    )


def test_catalog_normalizes_and_sorts_model_ids() -> None:
    assert parse_models(
        {
            "object": "list",
            "data": [
                {"id": "kimi-k2.7-code", "object": "model"},
                {"id": "glm-5.2", "object": "model"},
                {"id": "glm-5.2", "object": "model"},
            ],
        }
    ) == ("glm-5.2", "kimi-k2.7-code")


def test_catalog_rejects_malformed_responses() -> None:
    for payload in (None, {}, {"data": []}):
        try:
            parse_models(payload)
        except ValueError as error:
            assert "models" in str(error).lower()
        else:
            raise AssertionError("malformed catalog was accepted")


def test_models_use_offline_fallback_when_refresh_fails(monkeypatch) -> None:
    async def fail(**_kwargs):
        raise httpx.ConnectError("offline")

    monkeypatch.setattr("ness_cli.providers.opencode.adapter.fetch_model_ids", fail)
    adapter = OpenCodeProviderAdapter(_runtime())

    models = asyncio.run(adapter.models(refresh=True))

    assert tuple(item.id for item in models) == FALLBACK_MODEL_IDS
    assert models[0].is_default is True
    assert models[0].context_window == 1_048_576


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        ("gpt-5.6-luna", (1_050_000, ("high", "medium", "low", "none"), None, True)),
        ("glm-5.2", (1_048_576, ("high", "max"), None, False)),
        ("kimi-k2.7-code", (262_144, (), None, True)),
    ],
)
def test_cold_and_listed_opencode_metadata_match_expected_capabilities(model_id, expected):
    cold = OpenCodeProviderAdapter(_runtime()).model_info(model_id)
    listed = model_infos((model_id,), default_model="deepseek-v4-flash")[0]
    assert cold == listed
    assert (
        listed.context_window,
        listed.reasoning_efforts,
        listed.default_reasoning_effort,
        listed.supports_vision,
    ) == expected
    assert listed.input_price is None and listed.output_price is None


@pytest.mark.parametrize(
    ("model_id", "efforts", "default", "vision"),
    [
        ("gpt-5.6-luna", ("high", "medium", "low", "none"), None, True),
        ("deepseek-v4-flash", ("xhigh", "high"), None, False),
        ("glm-5.2", ("high", "max"), None, False),
        ("glm-5.3", (), None, False),
        ("grok-4.5", (), None, True),
        ("kimi-k3", (), None, True),
    ],
)
def test_opencode_overrides_preserve_provider_choices(
    model_id, efforts, default, vision
):
    info = OpenCodeProviderAdapter(_runtime()).model_info(model_id)
    assert (
        info.reasoning_efforts,
        info.default_reasoning_effort,
        info.supports_vision,
    ) == (efforts, default, vision)


def test_most_specific_family_wins_over_a_broader_provider_override():
    info = model_infos(("glm-5.3-flash:free",), default_model="deepseek-v4-flash")[0]
    assert info.context_window == 1_310_720
    assert info.reasoning_efforts == ("max", "high", "low")
    assert info.default_reasoning_effort == "max"
    assert info.supports_vision


@pytest.mark.parametrize(
    "model_id", ["gpt-5.60", "glm-5.30", "kimi-k2-unknown", "not-gpt-5.4"]
)
def test_unknown_opencode_versions_do_not_guess_capabilities(model_id):
    info = model_infos((model_id,), default_model="deepseek-v4-flash")[0]
    assert info.context_window is None
    assert info.reasoning_efforts == ()
    assert not info.supports_vision


def test_login_and_logout_change_only_opencode_secret(isolated_cli_env) -> None:
    store = ConfigStore(isolated_cli_env.config)
    store.write_secret("openai_api_key", "openrouter-key")
    manager = ConfigManager.load(isolated_cli_env.config, environment={})
    registry = ProviderRegistry.load(paths=isolated_cli_env.paths, config=manager)
    adapter = registry.get("opencode")

    result = asyncio.run(adapter.login(secret="  sk-go-test  "))

    assert result.status == "complete"
    assert adapter.is_authenticated() is True
    assert store.load_secrets() == {
        "openai_api_key": "openrouter-key",
        "opencode_api_key": "sk-go-test",
    }

    message = asyncio.run(adapter.logout())
    assert "Removed" in message
    assert store.load_secrets() == {"openai_api_key": "openrouter-key"}


def test_each_model_family_uses_its_documented_protocol() -> None:
    adapter = OpenCodeProviderAdapter(_runtime())

    responses = adapter.build_chat_model(
        "thread", model_name="gpt-5.6-luna", reasoning_effort="high"
    )
    completions = adapter.build_chat_model(
        "thread", model_name="glm-5.2", reasoning_effort="high"
    )
    messages = adapter.build_chat_model(
        "thread", model_name="minimax-m3", reasoning_effort=None
    )

    assert isinstance(responses, OpenCodeChatOpenAI)
    assert responses.use_responses_api is True
    assert isinstance(completions, OpenCodeChatOpenAI)
    assert completions.use_responses_api is False
    assert isinstance(messages, OpenRouterAnthropicMessages)
    assert messages.include_openrouter_extensions is False
    payload = messages._payload([])
    assert "session_id" not in payload
    assert "cache_control" not in payload


def test_model_wrapper_marks_subscription_billing() -> None:
    result = ChatResult(generations=[ChatGeneration(message=AIMessage(content="done"))])

    marked = OpenCodeChatOpenAI._mark_subscription(result)

    assert marked.generations[0].message.response_metadata["billing_mode"] == (
        "subscription"
    )


def test_usage_maps_all_rolling_windows() -> None:
    payload = {
        "usage": {
            "rolling": {
                "status": "ok",
                "percent": 12.5,
                "resetsAt": "2026-08-18T16:00:00Z",
            },
            "weekly": {
                "status": "ok",
                "percent": 40,
                "resetsAt": "2026-08-24T00:00:00Z",
            },
            "monthly": {
                "status": "ok",
                "percent": 75,
                "resetsAt": "2026-09-01T00:00:00Z",
            },
        }
    }

    buckets = _usage_buckets(payload)

    assert [item.window_minutes for item in buckets] == [300, 10_080, 43_200]
    assert [item.remaining_percent for item in buckets] == [87.5, 60.0, 25.0]
    assert buckets[0].resets_at == int(
        datetime(2026, 8, 18, 16, tzinfo=timezone.utc).timestamp()
    )


def test_status_uses_owned_key_and_caches(monkeypatch) -> None:
    calls: list[dict] = []
    payload = {
        "usage": {
            key: {
                "status": "ok",
                "percent": percent,
                "resetsAt": "2026-09-01T00:00:00Z",
            }
            for key, percent in (("rolling", 10), ("weekly", 20), ("monthly", 30))
        }
    }

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return payload

    class Client:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def get(self, url, *, headers):
            calls.append({"url": url, "headers": headers})
            return Response()

    monkeypatch.setattr("ness_cli.providers.opencode.adapter.httpx.AsyncClient", Client)
    adapter = OpenCodeProviderAdapter(_runtime("owned-key"))

    first = asyncio.run(adapter.status())
    second = asyncio.run(adapter.status())

    assert first is second
    assert len(calls) == 1
    assert calls[0]["headers"]["Authorization"] == "Bearer owned-key"
    assert len(first.limits) == 3


def test_status_keeps_auth_visible_when_usage_fails(monkeypatch) -> None:
    class Client:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def get(self, url, *, headers):
            request = httpx.Request("GET", url)
            response = httpx.Response(401, request=request)
            raise httpx.HTTPStatusError(
                "unauthorized", request=request, response=response
            )

    monkeypatch.setattr("ness_cli.providers.opencode.adapter.httpx.AsyncClient", Client)

    status = asyncio.run(
        OpenCodeProviderAdapter(_runtime("owned-key")).status(refresh=True)
    )

    assert status.auth.authenticated is True
    assert status.limits == ()
    assert status.warning and "Usage limits unavailable" in status.warning


def test_live_catalog_refresh_uses_owned_key(monkeypatch) -> None:
    async def fetch(*, api_key=None, timeout=15.0):
        assert api_key == "owned-key"
        return ("deepseek-v4-flash", "gpt-5.6-luna")

    monkeypatch.setattr("ness_cli.providers.opencode.adapter.fetch_model_ids", fetch)
    adapter = OpenCodeProviderAdapter(_runtime("owned-key"))

    models = asyncio.run(adapter.models(refresh=True))

    assert [item.id for item in models] == ["deepseek-v4-flash", "gpt-5.6-luna"]
    assert models[0].is_default is True
    assert models[1].supports_vision is True
