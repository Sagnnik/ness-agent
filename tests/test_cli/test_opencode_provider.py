from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import httpx

from ness_cli.config import settings
from ness_cli.config_store import load_secrets
from ness_cli.provider.opencode.adapter import (
    OpenCodeProviderAdapter,
    _usage_buckets,
)
from ness_cli.provider.opencode.catalog import parse_models
from ness_cli.provider.opencode.chat_model import OpenCodeChatOpenAI
from ness_cli.provider.openrouter.chat_model import OpenRouterAnthropicMessages


def test_opencode_catalog_parses_and_sorts_live_model_ids():
    assert parse_models(
        {
            "object": "list",
            "data": [
                {"id": "kimi-k2.7-code", "object": "model"},
                {"id": "glm-5.2", "object": "model"},
            ],
        }
    ) == ("glm-5.2", "kimi-k2.7-code")


def test_opencode_login_uses_a_separate_saved_secret(tmp_path, monkeypatch):
    monkeypatch.setenv("NESS_AGENT_CONFIG_DIR", str(tmp_path / "config"))
    previous = settings.opencode_api_key
    adapter = OpenCodeProviderAdapter()
    try:
        result = asyncio.run(adapter.login(secret="sk-go-test"))
        assert result.status == "complete"
        assert load_secrets() == {"opencode_api_key": "sk-go-test"}
        assert adapter.is_authenticated() is True

        message = asyncio.run(adapter.logout())
        assert "Removed" in message
        assert load_secrets() == {}
    finally:
        settings.opencode_api_key = previous


def test_opencode_routes_each_model_family_to_its_documented_protocol():
    previous = settings.opencode_api_key
    settings.opencode_api_key = "sk-test"
    try:
        adapter = OpenCodeProviderAdapter()
        responses = adapter.build_chat_model(
            "thread-1", model_name="gpt-5.6-luna", reasoning_effort="high"
        )
        completions = adapter.build_chat_model(
            "thread-1", model_name="glm-5.2", reasoning_effort="high"
        )
        messages = adapter.build_chat_model(
            "thread-1", model_name="minimax-m3", reasoning_effort=None
        )
    finally:
        settings.opencode_api_key = previous

    assert isinstance(responses, OpenCodeChatOpenAI)
    assert responses.use_responses_api is True
    assert isinstance(completions, OpenCodeChatOpenAI)
    assert completions.use_responses_api is False
    assert isinstance(messages, OpenRouterAnthropicMessages)
    assert messages.include_openrouter_extensions is False
    payload = messages._payload([], stream=False)
    assert "session_id" not in payload
    assert "cache_control" not in payload
    message = messages._message_from_response({"content": [], "usage": {}})
    assert message.response_metadata["billing_mode"] == "subscription"


def test_opencode_usage_maps_all_three_rolling_windows():
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
                "resetsAt": "2026-08-24T00:00:00+00:00",
            },
            "monthly": {
                "status": "ok",
                "percent": 75,
                "resetsAt": "2026-09-01T00:00:00Z",
            },
        }
    }

    buckets = _usage_buckets(payload)

    assert [bucket.window_minutes for bucket in buckets] == [300, 10_080, 43_200]
    assert [bucket.used_percent for bucket in buckets] == [12.5, 40.0, 75.0]
    assert [bucket.remaining_percent for bucket in buckets] == [87.5, 60.0, 25.0]
    assert buckets[0].resets_at == int(
        datetime(2026, 8, 18, 16, tzinfo=timezone.utc).timestamp()
    )


def test_opencode_status_uses_bearer_key_and_caches(monkeypatch):
    previous = settings.opencode_api_key
    settings.opencode_api_key = "sk-go-test"
    calls = 0
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

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return payload

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, url, *, headers):
            nonlocal calls
            calls += 1
            assert url.endswith("/zen/go/v1/usage")
            assert headers["Authorization"] == "Bearer sk-go-test"
            return FakeResponse()

    monkeypatch.setattr(
        "ness_cli.provider.opencode.adapter.httpx.AsyncClient", FakeClient
    )
    adapter = OpenCodeProviderAdapter()
    try:
        first = asyncio.run(adapter.status())
        second = asyncio.run(adapter.status())
    finally:
        settings.opencode_api_key = previous

    assert calls == 1
    assert first is second
    assert first.account.tier == "Go"
    assert len(first.limits) == 3


def test_opencode_status_keeps_auth_visible_when_usage_fails(monkeypatch):
    previous = settings.opencode_api_key
    settings.opencode_api_key = "sk-go-test"

    class FailingClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, url, *, headers):
            request = httpx.Request("GET", url)
            response = httpx.Response(401, request=request)
            raise httpx.HTTPStatusError(
                "unauthorized", request=request, response=response
            )

    monkeypatch.setattr(
        "ness_cli.provider.opencode.adapter.httpx.AsyncClient", FailingClient
    )
    try:
        status = asyncio.run(OpenCodeProviderAdapter().status(refresh=True))
    finally:
        settings.opencode_api_key = previous

    assert status.auth.authenticated is True
    assert status.limits == ()
    assert status.warning and "Usage limits unavailable" in status.warning


def test_opencode_models_refreshes_live_catalog(monkeypatch):
    async def fake_fetch_model_ids(*, api_key=None, timeout=15.0):
        assert api_key == "sk-test"
        return ("deepseek-v4-flash", "gpt-5.6-luna")

    previous = settings.opencode_api_key
    settings.opencode_api_key = "sk-test"
    monkeypatch.setattr(
        "ness_cli.provider.opencode.adapter.fetch_model_ids", fake_fetch_model_ids
    )
    try:
        models = asyncio.run(OpenCodeProviderAdapter().models(refresh=True))
    finally:
        settings.opencode_api_key = previous

    assert [model.id for model in models] == ["deepseek-v4-flash", "gpt-5.6-luna"]
    assert models[0].is_default is True
    assert models[0].supports_vision is False
    assert models[1].supports_vision is True
