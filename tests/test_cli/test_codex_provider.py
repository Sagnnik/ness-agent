from __future__ import annotations

import base64
import asyncio
import json
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from ness_cli.config import AVAILABLE_MODELS, settings
from ness_cli.provider.codex.adapter import CodexProviderAdapter
from ness_cli.provider.codex.auth import CodexAuth, _jwt_expiry
from ness_cli.provider.codex.chat_model import CodexSubscriptionChatModel
from ness_cli.provider.codex.transport import (
    CodexResponsesTransport,
    CodexStreamError,
    merge_streamed_response,
)
from ness_cli.provider.openrouter.adapter import OpenRouterProviderAdapter
from ness_cli.provider.profile import provider_profile, update_provider_profile
from ness_agent.tracing.cost import CostTracker
from evals.codex.codex_chat_model import CodexChatModel


def _jwt(expiry: int) -> str:
    payload = (
        base64.urlsafe_b64encode(json.dumps({"exp": expiry}).encode())
        .decode()
        .rstrip("=")
    )
    return f"header.{payload}.signature"


def test_provider_profile_roundtrip_is_namespaced(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("NESS_AGENT_CONFIG_DIR", str(tmp_path))
    update_provider_profile(
        "codex", {"model_name": "gpt-test", "reasoning_effort": "high"}
    )
    update_provider_profile("openrouter", {"model_name": "openai/gpt-test"})

    assert provider_profile("codex") == {
        "model_name": "gpt-test",
        "reasoning_effort": "high",
    }
    assert provider_profile("openrouter") == {"model_name": "openai/gpt-test"}


def test_codex_auth_reads_only_isolated_home(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("NESS_AGENT_CONFIG_DIR", str(tmp_path))
    home = tmp_path / "codex"
    home.mkdir()
    expiry = int(time.time()) + 3600
    (home / "auth.json").write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "tokens": {
                    "access_token": _jwt(expiry),
                    "account_id": "acct-test",
                    "refresh_token": "secret",
                },
            }
        ),
        encoding="utf-8",
    )

    credentials = CodexAuth().credentials()
    assert credentials is not None
    assert credentials.account_id == "acct-test"
    assert credentials.expires_at == expiry
    assert _jwt_expiry("invalid") is None


def test_codex_chat_model_preserves_function_call_history():
    model = CodexSubscriptionChatModel(model="gpt-test", reasoning_effort="high")
    instructions, items = model._input(
        [
            HumanMessage(content="inspect"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read",
                        "args": {"path": "a.py"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(content="contents", tool_call_id="call-1"),
        ]
    )

    assert instructions == ""
    assert items[1] == {
        "type": "function_call",
        "call_id": "call-1",
        "name": "read",
        "arguments": '{"path":"a.py"}',
    }
    assert items[2]["type"] == "function_call_output"
    assert items[2]["output"] == "contents"


def test_codex_chat_model_projects_image_tool_results():
    model = CodexSubscriptionChatModel(model="gpt-test")
    data_url = "data:image/png;base64,cG5n"

    _instructions, items = model._input(
        [
            ToolMessage(
                content=[
                    {"type": "text", "text": "Read image: code.png, 2x1"},
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url, "detail": "high"},
                    },
                ],
                tool_call_id="call-image",
            )
        ]
    )

    assert items == [
        {
            "type": "function_call_output",
            "call_id": "call-image",
            "output": [
                {"type": "input_text", "text": "Read image: code.png, 2x1"},
                {"type": "input_image", "image_url": data_url, "detail": "high"},
            ],
        }
    ]


def test_codex_eval_chat_model_projects_image_tool_results():
    model = CodexChatModel(model="gpt-test")
    data_url = "data:image/png;base64,cG5n"
    _instructions, items = model._input(
        [
            ToolMessage(
                content=[
                    {"type": "text", "text": "Read image: code.png, 2x1"},
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url, "detail": "high"},
                    },
                ],
                tool_call_id="call-image",
            )
        ]
    )

    assert items[0]["output"] == [
        {"type": "input_text", "text": "Read image: code.png, 2x1"},
        {"type": "input_image", "image_url": data_url, "detail": "high"},
    ]


def test_codex_prompt_cache_key_is_stable_and_thread_scoped():
    adapter = CodexProviderAdapter()
    model = adapter.build_chat_model(
        "thread-123",
        model_name="gpt-5.5-codex",
        reasoning_effort="high",
    )

    cache_key = str(model.prompt_cache_key)
    assert str(uuid.UUID(cache_key)) == cache_key
    payload = model._payload([HumanMessage(content="inspect")])  # type: ignore[attr-defined]
    assert payload["prompt_cache_key"] == cache_key
    assert "prompt_cache_options" not in payload


def test_codex_subscription_payload_omits_public_cache_controls():
    model = CodexSubscriptionChatModel(
        model="gpt-5.6-codex",
        prompt_cache_key="ness-agent:thread-123",
    )
    payload = model._payload(
        [
            SystemMessage(content="stable instructions"),
            HumanMessage(content="inspect"),
        ]
    )

    assert payload["prompt_cache_key"] == "ness-agent:thread-123"
    assert "prompt_cache_options" not in payload
    assert payload["instructions"] == "stable instructions"
    assert payload["input"][0]["role"] == "user"
    assert "prompt_cache_breakpoint" not in json.dumps(payload)


def test_codex_response_maps_usage_tools_and_subscription_billing():
    message = CodexSubscriptionChatModel._message(
        {
            "id": "resp-1",
            "model": "gpt-test",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "checking"}],
                },
                {
                    "type": "function_call",
                    "call_id": "call-1",
                    "name": "read",
                    "arguments": '{"path":"a.py"}',
                },
            ],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 4,
                "input_tokens_details": {
                    "cached_tokens": 3,
                    "cache_write_tokens": 6,
                },
            },
        }
    )

    assert message.content == "checking"
    assert message.tool_calls[0]["args"] == {"path": "a.py"}
    assert message.usage_metadata["input_token_details"]["cache_read"] == 3
    assert message.usage_metadata["input_token_details"]["cache_creation"] == 6
    assert message.response_metadata["cache_write_tokens"] == 6
    assert message.response_metadata["billing_mode"] == "subscription"


def _function_tool(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "test tool",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def test_codex_payload_normalizes_langchain_any_tool_choice():
    model = CodexSubscriptionChatModel(model="gpt-test")

    payload = model._payload(
        [HumanMessage(content="return structured output")],
        tools=[_function_tool("structured_output")],
        tool_choice="any",
    )

    assert payload["tool_choice"] == "required"


def test_codex_registry_binding_exposes_only_active_tools():
    model = CodexSubscriptionChatModel(model="gpt-test")
    registry = type(
        "Registry",
        (),
        {
            "active_tools": [_function_tool("active")],
            "all_tools": lambda self: [
                _function_tool("active"),
                _function_tool("deferred"),
            ],
        },
    )()

    bound = model.bind_tool_registry(registry)
    payload = bound._payload([HumanMessage(content="use a tool")])  # type: ignore[attr-defined]

    assert [tool["name"] for tool in payload["tools"]] == ["active"]


def test_sparse_completed_response_keeps_streamed_assistant_text():
    response = merge_streamed_response(
        {
            "id": "resp-1",
            "model": "gpt-test",
            "output": [],
            "usage": {"input_tokens": 10, "output_tokens": 3},
        },
        [{"type": "reasoning", "id": "reasoning-1"}],
        ["hel", "lo"],
    )
    message = CodexSubscriptionChatModel._message(response)

    assert message.content == "hello"
    assert message.usage_metadata["output_tokens"] == 3
    replayable = message.additional_kwargs["codex_output_items"]
    assert any(item.get("type") == "message" for item in replayable)


def test_streamed_output_items_replace_sparse_completed_output():
    streamed = [
        {
            "type": "function_call",
            "call_id": "call-1",
            "name": "read",
            "arguments": '{"path":"a.py"}',
        }
    ]
    response = merge_streamed_response({"output": []}, streamed, [])
    message = CodexSubscriptionChatModel._message(response)

    assert message.tool_calls[0]["name"] == "read"


class _TransportAuth:
    async def valid_credentials(self, *, force_refresh: bool = False):
        del force_refresh
        return SimpleNamespace(access_token="token", account_id="account")


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n"


def _patch_transport_client(monkeypatch, handler):
    async_client = httpx.AsyncClient
    mock_transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        "ness_cli.provider.codex.transport.httpx.AsyncClient",
        lambda **kwargs: async_client(transport=mock_transport, **kwargs),
    )


def test_codex_transport_retries_streamed_overload(monkeypatch):
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                200,
                headers={"Retry-After": "3"},
                text=_sse(
                    {
                        "type": "error",
                        "error": {
                            "type": "service_unavailable_error",
                            "code": "server_is_overloaded",
                            "message": "Our servers are currently overloaded.",
                        },
                    }
                ),
            )
        return httpx.Response(
            200,
            text=_sse(
                {
                    "type": "response.completed",
                    "response": {"id": "resp-ok", "output": []},
                }
            ),
        )

    _patch_transport_client(monkeypatch, handler)
    sleep = AsyncMock()
    monkeypatch.setattr("ness_cli.provider.codex.transport._sleep", sleep)
    monkeypatch.setattr(
        "ness_cli.provider.codex.transport.uniform", lambda start, end: 0.25
    )
    transport = CodexResponsesTransport(_TransportAuth(), max_retries=3)  # type: ignore[arg-type]

    response = asyncio.run(transport.create({"model": "gpt-test"}))

    assert response["id"] == "resp-ok"
    assert calls == 2
    sleep.assert_awaited_once_with(3.25)


def test_codex_transport_retries_nested_response_failure(monkeypatch):
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                200,
                text=_sse(
                    {
                        "type": "response.failed",
                        "response": {
                            "error": {
                                "type": "server_error",
                                "code": "internal_server_error",
                            }
                        },
                    }
                ),
            )
        return httpx.Response(
            200,
            text=_sse(
                {
                    "type": "response.completed",
                    "response": {"id": "resp-ok", "output": []},
                }
            ),
        )

    _patch_transport_client(monkeypatch, handler)
    sleep = AsyncMock()
    monkeypatch.setattr("ness_cli.provider.codex.transport._sleep", sleep)
    monkeypatch.setattr(
        "ness_cli.provider.codex.transport.uniform", lambda start, end: 0.0
    )
    transport = CodexResponsesTransport(_TransportAuth(), max_retries=1)  # type: ignore[arg-type]

    response = asyncio.run(transport.create({"model": "gpt-test"}))

    assert response["id"] == "resp-ok"
    assert calls == 2
    sleep.assert_awaited_once_with(1.0)


def test_codex_transport_does_not_retry_non_transient_stream_error(monkeypatch):
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            text=_sse(
                {
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "code": "invalid_model",
                        "message": "Unknown model.",
                    },
                }
            ),
        )

    _patch_transport_client(monkeypatch, handler)
    sleep = AsyncMock()
    monkeypatch.setattr("ness_cli.provider.codex.transport._sleep", sleep)
    transport = CodexResponsesTransport(_TransportAuth(), max_retries=3)  # type: ignore[arg-type]

    with pytest.raises(CodexStreamError, match="invalid_model"):
        asyncio.run(transport.create({"model": "missing"}))

    assert calls == 1
    sleep.assert_not_awaited()


def test_codex_transport_surfaces_backend_400_detail(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            headers={"x-request-id": "req-cache-400"},
            json={"detail": "Unsupported parameter: prompt_cache_options"},
        )

    _patch_transport_client(monkeypatch, handler)
    transport = CodexResponsesTransport(_TransportAuth(), max_retries=3)  # type: ignore[arg-type]

    with pytest.raises(httpx.HTTPStatusError) as caught:
        asyncio.run(transport.create({"model": "gpt-test"}))

    message = str(caught.value)
    assert "Unsupported parameter: prompt_cache_options" in message
    assert "req-cache-400" in message


def test_codex_transport_reports_append_only_cache_prefix(monkeypatch):
    session_headers: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        session_headers.append(request.headers.get("session_id"))
        return httpx.Response(
            200,
            text=_sse(
                {
                    "type": "response.completed",
                    "response": {"id": "resp-ok", "output": []},
                }
            ),
        )

    _patch_transport_client(monkeypatch, handler)
    transport = CodexResponsesTransport(_TransportAuth())  # type: ignore[arg-type]
    base = {
        "model": "gpt-test",
        "instructions": "stable",
        "tools": [{"type": "function", "name": "read"}],
        "prompt_cache_key": "ness-agent:thread-1",
    }
    first = asyncio.run(
        transport.create(
            {**base, "input": [{"role": "user", "content": "first"}]}
        )
    )
    second = asyncio.run(
        transport.create(
            {
                **base,
                "input": [
                    {"role": "user", "content": "first"},
                    {"role": "assistant", "content": "answer"},
                    {"role": "user", "content": "second"},
                ],
            }
        )
    )

    assert first["_cache_diagnostics"]["has_previous_request"] is False
    diagnostics = second["_cache_diagnostics"]
    assert diagnostics["stable_configuration"] is True
    assert diagnostics["append_only_prefix"] is True
    assert diagnostics["matching_input_items"] == 1
    assert session_headers == ["ness-agent:thread-1", "ness-agent:thread-1"]


def test_login_readiness_waits_for_account_state(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("NESS_AGENT_CONFIG_DIR", str(tmp_path))
    home = tmp_path / "codex"
    home.mkdir()
    (home / "auth.json").write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": _jwt(int(time.time()) + 3600),
                    "account_id": "acct-test",
                }
            }
        ),
        encoding="utf-8",
    )

    class EventuallyReadyServer:
        def __init__(self):
            self.calls = 0

        async def request(self, method, params):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("account state is still converging")
            if self.calls == 2:
                return {"account": None}
            return {"account": {"type": "chatgpt"}}

    server = EventuallyReadyServer()
    credentials = asyncio.run(CodexAuth(server).wait_until_ready(timeout=1))  # type: ignore[arg-type]

    assert credentials.account_id == "acct-test"
    assert server.calls == 3


def test_device_login_explains_chatgpt_security_prerequisite():
    adapter = CodexProviderAdapter()

    device = next(method for method in adapter.login_methods() if method.id == "device")
    message = adapter._login_error(
        "Enable device code authorization for Codex in ChatGPT Security Settings, "
        'then run "codex login --device-auth" again.'
    )

    assert device.guidance is not None
    assert "Settings > Security" in device.guidance
    assert "Device-code authorization is disabled" in message
    assert "Open browser" in message
    assert "/login" in message


def test_device_login_start_translates_security_error():
    adapter = CodexProviderAdapter()

    class DisabledDeviceAuthServer:
        async def start(self):
            return None

        async def request(self, method, params):
            raise RuntimeError(
                "Enable device code authorization for Codex in ChatGPT Security Settings"
            )

    adapter.server = DisabledDeviceAuthServer()  # type: ignore[assignment]
    result = asyncio.run(adapter.login(method="device"))

    assert result.status == "error"
    assert "Settings > Security" in result.message
    assert "Open browser" in result.message


def test_openrouter_adapter_owns_masked_key_login(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("NESS_AGENT_CONFIG_DIR", str(tmp_path))
    previous = settings.openai_api_key
    adapter = OpenRouterProviderAdapter()
    try:
        method = adapter.login_methods()[0]
        result = asyncio.run(adapter.login(method=method.id, secret="  sk-or-test  "))

        assert method.input_kind == "secret"
        assert method.input_label == "OpenRouter API key"
        assert result.status == "complete"
        assert settings.openai_api_key == "sk-or-test"
        assert (tmp_path / "secrets.json").exists()
    finally:
        settings.openai_api_key = previous


def test_openrouter_models_use_packaged_fallback_when_cache_is_cold(monkeypatch):
    monkeypatch.setattr(
        "ness_cli.provider.openrouter.adapter.cached_models",
        lambda: (),
    )

    models = asyncio.run(OpenRouterProviderAdapter().models(refresh=False))

    assert tuple(model.id for model in models) == AVAILABLE_MODELS


def test_subscription_metadata_disables_api_price_estimate():
    tracker = CostTracker(pricing={"gpt-test": (10.0, 20.0, 0.1)})
    usage = tracker.add(
        {"input_tokens": 1_000, "output_tokens": 500},
        "gpt-test",
        {"billing_mode": "subscription"},
    )
    assert usage is not None
    assert usage.cost_usd is None
    assert usage.cost_source is None


def test_codex_status_deduplicates_windows_and_exposes_weekly(monkeypatch):
    adapter = CodexProviderAdapter()

    class FakeAuth:
        def is_authenticated(self):
            return True

    class FakeServer:
        async def start(self):
            return None

        async def request(self, method, params):
            if method == "account/read":
                return {
                    "account": {
                        "type": "chatgpt",
                        "email": "user@example.com",
                        "planType": "plus",
                    }
                }
            snapshot = {
                "limitId": "codex",
                "limitName": "Codex",
                "primary": {
                    "usedPercent": 20,
                    "windowDurationMins": 300,
                    "resetsAt": 2_000_000_000,
                },
                "secondary": {
                    "usedPercent": 40,
                    "windowDurationMins": 10080,
                    "resetsAt": 2_000_100_000,
                },
            }
            return {
                "rateLimits": snapshot,
                "rateLimitsByLimitId": {"codex": snapshot},
                "rateLimitResetCredits": {"availableCount": 2},
            }

    adapter.auth = FakeAuth()  # type: ignore[assignment]
    adapter.server = FakeServer()  # type: ignore[assignment]
    status = asyncio.run(adapter.status(refresh=True))

    assert status.account.email == "user@example.com"
    assert status.account.tier == "plus"
    assert len(status.limits) == 2
    assert {bucket.window_minutes for bucket in status.limits} == {300, 10080}
    assert status.credits == "2 available"
