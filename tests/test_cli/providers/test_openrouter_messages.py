from __future__ import annotations

import asyncio
from unittest.mock import Mock

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from ness_cli.providers.openrouter.messages import OpenRouterAnthropicMessages


@tool
def local_tool(value: str) -> str:
    """Return a local value."""
    return value


@tool
def deferred_tool(value: str) -> str:
    """Return a deferred value."""
    return value


class _Registry:
    def __init__(self) -> None:
        self.active_mcp_tools: set[str] = set()

    def all_tools(self):
        return [local_tool, deferred_tool]

    def deferred_tool_names(self):
        return {"deferred_tool"} - self.active_mcp_tools


def _model(**overrides) -> OpenRouterAnthropicMessages:
    values = {
        "model": "anthropic/claude-sonnet-5",
        "api_key": "test-key",
        "session_id": "session-1",
    }
    values.update(overrides)
    return OpenRouterAnthropicMessages(**values)


def test_payload_preserves_system_images_tools_results_and_cache_metadata() -> None:
    model = _model(reasoning={"effort": "high"})
    assistant = AIMessage(
        content="",
        tool_calls=[{"name": "local_tool", "args": {"value": "x"}, "id": "t1"}],
    )

    payload = model._payload(
        [
            SystemMessage(content="stable"),
            HumanMessage(
                content=[
                    {"type": "text", "text": "inspect"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AAAA"},
                    },
                ]
            ),
            assistant,
            ToolMessage("result", tool_call_id="t1"),
        ]
    )

    assert payload["system"] == [{"type": "text", "text": "stable"}]
    assert payload["messages"][0]["content"][1] == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"},
    }
    assert payload["messages"][1]["content"] == [
        {"type": "tool_use", "id": "t1", "name": "local_tool", "input": {"value": "x"}}
    ]
    assert payload["messages"][2]["content"][0]["type"] == "tool_result"
    assert payload["session_id"] == "session-1"
    assert payload["cache_control"] == {"type": "ephemeral", "ttl": "5m"}
    assert payload["output_config"] == {"effort": "high"}


def test_reasoning_blocks_and_signatures_round_trip() -> None:
    model = _model()
    raw = [
        {"type": "thinking", "thinking": "inspect", "signature": "signed"},
        {"type": "redacted_thinking", "data": "opaque"},
        {"type": "text", "text": "done"},
    ]

    message = model._message_from_response({"content": raw, "usage": {}})
    payload = model._payload([HumanMessage("work"), message])

    assert message.additional_kwargs["reasoning_content"] == "inspect"
    assert payload["messages"][1]["content"] == raw


def test_bound_deferred_tools_are_stable_and_additions_are_explicit() -> None:
    registry = _Registry()
    first = _model().bind_tool_registry(registry)
    initial = first._payload([HumanMessage("hello")])
    registry.active_mcp_tools.add("deferred_tool")
    second = _model().bind_tool_registry(registry)

    assert initial["tools"][1]["defer_loading"] is True
    assert first._payload([HumanMessage("hello")])["tools"][1]["defer_loading"] is True
    assert second._payload([HumanMessage("hello")])["messages"][-1]["content"] == [
        {
            "type": "tool_addition",
            "tool": {"type": "tool_reference", "name": "deferred_tool"},
        }
    ]


def test_response_maps_text_tools_usage_and_billing() -> None:
    model = _model(billing_mode="subscription")
    message = model._message_from_response(
        {
            "model": "anthropic/claude-sonnet-5",
            "content": [
                {"type": "thinking", "thinking": "plan"},
                {"type": "text", "text": "done"},
                {"type": "tool_use", "id": "t1", "name": "read", "input": {"path": "a.py"}},
            ],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 4,
                "cache_read_input_tokens": 3,
            },
        }
    )

    assert message.content == "done"
    assert message.tool_calls[0]["name"] == "read"
    assert message.usage_metadata["total_tokens"] == 14
    assert message.usage_metadata["input_token_details"]["cache_read"] == 3
    assert message.response_metadata["billing_mode"] == "subscription"


def test_sync_transport_retries_transient_errors(monkeypatch) -> None:
    model = _model(max_retries=1)
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/messages")
    response = httpx.Response(
        200,
        request=request,
        json={"content": [{"type": "text", "text": "ok"}], "usage": {}},
    )
    post = Mock(side_effect=[httpx.ConnectError("temporary", request=request), response])
    sleep = Mock()
    monkeypatch.setattr("ness_cli.providers.openrouter.messages.httpx.post", post)
    monkeypatch.setattr("ness_cli.providers.openrouter.messages.time.sleep", sleep)

    result = model._generate([HumanMessage("hello")])

    assert post.call_count == 2
    sleep.assert_called_once()
    assert result.generations[0].message.content == "ok"


def test_terminal_http_error_is_not_retried(monkeypatch) -> None:
    model = _model(max_retries=3)
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/messages")
    response = httpx.Response(400, request=request, json={"error": "bad request"})
    post = Mock(return_value=response)
    monkeypatch.setattr("ness_cli.providers.openrouter.messages.httpx.post", post)

    with pytest.raises(httpx.HTTPStatusError):
        model._generate([HumanMessage("hello")])

    assert post.call_count == 1


def test_streamed_deltas_merge_to_the_nonstream_message(monkeypatch) -> None:
    events = [
        {"type": "message_start", "message": {"usage": {"input_tokens": 2}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "plan"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "done"}},
        {"type": "content_block_stop", "index": 1},
        {"type": "message_delta", "usage": {"output_tokens": 3}},
    ]

    class Response:
        def raise_for_status(self):
            return None

        async def aiter_lines(self):
            for event in events:
                import json

                yield "data: " + json.dumps(event)

    class Stream:
        async def __aenter__(self):
            return Response()

        async def __aexit__(self, *_args):
            return None

    class Client:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def stream(self, *_args, **_kwargs):
            return Stream()

    monkeypatch.setattr("ness_cli.providers.openrouter.messages.httpx.AsyncClient", Client)
    model = _model()

    async def collect():
        return [chunk async for chunk in model._astream_once([HumanMessage("work")])]

    chunks = asyncio.run(collect())
    merged = chunks[0].message
    for chunk in chunks[1:]:
        merged += chunk.message
    nonstream = model._message_from_response(
        {
            "content": [
                {"type": "thinking", "thinking": "plan"},
                {"type": "text", "text": "done"},
            ],
            "usage": {"input_tokens": 2, "output_tokens": 3},
        }
    )

    assert merged.content == nonstream.content
    assert merged.additional_kwargs["reasoning_content"] == "plan"
    assert merged.usage_metadata["total_tokens"] == 5


def test_tool_result_string_stays_a_string() -> None:
    model = OpenRouterAnthropicMessages(model="test", api_key="test")

    payload = model._payload([ToolMessage("contents", tool_call_id="t1")])

    assert payload["messages"][0]["content"][0]["content"] == "contents"


def test_image_tool_result_uses_anthropic_base64_source() -> None:
    model = OpenRouterAnthropicMessages(model="test", api_key="test")
    message = ToolMessage(
        content=[
            {"type": "text", "text": "Read image: code.png, 2x1"},
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/png;base64,cG5n",
                    "detail": "high",
                },
            },
        ],
        tool_call_id="t1",
    )

    payload = model._payload([message])

    assert payload["messages"][0]["content"] == [
        {
            "type": "tool_result",
            "tool_use_id": "t1",
            "content": [
                {"type": "text", "text": "Read image: code.png, 2x1"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "cG5n",
                    },
                },
            ],
        }
    ]


def test_direct_data_url_image_uses_anthropic_base64_source() -> None:
    model = OpenRouterAnthropicMessages(model="test", api_key="test")
    message = HumanMessage(
        content=[
            {"type": "text", "text": "inspect"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/webp;base64,d2VicA=="},
            },
        ]
    )

    payload = model._payload([message])

    assert payload["messages"][0]["content"][1] == {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/webp",
            "data": "d2VicA==",
        },
    }
