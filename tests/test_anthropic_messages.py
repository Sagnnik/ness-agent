from __future__ import annotations

from unittest.mock import patch

import httpx
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from ness_cli.provider.openrouter.chat_model import OpenRouterAnthropicMessages


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

    def tool_names(self):
        return ["local_tool", *sorted(self.active_mcp_tools)]

    def deferred_tool_names(self):
        return {"deferred_tool"} - self.active_mcp_tools


def test_messages_payload_uses_stable_deferred_tools_and_addition_blocks() -> None:
    registry = _Registry()
    model = OpenRouterAnthropicMessages(
        model="anthropic/claude-sonnet-5",
        api_key="test",
        session_id="session-1",
    )
    first_binding = model.bind_tool_registry(registry)

    first = first_binding._payload(
        [SystemMessage(content="stable"), HumanMessage(content="hello")]
    )
    registry.active_mcp_tools.add("deferred_tool")
    second_binding = model.bind_tool_registry(registry)
    second = second_binding._payload(
        [SystemMessage(content="stable"), HumanMessage(content="hello")]
    )

    assert first["session_id"] == "session-1"
    assert first["cache_control"] == {"type": "ephemeral", "ttl": "5m"}
    assert [tool["name"] for tool in first["tools"]] == [
        "local_tool",
        "deferred_tool",
    ]
    assert first["tools"][1]["defer_loading"] is True
    assert first_binding._payload(
        [SystemMessage(content="stable"), HumanMessage(content="hello")]
    )["tools"][1]["defer_loading"] is True
    assert [tool["name"] for tool in second["tools"]] == [
        "local_tool",
        "deferred_tool",
    ]
    assert second["messages"][-1]["content"] == [
        {
            "type": "tool_addition",
            "tool": {"type": "tool_reference", "name": "deferred_tool"},
        }
    ]


def test_tool_only_assistant_has_no_empty_text_block() -> None:
    model = OpenRouterAnthropicMessages(
        model="anthropic/claude-sonnet-5",
        api_key="test",
        session_id="session-1",
    )
    assistant = AIMessage(
        content="",
        tool_calls=[{"name": "local_tool", "args": {"value": "x"}, "id": "t1"}],
    )

    payload = model._payload(
        [HumanMessage(content="call it"), assistant, ToolMessage("x", tool_call_id="t1")]
    )

    assert payload["messages"][1]["content"] == [
        {
            "type": "tool_use",
            "id": "t1",
            "name": "local_tool",
            "input": {"value": "x"},
        }
    ]


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


def test_thinking_signature_round_trips_and_effort_uses_output_config() -> None:
    model = OpenRouterAnthropicMessages(
        model="anthropic/claude-sonnet-5",
        api_key="test",
        session_id="session-1",
        reasoning={"effort": "high"},
    )
    raw = [
        {"type": "thinking", "thinking": "inspect", "signature": "signed"},
        {"type": "redacted_thinking", "data": "opaque"},
        {"type": "text", "text": "done"},
    ]
    message = model._message_from_response({"content": raw, "usage": {}})

    payload = model._payload([HumanMessage(content="work"), message])

    assert payload["messages"][1]["content"] == raw
    assert payload["output_config"] == {"effort": "high"}
    assert "reasoning" not in payload


def test_sync_transport_retries_transient_failure() -> None:
    model = OpenRouterAnthropicMessages(
        model="anthropic/claude-sonnet-5",
        api_key="test",
        session_id="session-1",
        max_retries=1,
    )
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/messages")
    response = httpx.Response(
        200,
        request=request,
        json={"content": [{"type": "text", "text": "ok"}], "usage": {}},
    )
    with (
        patch(
            "ness_cli.provider.openrouter.chat_model.httpx.post",
            side_effect=[httpx.ConnectError("temporary", request=request), response],
        ) as post,
        patch("ness_cli.provider.openrouter.chat_model.time.sleep"),
    ):
        result = model._generate([HumanMessage(content="hello")])

    assert post.call_count == 2
    assert result.generations[0].message.content == "ok"
