from __future__ import annotations

import asyncio
from typing import Any

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from PIL import Image

from ness_agent import NessAgent, PromptLayers, PromptLayersConfig
from ness_agent.context.budget import IMAGE_TOKEN_ALLOWANCE, content_text, resolve_token_count
from ness_agent.graph.nodes import _normalize_tool_result
from ness_agent.hooks import Hook
from ness_agent.options import NessAgentOptions
from ness_agent.session import _tool_end_data
from ness_agent.tracing import TracingConfig
from ness_agent.tracing.tracer import InMemorySpan


class ImageToolModel:
    model = "image-tool-test"

    def __init__(self) -> None:
        self.calls: list[list[Any]] = []

    def bind_tools(self, _tools, **_kwargs):
        return self

    async def ainvoke(self, messages, **_kwargs):
        self.calls.append(list(messages))
        tool_result = next(
            (message for message in messages if isinstance(message, ToolMessage)),
            None,
        )
        if tool_result is not None:
            return AIMessage(content="I inspected the image.")
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "read",
                    "args": {"path": "test.png"},
                    "id": "call-image",
                }
            ],
        )


class CapturingTracer:
    def __init__(self) -> None:
        self.spans: list[InMemorySpan] = []

    def start_span(self, name, attributes=None, kind=None) -> InMemorySpan:
        span = InMemorySpan(name, attributes)
        self.spans.append(span)
        return span


def test_normalize_tool_result_preserves_strings_and_media() -> None:
    assert _normalize_tool_result("contents") == ("contents", "contents")
    result = [
        {"type": "text", "text": "Read image: test.png, 2x1"},
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,cG5n", "detail": "high"},
        },
    ]

    content, display_text = _normalize_tool_result(result)

    assert content == result
    assert display_text == "Read image: test.png, 2x1\n[image]"
    assert "cG5n" not in display_text
    assert content_text(content) == "Read image: test.png, 2x1 [image]"


@pytest.mark.parametrize("block_type", ["image_url", "image", "input_image"])
def test_image_token_estimate_counts_images_without_counting_payload(block_type) -> None:
    def message(payload, count):
        if block_type == "image_url":
            block = {"type": block_type, "image_url": {"url": payload}}
        elif block_type == "input_image":
            block = {"type": block_type, "image_url": payload}
        else:
            block = {
                "type": block_type,
                "source": {"type": "base64", "media_type": "image/png", "data": payload},
            }
        return ToolMessage(content=[block] * count, tool_call_id="image-call")

    small = message("data:image/png;base64,cG5n", 1)
    large = message("data:image/png;base64," + "A" * 300_000, 1)
    multiple = message("data:image/png;base64,cG5n", 3)
    placeholder = ToolMessage(content="[image]", tool_call_id="image-call")

    assert content_text(large.content) == "[image]"
    assert resolve_token_count([small]) == resolve_token_count([large])
    assert resolve_token_count([small]) == resolve_token_count([placeholder]) + IMAGE_TOKEN_ALLOWANCE
    assert resolve_token_count([multiple]) >= resolve_token_count([small]) + 2 * IMAGE_TOKEN_ALLOWANCE
    assert resolve_token_count([multiple], known_input_tokens=123) == 123


def test_tool_end_prefers_safe_display_text() -> None:
    message = ToolMessage(
        content=[
            {"type": "text", "text": "image"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,secret"},
            },
        ],
        tool_call_id="c1",
        name="read",
        additional_kwargs={"display_text": "image\n[image]", "duration_ms": 3},
    )

    assert _tool_end_data(message) == {
        "name": "read",
        "content": "image\n[image]",
        "id": "c1",
        "duration_ms": 3,
    }


def test_image_tool_result_reaches_second_model_request_without_side_channel_leaks(
    tmp_path,
) -> None:
    Image.new("RGB", (8, 6), "orange").save(tmp_path / "test.png")
    hook_payloads: list[dict[str, Any]] = []

    def post_tool(payload: dict[str, Any]) -> tuple[bool, str]:
        hook_payloads.append(payload)
        return True, "hook note"

    model = ImageToolModel()
    tracer = CapturingTracer()
    agent = NessAgent(
        model=model,
        tools=["read"],
        prompt=PromptLayers(PromptLayersConfig(l0="L0", persona="P")),
        options=NessAgentOptions(
            ness_dir=tmp_path / ".ness",
            project_root=tmp_path,
            enable_approval=False,
        ),
        hooks=[Hook(event="postToolUse", matcher="read", handler=post_tool)],
        tracer=tracer,
        tracing=TracingConfig(capture_messages=True),
    )
    session = agent.session(
        thread_id="image-flow",
        vision=True,
        git_available=False,
    )

    async def run() -> list[Any]:
        return [event async for event in session.stream("inspect test.png")]

    events = asyncio.run(run())

    tool_message = next(
        message
        for message in model.calls[1]
        if isinstance(message, ToolMessage)
    )
    assert isinstance(tool_message.content, list)
    assert tool_message.content[0] == {"type": "text", "text": "hook note"}
    image_block = tool_message.content[-1]
    data_url = image_block["image_url"]["url"]
    assert data_url.startswith("data:image/png;base64,")
    assert tool_message.additional_kwargs["display_text"] == (
        "hook note\n\nRead image: test.png, 8x6\n[image]"
    )

    marker = data_url.split(",", 1)[1][:20]
    assert hook_payloads[0]["result"] == "Read image: test.png, 8x6\n[image]"
    assert marker not in str(hook_payloads)
    assert marker not in str([event.data for event in events])
    assert marker not in str([span.attributes for span in tracer.spans])

    durable_events = agent.config.thread_store.load_thread_events("image-flow")
    assert marker not in str(durable_events)
    tool_event = next(event for event in durable_events if event["kind"] == "tool")
    assert tool_event["result"] == "hook note\n\nRead image: test.png, 8x6\n[image]"


def test_text_only_session_sends_omission_message(tmp_path) -> None:
    Image.new("RGB", (5, 4), "black").save(tmp_path / "test.png")
    model = ImageToolModel()
    agent = NessAgent(
        model=model,
        tools=["read"],
        prompt=PromptLayers(PromptLayersConfig(l0="L0", persona="P")),
        options=NessAgentOptions(
            ness_dir=tmp_path / ".ness",
            project_root=tmp_path,
            enable_approval=False,
        ),
    )
    session = agent.session(
        thread_id="text-only-image-flow",
        vision=False,
        git_available=False,
    )

    asyncio.run(session.run("inspect test.png"))

    tool_message = next(
        message for message in model.calls[1] if isinstance(message, ToolMessage)
    )
    assert tool_message.content == (
        "Read image: test.png, 5x4\n[image omitted: model is text-only]"
    )
    assert "base64" not in tool_message.content
