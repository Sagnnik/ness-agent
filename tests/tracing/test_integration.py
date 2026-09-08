"""Integration test: session instruments TURN + LLM_CALL + tool execution spans."""

from __future__ import annotations

import asyncio
from pathlib import Path
import tempfile

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool

from ness_agent import NessAgent, PromptLayers, PromptLayersConfig
from ness_agent.options import NessAgentOptions
from ness_agent.tracing import TracingConfig
from ness_agent.tracing.messages import serialize_messages
from ness_agent.tracing.semconv import (
    CACHE_HIT_RATE,
    CACHE_READ_TOKENS,
    GEN_AI_COMPLETION,
    GEN_AI_PROMPT,
    GEN_AI_TOOL_CALL_ARGUMENTS,
    GEN_AI_TOOL_CALL_RESULT,
    INPUT_TOKENS,
    LLM_CALL,
    OUTPUT_TOKENS,
    TOOL_EXEC,
    TURN,
)
from ness_agent.tracing.tracer import InMemorySpan, MultiTracer, NoopTracer


class StubChatModel(BaseChatModel):
    """Minimal chat model that supports ``bind_tools`` and emits usage_metadata."""

    response: str = "hi there"
    model: str = "stub-model"

    @property
    def _llm_type(self) -> str:
        return "stub"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # type: ignore[override]
        msg = AIMessage(
            content=self.response,
            usage_metadata={
                "input_tokens": 100,
                "output_tokens": 5,
                "total_tokens": 105,
                "input_token_details": {"cache_read": 20},
            },
            response_metadata={"model": self.model},
        )
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def bind_tools(self, tools, **kwargs):  # type: ignore[override]
        # accept tools without actually wiring them — sufficient for instrumentation tests
        return self


class ToolCallChatModel(StubChatModel):
    """Stub that issues a single tool call on its first invocation and a
    plain reply afterward — exercises the tool-execution span instrumentation.

    ``tool_name``/``tool_args`` configure the emitted tool call so truncate tests
    can point at a long-output tool.
    """

    tool_name: str = "ping"
    tool_args: dict = {}

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # type: ignore[override]
        has_tool_result = any(getattr(m, "tool_call_id", None) for m in messages)
        if has_tool_result:
            msg = AIMessage(
                content="done",
                usage_metadata={
                    "input_tokens": 50,
                    "output_tokens": 2,
                    "total_tokens": 52,
                    "input_token_details": {"cache_read": 0},
                },
                response_metadata={"model": self.model},
            )
        else:
            msg = AIMessage(
                content=f"calling {self.tool_name}",
                tool_calls=[{"name": self.tool_name, "args": dict(self.tool_args), "id": "call-1"}],
                usage_metadata={
                    "input_tokens": 50,
                    "output_tokens": 2,
                    "total_tokens": 52,
                    "input_token_details": {"cache_read": 0},
                },
                response_metadata={"model": self.model},
            )
        return ChatResult(generations=[ChatGeneration(message=msg)])


class CapturingTracer:
    def __init__(self) -> None:
        self.spans: list[InMemorySpan] = []

    def start_span(self, name, attributes=None, kind=None) -> InMemorySpan:
        span = InMemorySpan(name, attributes)
        self.spans.append(span)
        return span


def _agent(tmp_path: Path, tracer, *, capture_messages: bool = False, max_message_length: int = 10000) -> NessAgent:
    @tool
    def ping() -> str:
        """Return pong."""
        return "pong"

    return NessAgent(
        model=StubChatModel(),
        tools=[ping],
        prompt=PromptLayers(PromptLayersConfig(l0="L0", persona="P")),
        options=NessAgentOptions(ness_dir=tmp_path / ".ness", project_root=tmp_path),
        tracer=tracer,
        tracing=TracingConfig(capture_messages=capture_messages, max_message_length=max_message_length),
    )


def _run_session(session):
    async def _run():
        events = []
        async for ev in session.stream("hi"):
            events.append(ev)
        return events
    return asyncio.run(_run())


def test_session_emits_turn_and_llm_call_spans(tmp_path: Path):
    capture = CapturingTracer()
    agent = _agent(tmp_path, capture)
    session = agent.session(thread_id="trace-1")
    _run_session(session)
    names = [s.name for s in capture.spans]
    assert TURN in names
    assert LLM_CALL in names


def test_turn_span_attributes_thread_id_and_mode(tmp_path: Path):
    capture = CapturingTracer()
    agent = _agent(tmp_path, capture)
    session = agent.session(thread_id="trace-attr")
    _run_session(session)
    turn = next(s for s in capture.spans if s.name == TURN)
    assert turn.attributes["session.thread_id"] == "trace-attr"
    assert turn.attributes["session.mode"] == "act"


def test_llm_span_records_usage_attrs(tmp_path: Path):
    capture = CapturingTracer()
    agent = _agent(tmp_path, capture)
    session = agent.session(thread_id="trace-usage")
    _run_session(session)
    llm = next(s for s in capture.spans if s.name == LLM_CALL)
    assert llm.attributes[INPUT_TOKENS] == 100
    assert llm.attributes[OUTPUT_TOKENS] == 5
    assert llm.attributes[CACHE_READ_TOKENS] == 20
    assert llm.attributes[CACHE_HIT_RATE] == 0.2
    assert llm.attributes["gen_ai.request.model"] == "stub-model"


# ---------------------------------------------------------------------------
# gen_ai.prompt / gen_ai.completion / gen_ai.tool.call.* capture
# ---------------------------------------------------------------------------

def test_llm_span_records_prompt_and_completion_when_enabled(tmp_path: Path):
    capture = CapturingTracer()
    agent = _agent(tmp_path, capture, capture_messages=True)
    session = agent.session(thread_id="capture-llm")
    _run_session(session)
    llm = next(s for s in capture.spans if s.name == LLM_CALL)
    # prompt is a JSON array of OpenAI-style messages with role keys.
    import json as _json
    prompt = _json.loads(llm.attributes[GEN_AI_PROMPT])
    assert isinstance(prompt, list)
    assert any(m.get("role") == "system" for m in prompt)
    assert any(m.get("role") == "user" for m in prompt)
    completion = _json.loads(llm.attributes[GEN_AI_COMPLETION])
    assert isinstance(completion, list)
    assert completion[0]["role"] == "assistant"


def test_tool_span_records_arguments_and_result_when_enabled(tmp_path: Path):
    capture = CapturingTracer()
    agent = _tool_agent(tmp_path, capture, capture_messages=True)
    session = agent.session(thread_id="capture-tool")
    _run_session(session)
    tool_span = next(s for s in capture.spans if s.name.startswith("tool."))
    import json as _json
    args = _json.loads(tool_span.attributes[GEN_AI_TOOL_CALL_ARGUMENTS])
    assert args == {}  # ping() takes no args
    assert tool_span.attributes[GEN_AI_TOOL_CALL_RESULT] == "pong"


def test_capture_disabled_by_default(tmp_path: Path):
    capture = CapturingTracer()
    agent = _tool_agent(tmp_path, capture)  # capture_messages defaults to False
    session = agent.session(thread_id="capture-off")
    _run_session(session)
    llm = next(s for s in capture.spans if s.name == LLM_CALL)
    assert GEN_AI_PROMPT not in llm.attributes
    assert GEN_AI_COMPLETION not in llm.attributes
    tool_span = next(s for s in capture.spans if s.name.startswith("tool."))
    assert GEN_AI_TOOL_CALL_RESULT not in tool_span.attributes
    assert GEN_AI_TOOL_CALL_ARGUMENTS not in tool_span.attributes


def test_serialize_messages_strips_image_base64(tmp_path: Path):
    """Multi-modal HumanMessage with an image_url block should not contain
    the base64 blob in the serialised JSON — it should be replaced with a
    short text placeholder."""
    msg = HumanMessage(
        content=[
            {"type": "text", "text": "What is in this picture?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAE="}},
            {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}},
            {"type": "input_image", "image_url": "data:image/png;base64,INPUT_IMAGE_SECRET"},
        ]
    )
    raw = serialize_messages([msg])
    assert "iVBORw0KGgo" not in raw, "base64 data leaked into serialised prompt"
    assert "INPUT_IMAGE_SECRET" not in raw, "input_image data leaked into prompt"
    assert "https://example.com/photo.jpg" not in raw, "image URL leaked into serialised prompt"
    assert "[image_url [base64]]" in raw
    assert "[image_url [url]]" in raw
    assert "[input_image]" in raw
    assert "What is in this picture?" in raw


def test_tool_result_truncation_respects_max_length(tmp_path: Path):
    long_blob = "x" * 50000

    @tool
    def big() -> str:
        """Return a large string."""
        return long_blob

    capture = CapturingTracer()
    agent = NessAgent(
        model=ToolCallChatModel(tool_name="big"),
        tools=[big],
        prompt=PromptLayers(PromptLayersConfig(l0="L0", persona="P")),
        options=NessAgentOptions(ness_dir=tmp_path / ".ness", project_root=tmp_path),
        tracer=capture,
        tracing=TracingConfig(capture_messages=True, max_message_length=100),
    )
    reg = agent.config.tool_registry
    reg._include = {"big"}  # type: ignore[attr-defined]
    reg.bump_generation()
    session = agent.session(thread_id="capture-trunc")
    _run_session(session)
    tool_span = next(s for s in capture.spans if s.name.startswith("tool."))
    result = tool_span.attributes[GEN_AI_TOOL_CALL_RESULT]
    assert len(result) <= 100
    assert result.endswith("...[truncated]")


def _tool_agent(tmp_path: Path, tracer, *, capture_messages: bool = False, max_message_length: int = 10000) -> NessAgent:
    """Build an agent whose StubChatModel emits one real tool call so the
    tool-execution span is exercised end-to-end."""
    @tool
    def ping() -> str:
        """Return pong."""
        return "pong"

    agent = NessAgent(
        model=ToolCallChatModel(),
        tools=[ping],
        prompt=PromptLayers(PromptLayersConfig(l0="L0", persona="P")),
        options=NessAgentOptions(ness_dir=tmp_path / ".ness", project_root=tmp_path),
        tracer=tracer,
        tracing=TracingConfig(capture_messages=capture_messages, max_message_length=max_message_length),
    )
    # Custom user-supplied tools are filtered out of the active set by
    # ToolRegistry unless they are built-ins or explicitly included. Activate
    # the custom tool so the tools_node can resolve and invoke it.
    reg = agent.config.tool_registry
    reg._include = {"ping"}  # type: ignore[attr-defined]
    reg.bump_generation()
    return agent
