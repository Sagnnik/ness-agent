from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, messages_from_dict
from langchain_core.tools import tool

from ness_agent import NessAgent, NoOverlay, PromptLayers, PromptLayersConfig
from ness_agent.compaction import summarize
from ness_agent.context.budget import (
    ContextPressure,
    resolve_token_count,
    resolve_usable_context_budget,
)
from ness_agent.context.overlay import OverlayContext, OverlayProvider
from ness_cli.events import events_to_messages
from ness_agent.context.layers import AuxPrompts
from ness_agent.graph.nodes import make_nodes
from ness_agent.memory import MemoryStore
from ness_agent.options import MemoryConfig, NessAgentOptions
from ness_agent.persistence import ThreadStore
from ness_agent.reflection import finalize_session_reflection, run_reflection_gate
from ness_agent.tracing.cost import CostTracker


def _agent(tmp_path: Path, **kwargs):
    model = FakeListChatModel(responses=["hello"])

    @tool
    def ping() -> str:
        """Return pong."""
        return "pong"

    return NessAgent(
        model=model,
        tools=[ping],
        prompt=PromptLayers(PromptLayersConfig(l0="L0", persona="P")),
        options=NessAgentOptions(ness_dir=tmp_path / ".ness", project_root=tmp_path),
        **kwargs,
    )


def test_sessions_fork_mutable_config_and_share_project_services(tmp_path: Path):
    agent = _agent(tmp_path)
    first = agent.session(thread_id="session-one")
    second = agent.session(thread_id="session-two")

    assert first.config is not second.config
    assert first.config is not agent.config
    assert first.config.options is not second.config.options
    assert first.config.cost_tracker is not second.config.cost_tracker
    assert first.config.permission_store is not second.config.permission_store
    assert first.config.tool_registry is not second.config.tool_registry
    assert first.config.thread_store is second.config.thread_store is agent.config.thread_store
    assert first.config.memory_store is second.config.memory_store is agent.config.memory_store
    assert first.config.hook_runner is second.config.hook_runner is agent.config.hook_runner
    assert first.config.tracer is second.config.tracer is agent.config.tracer


def test_session_cost_is_local_and_live_usage_rolls_up_once(tmp_path: Path):
    agent = _agent(tmp_path)
    first = agent.session(thread_id="session-one")
    second = agent.session(thread_id="session-two")
    usage = {"input_tokens": 10, "output_tokens": 2}

    first.cost_tracker.add(usage, "model")

    assert first.cost_tracker.input_tokens == 10
    assert second.cost_tracker.input_tokens == 0
    assert agent.config.cost_tracker.input_tokens == 10

    first.cost_tracker.restore(usage, "model")
    assert first.cost_tracker.input_tokens == 20
    assert agent.config.cost_tracker.input_tokens == 10


def test_session_permissions_and_mcp_activation_are_isolated(tmp_path: Path):
    agent = _agent(tmp_path)
    first = agent.session(thread_id="session-one")
    second = agent.session(thread_id="session-two")

    first.config.permission_store.persist_rule(
        "shell:run:pytest*", "allow", scope="session"
    )
    assert first.config.permission_store.check("shell", {"command": "pytest"}) == "allow"
    assert second.config.permission_store.check("shell", {"command": "pytest"}) == "ask"
    first.config.permission_store.persist_rule("shell:run:ruff*", "allow")
    assert second.config.permission_store.check("shell", {"command": "ruff check"}) == "allow"

    @tool
    def mcp__demo__lookup() -> str:
        """Look something up."""
        return "ok"

    first.config.tool_registry.register_dynamic([mcp__demo__lookup])
    first.config.tool_registry.set_mcp_catalog(
        {"demo": {"tools": [{"name": "mcp__demo__lookup"}]}}
    )
    added, _ = first.config.tool_registry.activate_mcp(["mcp__demo__lookup"])
    assert added == ["mcp__demo__lookup"]
    assert "mcp__demo__lookup" in first.config.tool_registry.tool_names()
    assert "mcp__demo__lookup" not in second.config.tool_registry.tool_names()
    assert "demo" in second.config.tool_registry.mcp_catalog()


def test_model_reconfiguration_pins_existing_sessions_and_updates_defaults(tmp_path: Path):
    agent = _agent(tmp_path)
    existing = agent.session(thread_id="session-existing")
    selected = agent.session(thread_id="session-selected")
    replacement = FakeListChatModel(responses=["new"])

    agent.configure_default_models(
        model=replacement,
        reflection_model=replacement,
        context_window=64_000,
    )
    selected.configure_models(
        model=replacement,
        reflection_model=replacement,
        context_window=64_000,
        vision=False,
    )
    future = agent.session(thread_id="session-future")

    assert existing.config.model is not replacement
    assert existing.config.options.context_window != 64_000
    assert selected.config.model is replacement
    assert selected.config.options.context_window == 64_000
    assert future.config.model is replacement
    assert future.config.options.context_window == 64_000


def test_summarize_appends_human_instruction_to_exact_prefix():
    seen = {}

    class Model:
        async def ainvoke(self, messages, **kwargs):
            seen["messages"] = list(messages)
            seen["kwargs"] = kwargs
            return AIMessage(content="summary text")

    prefix = [HumanMessage(content="first"), AIMessage(content="answer")]
    result = asyncio.run(
        summarize(prefix, Model(), instruction="Summarize now", max_output_tokens=321)
    )
    assert result == "summary text"
    assert seen["messages"][:-1] == prefix
    assert seen["messages"][-1].content == "Summarize now"
    assert seen["kwargs"]["max_tokens"] == 321


def test_reflection_result_returns_bullets(tmp_path: Path):
    store = ThreadStore(threads_dir=tmp_path / "threads", default_model="m")
    memory = MemoryStore(MemoryConfig(), ness_dir=tmp_path / ".ness")

    class OkModel:
        def with_structured_output(self, _schema):
            return self

        async def ainvoke(self, _messages):
            return SimpleNamespace(
                model_dump=lambda: {
                    "new_bullet_points": ["Added rate limiter", "Wired auth"]
                },
                new_bullet_points=["Added rate limiter", "Wired auth"],
            )

    result = asyncio.run(
        run_reflection_gate(
            "session-bullets",
            [HumanMessage(content="hello")],
            OkModel(),
            1,
            memory=memory,
            persistence=store,
            aux_prompts=AuxPrompts(
                reflection=(
                    "t={thread_id} n={user_message_count} msgs={messages} "
                    "bullets={current_session_bullets} todos={todos}"
                )
            ),
        )
    )
    assert result.memory_updated is True
    assert result.bullets == ("Added rate limiter", "Wired auth")
    assert result.error == ""
    assert result.message_index == 1


def test_session_run_reflection_uses_unreflected_tail_and_updates_cursor(
    tmp_path: Path,
):
    seen: dict[str, str] = {}
    calls: list[None] = []

    class ReflectionModel:
        def with_structured_output(self, _schema):
            return self

        async def ainvoke(self, messages):
            calls.append(None)
            seen["prompt"] = str(messages[0].content)
            return SimpleNamespace(
                model_dump=lambda: {"new_bullet_points": ["Finished the parser"]},
                new_bullet_points=["Finished the parser"],
            )

    class App:
        def __init__(self):
            self.updates: list[dict] = []
            self.values = {
                "messages": [
                    HumanMessage(content="already reflected request"),
                    AIMessage(content="already reflected answer"),
                    HumanMessage(content="new parser request"),
                    AIMessage(content="new parser answer"),
                ],
                "last_reflection_index": 2,
                "todos": [],
            }

        async def aget_state(self, _config):
            return SimpleNamespace(values=self.values)

        async def aupdate_state(self, _config, updates):
            self.updates.append(dict(updates))
            self.values.update(updates)

    agent = _agent(tmp_path, reflection_model=ReflectionModel())
    session = agent.session(thread_id="manual-reflection", git_available=False)
    fake_app = App()
    session._app = fake_app

    result = asyncio.run(session.run_reflection())

    assert result.bullets == ("Finished the parser",)
    assert result.message_index == 4
    assert "new parser request" in seen["prompt"]
    assert "already reflected request" not in seen["prompt"]
    assert fake_app.updates == [{"last_reflection_index": 4}]

    empty = asyncio.run(session.run_reflection())
    assert empty.message_index is None
    assert len(calls) == 1


def test_agent_node_tool_loop_without_overlay(tmp_path: Path):
    """Subagents set overlay=None; tool-loop turns must not crash on render_overlay_delta."""
    store = ThreadStore(threads_dir=tmp_path / "threads", default_model="m")
    agent = _agent(tmp_path)
    cfg = replace(agent.config, overlay=None, thread_store=store)
    rt = make_nodes(cfg, thread_id="subagent-explore-test", mode="act", git_available=False)

    async def fake_ainvoke(_msgs):
        return AIMessage(content="ok")

    bind = SimpleNamespace(ainvoke=fake_ainvoke)
    state = {
        "messages": [HumanMessage(content="find routes")],
        "mode": "act",
        "todos": [],
    }

    with patch.object(cfg.tool_registry, "bind_model", return_value=bind):
        asyncio.run(rt.agent_node(state))
        tool_loop = {
            **state,
            "messages": [
                HumanMessage(content="find routes"),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "ping", "args": {}, "id": "c1"}],
                ),
                ToolMessage(content="pong", tool_call_id="c1", name="ping"),
            ],
        }
        updates = asyncio.run(rt.agent_node(tool_loop))

    assert updates["messages"][0].content == "ok"


def test_gate_provides_overlay_note_without_agent_pressure_recalculation(tmp_path: Path):
    class RecordingOverlay(OverlayProvider):
        def __init__(self):
            self.notes: list[str] = []

        def sections(self, _state, ctx: OverlayContext) -> dict[str, str]:
            self.notes.append(ctx.compaction_note)
            return {"compaction": ctx.compaction_note} if ctx.compaction_note else {}

    overlay = RecordingOverlay()
    agent = _agent(tmp_path, overlay=overlay)
    rt = make_nodes(agent.config, thread_id="gate-note", mode="act", git_available=False)
    pressure = ContextPressure(
        token_count=750,
        context_limit=1_000,
        ratio=0.75,
        warning=True,
        should_compact=False,
        safety_threshold_reached=False,
        hard_threshold_reached=False,
    )

    async def fake_ainvoke(_messages):
        return AIMessage(content="ok")

    state = {
        "messages": [HumanMessage(content="check context")],
        "mode": "act",
        "todos": [],
    }
    with (
        patch("ness_agent.graph.nodes.calculate_context_pressure", return_value=pressure) as calculate,
        patch.object(
            agent.config.tool_registry,
            "bind_model",
            return_value=SimpleNamespace(ainvoke=fake_ainvoke),
        ),
    ):
        gate_updates = asyncio.run(rt.context_gate(state))
        agent_updates = asyncio.run(rt.agent_node({**state, **gate_updates}))

    assert calculate.call_count == 1
    assert overlay.notes == [
        "Context ~750 tokens (75% of the configured context limit). "
        "Summary compaction may begin soon."
    ]
    assert agent_updates["compaction_status"] == {}


def test_agent_node_fallback_token_count_uses_final_overlay_payload(tmp_path: Path):
    class StaticOverlay(OverlayProvider):
        def sections(self, _state, _ctx: OverlayContext) -> dict[str, str]:
            return {"test": "Overlay content that must be counted."}

    agent = _agent(tmp_path, overlay=StaticOverlay())
    rt = make_nodes(agent.config, thread_id="fallback-token-count", mode="act", git_available=False)
    seen: list = []

    async def fake_ainvoke(messages):
        seen[:] = list(messages)
        return AIMessage(content="ok")

    with patch.object(
        agent.config.tool_registry,
        "bind_model",
        return_value=SimpleNamespace(ainvoke=fake_ainvoke),
    ):
        updates = asyncio.run(rt.agent_node({
            "messages": [HumanMessage(content="count this request")],
            "mode": "act",
            "todos": [],
        }))

    assert any(
        (message.additional_kwargs or {}).get("ness_internal") == "overlay"
        for message in seen
    )
    assert updates["last_input_tokens"] == resolve_token_count(
        seen,
        known_input_tokens=None,
    )


def test_summarize_rejects_provider_failure_and_tool_calls():
    class OkModel:
        async def ainvoke(self, _messages, **_kwargs):
            return SimpleNamespace(
                content="summary text",
                usage_metadata=None,
                response_metadata={},
            )

    class BoomModel:
        async def ainvoke(self, _messages, **_kwargs):
            raise RuntimeError("summarizer down")
    assert asyncio.run(summarize([HumanMessage(content="hello")], OkModel())) == "summary text"
    with pytest.raises(RuntimeError, match="summarizer down"):
        asyncio.run(summarize([HumanMessage(content="hello")], BoomModel()))

    class ToolModel:
        async def ainvoke(self, _messages, **_kwargs):
            return AIMessage(
                content="",
                tool_calls=[{"name": "ping", "args": {}, "id": "c1"}],
            )

    with pytest.raises(RuntimeError, match="tool call"):
        asyncio.run(summarize([HumanMessage(content="hello")], ToolModel()))


def test_forced_compaction_forks_cached_prefix_and_retains_active_user(tmp_path: Path):
    class RecordingModel:
        model = "recording"

        def __init__(self):
            self.calls = []
            self.responses = [
                "first answer",
                "summary text",
                "second answer",
                "resumed answer",
            ]

        def bind_tools(self, _tools):
            return self

        async def ainvoke(self, messages, **kwargs):
            self.calls.append((list(messages), dict(kwargs)))
            return AIMessage(content=self.responses[len(self.calls) - 1])

    model = RecordingModel()
    agent = NessAgent(
        model=model,
        tools=[],
        prompt=PromptLayers(PromptLayersConfig(l0="L0", persona="P")),
        overlay=NoOverlay(),
        options=NessAgentOptions(ness_dir=tmp_path / ".ness", project_root=tmp_path),
    )
    session = agent.session(thread_id="session-cache-safe", git_available=False)
    asyncio.run(session.run("first task"))
    session.request_compact()
    asyncio.run(session.run("active task"))

    first_request = model.calls[0][0]
    summary_request = model.calls[1][0]
    post_compaction = model.calls[2][0]
    assert summary_request[: len(first_request)] == first_request
    assert "HARNESS COMPACTION RULES" in summary_request[-1].content
    assert any(m.content == "active task" for m in post_compaction)
    assert any("<compacted-history>" in str(m.content) for m in post_compaction)
    assert not any(m.content == "first task" for m in post_compaction)

    checkpoint = next(
        event
        for event in agent.config.thread_store.load_thread_events("session-cache-safe")
        if event.get("kind") == "compaction_llm"
    )
    assert checkpoint["active_user_seq"] is None
    durable_suffix = messages_from_dict(checkpoint["active_suffix"])
    assert any(message.content == "active task" for message in durable_suffix)

    resumed = agent.session(thread_id="session-cache-safe", git_available=False)
    resumed.bootstrap(events_to_messages(
        agent.config.thread_store.load_thread_events("session-cache-safe")
    ))
    asyncio.run(resumed.run("follow up"))
    resumed_request = model.calls[3][0]
    assert any(message.content == "active task" for message in resumed_request)
    assert any(message.content == "follow up" for message in resumed_request)


def test_failed_binding_does_not_replace_last_successful_compaction_parent(tmp_path: Path):
    class Binding:
        def __init__(self, responses=(), *, fail=False):
            self.responses = list(responses)
            self.fail = fail
            self.calls = []

        async def ainvoke(self, messages, **_kwargs):
            self.calls.append(list(messages))
            if self.fail:
                raise RuntimeError("schema generation failed")
            return AIMessage(content=self.responses.pop(0))

    agent = _agent(tmp_path)
    rt = make_nodes(
        agent.config,
        thread_id="session-binding-failure",
        mode="act",
        git_available=False,
    )
    successful = Binding(["first answer", "summary from successful binding"])
    failed = Binding(fail=True)
    first_state = {"messages": [HumanMessage(content="first task")], "mode": "act"}

    with patch.object(
        agent.config.tool_registry,
        "bind_model",
        side_effect=[successful, failed],
    ):
        first = asyncio.run(rt.agent_node(first_state))
        assert rt.last_bound_model is successful

        @tool("mcp__review__new_schema")
        def new_schema_tool() -> str:
            """Dynamically activated schema used by the failed request."""
            return "new"

        agent.config.tool_registry.register_dynamic([new_schema_tool])
        added, unknown = agent.config.tool_registry.activate_mcp(
            ["mcp__review__new_schema"]
        )
        assert added == ["mcp__review__new_schema"]
        assert unknown == []
        second_state = {
            "messages": [
                *first_state["messages"],
                *first["messages"],
                HumanMessage(content="active task", id="active-binding-turn"),
            ],
            "model_context_messages": first["model_context_messages"],
            "model_context_source_count": first["model_context_source_count"],
            "model_system_message": first["model_system_message"],
            "last_input_tokens": first["last_input_tokens"],
            "mode": "act",
        }
        with pytest.raises(RuntimeError, match="schema generation failed"):
            asyncio.run(rt.agent_node(second_state))
        assert rt.last_bound_model is successful

        compacted = asyncio.run(rt.context_gate({
            **second_state,
            "force_compact": True,
        }))

    assert compacted["compaction_status"]["compacted"] is True
    assert len(successful.calls) == 2
    assert successful.calls[-1][0] == first["model_system_message"]
    assert "HARNESS COMPACTION RULES" in successful.calls[-1][-1].content


def test_session_pressure_includes_stable_system_prefix(tmp_path: Path):
    agent = NessAgent(
        model=FakeListChatModel(responses=["unused"]),
        tools=[],
        prompt=PromptLayers(PromptLayersConfig(l0="system " * 1200, persona="P")),
        overlay=NoOverlay(),
        options=NessAgentOptions(
            context_window=8_000,
            compaction_buffer_tokens=1_000,
            compaction_summary_max_tokens=500,
            ness_dir=tmp_path / ".ness",
            project_root=tmp_path,
        ),
    )
    session = agent.session(thread_id="session-pressure", git_available=False)
    human = HumanMessage(content="short conversation")

    class SnapshotApp:
        async def aget_state(self, _config):
            return SimpleNamespace(values={"messages": [human]})

    session._app = SnapshotApp()
    asyncio.run(session.refresh_context_snapshot())
    expected = resolve_token_count(
        [session.build_system_message(), human], known_input_tokens=None
    )
    conversation_only = resolve_token_count([human], known_input_tokens=None)
    assert session.context_used == expected
    assert session.context_used > conversation_only


def test_incremental_input_tokens_counts_only_new_semantic_tail():
    from langchain_core.messages import SystemMessage

    from ness_agent.graph.helpers import _incremental_input_tokens

    user = HumanMessage(content="task")
    overlay = HumanMessage(
        content="<system-reminder>plan</system-reminder>",
        additional_kwargs={"ness_internal": "overlay"},
    )
    tool = ToolMessage(content="big output", tool_call_id="c1", name="read")
    system = SystemMessage(content="sys")

    stored_context = [user, overlay]
    conversation = [user, overlay, tool]
    tool_tokens = resolve_token_count([tool], known_input_tokens=None)

    assert _incremental_input_tokens(
        conversation=conversation,
        stored_context=stored_context,
        stored_system=system,
        current_system=system,
        last_input=1000,
    ) == 1000 + tool_tokens


def test_compaction_retains_multi_step_active_tool_trajectory(tmp_path: Path):
    class SummaryBinding:
        def __init__(self):
            self.calls = []

        async def ainvoke(self, messages, **_kwargs):
            self.calls.append(list(messages))
            return AIMessage(content="completed history summary")

    agent = _agent(tmp_path)
    rt = make_nodes(
        agent.config,
        thread_id="session-tool-trajectory",
        mode="act",
        git_available=False,
    )
    binding = SummaryBinding()
    rt.last_bound_model = binding
    active_user = HumanMessage(content="inspect and continue", id="active-tool-turn")
    tool_call = AIMessage(
        content="",
        tool_calls=[{"name": "ping", "args": {}, "id": "call-1"}],
    )
    tool_result = ToolMessage(
        content="pong",
        name="ping",
        tool_call_id="call-1",
    )
    updates = asyncio.run(rt.context_gate({
        "messages": [
            HumanMessage(content="completed task"),
            AIMessage(content="completed answer"),
            active_user,
            tool_call,
            tool_result,
        ],
        "force_compact": True,
        "mode": "act",
    }))

    retained = updates["model_context_messages"]
    assert "<compacted-history>" in retained[0].content
    assert retained[1:] == [active_user, tool_call, tool_result]
    assert updates["compaction_status"]["overlay_note"].startswith(
        "Conversation was summarized at this model boundary."
    )
    checkpoint = agent.config.thread_store.load_thread_events(
        "session-tool-trajectory"
    )[-1]
    durable = messages_from_dict(checkpoint["active_suffix"])
    assert [message.type for message in durable] == ["human", "ai", "tool"]
    assert durable[-1].tool_call_id == "call-1"


def test_summary_failure_is_suppressed_for_same_active_turn(tmp_path: Path):
    class FailingSummaryBinding:
        def __init__(self):
            self.calls = 0

        async def ainvoke(self, _messages, **_kwargs):
            self.calls += 1
            raise RuntimeError("summary unavailable")

    agent = _agent(tmp_path)
    rt = make_nodes(
        agent.config,
        thread_id="session-summary-retry",
        mode="act",
        git_available=False,
    )
    binding = FailingSummaryBinding()
    rt.last_bound_model = binding
    active = HumanMessage(content="active task", id="same-active-turn")
    state = {
        "messages": [
            HumanMessage(content="old task"),
            AIMessage(content="old answer"),
            active,
        ],
        "mode": "act",
    }
    pressure = ContextPressure(
        token_count=850,
        context_limit=1_000,
        ratio=0.85,
        warning=True,
        should_compact=True,
        safety_threshold_reached=False,
        hard_threshold_reached=False,
    )
    with patch("ness_agent.graph.nodes.calculate_context_pressure", return_value=pressure):
        failed = asyncio.run(rt.context_gate(state))
        assert failed["compaction_status"]["skip_reason"] == "failed"
        assert "Summary compaction is due." in failed["compaction_status"]["overlay_note"]
        suppressed = asyncio.run(rt.context_gate({
            **state,
            "compaction_failed_turn_id": failed["compaction_failed_turn_id"],
        }))

    assert suppressed["compaction_status"]["skip_reason"] == "retry_suppressed"
    assert binding.calls == 1


def test_ordinary_calls_retain_exact_prior_wire_prefix(tmp_path: Path):
    class RecordingModel:
        model = "recording"

        def __init__(self):
            self.calls = []

        def bind_tools(self, _tools):
            return self

        async def ainvoke(self, messages, **_kwargs):
            self.calls.append(list(messages))
            return AIMessage(content=f"answer-{len(self.calls)}")

    model = RecordingModel()
    agent = NessAgent(
        model=model,
        tools=[],
        prompt=PromptLayers(PromptLayersConfig(l0="L0", persona="P")),
        options=NessAgentOptions(ness_dir=tmp_path / ".ness", project_root=tmp_path),
    )
    session = agent.session(thread_id="session-prefix", mode="plan", git_available=False)
    asyncio.run(session.run("one"))
    asyncio.run(session.run("two"))
    assert model.calls[1][: len(model.calls[0])] == model.calls[0]
    state = asyncio.run(session.get_state())
    assert not any(
        (m.additional_kwargs or {}).get("ness_internal") == "overlay"
        for m in state["messages"]
    )
    assert any(
        (m.additional_kwargs or {}).get("ness_internal") == "overlay"
        for m in state["model_context_messages"]
    )


def test_usage_event_always_logged_with_model(tmp_path: Path):
    model = FakeListChatModel(responses=["hello"])
    object.__setattr__(model, "model", "usage-model")
    store = ThreadStore(threads_dir=tmp_path / "threads", default_model="")

    @tool
    def ping() -> str:
        """Return pong."""
        return "pong"

    agent = NessAgent(
        model=model,
        tools=[ping],
        prompt=PromptLayers(PromptLayersConfig(l0="L0", persona="P")),
        options=NessAgentOptions(ness_dir=tmp_path / ".ness", project_root=tmp_path),
        cost_tracker=CostTracker(),
    )
    agent.config.thread_store = store
    agent.config.cost_tracker.add = lambda *a, **k: None  # type: ignore[method-assign]
    rt = make_nodes(agent.config, thread_id="session-u", mode="act", git_available=False)

    response = AIMessage(content="ok")
    response.usage_metadata = {"input_tokens": 5, "output_tokens": 1}

    async def fake_ainvoke(_msgs):
        return response

    with patch.object(
            agent.config.tool_registry,
            "bind_model",
            return_value=SimpleNamespace(ainvoke=fake_ainvoke),
        ):
        asyncio.run(
            rt.agent_node(
                {
                    "messages": [HumanMessage(content="hi")],
                    "mode": "act",
                    "todos": [],
                    "last_input_tokens": 0,
                }
            )
        )

    events = agent.config.thread_store.load_thread_events("session-u")
    usage = next(e for e in events if e.get("kind") == "usage")
    assert usage["model"] == "usage-model"


def test_options_context_window_drives_usable_budget():
    opts = NessAgentOptions(
        context_window=100_000,
        compaction_buffer_tokens=10_000,
        compaction_summary_max_tokens=2_000,
    )
    assert resolve_usable_context_budget("any-model", opts) == 100_000
    assert resolve_usable_context_budget("any-model", None) == 120_000
    assert resolve_usable_context_budget(
        "any-model",
        NessAgentOptions(context_window=None, compaction_token_budget=50_000),
    ) == 50_000


def test_agent_spec_resolves_backends(tmp_path: Path):
    from ness_agent import AgentSpec, NessAgent
    from ness_agent.tracing.tracer import NoopTracer

    model = FakeListChatModel(responses=["ok"])

    @tool
    def ping() -> str:
        """Return pong."""
        return "pong"

    spec = AgentSpec(
        model=model,
        tools=[ping],
        prompt=PromptLayers(PromptLayersConfig(l0="L0", persona="P")),
        options=NessAgentOptions(ness_dir=tmp_path / ".ness", project_root=tmp_path),
    )
    agent = NessAgent.from_spec(spec)
    cfg = agent.config
    assert cfg.memory_store is not None
    assert cfg.thread_store is not None
    assert cfg.permission_store is not None
    assert cfg.tool_registry is not None
    assert cfg.cost_tracker is not None
    assert isinstance(cfg.tracer, NoopTracer)
    assert not hasattr(cfg, "budget")
    assert not hasattr(cfg, "permission_policy")
    assert not hasattr(cfg, "mcp_config")


def test_aggregate_usage_sums_calls_and_costs():
    from ness_agent.types import UsageEvent, aggregate_usage

    assert aggregate_usage([]) is None
    total = aggregate_usage(
        [
            UsageEvent(
                "m", 10, 8, 2, 3, 0.01, calls=1, cache_write_input_tokens=7
            ),
            UsageEvent(
                "m", 20, 15, 5, 4, 0.02, calls=1, cache_write_input_tokens=11
            ),
        ]
    )
    assert total is not None
    assert total.model == "m"
    assert total.input_tokens == 30
    assert total.uncached_input_tokens == 23
    assert total.cached_input_tokens == 7
    assert total.cache_write_input_tokens == 18
    assert total.output_tokens == 7
    assert total.cost_usd == 0.03
    assert total.calls == 2

    mixed = aggregate_usage(
        [
            UsageEvent("a", 1, 1, 0, 1, None),
            UsageEvent("b", 2, 2, 0, 1, 0.5),
        ]
    )
    assert mixed is not None
    assert mixed.model == "*"
    assert mixed.cost_usd == 0.5


def test_run_result_usage_total_accumulates_bridge_events(tmp_path: Path):
    """Session.run exposes usage_total as the sum of per-call usage events."""
    from ness_agent.types import UsageEvent, aggregate_usage

    model = FakeListChatModel(responses=["done"])
    agent = NessAgent(
        model=model,
        tools=[],
        prompt=PromptLayers(PromptLayersConfig(l0="L0", persona="P")),
        options=NessAgentOptions(
            ness_dir=tmp_path / ".ness",
            project_root=tmp_path,
            enable_approval=False,
            auto_save_threads=False,
        ),
    )
    session = agent.session(thread_id="t-usage-total")

    # Simulate the usage bridge the agent node would fire mid-turn.
    from ness_agent.session import _active_session
    from ness_agent.session_context import reset_session_context

    async def _run():
        ctx_token = session._install_session_runtime()
        session._last_usage = None
        session._turn_usages = []
        token = _active_session.set(session)
        try:
            bridge = session.config._usage_bridge
            bridge(UsageEvent("m", 100, 90, 10, 5, 0.1))
            bridge(UsageEvent("m", 200, 150, 50, 8, 0.2))
            assert session._last_usage is not None
            assert session._last_usage.input_tokens == 200
            total = aggregate_usage(session._turn_usages)
            assert total is not None
            assert total.input_tokens == 300
            assert total.calls == 2
            assert total.cost_usd == pytest.approx(0.3)
        finally:
            _active_session.reset(token)
            reset_session_context(ctx_token)

    asyncio.run(_run())


def test_run_result_exposes_only_aggregate_usage_field():
    from ness_agent.types import RunResult

    result = RunResult(assistant_message="done", todos=[], events=[])

    assert result.usage_total is None
    assert not hasattr(result, "usage")
