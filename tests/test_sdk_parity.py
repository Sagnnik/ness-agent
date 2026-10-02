from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from ness_agent import (
    CodingOverlay,
    NessAgent,
    PromptLayers,
    PromptLayersConfig,
    Session,
)
from ness_agent.graph.nodes import make_nodes
from ness_agent.options import ModeConfig, NessAgentOptions


def _agent(**kwargs):
    model = FakeListChatModel(responses=["hello"])

    @tool
    def ping() -> str:
        """Return pong."""
        return "pong"

    tools = kwargs.pop("tools", [ping])
    if "options" not in kwargs:
        root = Path(tempfile.mkdtemp(prefix="ness-sdk-parity-"))
        kwargs["options"] = NessAgentOptions(
            project_root=root,
            ness_dir=root / ".ness",
        )
    return NessAgent(
        model=model,
        tools=tools,
        prompt=PromptLayers(PromptLayersConfig(l0="L0", persona="P")),
        **kwargs,
    )


def test_toggle_mode():
    agent = _agent()
    session = agent.session(thread_id="t-toggle")
    assert session.toggle_mode() == "plan"
    assert session.toggle_mode() == "act"


def test_preview_context_system_and_l3(tmp_path: Path):
    from ness_agent import ContextPreview

    agent = _agent(
        options=NessAgentOptions(project_root=tmp_path, ness_dir=tmp_path / ".ness"),
        overlay=CodingOverlay(plan_mode_template="PLAN BODY"),
    )
    session = agent.session(thread_id="t-preview", mode="act", git_available=False)

    async def _run():
        act = await session.preview_context()
        assert isinstance(act, ContextPreview)
        assert "L0" in act.system_message
        assert act.mode == "act"
        assert "plan_mode" not in act.overlay_sections
        assert act.overlay_reminder == "" or "<system-reminder>" in act.overlay_reminder

        plan = await session.preview_context(mode="plan")
        assert plan.mode == "plan"
        assert "plan_mode" in plan.overlay_sections
        assert "PLAN BODY" in plan.overlay
        assert plan.overlay_reminder.startswith("<system-reminder>")
        assert plan.system_message  # same L0–L2 shape for both modes

    asyncio.run(_run())


def test_plan_mode_gates_writes_without_mode_config(tmp_path: Path):
    """Plan-mode write gating must not require ModeConfig to be present."""
    agent = _agent(
        tools=["write", "read"],
        options=NessAgentOptions(project_root=tmp_path, ness_dir=tmp_path / ".ness"),
        modes=None,
    )
    assert agent.config.modes is None
    rt = make_nodes(agent.config, thread_id="t-plan-gate", mode="plan", git_available=False)

    async def _run():
        ai = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "write",
                    "args": {"path": "x.py", "content": "print(1)"},
                    "id": "w1",
                }
            ],
        )
        route = await rt.route_after_agent({"messages": [ai], "mode": "plan"})
        assert route == "tools"
        out = await rt.tools_node({"messages": [ai], "mode": "plan", "todos": []})
        msgs = out["messages"]
        assert len(msgs) == 1
        assert "Unavailable in plan mode" in msgs[0].content

    asyncio.run(_run())


def test_plan_mode_readonly_false_allows_mutating_tools(tmp_path: Path):
    agent = _agent(
        tools=["write"],
        options=NessAgentOptions(
            project_root=tmp_path,
            ness_dir=tmp_path / ".ness",
            enable_approval=False,
        ),
        modes=ModeConfig(plan_mode_readonly=False),
    )
    rt = make_nodes(agent.config, thread_id="t-plan-rw", mode="plan", git_available=False)

    async def _run():
        ai = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "write",
                    "args": {"path": str(tmp_path / "ok.py"), "content": "x=1\n"},
                    "id": "w2",
                }
            ],
        )
        route = await rt.route_after_agent({"messages": [ai], "mode": "plan"})
        assert route == "tools"
        out = await rt.tools_node({"messages": [ai], "mode": "plan", "todos": []})
        assert "Unavailable in plan mode" not in out["messages"][0].content

    asyncio.run(_run())


def test_yolo_bypasses_deny_rules_but_not_plan_mode(tmp_path: Path):
    agent = _agent(
        tools=["write"],
        options=NessAgentOptions(
            project_root=tmp_path,
            ness_dir=tmp_path / ".ness",
            enable_approval=True,
            yolo_mode=True,
        ),
    )
    cfg = agent.config
    rt = make_nodes(agent.config, thread_id="t-yolo", mode="act", git_available=False)

    async def _run():
        target = tmp_path / "yolo.py"
        ai = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "write",
                    "args": {"path": str(target), "content": "allowed = True\n"},
                    "id": "y1",
                }
            ],
        )
        assert (
            await rt.route_after_agent({"messages": [ai], "mode": "act"})
            == "tools"
        )
        cfg.permission_store.persist_rule("write:*", "deny", scope="session")
        out = await rt.tools_node({"messages": [ai], "mode": "act", "todos": []})
        assert "Denied by permission rule" not in out["messages"][0].content

        plan_out = await rt.tools_node(
            {"messages": [ai], "mode": "plan", "todos": []}
        )
        assert "Unavailable in plan mode" in plan_out["messages"][0].content

    asyncio.run(_run())


def test_approval_session_and_never_persist(tmp_path: Path):
    from ness_agent.types import ApprovalHandler

    decisions = iter(["session", "never"])

    class TestHandler(ApprovalHandler):
        async def __call__(self, name: str, args: dict) -> str:
            return next(decisions)

    agent = _agent(
        approval_handler=TestHandler(),
        options=NessAgentOptions(
            project_root=tmp_path,
            ness_dir=tmp_path / ".ness",
            enable_approval=True,
            auto_save_threads=True,
        ),
    )
    cfg = agent.config
    cfg.thread_store.auto_save = True
    rt = make_nodes(cfg, thread_id="t-appr", mode="act", git_available=False)

    async def _run():
        # Destructive shell cmds not in allow/deny lists → ask
        # Use distinct first tokens so session rules don't collide (python* vs cargo*).
        args_session = {"action": "run", "command": "npm run build"}
        ai = AIMessage(
            content="",
            tool_calls=[{"name": "shell", "args": args_session, "id": "c1"}],
        )
        result = await rt.approval_gate({"messages": [ai], "approval_declined": {}})
        assert result.get("approval_declined") == {}
        assert cfg.permission_store.check("shell", args_session) == "allow"

        args_never = {"action": "run", "command": "cargo test"}
        ai2 = AIMessage(
            content="",
            tool_calls=[{"name": "shell", "args": args_never, "id": "c2"}],
        )
        result2 = await rt.approval_gate({"messages": [ai2], "approval_declined": {}})
        denials = result2.get("approval_declined") or {}
        assert "c2" in denials
        assert "Denied by persisted permission rule" in denials["c2"]
        assert cfg.permission_store.check("shell", args_never) == "deny"

    asyncio.run(_run())


def test_approval_deny_preserves_sibling_tools(tmp_path: Path):
    """Denying one gated call must not cancel independent siblings in the batch."""
    from ness_agent.types import ApprovalHandler
    from ness_agent.session_context import SessionContext, set_session_context, reset_session_context

    class DenyOnce(ApprovalHandler):
        async def __call__(self, name: str, args: dict) -> str:
            return "no"

    agent = _agent(
        tools=["read", "shell"],
        approval_handler=DenyOnce(),
        options=NessAgentOptions(
            project_root=tmp_path,
            ness_dir=tmp_path / ".ness",
            enable_approval=True,
            auto_save_threads=True,
        ),
    )
    cfg = agent.config
    cfg.thread_store.auto_save = True
    rt = make_nodes(cfg, thread_id="t-partial-deny", mode="act", git_available=False)

    safe_path = tmp_path / "note.txt"
    safe_path.write_text("hello sibling\n", encoding="utf-8")
    ctx_token = set_session_context(
        SessionContext(
            permissions=cfg.permission_store,
            options=cfg.options,
            thread_store=cfg.thread_store,
            ness_dir=cfg.options.ness_dir or (tmp_path / ".ness"),
            project_root=tmp_path,
            agent_config=cfg,
        )
    )

    async def _run():
        ai = AIMessage(
            content="",
            tool_calls=[
                {"name": "read", "args": {"path": str(safe_path)}, "id": "safe"},
                {
                    "name": "shell",
                    "args": {"action": "run", "command": "npm run build"},
                    "id": "gated",
                },
            ],
        )
        with patch("warnings.warn") as warn:
            gate = await rt.approval_gate({"messages": [ai], "approval_declined": {}})
            warn.assert_not_called()

        denials = gate.get("approval_declined") or {}
        assert list(denials) == ["gated"]
        assert "Denied by user approval: shell" in denials["gated"]
        assert not gate.get("messages")

        route = await rt.route_after_approval({"messages": [ai], "approval_declined": denials})
        assert route == "tools"

        out = await rt.tools_node(
            {
                "messages": [ai],
                "approval_declined": denials,
                "todos": [],
                "mode": "act",
            }
        )
        by_id = {msg.tool_call_id: msg for msg in out["messages"]}
        assert "hello sibling" in by_id["safe"].content
        assert "Denied by user approval: shell" in by_id["gated"].content
        assert out.get("approval_declined") == {}

    try:
        asyncio.run(_run())
    finally:
        reset_session_context(ctx_token)


@pytest.mark.parametrize("with_prior", [False, True])
@pytest.mark.parametrize("model_fails", [False, True])
def test_session_context_restored_on_success_and_failure(
    tmp_path: Path, with_prior, model_fails
):
    from ness_agent.session_context import (
        SessionContext,
        set_session_context,
        reset_session_context,
        try_get_session_context,
    )

    error = RuntimeError("context-cleanup model failure") if model_fails else None
    agent = _bindable_agent(tmp_path, error=error)
    session = agent.session(thread_id="t-ctx-reset")
    prior = (
        SessionContext(
            permissions=agent.config.permission_store,
            options=agent.config.options,
            thread_store=agent.config.thread_store,
            ness_dir=tmp_path / ".ness",
            project_root=tmp_path,
            agent_config=agent.config,
            available_skills={"marker": {"name": "marker"}},
        )
        if with_prior
        else None
    )
    token = set_session_context(prior)
    try:
        async def exercise():
            assert try_get_session_context() is prior
            events = [event async for event in session.stream("hello")]
            if model_fails:
                assert any(
                    event.kind == "error"
                    and "context-cleanup model failure" in event.data["message"]
                    for event in events
                )
            else:
                assert not any(event.kind == "error" for event in events)
                assert any(
                    event.kind == "assistant_final"
                    and event.data["content"] == "FINAL-ANSWER"
                    for event in events
                )
            restored = try_get_session_context()
            assert restored is prior
            if with_prior:
                assert restored.available_skills == {"marker": {"name": "marker"}}

        asyncio.run(exercise())
    finally:
        reset_session_context(token)


# ---------------------------------------------------------------------------
# Per-Session runtime hooks, bootstrap, cancel, finalize
# (Phase 1d additions — domain-agnostic SDK Session behaviour)
# ---------------------------------------------------------------------------


class _FakeApp:
    """Stand-in compiled graph for SDK Session stream/cancel tests.

    Yields a fixed stream of astream_events chunks; triggers the Session's
    cancel token after a configurable number of events; and records any
    aupdate_state calls so the cancel-finalize path is observable without
    requiring a real langgraph run.
    """

    def __init__(
        self,
        events: list[dict],
        *,
        trigger_cancel_after: int | None = None,
        session: Session | None = None,
        snapshot_messages: list | None = None,
    ) -> None:
        self._events = list(events)
        self._trigger_after = trigger_cancel_after
        self._session = session
        self._snapshot_messages = snapshot_messages or []
        self.updates: list[dict] = []
        self.last_payload: dict | None = None

    async def astream_events(self, payload, *, config=None, version="v2"):
        self.last_payload = payload
        if self._trigger_after == 0 and self._session is not None:
            self._session.cancel()
        for index, event in enumerate(self._events):
            yield event
            if (
                self._trigger_after is not None
                and self._session is not None
                and index + 1 >= self._trigger_after
            ):
                self._session.cancel()

    async def aget_state(self, config):
        return type(
            "Snapshot",
            (),
            {"values": {"messages": list(self._snapshot_messages)}},
        )()

    async def aupdate_state(self, config, updates):
        self.updates.append(updates)


def _stream_session(session, message="hi", **kwargs):
    async def _run():
        events = []
        async for ev in session.stream(message, **kwargs):
            events.append(ev)
        return events

    return asyncio.run(_run())


def test_bootstrap_seeds_exactly_one_turn():
    agent = _agent()
    session = agent.session(thread_id="t-boot")
    fake = _FakeApp([{"event": "on_chain_end", "name": "agent", "data": {"output": {"messages": []}}}])
    session._app = fake

    session.bootstrap([HumanMessage(content="seed")])
    _stream_session(session)
    assert fake.last_payload is not None
    assert [message.content for message in fake.last_payload["messages"]] == [
        "seed", "hi"
    ]

    _stream_session(session, "next turn")
    assert [message.content for message in fake.last_payload["messages"]] == ["next turn"]


@pytest.mark.parametrize("with_hook", [False, True])
def test_plan_turn_delivers_real_answer_to_hook_or_event(tmp_path: Path, with_hook):
    agent = _bindable_agent(tmp_path, "PLAN OK")
    received = []
    session = agent.session(
        thread_id="t-plan",
        mode="plan",
        on_plan_turn=received.append if with_hook else None,
    )
    events = _stream_session(session)
    assert not any(event.kind == "error" for event in events)
    finals = [event.data["content"] for event in events if event.kind == "assistant_final"]
    assert finals[-1] == "PLAN OK"
    plans = [event.data["text"] for event in events if event.kind == "plan_turn"]
    assert received == (["PLAN OK"] if with_hook else [])
    assert plans == ([] if with_hook else ["PLAN OK"])


def test_session_cancel_synthesises_failed_toolmessage():
    agent = _agent()
    session = agent.session(thread_id="t-cancel-tools")
    # The snapshot the finalize path will read: an AIMessage with a pending
    # tool_call and no matching ToolMessage → must synthesise one.
    pending_ai = AIMessage(
        content="",
        tool_calls=[{"name": "ping", "args": {}, "id": "c1", "type": "tool_call"}],
    )
    fake = _FakeApp(
        [
            {
                "event": "on_chat_model_end",
                "name": "agent",
                "data": {
                    "output": {
                        "messages": [pending_ai],
                    }
                },
            },
        ],
        trigger_cancel_after=1,
        session=session,
        snapshot_messages=[pending_ai],
    )
    session._app = fake

    events = _stream_session(session)

    assert any(ev.kind == "interrupted" for ev in events)
    # The synthetic failed ToolMessage must have been written back.
    assert fake.updates, "aupdate_state was not called by finalize"
    flat = []
    for upd in fake.updates:
        flat.extend(upd.get("messages", []))
    synth = [m for m in flat if isinstance(m, ToolMessage)]
    assert synth, "no synthetic ToolMessage was written"
    assert synth[0].tool_call_id == "c1"
    assert "interrupted" in str(synth[0].content).lower()


def test_interruption_marker_on_empty_cancel():
    agent = _agent()
    session = agent.session(thread_id="t-cancel-empty")
    fake = _FakeApp(
        [
            {"event": "on_chat_model_start", "name": "agent"},
        ],
        trigger_cancel_after=1,
        session=session,
        snapshot_messages=[],
    )
    session._app = fake

    events = _stream_session(session)

    assert any(ev.kind == "interrupted" for ev in events)
    # No partial text and no pending tools → marker AIMessage written.
    flat = []
    for upd in fake.updates:
        flat.extend(upd.get("messages", []))
    markers = [m for m in flat if isinstance(m, AIMessage)]
    assert markers, "interruption marker AIMessage was not written"
    assert session._cfg.options.interruption_marker in str(markers[-1].content)


def test_answered_image_blocks_remain_until_compaction():
    agent = _agent()
    session = agent.session(thread_id="t-imgstrip", vision=True)

    # Canonical history retains the exact image-bearing message after an
    # answer. Summary compaction is the only operation allowed to replace it.
    prior_human = HumanMessage(
        content=[
            {"type": "text", "text": "look at this"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,xxx"}},
        ],
        id="img-1",
    )
    prior_ai = AIMessage(content="got it")

    class _StripFakeApp:
        def __init__(self):
            self.updates = []
            self.last_payload = None
            self._messages = [prior_human, prior_ai]

        async def astream_events(self, payload, *, config=None, version="v2"):
            self.last_payload = payload
            yield {
                "event": "on_chain_end",
                "name": "agent",
                "data": {"output": {"messages": []}},
            }

        async def aget_state(self, config):
            return type(
                "Snapshot",
                (),
                {"values": {"messages": list(self._messages)}},
            )()

        async def aupdate_state(self, config, updates):
            self.updates.append(updates)
            # Subsequent aget_state calls reflect the replacement.
            for m in updates.get("messages", []):
                if getattr(m, "id", None) == "img-1":
                    self._messages[0] = m

    fake = _StripFakeApp()
    session._app = fake

    _stream_session(session, "next turn")

    assert fake.updates == []
    assert isinstance(fake._messages[0].content, list)
    assert fake._messages[0].content == prior_human.content


def test_vision_disabled_emits_warning_and_drops_images():
    agent = _agent()
    session = agent.session(thread_id="t-vision-off", vision=False)
    fake = _FakeApp(
        [{"event": "on_chain_end", "name": "agent", "data": {"output": {"messages": []}}}]
    )
    session._app = fake

    events = _stream_session(session, "see this [Image #1]", images=["data:image/png;base64,zz"])

    assert any(ev.kind == "warning" for ev in events)
    # The payload message is text-only (no image_url blocks). TUI
    # ``[Image #N]`` placeholder stripping is adapter-owned — Session
    # forwards the text as given.
    msgs = fake.last_payload["messages"]
    user_msg = msgs[-1]
    assert isinstance(user_msg.content, str)
    assert "[Image #1]" in user_msg.content
    assert not isinstance(user_msg.content, list)


# ---------------------------------------------------------------------------
# Regression tests for the independent reviewer's findings.
# ---------------------------------------------------------------------------


class _BindableFakeModel:
    """Duck-typed chat model whose ``bind_tools`` works (unlike
    ``FakeListChatModel``) so a real langgraph run can complete. Returns a
    fixed ``AIMessage`` so the agent node's ``on_chain_end`` output carries
    it and the SDK can emit ``assistant_final``.
    """

    def __init__(self, text: str = "FINAL-ANSWER", error: Exception | None = None) -> None:
        self.text = text
        self.error = error
        self.calls = 0

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages, **kwargs):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return AIMessage(content=self.text)

    @property
    def model(self):
        return "bindfake"


def _bindable_agent(
    tmp_path: Path, text: str = "FINAL-ANSWER", *, error: Exception | None = None
):
    @tool
    def ping() -> str:
        """Return pong."""
        return "pong"

    from ness_agent.options import NessAgentOptions

    return NessAgent(
        model=_BindableFakeModel(text, error),
        tools=[ping],
        prompt=PromptLayers(PromptLayersConfig(l0="L0", persona="P")),
        options=NessAgentOptions(project_root=tmp_path, ness_dir=tmp_path / ".ness"),
    )


def test_assistant_final_emitted_for_real_graph_run(tmp_path: Path):
    """Finding 2: ``assistant_final`` was never emitted because the SDK
    filtered ``on_chat_model_end`` by ``name == "agent"`` (real model events
    carry the model runnable name). Emitting it from ``on_chain_end`` (agent)
    — which carries only the agent response message — restores it and
    excludes the compaction summarizer that runs in the same node.
    """
    agent = _bindable_agent(tmp_path, "FINAL-ANSWER")
    session = agent.session(thread_id="t-final")

    async def _run():
        kinds, finals = [], []
        async for ev in session.stream("hi"):
            kinds.append(ev.kind)
            if ev.kind == "assistant_final":
                finals.append(ev.data.get("content"))
        return kinds, finals

    kinds, finals = asyncio.run(_run())
    assert "assistant_final" in kinds, f"assistant_final missing: {kinds}"
    assert finals[-1] == "FINAL-ANSWER"


def test_mode_override_is_turn_only_and_restores(tmp_path: Path):
    """Finding 9: the ``mode`` kwarg docstring promises "this turn only", but
    ``set_mode`` permanently mutated session mode. After the override turn the
    session must return to its prior mode (and must not schedule a spurious
    plan->act compaction checkpoint for the next turn).
    """
    agent = _bindable_agent(tmp_path, "plan text")
    session = agent.session(thread_id="t-mode", mode="act")
    assert session.mode == "act"

    async def _run():
        async for _ in session.stream("hi", mode="plan"):
            pass

    asyncio.run(_run())

    # Restore: the override was this-turn-only.
    assert session.mode == "act", "mode override leaked across turns"


def test_staged_skills_replace_append_override_and_consume_once():
    agent = _agent()
    session = agent.session(thread_id="t-skills")
    fake = _FakeApp([])
    session._app = fake

    session.requested_skills(["stale"])
    session.requested_skills(["a", "b"])
    session.stage_skills(["b", "c"])

    _stream_session(session, requested_skills=["explicit"])
    assert fake.last_payload["requested_skills"] == ["explicit"]

    _stream_session(session, "consume staged skills")
    assert fake.last_payload["requested_skills"] == ["a", "b", "c"]

    _stream_session(session, "no skills left")
    assert fake.last_payload["requested_skills"] == []


@pytest.mark.parametrize("entrypoint", ["run", "stream"])
def test_existing_active_skills_calls_use_requested_state(entrypoint):
    agent = _agent()
    session = agent.session(thread_id="t-legacy-skills")
    fake = _FakeApp([])
    session._app = fake
    session.active_skills(["staged"])

    async def run():
        if entrypoint == "run":
            await session.run("explicit", active_skills=["legacy"])
            assert fake.last_payload["requested_skills"] == ["legacy"]
            await session.run("consume staged")
        else:
            async for _ in session.stream("explicit", active_skills=["legacy"]):
                pass
            assert fake.last_payload["requested_skills"] == ["legacy"]
            async for _ in session.stream("consume staged"):
                pass
        assert fake.last_payload["requested_skills"] == ["staged"]

    asyncio.run(run())
