"""Tests for the CodingSession adapter (Phase 2).

Integration tests use a real :class:`NessAgent` + ``FakeListChatModel`` against
a temp project root, so the per-turn checkpoint orchestration (thread_store
writes, save_checkpoint calls) is exercised end-to-end. Targeted unit tests
mock the underlying SDK ``Session`` to test the durable-compaction relocation
and the adapter-owned rollback mutation tracking without the langgraph run.
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from unittest.mock import PropertyMock, patch

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.tools import tool

from ness_agent import NessAgent, PromptLayers, PromptLayersConfig, SessionEvent
from ness_agent.options import NessAgentOptions
from ness_cli import CodingSession
from ness_cli.events import events_to_messages


def _make_agent(tmp_path: Path):
    """Build a NessAgent on a temp project root with auto-save enabled."""

    @tool
    def ping() -> str:
        """Return pong."""
        return "pong"

    agent = NessAgent(
        model=FakeListChatModel(responses=["ok"]),
        tools=[ping],
        prompt=PromptLayers(PromptLayersConfig(l0="L0", persona="P")),
        options=NessAgentOptions(
            project_root=tmp_path,
            ness_dir=tmp_path / ".ness",
            auto_save_threads=True,
        ),
    )
    agent.config.thread_store.auto_save = True
    return agent


@pytest.fixture
def coding(tmp_path: Path):
    """A CodingSession on a fresh temp project root, act mode, vision off."""
    return CodingSession(
        _make_agent(tmp_path),
        thread_id="t-cli-1",
        mode="act",
        vision=False,
    )


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def test_run_turn_writes_checkpoint_and_user_event(coding):
    # Capture the seq assigned by append_event so the checkpoint lookup is
    # exact (load_thread_events does not return the seq column).
    captured_seq = {}

    original_append = coding.thread_store.append_event

    def _spy_append(thread_id, event):
        seq = original_append(thread_id, event)
        if event.get("kind") == "user":
            captured_seq["seq"] = seq
        return seq

    coding.thread_store.append_event = _spy_append

    events = _run(_collect(coding.run_turn("hello")))

    # The user event was persisted to the durable events table.
    durable = coding.thread_store.load_thread_events(coding.thread_id)
    assert any(e.get("kind") == "user" and "hello" in str(e.get("content", "")) for e in durable), durable

    # The checkpoint row was written for that exact user_seq.
    seq = captured_seq.get("seq")
    assert seq is not None, "append_event did not return a seq for the user event"
    cp = coding.thread_store.get_checkpoint(coding.thread_id, seq)
    assert cp is not None, f"save_checkpoint did not write a row for seq={seq}"

    # The SDK emitted at least one assistant event (FakeListChatModel "ok").
    assert events, f"no SessionEvents were yielded: {events!r}"
    assert any(ev.kind in ("assistant_delta", "assistant_final", "error") for ev in events), events


def test_set_name_delegates_to_public_session_api(coding):
    assert coding.set_name("  Named   session ") is True
    assert coding.thread_store.thread_exists(coding.thread_id)
    with sqlite3.connect(coding.thread_store.threads_db) as conn:
        name = conn.execute(
            "SELECT name FROM threads WHERE thread_id = ?", (coding.thread_id,)
        ).fetchone()[0]
    assert name == "Named session"


def test_resume_named_thread_without_events(coding):
    coding.thread_store.set_thread_name("session-empty", "Empty session")

    assert _run(coding.resume("session-empty", replay_cost=False)) is True
    assert coding.thread_id == "session-empty"
    assert coding.session._pending_bootstrap == []


def test_resume_bootstraps_via_session_bootstrap(tmp_path: Path):
    coding = CodingSession(
        _make_agent(tmp_path),
        thread_id="t-resume-1",
        mode="act",
    )

    # Seed a prior thread with events we can resume from.
    other = "t-resume-src"
    coding.thread_store.append_event(other, {"kind": "user", "content": "first"})
    coding.thread_store.append_event(
        other,
        {"kind": "assistant", "content": "ok", "tool_calls": []},
    )

    _run(coding.resume(other, replay_cost=False))

    # The bootstrap list should be staged on the underlying Session and the
    # adapter's thread_id should now point at the resumed thread.
    assert coding.thread_id == other
    assert coding.session._pending_bootstrap, "bootstrap() did not stage messages"
    assert any(
        getattr(m, "content", "") == "first"
        for m in coding.session._pending_bootstrap
    )


def test_clone_for_thread_keeps_source_session_independent(coding):
    target = "session-cloned"
    coding.thread_store.append_event(
        target, {"kind": "user", "content": "background work"}
    )
    source_id = coding.thread_id

    clone = _run(coding.clone_for_thread(target))

    assert coding.thread_id == source_id
    assert clone.thread_id == target
    assert clone is not coding
    assert clone.session is not coding.session
    assert clone.session.checkpointer is not coding.session.checkpointer
    assert clone.session._pending_bootstrap


def test_resume_rebuilds_the_selected_threads_local_cost(coding):
    for thread_id in ("t-cost-a", "t-cost-b"):
        coding.thread_store.append_event(
            thread_id, {"kind": "user", "content": thread_id}
        )
        coding.thread_store.append_event(
            thread_id,
            {
                "kind": "usage",
                "model": "test",
                "input_tokens": 10,
                "output_tokens": 1,
                "cost_usd": 0.1,
            },
        )

    with patch(
        "ness_cli.coding_session.restore_cost_from_events"
    ) as restore:
        _run(coding.resume("t-cost-a"))
        _run(coding.resume("t-cost-b"))
        _run(coding.resume("t-cost-a"))
        _run(coding.resume("t-cost-b"))

    assert [call.args[0][0]["content"] for call in restore.call_args_list] == [
        "t-cost-a",
        "t-cost-b",
        "t-cost-a",
        "t-cost-b",
    ]


def test_reload_model_refreshes_vision_capability(coding):
    model = coding.cfg.model
    with (
        patch("ness_cli.chat_model.create_model", return_value=model),
        patch("ness_cli.chat_model.create_reflection_model", return_value=model),
        patch("ness_cli.config.context_window_for", return_value=128_000),
        patch(
            "ness_cli.config.Settings.supports_vision",
            new_callable=PropertyMock,
            return_value=True,
        ),
    ):
        coding.reload_model()

    assert coding._vision is True
    assert coding.session._vision is True


def test_reload_model_updates_selected_and_future_but_not_existing_sibling(coding):
    sibling = CodingSession(coding.agent, thread_id="t-existing", vision=False)
    old_model = sibling.cfg.model
    new_model = FakeListChatModel(responses=["new"])

    with (
        patch("ness_cli.chat_model.create_model", return_value=new_model),
        patch("ness_cli.chat_model.create_reflection_model", return_value=new_model),
        patch("ness_cli.chat_model.active_model_name", return_value="new-model"),
        patch("ness_cli.chat_model.active_provider_id", return_value="openrouter"),
        patch("ness_cli.chat_model.active_reasoning_effort", return_value="high"),
        patch("ness_cli.config.context_window_for", return_value=64_000),
        patch(
            "ness_cli.config.Settings.supports_vision",
            new_callable=PropertyMock,
            return_value=True,
        ),
    ):
        coding.reload_model()
        future = coding.new_for_thread("t-future")

    assert sibling.cfg.model is old_model
    assert sibling.cfg.options.context_window != 64_000
    assert coding.cfg.model is new_model
    assert coding.cfg.options.context_window == 64_000
    assert coding.model_name == "new-model"
    assert coding.reasoning_effort == "high"
    assert future.cfg.model is new_model
    assert future.cfg.options.context_window == 64_000


def test_rollback_truncates_and_restores_files(coding):
    # Seed events; then rollback to the SECOND user_seq, which truncates the
    # tail while leaving the first user turn intact.
    s1 = coding.thread_store.append_event(coding.thread_id, {"kind": "user", "content": "turn-1"})
    coding.thread_store.append_event(
        coding.thread_id, {"kind": "assistant", "content": "a1", "tool_calls": []}
    )
    s2 = coding.thread_store.append_event(coding.thread_id, {"kind": "user", "content": "turn-2"})
    coding.thread_store.append_event(
        coding.thread_id, {"kind": "assistant", "content": "a2", "tool_calls": []}
    )

    before = coding.thread_store.load_thread_events(coding.thread_id)
    assert len(before) == 4

    # seq starts at 0; user events land on even seqs (0, 2).
    assert s1 == 0 and s2 == 2, (s1, s2)

    # Create a synthetic checkpoint for s2 so rollback has a row to hit.
    coding.thread_store.save_checkpoint(coding.thread_id, s2, None, "")

    msg = _run(coding.rollback_to(s2))

    after = coding.thread_store.load_thread_events(coding.thread_id)
    # Truncate_after deletes seq >= s2: the first turn (turn-1 + a1) survives.
    assert len(after) == 2, after
    assert after[0]["kind"] == "user"
    assert after[0]["content"] == "turn-1"
    assert after[1]["kind"] == "assistant"
    # The returned message should not be the "no checkpoint" error.
    assert "No checkpoint" not in msg


def test_rollback_missing_checkpoint_returns_error(coding):
    coding.thread_store.append_event(coding.thread_id, {"kind": "user", "content": "x"})
    msg = _run(coding.rollback_to(99))
    assert "No checkpoint" in msg


def test_compaction_event_durable_logged_by_adapter(coding):
    """The adapter consumes a `compaction` SessionEvent into a durable row.

    This is the caveat-1 relocation: the SDK no longer writes ``compact`` rows
    into the thread_store; the adapter is the single owner of the durable
    compaction log.
    """
    fake_event = SessionEvent(
        "compaction",
        {"reason": "pre_act_user", "info": "Context ~120k tokens.", "forced": True},
    )

    # Patch the underlying Session.stream to emit only the fake compaction
    # event — avoids requiring a real compaction trigger.
    async def _fake_stream(*args, **kwargs):
        yield fake_event

    with patch.object(coding.session, "stream", _fake_stream):
        _run(_collect(coding.run_turn("go")))

    durable = coding.thread_store.load_thread_events(coding.thread_id)
    compacts = [e for e in durable if e.get("kind") == "compact"]
    assert compacts, "adapter did not durable-log the compaction event"
    assert "pre_act_user" in compacts[0].get("content", "")
    assert "[forced]" in compacts[0].get("content", "")


def _seed_tool_turn(coding, user_seq: int, tool_events: list[dict]) -> None:
    """Persist a user event + checkpoint at ``user_seq``, then durable tool rows.

    Mirrors what a real turn writes: the user event lands at ``user_seq``
    (tests pad with filler events when ``user_seq > 0``) and the graph's
    tools node appends ``_tool_event``-shaped rows after it.
    """
    for i in range(user_seq):
        coding.thread_store.append_event(
            coding.thread_id, {"kind": "filler", "content": str(i)}
        )
    coding.thread_store.append_event(coding.thread_id, {"kind": "user", "content": "x"})
    coding.thread_store.save_checkpoint(coding.thread_id, user_seq, "HEAD", "")
    for ev in tool_events:
        coding.thread_store.append_event(coding.thread_id, ev)


def _tool_event(tool: str, args: dict, *, result: str = "ok", exit: str = "ok") -> dict:
    return {
        "kind": "tool",
        "tool": tool,
        "args": args,
        "result": result,
        "call_id": "c1",
        "duration_ms": 1,
        "exit": exit,
    }


def test_record_turn_mutations_records_paths(coding):
    _seed_tool_turn(coding, 1, [_tool_event("write", {"path": "src/app.py"})])

    coding._record_turn_mutations(1)

    cp = coding.thread_store.get_checkpoint(coding.thread_id, 1)
    assert cp is not None
    paths = cp.get("modified_paths") or "[]"
    assert "src/app.py" in paths


def test_record_turn_mutations_shell_is_full_tree_sentinel(coding):
    _seed_tool_turn(coding, 1, [_tool_event("shell", {"command": "rm -rf build/"})])

    coding._record_turn_mutations(1)

    cp = coding.thread_store.get_checkpoint(coding.thread_id, 1)
    assert cp is not None
    paths = cp.get("modified_paths") or ""
    assert '"*"' in paths


def test_record_turn_mutations_skips_denied_mode_gated_and_tool_errors(coding):
    """The skips mirror the graph's own gates: permission/hook denies and
    plan-mode gating never executed, and a tool that raised (``Error:``
    prefix) touched nothing."""
    _seed_tool_turn(
        coding,
        1,
        [
            _tool_event("write", {"path": "denied.py"}, exit="denied"),
            _tool_event("write", {"path": "gated.py"}, exit="mode_gated"),
            _tool_event("edit", {"path": "boom.py"}, result="Error: disk full", exit="error"),
        ],
    )

    coding._record_turn_mutations(1)

    cp = coding.thread_store.get_checkpoint(coding.thread_id, 1)
    assert cp is not None
    assert (cp.get("modified_paths") or "") in ("", "[]")


def test_expand_documents_on_send_and_replay(tmp_path: Path):
    """@file mentions expand on send AND re-expand on events_to_messages replay.

    A symmetric property: both the live run_turn path and the resume rollback
    path re-read the file contents against current disk, so the model always
    sees fresh content.
    """
    (tmp_path / "alpha.txt").write_text("ALPHA-CONTENT-v1", encoding="utf-8")

    # auto-save off here — we don't run a real turn, only probe the expansion.
    agent = NessAgent(
        model=FakeListChatModel(responses=["ok"]),
        tools=[],
        prompt=PromptLayers(PromptLayersConfig(l0="L0", persona="P")),
        options=NessAgentOptions(
            project_root=tmp_path,
            ness_dir=tmp_path / ".ness",
            auto_save_threads=False,
        ),
    )
    coding = CodingSession(agent, thread_id="t-mentions", mode="act")

    from ness_cli.mentions import expand_documents

    expanded = expand_documents("@alpha.txt see this", coding.permission_store)
    assert "ALPHA-CONTENT-v1" in expanded
    assert "@alpha.txt" in expanded  # tag preserved verbatim

    # Simulate a persisted user event and confirm events_to_messages
    # re-expands against the (now-modified) disk file.
    events = [{"kind": "user", "content": "@alpha.txt see this"}]
    (tmp_path / "alpha.txt").write_text("ALPHA-CONTENT-v2", encoding="utf-8")
    msgs = events_to_messages(events, permission_store=coding.permission_store)
    assert msgs, "events_to_messages produced no messages"
    assert "ALPHA-CONTENT-v2" in str(msgs[0].content)
    assert "ALPHA-CONTENT-v1" not in str(msgs[0].content)


def test_vision_flag_forwarded_to_session(tmp_path: Path):
    agent = _make_agent(tmp_path)
    coding = CodingSession(agent, thread_id="t-vision-flag", vision=True)
    assert coding.session._vision is True

    codingoff = CodingSession(agent, thread_id="t-vision-off", vision=False)
    assert codingoff.session._vision is False


def test_plan_autosave_writes_plan_file(coding, tmp_path: Path):
    """Success-path plan text accumulates and autosaves to <plans>/."""
    coding.set_mode("plan")
    coding._handle_plan_turn_text("Step 1: scaffold")
    coding._handle_plan_turn_text("Step 2: wire up")
    coding._autosave_plan_turn()

    plans_dir = coding.ness_dir / "plans"
    plans = sorted(plans_dir.glob(f"*-{coding.thread_id}.md"))
    assert plans, "plan file was not autosaved"
    content = plans[-1].read_text(encoding="utf-8")
    # The autosave keeps the LAST non-empty entry (per plan_autosave_text).
    assert "Step 2: wire up" in content


def test_interrupted_plan_turn_text_is_suffixed(coding):
    """on_interrupt in plan mode appends the convention suffix to the partial text."""
    coding.set_mode("plan")
    coding._on_interrupt("partial plan text")

    assert coding._plan_turn_texts
    assert coding._plan_turn_texts[-1].endswith("… [interrupted]")
    assert "partial plan text" in coding._plan_turn_texts[-1]


# --- helpers ------------------------------------------------------------


async def _collect(agen):
    out: list[SessionEvent] = []
    async for ev in agen:
        out.append(ev)
    return out


def test_request_compact_writes_durable_row(coding):
    """Manual compaction requests are durable-logged by the adapter (the SDK
    only sets the force flag; the adapter owns the ``compact`` log)."""
    coding.request_compact()
    durable = [
        e
        for e in coding.thread_store.load_thread_events(coding.thread_id)
        if e.get("kind") == "compact"
    ]
    assert durable, "manual compaction was not durable-logged"
    assert "manual compaction" in durable[0].get("content", "")
    assert coding.session._force_compact is True


def test_resume_unknown_thread_returns_false_and_keeps_current(tmp_path: Path):
    """Resuming a thread with no persisted events is a no-op (the TUI shows
    "No saved thread") — the current thread is NOT archived or switched."""
    coding = CodingSession(
        _make_agent(tmp_path), thread_id="t-live", mode="act"
    )
    coding.thread_store.append_event("t-live", {"kind": "user", "content": "x"})

    ok = _run(coding.resume("t-does-not-exist"))
    assert ok is False
    assert coding.thread_id == "t-live"


class _BindableFakeModel:
    """bind_tools-capable fake so a REAL langgraph run completes (vs
    FakeListChatModel). Cycles fixed responses."""

    def __init__(self, texts: list[str]) -> None:
        self._texts = list(texts)
        self.calls = 0

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages, **kwargs):
        text = self._texts[min(self.calls, len(self._texts) - 1)]
        self.calls += 1
        from langchain_core.messages import AIMessage

        return AIMessage(content=text)

    @property
    def model(self):
        return "bindfake"


def _make_bindable_agent(tmp_path: Path, texts: list[str]):
    @tool
    def ping() -> str:
        """Return pong."""
        return "pong"

    agent = NessAgent(
        model=_BindableFakeModel(texts),
        tools=[ping],
        prompt=PromptLayers(PromptLayersConfig(l0="L0", persona="P")),
        options=NessAgentOptions(
            project_root=tmp_path,
            ness_dir=tmp_path / ".ness",
            auto_save_threads=True,
        ),
    )
    agent.config.thread_store.auto_save = True
    return agent


def test_rollback_rebuilds_graph_state_from_events(tmp_path: Path):
    """Regression: in-process rollback must NOT resurrect truncated turns or
    duplicate the replayed prefix in the checkpointer.

    Previously resume() reused the Session's MemorySaver; the replayed
    bootstrap merged with the stale checkpoint, so the rolled-back turn
    survived in graph state and the surviving history appeared twice.
    """
    agent = _make_bindable_agent(tmp_path, ["r-one", "r-two", "r-three"])
    coding = CodingSession(agent, thread_id="t-rb", mode="act")

    _run(_collect(coding.run_turn("turn-one")))
    _run(_collect(coding.run_turn("turn-two")))

    durable = coding.thread_store.load_thread_events(coding.thread_id)
    user_seqs = [
        i for i, e in enumerate(durable) if e.get("kind") == "user"
    ]
    assert len(user_seqs) == 2

    msg = _run(coding.rollback_to(user_seqs[1]))
    assert "No checkpoint" not in msg

    # The stale checkpoint must be gone (the replay is staged via bootstrap
    # and only re-enters the graph on the next turn).
    cfg = {"configurable": {"thread_id": coding.thread_id}}
    snap = _run(coding.app.aget_state(cfg))
    assert not (snap.values or {}).get("messages"), (
        "stale pre-rollback messages survived in the checkpointer"
    )

    # After the next turn, graph state must equal the truncated history plus
    # the new turn — no resurrection, no duplicated prefix.
    _run(_collect(coding.run_turn("turn-three")))
    snap = _run(coding.app.aget_state(cfg))
    contents = [str(m.content) for m in snap.values["messages"]]
    assert contents.count("turn-one") == 1, contents
    assert "turn-three" in contents
    assert not any("turn-two" in c or "r-two" in c for c in contents), contents
