from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from ness_agent import PermissionStore

from ness_cli.session.events import (
    DurableEvent,
    ToolMutationEvent,
    compact_event,
    parse_events,
    user_event,
)
from ness_cli.session.mentions import expand_documents, extract_mentions
from ness_cli.session.plans import PlanCapture, PlanStore, final_plan_text
from ness_cli.session.replay import events_to_messages, restore_cost
from ness_cli.session.rollback import mutated_paths


def test_durable_events_copy_input_and_preserve_unknown_kinds():
    source = {"kind": "future", "content": {"value": 1}}
    event = DurableEvent.parse(source, fallback_seq=4)
    source["content"]["value"] = 2
    assert event.kind == "future"
    assert event.seq == 4
    assert event.as_dict()["content"] == {"value": 1}


def test_event_helpers_are_stable_and_parse_invalid_sequence():
    assert user_event("hello", images=("data:image/png;base64,x",)) == {
        "kind": "user",
        "content": "hello",
        "images": ["data:image/png;base64,x"],
    }
    assert "forced" in compact_event(
        {"notice_reason": "threshold", "forced": True, "info": "trimmed"}
    )["content"]
    assert parse_events([{"kind": "user", "seq": True}])[0].seq == 0


@pytest.mark.parametrize(
    ("tool", "arguments", "expected"),
    [
        ("write", {"path": "a.py"}, ("a.py",)),
        ("EDIT", {"path": "b.py"}, ("b.py",)),
        ("shell", {"command": "touch x"}, ("*",)),
        ("read", {"path": "a.py"}, ()),
    ],
)
def test_mutated_paths_tracks_only_workspace_mutators(tool, arguments, expected):
    assert mutated_paths(tool, arguments) == expected


def test_tool_mutation_event_rejects_non_tool_and_freezes_arguments():
    assert ToolMutationEvent.parse(DurableEvent.parse({"kind": "user"}, fallback_seq=0)) is None
    parsed = ToolMutationEvent.parse(
        DurableEvent.parse(
            {"kind": "tool", "tool": "write", "args": {"path": "a"}, "result": "ok"},
            fallback_seq=1,
        )
    )
    assert parsed is not None and dict(parsed.arguments) == {"path": "a"}
    with pytest.raises(TypeError):
        parsed.arguments["path"] = "b"  # type: ignore[index]


def test_replay_pairs_tools_and_ignores_non_message_events():
    messages = events_to_messages(
        [
            {"kind": "user", "content": "run it"},
            {
                "kind": "assistant",
                "content": "",
                "tool_calls": [{"name": "read", "args": {"path": "a"}, "id": "c1"}],
            },
            {"kind": "usage", "input_tokens": 2},
            {"kind": "tool", "tool": "read", "result": "body"},
            {"kind": "assistant", "content": "done"},
        ]
    )
    assert [type(message) for message in messages] == [
        HumanMessage,
        AIMessage,
        ToolMessage,
        AIMessage,
    ]
    assert messages[2].tool_call_id == "c1"


def test_replay_uses_latest_compaction_and_valid_active_suffix():
    messages = events_to_messages(
        [
            {"kind": "user", "content": "discarded"},
            {"kind": "assistant", "content": "discarded answer"},
            {
                "kind": "compaction_llm",
                "source_event_seq": 1,
                "response": "old summary",
            },
            {"kind": "user", "content": "intermediate"},
            {"kind": "assistant", "content": "intermediate answer"},
            {
                "kind": "compaction_llm",
                "source_event_seq": 4,
                "response": "latest summary",
                "active_suffix": [],
            },
            {"kind": "user", "content": "continue"},
            {"kind": "assistant", "content": "continued answer"},
        ]
    )
    assert len(messages) == 3
    assert messages[0].content.startswith("<compacted-history>")
    assert "latest summary" in messages[0].content
    assert "old summary" not in messages[0].content
    assert messages[1].content == "continue"
    assert messages[2].content == "continued answer"


def test_replay_images_respects_vision_flag():
    event = [{"kind": "user", "content": "look", "images": ["data:image/png;base64,x"]}]
    assert isinstance(events_to_messages(event, vision=True)[0].content, list)
    assert events_to_messages(event, vision=False)[0].content == "look"


def test_restore_cost_skips_inherited_usage():
    tracker = SimpleNamespace(calls=[])
    tracker.restore = lambda *args: tracker.calls.append(args)
    restore_cost(
        [
            {"kind": "usage", "input_tokens": 3, "output_tokens": 2, "model": "m", "cost_usd": .1},
            {"kind": "usage", "input_tokens": 99, "inherited": True},
        ],
        tracker,
    )
    assert len(tracker.calls) == 1
    assert tracker.calls[0][0]["input_tokens"] == 3
    assert tracker.calls[0][2] == {"cost": .1}


def test_mentions_keep_source_order_and_expand_project_files(tmp_path):
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    store = PermissionStore(ness_dir=tmp_path / ".ness", project_root=tmp_path)
    source, paths = extract_mentions("Compare @a.txt with @missing.txt")
    expanded = expand_documents(source, store)
    assert paths == ("a.txt", "missing.txt")
    assert expanded.index("alpha") < expanded.index("does not exist") < expanded.index(source)


def test_plan_capture_saves_only_last_nonempty_plan(tmp_path):
    capture = PlanCapture(
        PlanStore(tmp_path / "plans"),
        thread_id=lambda: "thread",
        in_plan_mode=lambda: True,
    )
    capture.begin_turn()
    capture.on_plan_turn("draft")
    capture.on_plan_turn(" final plan ")
    path = capture.finish_turn()
    assert path is not None and path.read_text(encoding="utf-8") == "final plan\n"
    assert capture.finish_turn() is None
    assert final_plan_text(["", " a ", "b"]) == "b"


def test_plan_interrupt_is_saved_as_interrupted(tmp_path):
    capture = PlanCapture(
        PlanStore(tmp_path), thread_id=lambda: "t", in_plan_mode=lambda: True
    )
    capture.begin_turn()
    assert capture.on_interrupt("partial") == "partial"
    path = capture.finish_turn()
    assert path is not None and "[interrupted]" in path.read_text(encoding="utf-8")
