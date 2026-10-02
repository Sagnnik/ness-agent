from __future__ import annotations

from langchain_core.messages import HumanMessage, message_to_dict

from ness_agent.persistence import ThreadStore
from ness_cli.session.replay import events_to_messages


def _seed_summary_boundary(store: ThreadStore, thread_id: str) -> tuple[int, int]:
    store.append_event(thread_id, {"kind": "user", "content": "old task"})
    store.append_event(thread_id, {"kind": "assistant", "content": "old answer"})
    active_seq = store.append_event(
        thread_id, {"kind": "user", "content": "active task"}
    )
    store.append_compaction_checkpoint(
        thread_id,
        {
            "response": "old work summarized",
            "active_suffix": [
                message_to_dict(HumanMessage(content="active task"))
            ],
        },
        active_turn=True,
    )
    store.append_event(
        thread_id, {"kind": "assistant", "content": "active answer"}
    )
    next_seq = store.append_event(
        thread_id, {"kind": "user", "content": "next task"}
    )
    return active_seq, next_seq


def test_copy_thread_prefix_preserves_lineage_without_double_cost(tmp_path) -> None:
    store = ThreadStore(tmp_path / "threads", default_model="test-model")
    source = "session-source"
    first = store.append_event(source, {"kind": "user", "content": "first"})
    store.save_checkpoint(source, first, "git-a", "memory-a")
    store.append_event(source, {"kind": "assistant", "content": "answer"})
    store.append_event(
        source,
        {
            "kind": "usage",
            "model": "test-model",
            "input_tokens": 100,
            "output_tokens": 10,
            "cost_usd": 1.0,
        },
    )
    second = store.append_event(source, {"kind": "user", "content": "second"})
    store.save_checkpoint(source, second, "git-b", "memory-b")

    copied = store.copy_thread_prefix(source, "session-child", second)

    assert [event["kind"] for event in copied] == ["user", "assistant", "usage"]
    assert copied[-1]["inherited"] is True
    child = next(
        item for item in store.list_threads(10) if item["thread_id"] == "session-child"
    )
    assert child["fork_root_id"] == source
    assert child["fork_parent_id"] == source
    assert child["forked_from_seq"] == second
    assert child["turn_count"] == 1
    assert child["total_cost_usd"] == 0.0
    root = next(
        item for item in store.list_threads(10) if item["thread_id"] == source
    )
    assert root["fork_count"] == 1
    assert root["fork_index"] == 0
    assert child["fork_count"] == 1
    assert child["fork_index"] == 1


def test_fork_of_fork_keeps_root_lineage(tmp_path) -> None:
    store = ThreadStore(tmp_path / "threads")
    source = "session-root"
    boundary = store.append_event(source, {"kind": "user", "content": "root"})
    store.save_checkpoint(source, boundary, None, "")
    store.copy_thread_prefix(source, "session-child", boundary)
    child_boundary = store.append_event(
        "session-child", {"kind": "user", "content": "branch"}
    )
    store.save_checkpoint("session-child", child_boundary, None, "")

    store.copy_thread_prefix(
        "session-child",
        "session-grandchild",
        child_boundary,
    )

    rows = {row["thread_id"]: row for row in store.list_threads(10)}
    assert rows["session-grandchild"]["fork_root_id"] == source
    assert rows[source]["fork_count"] == 2
    assert rows[source]["fork_index"] == 0
    assert rows["session-child"]["fork_index"] == 1
    assert rows["session-grandchild"]["fork_index"] == 2


def test_fork_and_rollback_replay_on_both_sides_of_summary_checkpoint(tmp_path) -> None:
    store = ThreadStore(tmp_path / "summary-boundaries")
    active_seq, next_seq = _seed_summary_boundary(store, "session-source-summary")

    before = store.copy_thread_prefix(
        "session-source-summary", "session-fork-before-summary", active_seq
    )
    after = store.copy_thread_prefix(
        "session-source-summary", "session-fork-after-summary", next_seq
    )
    before_contents = [str(message.content) for message in events_to_messages(before)]
    after_contents = [str(message.content) for message in events_to_messages(after)]
    assert "old task" in before_contents
    assert not any("compacted-history" in content for content in before_contents)
    assert "active task" in after_contents
    assert "active answer" in after_contents
    assert any("compacted-history" in content for content in after_contents)

    rollback_active, _ = _seed_summary_boundary(store, "session-rollback-before")
    store.truncate_after("session-rollback-before", rollback_active)
    rolled_before = events_to_messages(
        store.load_thread_events("session-rollback-before")
    )
    assert [str(message.content) for message in rolled_before] == [
        "old task",
        "old answer",
    ]

    _, rollback_next = _seed_summary_boundary(store, "session-rollback-after")
    store.truncate_after("session-rollback-after", rollback_next)
    rolled_after = [
        str(message.content)
        for message in events_to_messages(
            store.load_thread_events("session-rollback-after")
        )
    ]
    assert "active task" in rolled_after
    assert "active answer" in rolled_after
    assert any("compacted-history" in content for content in rolled_after)
