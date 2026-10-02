from __future__ import annotations

import tempfile
import threading
import unittest
import sqlite3
from pathlib import Path

from ness_agent.persistence import ThreadStore
from ness_cli.session.replay import events_to_messages
from langchain_core.messages import HumanMessage, message_to_dict


class SessionStorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.ness_dir = Path(self._tmpdir.name)
        self.store = ThreadStore(threads_dir=self.ness_dir / "threads", auto_save=True)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_append_and_load_roundtrip(self) -> None:
        self.store.append_event("session-abc", {"kind": "user", "content": "hello"})
        self.store.append_event(
            "session-abc",
            {
                "kind": "assistant",
                "content": "",
                "tool_calls": [{"name": "read", "args": {"path": "a"}, "id": "call-1"}],
            },
        )
        self.store.append_event(
            "session-abc",
            {
                "kind": "tool",
                "tool": "read",
                "args": {"path": "a"},
                "result": "file contents",
                "call_id": "call-1",
                "duration_ms": 1,
                "exit": "ok",
            },
        )

        events = self.store.load_thread_events("session-abc")
        self.assertEqual(len(events), 3)
        self.assertEqual(events[0]["kind"], "user")
        self.assertIn("t", events[0])
        self.assertEqual(events[1]["tool_calls"][0]["id"], "call-1")
        self.assertEqual(events[2]["result"], "file contents")

    def test_load_thread_events_since_filters_by_seq(self) -> None:
        self.store.append_event("session-since", {"kind": "user", "content": "before"})
        start = self.store.append_event(
            "session-since",
            {"kind": "goal", "phase": "start", "goal": "do the thing"},
        )
        self.store.append_event("session-since", {"kind": "user", "content": "do the thing"})
        self.store.append_event(
            "session-since",
            {"kind": "assistant", "content": "done", "tool_calls": []},
        )

        sliced = self.store.load_thread_events_since("session-since", start)
        self.assertEqual(len(sliced), 3)
        self.assertEqual(sliced[0]["kind"], "goal")
        self.assertEqual(sliced[0]["seq"], start)
        self.assertEqual(sliced[1]["content"], "do the thing")
        self.assertNotIn("before", [e.get("content") for e in sliced])
        # Full load still omits seq so existing callers stay unchanged.
        full = self.store.load_thread_events("session-since")
        self.assertEqual(len(full), 4)
        self.assertNotIn("seq", full[0])

    def test_usage_updates_thread_aggregates(self) -> None:
        self.store.append_event("session-cost", {"kind": "user", "content": "hi"})
        self.store.append_event(
            "session-cost",
            {
                "kind": "usage",
                "model": "gpt-4o-mini",
                "input_tokens": 100,
                "cached_input_tokens": 20,
                "output_tokens": 10,
                "cost_usd": 0.01,
            },
        )

        rows = self.store.list_threads(5)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["turn_count"], 1)
        self.assertEqual(rows[0]["input_tokens"], 100)
        self.assertEqual(rows[0]["cached_input_tokens"], 20)
        self.assertEqual(rows[0]["output_tokens"], 10)
        self.assertAlmostEqual(rows[0]["total_cost_usd"], 0.01)

    def test_list_threads_filters_to_session_prefix(self) -> None:
        self.store.append_event("session-visible", {"kind": "user", "content": "a"})
        self.store.append_event("subagent-explore-deadbeef", {"kind": "user", "content": "b"})
        self.store.append_event("todo-graph-abc", {"kind": "user", "content": "c"})

        rows = self.store.list_threads(10)
        self.assertEqual([row["thread_id"] for row in rows], ["session-visible"])
        self.assertIsNone(self._thread_row("subagent-explore-deadbeef"))
        self.assertEqual(self.store.load_thread_events("subagent-explore-deadbeef"), [])

    def test_archive_thread_idempotent(self) -> None:
        self.store.append_event("session-archive", {"kind": "user", "content": "archive me please"})
        first = self.store.archive_thread("session-archive")
        second = self.store.archive_thread("session-archive")
        self.assertIn("Archived thread", first)
        self.assertIn("already archived", second)

    def test_set_thread_name_before_first_turn_and_preserve_on_archive(self) -> None:
        self.assertTrue(self.store.set_thread_name("session-named", "  Release   prep  "))

        row = self.store.list_threads(5)[0]
        self.assertEqual(row["name"], "Release prep")
        self.assertEqual(row["summary"], "")
        self.assertTrue(self.store.thread_exists("session-named"))

        self.store.append_event(
            "session-named", {"kind": "user", "content": "original first message"}
        )
        self.store.archive_thread("session-named")
        row = self.store.list_threads(5)[0]
        self.assertEqual(row["name"], "Release prep")
        self.assertEqual(row["summary"], "original first message")

    def test_named_empty_thread_can_be_archived(self) -> None:
        self.store.set_thread_name("session-empty", "Empty but named")
        result = self.store.archive_thread("session-empty")

        self.assertIn("Archived thread", result)
        row = self.store.list_threads(5)[0]
        self.assertEqual(row["name"], "Empty but named")
        self.assertIn("archived_at", row)

    def test_set_thread_name_validates_length_and_empty_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            self.store.set_thread_name("session-empty-name", "  \n  ")
        with self.assertRaisesRegex(ValueError, "80 characters"):
            self.store.set_thread_name("session-long-name", "x" * 81)

    def test_set_thread_name_is_disabled_with_autosave(self) -> None:
        store = ThreadStore(
            threads_dir=self.ness_dir / "threads-name-off", auto_save=False
        )
        self.assertFalse(store.set_thread_name("session-off", "Not persisted"))
        self.assertFalse(store.threads_db.exists())

    def test_old_thread_schema_fails_without_migrating(self) -> None:
        threads_dir = self.ness_dir / "threads-old"
        threads_dir.mkdir(parents=True)
        db = threads_dir / "threads.db"
        with sqlite3.connect(db) as conn:
            conn.execute("CREATE TABLE threads (thread_id TEXT PRIMARY KEY)")
            conn.commit()

        with self.assertRaisesRegex(RuntimeError, "Incompatible thread database schema"):
            ThreadStore(threads_dir=threads_dir)

        with sqlite3.connect(db) as conn:
            columns = [row[1] for row in conn.execute("PRAGMA table_info(threads)")]
        self.assertEqual(columns, ["thread_id"])

    def test_subagent_registration(self) -> None:
        self.store.append_event("session-parent", {"kind": "user", "content": "run subagents"})
        self.store.register_subagent(
            "session-parent",
            "subagent-explore-111",
            agent_name="explore",
            label="scan repo",
        )
        self.store.complete_subagent(
            "subagent-explore-111",
            status="ok",
            output="found main.py",
            duration_ms=42,
        )

        self.assertIsNotNone(self._thread_row("session-parent"))
        self.assertIsNone(self._thread_row("subagent-explore-111"))
        rows = self.store.list_subagents("session-parent")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["agent_name"], "explore")
        self.assertEqual(rows[0]["status"], "ok")
        self.assertEqual(rows[0]["output"], "found main.py")

    def test_subagent_usage_rolls_up_to_parent(self) -> None:
        self.store.append_event("session-parent-cost", {"kind": "user", "content": "run subagents"})
        self.store.register_subagent(
            "session-parent-cost",
            "subagent-explore-cost",
            agent_name="explore",
            label="scan",
        )
        self.store.append_event(
            "subagent-explore-cost",
            {
                "kind": "usage",
                "model": "gpt-4o-mini",
                "input_tokens": 50,
                "cached_input_tokens": 10,
                "output_tokens": 5,
                "cost_usd": 0.005,
            },
        )

        self.assertEqual(self.store.load_thread_events("subagent-explore-cost"), [])
        rows = self.store.list_threads(5)
        parent = next(row for row in rows if row["thread_id"] == "session-parent-cost")
        self.assertEqual(parent["input_tokens"], 50)
        self.assertEqual(parent["cached_input_tokens"], 10)
        self.assertEqual(parent["output_tokens"], 5)
        self.assertAlmostEqual(parent["total_cost_usd"], 0.005)

    def _thread_row(self, thread_id: str) -> dict | None:
        with sqlite3.connect(self.store.threads_db) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT thread_id FROM threads WHERE thread_id = ?",
                (thread_id,),
            ).fetchone()
        return dict(row) if row else None

    def test_concurrent_append_same_thread(self) -> None:
        errors: list[Exception] = []

        def worker(index: int) -> None:
            try:
                self.store.append_event(
                    "session-concurrent",
                    {"kind": "usage", "input_tokens": 1, "output_tokens": 1, "cost_usd": 0.0},
                )
            except Exception as exc:
                errors.append(exc)

        self.store.append_event("session-concurrent", {"kind": "user", "content": "start"})
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(errors, [])
        events = self.store.load_thread_events("session-concurrent")
        self.assertEqual(len(events), 9)

    def test_autosave_disabled_skips_writes(self) -> None:
        store = ThreadStore(threads_dir=self.ness_dir / "threads-off", auto_save=False)
        store.append_event("session-off", {"kind": "user", "content": "nope"})
        self.assertFalse(store.threads_db.exists())

    def test_compaction_checkpoint_records_active_turn_boundary(self) -> None:
        self.store.append_event("session-compact", {"kind": "user", "content": "old"})
        self.store.append_event("session-compact", {"kind": "assistant", "content": "done"})
        active_seq = self.store.append_event(
            "session-compact", {"kind": "user", "content": "active"}
        )
        self.store.append_compaction_checkpoint(
            "session-compact",
            {
                "response": "summary",
                "before_tokens": 100,
                "after_tokens": 10,
                "active_suffix": [message_to_dict(HumanMessage(content="active"))],
            },
            active_turn=True,
        )
        event = self.store.load_thread_events("session-compact")[-1]
        self.assertEqual(event["active_user_seq"], active_seq)
        self.assertEqual(event["source_event_seq"], active_seq)

    def test_compaction_checkpoint_without_sdk_user_event_retains_active_suffix(self) -> None:
        self.store.append_event(
            "session-sdk-compact", {"kind": "assistant", "content": "old answer"}
        )
        self.store.append_compaction_checkpoint(
            "session-sdk-compact",
            {
                "response": "completed work summarized",
                "active_suffix": [
                    message_to_dict(HumanMessage(content="active SDK request"))
                ],
            },
            active_turn=True,
        )
        events = self.store.load_thread_events("session-sdk-compact")
        checkpoint = events[-1]
        self.assertIsNone(checkpoint["active_user_seq"])
        self.assertEqual(checkpoint["source_event_seq"], 0)

        messages = events_to_messages(events)
        contents = [str(message.content) for message in messages]
        self.assertTrue(contents[0].startswith("<compacted-history>"))
        self.assertIn("active SDK request", contents)


class ResumeReplayTests(unittest.TestCase):
    def test_answered_image_event_remains_structured_until_compaction(self) -> None:
        messages = events_to_messages([
            {
                "kind": "user",
                "content": "inspect this",
                "images": ["data:image/png;base64,abc"],
            },
            {"kind": "assistant", "content": "done"},
        ], vision=True)
        self.assertIsInstance(messages[0].content, list)
        self.assertEqual(messages[0].content[1]["type"], "image_url")

    def test_events_to_messages_replays_tool_chain(self) -> None:
        events = [
            {"kind": "user", "content": "read file"},
            {
                "kind": "assistant",
                "content": "",
                "tool_calls": [
                    {"name": "read", "args": {"path": "a"}, "id": "call-1", "type": "tool_call"}
                ],
            },
            {
                "kind": "tool",
                "tool": "read",
                "args": {"path": "a"},
                "result": "contents",
                "call_id": "call-1",
            },
            {"kind": "assistant", "content": "done"},
        ]
        messages = events_to_messages(events)
        self.assertEqual(len(messages), 4)
        self.assertEqual(messages[1].tool_calls[0]["id"], "call-1")
        self.assertEqual(messages[2].tool_call_id, "call-1")
        self.assertEqual(messages[2].content, "contents")

    def test_spawn_subagent_result_enrichment(self) -> None:
        short = "status=ok"
        subagents = [
            {
                "subagent_thread_id": "subagent-explore-111",
                "agent_name": "explore",
                "label": "",
                "status": "ok",
                "duration_ms": 10,
                "output": "detailed findings from subagent run",
            }
        ]
        messages = events_to_messages(
            [
                {
                    "kind": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "name": "spawn_subagent",
                            "args": {},
                            "id": "call-1",
                            "type": "tool_call",
                        }
                    ],
                },
                {
                    "kind": "tool",
                    "tool": "spawn_subagent",
                    "result": short,
                    "call_id": "call-1",
                },
            ],
            subagents=subagents,
        )
        enriched = str(messages[-1].content)
        self.assertIn("detailed findings", enriched)
        self.assertGreater(len(enriched), len(short))


if __name__ == "__main__":
    unittest.main()
