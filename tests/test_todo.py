from __future__ import annotations

import os
import unittest
import uuid

os.environ.setdefault("OPENAI_API_KEY", "test")

from ness_agent.context.coding_overlay import CodingOverlay
from ness_agent.context.overlay import OverlayContext
from ness_agent.tools.todo import get_thread_todos, render_todos, set_current_thread, set_thread_todos, todo


class TodoToolTests(unittest.TestCase):
    def setUp(self):
        self.thread_id = f"todo-{uuid.uuid4().hex}"
        set_current_thread(self.thread_id)
        set_thread_todos(self.thread_id, [])

    def test_replace_keeps_existing_contract(self):
        result = todo.invoke(
            {
                "todos": [
                    {"content": "Plan change", "status": "completed"},
                    {"id": "custom", "content": "Run tests", "status": "pending"},
                ]
            }
        )

        self.assertEqual(result, "Updated 2 todos")
        self.assertEqual(
            get_thread_todos(self.thread_id),
            [
                {"id": "1", "content": "Plan change", "status": "completed"},
                {"id": "custom", "content": "Run tests", "status": "pending"},
            ],
        )

    def test_replace_updates_status_in_full_list(self):
        todo.invoke(
            {
                "todos": [
                    {"id": "1", "content": "First", "status": "pending"},
                    {"id": "2", "content": "Second", "status": "pending"},
                ]
            }
        )
        result = todo.invoke(
            {
                "todos": [
                    {"id": "2", "content": "Second", "status": "completed"},
                    {"id": "1", "content": "First", "status": "in_progress"},
                ]
            }
        )
        self.assertEqual(result, "Updated 2 todos")
        self.assertEqual(
            [(t["id"], t["status"]) for t in get_thread_todos(self.thread_id)],
            [("2", "completed"), ("1", "in_progress")],
        )

    def test_invalid_status_rejected_by_schema(self):
        with self.assertRaises(Exception):
            todo.invoke({"todos": [{"content": "x", "status": "blocked"}]})

    def test_get_thread_todos_returns_copies(self):
        set_thread_todos(self.thread_id, [{"id": "1", "content": "Stable", "status": "pending"}])

        todos = get_thread_todos(self.thread_id)
        todos[0]["status"] = "completed"

        self.assertEqual(get_thread_todos(self.thread_id)[0]["status"], "pending")

    def test_schema_requires_todos(self):
        schema = todo.args_schema.model_json_schema()
        self.assertIn("todos", schema.get("required", []))
        self.assertNotIn("action", schema.get("properties", {}))

class RenderTodosTests(unittest.TestCase):
    def test_render_todos_omits_completed(self):
        todos = [
            {"id": "1", "content": "Done", "status": "completed"},
            {"id": "2", "content": "Active", "status": "in_progress"},
        ]
        rendered = render_todos(todos)
        self.assertIn("- [in_progress] 2: Active", rendered)
        self.assertNotIn("Done", rendered)

    def test_render_todos_empty_when_all_completed(self):
        todos = [{"id": "1", "content": "Done", "status": "completed"}]
        self.assertEqual(render_todos(todos), "")
        self.assertEqual(render_todos([]), "")
        self.assertEqual(render_todos(None), "")

    def test_working_state_overlay_omits_todos_section_when_empty(self):
        overlay = CodingOverlay()
        ctx = OverlayContext(
            thread_id="t",
            mode="act",
            messages=[],
            todos=[],
            session_memory="",
            compaction_note="",
            mode_switch="",
        )
        sections = overlay.sections({}, ctx)
        self.assertNotIn("todos", sections)

    def test_working_state_overlay_includes_active_todos(self):
        overlay = CodingOverlay()
        ctx = OverlayContext(
            thread_id="t",
            mode="act",
            messages=[],
            todos=[{"id": "1", "content": "Ship it", "status": "pending"}],
            session_memory="",
            compaction_note="",
            mode_switch="",
        )
        sections = overlay.sections({}, ctx)
        self.assertIn("todos", sections)
        self.assertIn("TODOS\n- [pending] 1: Ship it", sections["todos"])

if __name__ == "__main__":
    unittest.main()
