from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("OPENAI_API_KEY", "test")

from langchain_core.messages import AIMessage

import ness_agent.tools.subagents as subagents
from ness_agent.tools.subagents import set_subagent_runtime, spawn_subagent, subagent_runs_active

from tests.sdk_fixtures import SessionContextTestMixin


class ConcurrentEchoModel:
    def __init__(self, text: str = "done", delay: float = 0):
        self.text = text
        self.delay = delay
        self.calls = 0
        self.in_flight = 0
        self.max_in_flight = 0
        self.bound_tools: list[list[str]] = []

    def bind_tools(self, tools):
        self.bound_tools.append([tool.name for tool in tools])
        return self

    async def ainvoke(self, _messages):
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            self.calls += 1
            return AIMessage(content=f"{self.text} {self.calls}")
        finally:
            self.in_flight -= 1


class SubagentToolTests(SessionContextTestMixin, unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.install_ctx(root, tools=["read", "grep", "glob", "write"])
        self.agents_dir = self.ctx.ness_dir / "agents"
        self.agents_dir.mkdir(parents=True, exist_ok=True)
        set_subagent_runtime(None)
        subagents._global_concurrency_semaphore = None

    def tearDown(self):
        set_subagent_runtime(None)
        subagents._global_concurrency_semaphore = None
        self.uninstall_ctx()
        self.temp.cleanup()

    def write_agent(self, name: str, tools: list[str], body: str = "Return concise findings.") -> None:
        tool_list = ", ".join(tools)
        (self.agents_dir / f"{name}.md").write_text(
            f"---\ntools: [{tool_list}]\n---\n{body}\n",
            encoding="utf-8",
        )

    async def test_spawn_subagent_runs_tasks_concurrently(self):
        self.write_agent("explore", ["read", "grep", "glob"])
        model = ConcurrentEchoModel(delay=0.15)
        set_subagent_runtime(model)

        result = await spawn_subagent.ainvoke(
            {
                "tasks": [
                    {"name": "explore", "prompt": "inspect auth", "label": "auth"},
                    {"name": "explore", "prompt": "inspect storage", "label": "storage"},
                ],
                "max_concurrency": 2,
                "timeout": 5,
            }
        )

        self.assertIn("status=ok", result)
        self.assertIn("tasks_total=2", result)
        self.assertIn("label=auth", result)
        self.assertIn("label=storage", result)
        self.assertIn("thread_id=subagent-explore-", result)
        self.assertEqual(model.calls, 2)
        self.assertEqual(model.max_in_flight, 2)
        self.assertEqual(subagent_runs_active(), 0)

    async def test_spawn_subagent_rejects_write_capable_batch_before_running(self):
        self.write_agent("explore", ["read"])
        self.write_agent("exec", ["read", "write"])
        model = ConcurrentEchoModel()
        set_subagent_runtime(model)

        result = await spawn_subagent.ainvoke(
            {
                "tasks": [
                    {"name": "explore", "prompt": "inspect first"},
                    {"name": "exec", "prompt": "change code"},
                ]
            }
        )

        self.assertIn("status=partial", result)
        self.assertIn("tasks_ok=1", result)
        self.assertIn("tasks_failed=1", result)
        self.assertIn("unsafe tools for subagent exec: write", result)
        self.assertEqual(model.calls, 1)

    async def test_spawn_subagent_rejects_unknown_tool_names(self):
        self.write_agent("stale", ["read_file", "bash"])
        model = ConcurrentEchoModel()
        set_subagent_runtime(model)

        result = await spawn_subagent.ainvoke(
            {"tasks": [{"name": "stale", "prompt": "inspect"}]}
        )

        self.assertIn("Error: subagent stale failed:", result)
        self.assertIn("unknown tools for subagent stale: bash", result)
        self.assertEqual(model.calls, 0)

    async def test_spawn_subagent_single_rejects_write_capable_agent(self):
        self.write_agent("exec", ["read", "write"])
        model = ConcurrentEchoModel()
        set_subagent_runtime(model)

        result = await spawn_subagent.ainvoke(
            {"tasks": [{"name": "exec", "prompt": "change code"}]}
        )

        self.assertIn("unsafe tools for subagent exec: write", result)
        self.assertEqual(model.calls, 0)

    async def test_spawn_subagent_single_succeeds_with_read_only_tools(self):
        self.write_agent("explore", ["read"])
        model = ConcurrentEchoModel(text="single ok")
        set_subagent_runtime(model)

        result = await spawn_subagent.ainvoke(
            {"tasks": [{"name": "explore", "prompt": "inspect"}]}
        )

        self.assertIn("single ok", result)
        self.assertEqual(model.calls, 1)

    async def test_spawn_subagent_timeout_is_reported_per_task(self):
        self.write_agent("slow", ["read"])
        model = ConcurrentEchoModel(delay=1.2)
        set_subagent_runtime(model)

        result = await spawn_subagent.ainvoke(
            {
                "tasks": [{"name": "slow", "prompt": "inspect slowly"}],
                "max_concurrency": 1,
                "timeout": 1,
            }
        )

        self.assertIn("Error: subagent slow timeout:", result)
        self.assertIn("timed out after 1s", result)

    async def test_spawn_subagent_requires_tasks(self):
        self.write_agent("explore", ["read"])
        model = ConcurrentEchoModel()
        set_subagent_runtime(model)

        with self.assertRaises(Exception):
            await spawn_subagent.ainvoke({})
        self.assertEqual(model.calls, 0)

    async def test_spawn_subagent_rejects_nested_subagent_tool(self):
        self.write_agent("nested", ["read", "spawn_subagent"])
        model = ConcurrentEchoModel()
        set_subagent_runtime(model)

        result = await spawn_subagent.ainvoke(
            {"tasks": [{"name": "nested", "prompt": "inspect"}]}
        )

        self.assertIn("unsafe tools for subagent nested: spawn_subagent", result)
        self.assertEqual(model.calls, 0)

    async def test_thread_id_is_consistent_between_graph_and_invoke_config(self):
        self.write_agent("explore", ["read"])
        model = ConcurrentEchoModel()
        set_subagent_runtime(model)
        seen: dict[str, str | None] = {}

        class FakeApp:
            async def ainvoke(self, state, config):
                seen["config_thread_id"] = config["configurable"]["thread_id"]
                seen["state_agent_mode"] = state["mode"]
                return {"messages": [AIMessage(content="thread ok")]}

        def fake_build_graph(cfg, thread_id, mode=None, **_kwargs):
            seen["thread_id"] = thread_id
            seen["mode"] = mode
            seen["tool_names"] = ",".join(tool.name for tool in cfg.tools)
            return FakeApp()

        with mock.patch("ness_agent.graph.builder.build_graph", side_effect=fake_build_graph):
            result = await spawn_subagent.ainvoke(
                {"tasks": [{"name": "explore", "prompt": "inspect"}]}
            )

        self.assertIn("thread ok", result)
        self.assertEqual(seen["thread_id"], seen["config_thread_id"])
        self.assertEqual(seen["mode"], "act")
        self.assertEqual(seen["state_agent_mode"], "act")
        self.assertEqual(seen["tool_names"], "read")

    async def test_spawn_subagent_rejects_unknown_agent_before_running(self):
        self.write_agent("explore", ["read"])
        model = ConcurrentEchoModel(text="partial ok")
        set_subagent_runtime(model)

        result = await spawn_subagent.ainvoke(
            {
                "tasks": [
                    {"name": "explore", "prompt": "inspect first"},
                    {"name": "missing", "prompt": "inspect second"},
                ]
            }
        )

        self.assertIn("status=error", result)
        self.assertIn("unknown agent 'missing'", result)
        self.assertIn("available agents: explore", result)
        self.assertEqual(model.calls, 0)

    async def test_spawn_subagent_rejects_unknown_agent_in_single_mode(self):
        model = ConcurrentEchoModel()
        set_subagent_runtime(model)

        result = await spawn_subagent.ainvoke(
            {"tasks": [{"name": "missing", "prompt": "inspect"}]}
        )

        self.assertIn("unknown agent 'missing'", result)
        self.assertEqual(model.calls, 0)

    async def test_global_concurrency_cap_limits_across_concurrent_calls(self):
        subagents._global_concurrency_semaphore = None

        self.write_agent("explore", ["read"])
        model = ConcurrentEchoModel(delay=0.2)
        set_subagent_runtime(model)

        cap = subagents.MAX_CONCURRENCY

        async def fire_batch():
            return await spawn_subagent.ainvoke(
                {
                    "tasks": [
                        {"name": "explore", "prompt": f"task {n}", "label": f"l{n}"}
                        for n in range(cap)
                    ],
                    "max_concurrency": cap,
                    "timeout": 10,
                }
            )

        await asyncio.gather(fire_batch(), fire_batch())

        self.assertEqual(model.calls, cap * 2)
        self.assertLessEqual(model.max_in_flight, cap)
        self.assertEqual(subagent_runs_active(), 0)


if __name__ == "__main__":
    unittest.main()
