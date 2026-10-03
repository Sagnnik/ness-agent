import asyncio
import shlex
import sys
import time
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from ness_agent import NessAgent, AgentSpec, PromptLayersConfig
from ness_agent.options import NessAgentOptions
from ness_agent.tools.shell_processes import _process_group_alive


class ShellModel:
    def __init__(self, command):
        self.command = command
        self.calls = 0

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "shell",
                        "args": {"command": self.command, "timeout": 900},
                        "id": "shell-call",
                    }
                ],
            )
        return AIMessage(content="done")


def make_agent(tmp_path):
    command = shlex.join(
        [
            sys.executable,
            "-c",
            "import os,pathlib,time; print('before stop',flush=True); "
            "pathlib.Path('ready.pid').write_text(str(os.getpid())); time.sleep(30)",
        ]
    )
    return NessAgent.from_spec(
        AgentSpec(
            model=ShellModel(command),
            tools=["shell"],
            prompt=PromptLayersConfig(l0="test"),
            options=NessAgentOptions(
                project_root=tmp_path,
                ness_dir=tmp_path / ".ness",
                enable_approval=False,
                auto_save_threads=False,
            ),
        )
    )


async def wait_until(predicate):
    end = time.monotonic() + 5
    while not predicate():
        if time.monotonic() > end:
            pytest.fail("Shell did not reach expected state")
        await asyncio.sleep(0.02)


@pytest.mark.parametrize("cancellation", ["cooperative", "hard", "close"])
def test_session_cancellation_stops_foreground_shell(tmp_path, cancellation):
    async def run():
        agent = make_agent(tmp_path)
        session = agent.session(thread_id="cancel")
        task = asyncio.create_task(session.run("run command"))
        try:
            await wait_until((tmp_path / "ready.pid").exists)
            manager = session._shell_process_manager
            [job] = list(manager._jobs.values())
            if cancellation == "cooperative":
                session.cancel()
            elif cancellation == "hard":
                task.cancel()
            else:
                await session.close()
            if cancellation == "hard":
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(task, 5)
            else:
                result = await asyncio.wait_for(task, 5)
                assert not any(event.kind == "error" for event in result.events)
            await wait_until(lambda: not _process_group_alive(job["pgid"]))
            assert "before stop" in Path(job["log_path"]).read_text()
            if cancellation != "close":
                # Cancellation cannot leak into the next turn's shell context.
                session.config.model.command = "printf next-turn"
                session.config.model.calls = 0
                result = await session.run("continue")
                assert not any(event.kind == "error" for event in result.events)
                assert any(
                    e.kind == "tool_end"
                    and "status=ok" in e.data["content"]
                    and "next-turn" in e.data["content"]
                    for e in result.events
                )
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            await session.close()

    asyncio.run(run())


def test_session_deadline_reaches_foreground_tool(tmp_path):
    async def run():
        session = make_agent(tmp_path).session(
            thread_id="deadline", deadline=time.monotonic() + 0.3
        )
        try:
            result = await session.run("run command")
            tool_results = [
                e.data["content"] for e in result.events if e.kind == "tool_end"
            ]
            assert any(
                "status=timeout" in content and "timeout_reason=deadline" in content
                for content in tool_results
            )
            assert not any(event.kind == "error" for event in result.events)
        finally:
            await session.close()

    asyncio.run(run())


def test_closing_stream_stops_foreground_shell(tmp_path):
    async def run():
        session = make_agent(tmp_path).session(thread_id="closed-stream")
        stream = session.stream("run command")
        try:
            async for event in stream:
                if event.kind == "tool_start":
                    break
            await wait_until((tmp_path / "ready.pid").exists)
            [job] = list(session._shell_process_manager._jobs.values())
            await stream.aclose()
            await wait_until(lambda: not _process_group_alive(job["pgid"]))
        finally:
            await stream.aclose()
            await session.close()

    asyncio.run(run())


@pytest.mark.parametrize("deadline", [True, "later", float("nan"), float("inf")])
def test_invalid_deadline_rejected(tmp_path, deadline):
    with pytest.raises(ValueError, match="deadline"):
        make_agent(tmp_path).session(thread_id="invalid", deadline=deadline)
