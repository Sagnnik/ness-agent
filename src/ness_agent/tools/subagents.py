from __future__ import annotations

import asyncio
import re
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Annotated, Any

import yaml
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ness_agent.context.layers import PromptLayers
from ness_agent.session_context import get_session_context

DEFAULT_AGENT_TOOLS = ("read", "grep", "glob", "web_search", "fetch_url", "skill_view")
MAX_BATCH_TASKS = 8
DEFAULT_MAX_CONCURRENCY = 3 # max concurrent subagents to run
MAX_CONCURRENCY = 8
DEFAULT_TIMEOUT_SECONDS = 300 # timeout for a single subagent
MAX_TIMEOUT_SECONDS = 1_800

# Syntactic guard before names are used in filesystem paths (blocks traversal chars).
AGENT_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")

DEFAULT_SUBAGENT_TEMPLATE = """You are the {agent_name} Ness Agent subagent.

You were spawned by a parent agent to handle one scoped assignment. You run in an isolated session with no access to the parent conversation. Your final assistant message is returned directly to the parent—there is no follow-up turn with you.

Parent context:
{parent_context}

Role-specific instructions:
{agent_body}

Operating constraints:
- Read-only only. Use only the tools bound to this run.
- Stay within the parent request above.
- Prefer targeted investigation.
- Return one concise final message for the parent agent.
"""

# ContextVars for subagent specific memory pockets to avoid collisions between parallel subagent runs
_subagent_model: ContextVar[Any | None] = ContextVar("subagent_model", default=None)
_parent_thread_id: ContextVar[str | None] = ContextVar("parent_thread_id", default=None)
_active_subagent_runs_by_parent: dict[str, int] = {} # track subagents run by paren't thread id

# Cross-call concurrency cap shared by every spawn_subagent invocation in this
# process. A single spawn_subagent call already bounds in-call concurrency via an
# asyncio.Semaphore, but without this global gate the main agent could fire
# multiple spawn_subagent calls in one turn and exceed MAX_CONCURRENCY. Lazily
# created on first use so we bind to the running event loop.
_global_concurrency_semaphore: asyncio.Semaphore | None = None


def _agents_dir() -> Path:
    from ness_agent.session_context import try_get_session_context

    rt = try_get_session_context()
    if rt is not None:
        return rt.ness_dir / "agents"
    return Path(".ness") / "agents"


def _build_subagent_prompt(agent_name: str, agent_body: str, parent_context: str = "") -> str:
    rt = get_session_context()
    template = DEFAULT_SUBAGENT_TEMPLATE
    cfg = rt.agent_config
    if cfg is not None and cfg.aux_prompts.subagent:
        src = cfg.aux_prompts.subagent
        if callable(src):
            template = src()
        else:
            text = str(src)
            if "\n" not in text and len(text) < 512:
                p = Path(text)
                try:
                    if p.exists() and p.is_file():
                        template = p.read_text(encoding="utf-8").strip()
                    else:
                        template = text
                except OSError:
                    template = text
            else:
                template = text
    elif cfg is not None and cfg.subagents and cfg.subagents.prompt_template:
        template = cfg.subagents.prompt_template
    return template.format(
        agent_name=agent_name,
        agent_body=agent_body.strip(),
        parent_context=parent_context.strip(),
    )


def _global_semaphore() -> asyncio.Semaphore:
    global _global_concurrency_semaphore
    if _global_concurrency_semaphore is None:
        _global_concurrency_semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    return _global_concurrency_semaphore


@dataclass(frozen=True)
class SubagentTask:
    name: str
    prompt: str
    label: str = ""


class SubagentTaskItem(BaseModel):
    """One batch task for ``spawn_subagent(tasks=[...])``."""

    name: str = Field(description="Agent name matching .ness/agents/<name>.md")
    prompt: str = Field(description="Assignment for this subagent.")
    label: str | None = Field(default=None, description="Optional short label for the result.")


@dataclass(frozen=True)
class PreparedTask:
    task: SubagentTask
    agent_prompt: str
    tools: list[Any]


@dataclass(frozen=True)
class SubagentRunResult:
    index: int
    name: str
    label: str
    status: str
    duration_ms: int
    thread_id: str
    output: str

# sets the subagent runtime context before main agent call; setter doesn't need to be async
def set_subagent_runtime(model: Any | None, parent_thread_id: str | None = None) -> None:
    _subagent_model.set(model)
    if parent_thread_id is not None:
        _parent_thread_id.set(parent_thread_id)


def subagent_runs_active(parent_thread_id: str | None = None) -> int:
    """Number of executing child graphs, optionally scoped to one parent."""
    if parent_thread_id is not None:
        return _active_subagent_runs_by_parent.get(parent_thread_id, 0)
    return sum(_active_subagent_runs_by_parent.values())

def _available_agent_names() -> frozenset[str]:
    agents_dir = _agents_dir()
    if not agents_dir.is_dir():
        return frozenset()
    return frozenset(
        path.stem for path in agents_dir.glob("*.md") if path.is_file() and path.stem
    )


def _spawn_subagent_description() -> str:
    agents = ", ".join(sorted(_available_agent_names())) or "none"
    return f"""Run one or more read-only Ness Agent subagents and wait for all results.

REQUIRED shape: always pass ``tasks`` as a non-empty list. Never use top-level
name/prompt — those fields are invalid. A single investigation is still a
one-item list:
  spawn_subagent(tasks=[{{"name": "explore", "prompt": "Find route handlers"}}])
Parallel exploration:
  spawn_subagent(tasks=[
    {{"name": "explore", "prompt": "Find routes", "label": "routes"}},
    {{"name": "explore", "prompt": "Find tests", "label": "tests"}},
  ])

Each task object requires name and prompt; label is optional.
max_concurrency defaults to {DEFAULT_MAX_CONCURRENCY}; timeout defaults to {DEFAULT_TIMEOUT_SECONDS}s.

Agent names must match a file in .ness/agents/<name>.md.
Available agents: {agents}."""


@tool(description=_spawn_subagent_description())
async def spawn_subagent(
    tasks: Annotated[list[SubagentTaskItem], Field(min_length=1)],
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> str:
    """Run one or more isolated sub-agents for parallel investigation and wait for results."""
    started = time.time()
    runtime = _require_subagent_model()
    if isinstance(runtime, str):
        return _call_error(runtime, started, 0)

    model = runtime

    validated = _normalize_tasks(tasks)
    if isinstance(validated, str):
        return _call_error(validated, started, len(tasks))

    concurrency = _bounded_int(max_concurrency, "max_concurrency", 1, MAX_CONCURRENCY)
    if isinstance(concurrency, str):
        return _call_error(concurrency, started, len(validated))

    timeout_seconds = _bounded_int(timeout, "timeout", 1, MAX_TIMEOUT_SECONDS)
    if isinstance(timeout_seconds, str):
        return _call_error(timeout_seconds, started, len(validated))

    in_call_semaphore = asyncio.Semaphore(min(concurrency, len(validated)))

    async def run_one(index: int, task: SubagentTask) -> SubagentRunResult:
        async with in_call_semaphore:
            prepared = _prepare_task(task)
            if isinstance(prepared, str):
                return SubagentRunResult(
                    index=index,
                    name=task.name,
                    label=task.label,
                    status="failed",
                    duration_ms=0,
                    thread_id=f"subagent-{task.name}-prep-failed",
                    output=prepared,
                )
            async with _global_semaphore():
                return await _run_prepared_with_timeout(
                    index=index,
                    prepared=prepared,
                    model=model,
                    timeout_seconds=timeout_seconds,
                )

    results = await asyncio.gather(
        *(run_one(index, task) for index, task in enumerate(validated, start=1))
    )
    if len(results) == 1:
        return _format_single_result(results[0])
    return _format_batch_result(results, int((time.time() - started) * 1000))


def _tool_error(message: str) -> str:
    return f"Error: {message}"


def _call_error(message: str, started: float, task_count: int) -> str:
    duration_ms = int((time.time() - started) * 1000)
    if task_count <= 1:
        return f"{_tool_error(message)} (duration_ms={duration_ms})"
    return _format_batch_error(message, duration_ms, task_count)


def _require_subagent_model() -> Any | str:
    model = _subagent_model.get()
    if model is None:
        return "no model available for subagent"
    return model


def _normalize_tasks(raw_tasks: list[SubagentTaskItem]) -> list[SubagentTask] | str:
    if len(raw_tasks) > MAX_BATCH_TASKS:
        return f"spawn_subagent supports at most {MAX_BATCH_TASKS} tasks"

    normalized: list[SubagentTask] = []
    for index, task in enumerate(raw_tasks, start=1):
        item = _normalize_task(task, index)
        if isinstance(item, str):
            return item
        normalized.append(item)
    return normalized


def _normalize_task(raw_task: SubagentTaskItem, index: int) -> SubagentTask | str:
    name = str(raw_task.name or "").strip()
    prompt = str(raw_task.prompt or "").strip()
    label = str(raw_task.label or "").strip()
    if error := _validate_agent_name(name):
        return error
    if not prompt:
        return f"task {index} prompt cannot be empty"
    return SubagentTask(name=name, prompt=prompt, label=label)


def _bounded_int(value: int, name: str, minimum: int, maximum: int) -> int | str:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return f"{name} must be an integer"
    if parsed < minimum:
        return f"{name} must be at least {minimum}"
    return min(parsed, maximum)


def _prepare_task(task: SubagentTask) -> PreparedTask | str:
    meta = _load_agent(task.name)
    if isinstance(meta, str):
        return meta

    tool_names = _agent_tool_names(meta, task.name)
    if isinstance(tool_names, str):
        return tool_names

    tools = _resolve_tools(task.name, tool_names)
    if isinstance(tools, str):
        return tools

    parent_context = _parent_context(task)
    agent_prompt = _build_subagent_prompt(task.name, meta.get("prompt", ""), parent_context)
    return PreparedTask(task=task, agent_prompt=agent_prompt, tools=tools)


def _parent_context(task: SubagentTask) -> str:
    lines = []
    if task.label:
        lines.append(f"Task label: {task.label}")
    lines.append(f"Parent request: {task.prompt}")
    return "\n".join(lines)


async def _run_prepared_with_timeout(
    index: int,
    prepared: PreparedTask,
    model: Any,
    timeout_seconds: int,
) -> SubagentRunResult:
    task = prepared.task
    started = time.time()
    thread_id = f"subagent-{task.name}-{uuid.uuid4().hex[:8]}"
    parent_thread_id = _parent_thread_id.get()

    # register the subagent with the parent thread in threads db
    if parent_thread_id:
        get_session_context().thread_store.register_subagent(
            parent_thread_id,
            thread_id,
            agent_name=task.name,
            label=task.label,
        )

    try:
        # .wait_for(): wrap a single coroutine with timeout and handle exceptions
        output = await asyncio.wait_for(
            _invoke_subagent(
                prepared=prepared,
                model=model,
                thread_id=thread_id,
            ),
            timeout=timeout_seconds,
        )
        result = SubagentRunResult(
            index=index,
            name=task.name,
            label=task.label,
            status="ok",
            duration_ms=int((time.time() - started) * 1000),
            thread_id=thread_id,
            output=output,
        )
    # handle exceptions
    except TimeoutError:
        result = SubagentRunResult(
            index=index,
            name=task.name,
            label=task.label,
            status="timeout",
            duration_ms=int((time.time() - started) * 1000),
            thread_id=thread_id,
            output=f"timed out after {timeout_seconds}s",
        )
    except Exception as exc:
        message = str(exc)
        if message.startswith("Error: "):
            message = message[7:]
        result = SubagentRunResult(
            index=index,
            name=task.name,
            label=task.label,
            status="failed",
            duration_ms=int((time.time() - started) * 1000),
            thread_id=thread_id,
            output=message,
        )

    # complete the subagent run with the result stored in the threads db
    if parent_thread_id:
        get_session_context().thread_store.complete_subagent(
            thread_id,
            status=result.status,
            output=result.output,
            duration_ms=result.duration_ms,
        )
    return result


async def _invoke_subagent(
    prepared: PreparedTask,
    model: Any,
    thread_id: str,
) -> str:
    """Main subagent invocation using the SDK graph builder."""
    parent_thread_id = _parent_thread_id.get() or ""
    _active_subagent_runs_by_parent[parent_thread_id] = (
        _active_subagent_runs_by_parent.get(parent_thread_id, 0) + 1
    )
    try:
        from ness_agent.graph.builder import build_graph
        from ness_agent.tools import ToolRegistry

        rt = get_session_context()
        parent = rt.agent_config
        if parent is None:
            raise RuntimeError("Subagent requires agent_config on SessionContext")

        child_options = replace(
            parent.options,
            enable_approval=False,
            session_end_reflection=False,
            reflection_token_ratio=0.0,
        )
        child_cfg = replace(
            parent,
            model=model or parent.model,
            # Child graphs have no overlay, so their skill catalog belongs in L1.
            prompts=PromptLayers(replace(parent.prompts.config, include_skill_catalog=True)),
            tools=list(prepared.tools),
            tool_registry=ToolRegistry(prepared.tools),
            overlay=None,
            modes=None,
            approval_handler=None,
            question_handler=None,
            options=child_options,
            cost_tracker=parent.cost_tracker,
            tracer=parent.tracer,
            memory_store=parent.memory_store,
            thread_store=parent.thread_store,
            permission_store=parent.permission_store,
            hook_runner=parent.hook_runner,
            skill_loader=parent.skill_loader,
        )
        app = build_graph(
            child_cfg,
            thread_id=thread_id,
            mode="act",
            git_available=False,
        )
        result = await app.ainvoke(
            {
                "messages": [HumanMessage(content=prepared.agent_prompt)],
                "approval_declined": {},
                "todos": [],
                "mode": "act",
            },
            config={"configurable": {"thread_id": thread_id}},
        )
        messages = result.get("messages", [])
        # Only the truly final message is the answer. Returning an earlier AI
        # message when the run ended on a tool/human message would silently
        # surface stale output (e.g. after an interrupt or tool error).
        if not messages:
            return "Subagent completed without producing any messages"
        last = messages[-1]
        if getattr(last, "type", "") in ("ai", "assistant"):
            final = last.content
        else:
            return (
                "Subagent did not finish with an assistant message "
                f"(last message type: {getattr(last, 'type', 'unknown')})"
            )
        return str(final or "Subagent completed with an empty final message")
    finally:
        remaining = _active_subagent_runs_by_parent.get(parent_thread_id, 1) - 1
        if remaining > 0:
            _active_subagent_runs_by_parent[parent_thread_id] = remaining
        else:
            _active_subagent_runs_by_parent.pop(parent_thread_id, None)


def _load_agent(name: str) -> dict[str, Any] | str:
    # load the agent specific instructions from the .ness/agents folder and put them in the prompt
    path = _agents_dir() / f"{name}.md"
    if not path.exists():
        return f"No agent definition at {path}"

    text = path.read_text(encoding="utf-8")
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            meta = yaml.safe_load(parts[1]) or {}
            if not isinstance(meta, dict):
                meta = {}
            meta["prompt"] = parts[2].strip()
            return meta
    return {"prompt": text.strip(), "tools": list(DEFAULT_AGENT_TOOLS)}


def _validate_agent_name(name: str) -> str | None:
    if not name or not AGENT_NAME_RE.fullmatch(name):
        return (
            "agent name must start with a letter or number and contain only letters, "
            "numbers, dots, underscores, or hyphens"
        )
    available = _available_agent_names()
    if name not in available:
        listed = ", ".join(sorted(available)) or "none"
        return f"unknown agent '{name}'; available agents: {listed}"
    return None


def _agent_tool_names(meta: dict[str, Any], agent_name: str) -> set[str] | str:
    raw_tools = meta.get("tools", list(DEFAULT_AGENT_TOOLS))
    if not all(isinstance(item, str) for item in raw_tools):
        return f"tools for subagent {agent_name} must be a list of tool names"
    tool_names = {item.strip() for item in raw_tools if item.strip()}
    if not tool_names:
        return f"subagent {agent_name} has no configured tools"
    return tool_names


def _resolve_tools(agent_name: str, tool_names: set[str]) -> list[Any] | str:
    # TODO: In future if subagents need more tool freedom; change access here
    # resolve tools from string to tool objects
    from ness_agent.tools import READ_ONLY_TOOLS, TOOL_MAP

    known_tool_names = set(TOOL_MAP)
    # reject unknown tools
    unknown = sorted(tool_names - known_tool_names)
    if unknown:
        return f"unknown tools for subagent {agent_name}: {', '.join(unknown)}"

    # reject unsafe tools
    subagent_safe_tools = set(READ_ONLY_TOOLS) - {
        "spawn_subagent",
        "todo",
        "search_tools",
        "add_tools",
    }
    unsafe = sorted(tool_names - subagent_safe_tools)
    if unsafe:
        return (
            "spawn_subagent only allows read-only native tools; "
            f"unsafe tools for subagent {agent_name}: {', '.join(unsafe)}"
        )

    return [TOOL_MAP[name] for name in sorted(tool_names)]


def _format_single_result(result: SubagentRunResult) -> str:
    if result.status == "ok":
        return result.output
    return f"Error: subagent {result.name} {result.status}: {result.output}"


def _format_batch_result(results: list[SubagentRunResult], duration_ms: int) -> str:
    ok = sum(1 for result in results if result.status == "ok")
    failed = len(results) - ok
    status = "ok" if failed == 0 else "failed" if ok == 0 else "partial"
    lines = [
        f"status={status}",
        f"duration_ms={duration_ms}",
        f"tasks_total={len(results)}",
        f"tasks_ok={ok}",
        f"tasks_failed={failed}",
        "",
    ]
    for result in results:
        heading = (
            f"[{result.index}] name={result.name} status={result.status} "
            f"duration_ms={result.duration_ms} thread_id={result.thread_id}"
        )
        if result.label:
            heading += f" label={result.label}"
        output = result.output.strip()
        if result.status != "ok":
            output = f"error={output}"
        lines.extend([heading, output, ""])
    return "\n".join(lines).rstrip()


def _format_batch_error(error: str, duration_ms: int, task_count: int = 0) -> str:
    return "\n".join(
        [
            "status=error",
            f"duration_ms={duration_ms}",
            f"tasks_total={task_count}",
            "tasks_ok=0",
            f"tasks_failed={task_count}",
            f"error={error}",
        ]
    )
