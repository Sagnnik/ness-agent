# SDK guide

The **Ness Agent SDK** (`ness-agent` on PyPI) is a LangGraph-based agent harness you can embed in your own apps, scripts, and internal tools. It provides the agent loop, built-in tools, permissions, memory, skills, hooks, compaction, reflection, and optional tracing.

The **Ness CLI** is a reference coding adapter built on top of this SDK (`ness_cli`).

See also: [SDK API reference](sdk-api.md) · [Architecture](architecture.md) · [Configuration](configuration.md) · [CLI guide](cli.md)

---

## Installation

```bash
pip install ness-agent
```

Optional OpenTelemetry tracing:

```bash
pip install ness-agent[tracing]
```

Requires **Python 3.12+**.

---

## Quick start

Omit `tools=` and `overlay=` and you get a working coding agent: all SDK built-in tools plus `CodingOverlay` (plan/act, git snapshot, todos, compaction note, session memory, skills). Default instruction texts come from `ness_agent.instructions`.

```python
import asyncio

from langchain_openai import ChatOpenAI

from ness_agent import NessAgent, PromptLayersConfig


async def main() -> None:
    agent = NessAgent(
        model=ChatOpenAI(model="gpt-4o"),
        prompt=PromptLayersConfig(),  # default L0 from ness_agent.instructions.L0_HARNESS
        # tools=      omitted → all SDK built-ins
        # overlay=    omitted → CodingOverlay
        # aux_prompts= omitted → compaction / reflection / subagent defaults
    )
    session = agent.session(thread_id="proj-1")
    # session.toggle_mode() flips plan ↔ act
    result = await session.run("Plan then implement: add a rate limiter on /api/login")
    print(result.assistant_message)
    print(result.usage_total)  # aggregate of every model call in the turn


asyncio.run(main())
```

`tools=` accepts a mix of `BaseTool` instances, plain callables (auto-wrapped), and built-in name strings (`"read"`, `"grep"`, `"shell"`, …). Pass `overlay=NoOverlay()` to drop L3 entirely. Instruction bodies are importable — e.g. `from ness_agent.instructions import L0_HARNESS, PLAN_MODE`.

### Project agents and concurrent sessions

A `NessAgent` is a project-scoped runtime: it owns shared persistence, memory, hooks, skill and tool catalogs, tracing, pricing, and defaults. Each call to `agent.session(...)` creates a separate effective runtime with its own graph/checkpointer, model fields, copied options, temporary permission rules, active MCP set, cancellation state, and cost tracker.

```python
agent = NessAgent(model=default_model, prompt=PromptLayersConfig())

first = agent.session(thread_id="thread-a")
second = agent.session(thread_id="thread-b", model=specialized_model)

await asyncio.gather(
    first.run("inspect the API"),
    second.run("inspect the database"),
)
```

Use a different `Session` for each concurrent thread; two simultaneous `run()`/`stream()` calls on the same `Session` are unsupported. Use a different `NessAgent` for an unrelated project because project paths and shared stores belong to the agent.

Changing defaults affects future sessions only:

```python
agent.configure_default_models(
    model=new_default,
    reflection_model=new_reflection_default,
    context_window=200_000,
)

# Existing sessions remain pinned. This session inherits the new defaults.
third = agent.session(thread_id="thread-c")
```

To deliberately change one live session, call `session.configure_models(...)`. Read thread usage from `session.cost_tracker`; `agent.config.cost_tracker` is the live current-process aggregate across sessions. Replayed history is restored into the session tracker without being counted as new aggregate spend.

### What the host owns

Bare `Session.run` is the turn engine only. Your application still needs to:

- Supply `l2_context` in the prompt when the model needs project or domain structure (not auto-loaded).
- Persist user events / resume via `ThreadStore` if you want durable threads (the coding CLI does this around the graph).
- Own MCP config, trust, and auth when connecting servers — the SDK does not read `.ness/mcp.json`.

For domain sketches (RAG, research, support), persistence helpers, and tracing recipes, see [SDK examples](../src/sdk_example_usage.md).

### Custom overlay and metadata

Replace the default coding overlay when your product has its own working state. Put per-turn facts on `session.metadata`; your `OverlayProvider` reads them into named L3 sections:

```python
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from ness_agent import NessAgent, OverlayContext, OverlayProvider


@tool
def vector_search(query: str, top_k: int = 5) -> str:
    """Semantic search over the indexed knowledge base."""
    ...


class RAGOverlay(OverlayProvider):
    def sections(self, state, ctx: OverlayContext) -> dict[str, str]:
        sections = {}
        retrieval = ctx.metadata.get("retrieval_summary", "")
        if retrieval:
            sections["retrieval_context"] = f"RETRIEVED THIS TURN\n{retrieval}"
        return sections


agent = NessAgent(
    model=ChatOpenAI(model="gpt-4o", temperature=0),
    tools=[vector_search],
    prompt={
        "l0": "Answer only from retrieved sources. Cite doc_id. Do not invent facts.",
        "persona": "Citation-first research assistant.",
        "l2_context": kb_catalog.describe(),  # app-supplied
        "l2_header": "KNOWLEDGE BASE",
        "include_git_line": False,
        "include_skill_catalog": False,
    },
    overlay=RAGOverlay(),
)


async def answer(user_id: str, question: str) -> str:
    session = agent.session(thread_id=f"user-{user_id}")
    session.metadata["retrieval_summary"] = retriever.retrieve(question).summary
    result = await session.run(question)
    return result.assistant_message
```

Stable section names matter: the harness sends only changed L3 sections during a tool loop. Mutate `session.metadata` in place so later turns see updates; reassignment (`session.metadata = {...}`) needs `rebuild_graph()`.

---

## Coding adapter (optional)

If you want the same wiring as the Ness CLI — OpenRouter models, `.ness/` paths, plan/act overlays, pricing — use the coding adapter:

```python
from ness_cli.factory import build_coding_session

coding = build_coding_session(thread_id="session-abc123")
async for event in coding.run_turn("add a rate limiter"):
    ...
```

This module is included in the same `ness-agent` package; the CLI entry point is `ness`.

---

## Public API

Core exports from `ness_agent`:

| Symbol | Purpose |
|--------|---------|
| `NessAgent`, `AgentSpec`, `NessAgentConfig` | Agent configuration and construction |
| `Session` | Run turns, stream events, manage thread state |
| `PromptLayers`, `PromptLayersConfig` | L0–L2 prompt assembly |
| `NessAgentOptions`, `MemoryConfig`, `ModeConfig` | Behavior toggles |
| `ToolRegistry`, `coding_tools` | Built-in and custom tools |
| `MCPRuntime`, `MCPServerSpec`, `MCPServerState` | Adapter-neutral MCP connections and discovered LangChain tools |
| `PermissionStore`, `HookRunner`, `SkillLoader` | Policy and extension points |
| `merge_skill_dirs`, `default_skill_search_dirs` | Opt-in well-known agent skill roots |
| `ThreadStore`, `MemoryStore` | Persistence backends |
| `CostTracker`, `TracingConfig`, `Tracer` | Usage and observability |
| `CodingOverlay`, `OverlayProvider`, `NoOverlay` | Default or custom L3 working-state overlay |
| `summarize` | Cache-safe summary fork using exact parent messages and bound model |

Import smoke test: `tests/test_sdk_smoke.py`. Longer embedding examples: [src/sdk_example_usage.md](../src/sdk_example_usage.md).

`Session.run()` returns a `RunResult`. Use `assistant_message` for the final text and `usage_total` for the aggregate usage of every model call in that turn. The former single-call `usage` attribute has been removed; replace `result.usage` with `result.usage_total` when upgrading.

## Skills

Skills are directories containing a `SKILL.md` (YAML frontmatter with `name` and `description`, plus the instruction body). Available skills appear as a one-line catalog in L1; the model loads full bodies on demand via the `skill_view` tool.

The SDK scans **exactly** the roots you configure — it never adds directories implicitly:

- `skills_dir=Path(...)` — scan this one directory (nested `category/skill/SKILL.md` layouts supported).
- `skills_dirs=[Path(...), ...]` — an explicit, exhaustive root list; earlier roots win on name collisions. Mutually exclusive with `skills_dir`.
- Both `None` (the default) — skills disabled.

To also load the well-known agent skill roots (`.agents/skills`, `.claude/skills`, `.codex/skills`, `.cursor/skills`, and their `~/` equivalents), opt in explicitly:

```python
from pathlib import Path
from ness_agent import NessAgent, merge_skill_dirs

agent = NessAgent(
    model=model,
    prompt=prompt,
    skills_dirs=merge_skill_dirs(project_root, project_root / ".ness" / "skills"),
)
```

`merge_skill_dirs(project_root, skills_dir)` returns your directory first, then the well-known project-local roots, then the user-global ones, deduped by resolved path (`default_skill_search_dirs(project_root)` returns just the well-known roots). Pass `project_rels=` / `global_rels=` to restrict which project-local and user-global roots are included (e.g. `global_rels=()` opts out of global roots entirely) — the Ness CLI uses `global_rels=(".agents/skills",)`, trusting only `~/.agents/skills` globally; your own application chooses whichever roots it trusts.

## MCP in an SDK application

`MCPRuntime` connects fully resolved server specifications without depending on Ness project files, trust prompts, terminal output, or credential storage. Start the runtime before constructing an agent, then pass its discovered tools to any LangChain-compatible application:

```python
from ness_agent import MCPRuntime, MCPServerSpec, NessAgent

runtime = MCPRuntime(http_auth_factory=my_optional_auth_factory)
await runtime.start(
    [
        MCPServerSpec(
            name="knowledge",
            transport="http",
            url="https://example.com/mcp",
            headers=(("X-Application", "my-app"),),
        )
    ]
)

agent = NessAgent(
    model=model,
    prompt=prompt,
    tools=list(runtime.tools.values()),
)

try:
    result = await agent.session().run("Search the connected knowledge source")
finally:
    await runtime.stop()
```

The embedding application decides where server configuration comes from and how users approve or authenticate connections. `HTTPAuthFactory` can provide an `httpx` authentication object for each resolved HTTP spec.

For signatures and contracts for every public export in `ness_agent.__all__`, see the [SDK API reference](sdk-api.md).

---

## Prompt layers

The SDK splits prompts into L0–L3 layers for stable prefix caching. See [Architecture → Prompt layers](architecture.md#prompt-layers) for the full model.

When using the SDK directly, you supply L0–L2 via `PromptLayers` / `PromptLayersConfig` (or a plain mapping). Omitting `overlay=` installs `CodingOverlay`; pass a custom `OverlayProvider` or `NoOverlay()` as shown above.

## Cache-safe summarization

Automatic compaction uses the main agent model and its bound tools. For custom flows, pass the exact parent request and the same bound runnable:

```python
from ness_agent import summarize

summary = await summarize(
    exact_parent_messages,
    bound_parent_model,
    instruction="Summarize completed work for continuation.",
    max_output_tokens=4096,
)
```

Constructing a separate model, changing tools, or replacing the system prompt prevents reuse of the parent's cached prefix.

---

## Tracing

Install the tracing extra, then pass `TracingConfig` when constructing the agent:

```python
from ness_agent import NessAgent, TracingConfig

agent = NessAgent(
    model=model,
    prompt=prompt,
    tracing=TracingConfig(enabled=True, exporter="console"),
)
```

See `tests/tracing/` for integration examples.

---

## Stability

Ness Agent is **0.x experimental**. Public APIs may change until 1.0. Pin versions in production and watch [CHANGELOG](../CHANGELOG.md).
