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
    try:
        result = await session.run("Plan then implement: add a rate limiter on /api/login")
        print(result.assistant_message)
        print(result.usage_total)  # aggregate of every model call in the turn
    finally:
        await session.close()  # stop any background shell jobs owned by this session


asyncio.run(main())
```

`tools=` accepts a mix of `BaseTool` instances, plain callables (auto-wrapped), and built-in name strings (`"read"`, `"grep"`, `"shell"`, …). Pass `overlay=NoOverlay()` to drop L3 entirely. Instruction bodies are importable — e.g. `from ness_agent.instructions import L0_HARNESS, PLAN_MODE`.

In a vision-capable session, the built-in `read` tool returns supported raster
images (PNG, JPEG, WebP, GIF, PPM, BMP, and TIFF) as structured image content.
It applies EXIF orientation, limits the long edge to 2000px, re-encodes to PNG,
and enforces a 5 MB normalized-payload ceiling. With `vision=False`, the model
receives a visible omission marker instead. PDFs and videos must first be
rendered or have frames extracted to a supported raster format. Persisted
events, hooks, traces, and display events receive a redacted text form rather
than the base64 image payload.

### Project agents and concurrent sessions

A `NessAgent` is a project-scoped runtime: it owns shared persistence, memory, hooks, skill and tool catalogs, tracing, pricing, and defaults. Each call to `agent.session(...)` creates a separate effective runtime with its own graph/checkpointer, model fields, copied options, temporary permission rules, active MCP set, cancellation state, and cost tracker.

Each session also has its own `ThreadStore` view. These views share the same
database and writer lock, while `auto_save` belongs to the view. New sessions
initialize it from the agent's `options.auto_save_threads` default. Changing
that default affects future sessions; existing sessions retain their policy.
All SDK persistence writes, including compaction and subagent records, use
the session's view. Disabling autosave leaves saved history readable.

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

Applications should construct `NessAgent` and use its public `Session` API. The `ness_cli` runtime and session modules ship in the same distribution, but are internal CLI implementation APIs, without a compatibility guarantee for embedding. The following example shows the current wiring for CLI maintainers. It requires a configured provider and may start trusted project MCP servers.

```python
import asyncio
from pathlib import Path

from ness_cli.runtime import HeadlessOptions, open_headless_runtime


async def main() -> None:
    options = HeadlessOptions(project_root=Path.cwd())
    # To resume instead, also set resume_thread_id="session-<saved-id>".
    async with open_headless_runtime(options) as runtime:
        coding = await runtime.session()
        async for event in coding.run_turn("add a rate limiter"):
            print(event.kind, event.data)


asyncio.run(main())
```

The async context manager owns runtime cleanup. Interactive Ness uses `open_interactive_runtime`, then `initial_session`, `new_session`, or `resume_session`. To start a fresh thread, create a new session; the adapter has no `reset` method. The console entry point is `ness_cli.main:main`, exposed as `ness`.

For an offline construction and resume example with no credentials, see the [SDK persistence recipe](../src/sdk_example_usage.md#sdk-persistence-recipe). That recipe explains the internal replay helper separately from the public SDK primitives.

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

Skills are directories containing a `SKILL.md` (YAML frontmatter with `name` and `description`, plus the instruction body). Available skills appear as a one-line catalog in L1 by default; the model loads full bodies on demand via the `skill_view` tool.

The SDK scans **exactly** the roots you configure — it never adds directories implicitly:

- `skills_dir=Path(...)` — scan this one directory (nested `category/skill/SKILL.md` layouts supported).
- `skills_dirs=[Path(...), ...]` — an explicit, exhaustive root list; earlier roots win on name collisions. Mutually exclusive with `skills_dir`.
- Both `None` (the default) — skills disabled.

To also load the well-known agent skill roots (`.agents/skills`, `.claude/skills`, `.codex/skills`, `.cursor/skills`, and their `~/` equivalents), opt in explicitly:

```python
from pathlib import Path
from ness_agent import NessAgent, default_skill_search_dirs

agent = NessAgent(
    model=model,
    prompt=prompt,
    skills_dirs=default_skill_search_dirs(project_root),
)
```

`default_skill_search_dirs(project_root)` returns the well-known project-local roots, then the user-global ones, with `.agents/skills` first in each scope. The Ness CLI uses this list. `merge_skill_dirs(project_root, skills_dir)` puts an application-specific directory first, then those shared roots, deduped by resolved path. Pass `project_rels=` / `global_rels=` to restrict either set; `global_rels=()` excludes all user-global roots. Your application chooses its own roots.

Skill defaults and their persistence belong to the application. Each new session has a fresh loader for the configured directories; snapshots and disabled IDs installed on `agent.config.skill_loader` are not inherited. Configure each session through the public methods, and reapply your choices when resuming:

```python
from ness_agent import SkillLoader

records = SkillLoader(skills_dirs=my_skill_roots).discover()
session = agent.session(thread_id=thread_id)
session.configure_skills(records, disabled_skill_ids=my_disabled_ids)

# Replace this session's disabled-ID set when the application's choices change.
session.set_skill_access(updated_disabled_ids)
```

`discover()` returns logical records with IDs, bodies, and source paths, grouping exact bundle copies. `configure_skills()` freezes those records for that session. Without a snapshot, directories are rediscovered as context is built; IDs are content fingerprints and change when bundle files change. The application owns refresh and access policy. The built-in viewer requires local source directories, so an application fetching remote bundles should materialize their files locally before configuring the session.

`session.stage_skills(names)` or `session.run(..., requested_skills=names)` requests that the model load available skills; it does not inject bodies directly. These requests are rendered by the overlay. With `include_skill_catalog=False`, the catalog also moves to the overlay, where it is sent on the first turn, after access changes, and after context rebuilds. Custom overlays must render `ctx.skill_catalog` and `ctx.requested_skills` to support that flow. `NoOverlay()` renders neither. Built-in subagents use an L1 catalog because they have no overlay.

`session.requested_skills(names)` replaces requests staged for the next turn. The older `session.active_skills(names)` and `active_skills=` argument on `run()`/`stream()` remain compatibility aliases; an explicit `requested_skills=` takes precedence.

Disabling a skill blocks future `skill_view` lookups and omits it from current requests. Viewed bodies stay in conversation history; there is no separate loaded-skill list. After compaction, the agent receives a general reminder to reload any skill instructions it needs. Other enabled file tools retain their own permissions.

To exclude skill tools and instructions as well as discovery, use application-authored prompts, an explicit tool list without `skill_view`, and `NoOverlay()` or an application overlay without skill sections:

```python
from ness_agent import NessAgent, NoOverlay, PromptLayersConfig

agent = NessAgent(
    model=model,
    prompt=PromptLayersConfig(
        l0="Follow the application's instructions.",
        persona="Application assistant.",
        include_git_line=False,
        include_skill_catalog=False,
    ),
    tools=[],
    skills_dirs=[],
    overlay=NoOverlay(),
)
```

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

`ness_agent.mcp` also exposes shared helpers for diagnostic text. `redact_url(value)` strips URL credentials, query parameters, and fragments. `redact_text(value, secrets, fallback="[redacted]")` replaces literal secrets, longest first. A matching secret shorter than four characters replaces the whole message with the fallback. These helpers format text for display; callers still need to validate connection URLs and escape text for their output format. The SDK MCP runtime and CLI use these same helpers.

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

### Shell execution and outside-project access

Configure shell limits through SDK options:

```python
options = NessAgentOptions(
    project_root=Path("/app"),
    ness_dir=Path("/logs/agent/ness"),
    shell_default_timeout=30,
    shell_max_timeout=1800,
)
```

Normal-mode file tools ask for approval before reading or writing outside the project. The existing once, session, and always decisions control how long the path grant lasts. Read approval does not grant writes. Configure an approval handler for interactive grants; without one, outside access is denied. YOLO bypasses file-path and protected-write restrictions as well as tool approvals and deny rules. OS permissions still apply. Neither mode sandboxes shell commands; use OS or container isolation for that.

Foreground commands return a retained log path and execution ID. The agent can retrieve the full log with `shell(action="read", job_id=..., offset=0)` and continue using the returned `next_offset`. Cancelling the session stops active foreground process groups. For a host deadline, pass `deadline=time.monotonic() + remaining_seconds` when creating the session; the host still owns the overall task timeout and session cleanup.
