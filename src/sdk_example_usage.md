# Ness Agent SDK examples

Domain-agnostic agent harness in `src/ness_agent/`. The coding CLI adapter lives in `src/ness_cli/` (tools, overlays, pricing, OpenRouter wiring).

Construct agents with `NessAgent(...)` kwargs (builds an `AgentSpec` internally) or `NessAgent.from_spec(AgentSpec(...))`. Compaction budget knobs live on `NessAgentOptions` — there is no separate budget config.

**App responsibilities (not done by bare `Session.run`):**

- Supply `l2_context` in the prompt when the model needs project/domain structure (SDK does not auto-load repo context).
- Append user events / call `thread_store.save_checkpoint` if you want resumable threads (the coding CLI does this around the graph).
- Pass `cost_tracker=make_sdk_cost_tracker()` from `ness_cli.config` when you want estimated USD for non-provider-cost models.
- Supply resolved MCP server settings and own connection approval, trust, and authentication policy when adding MCP (the bare SDK does not read Ness CLI project files).

---

## Minimal coding agent (zero boilerplate)

`tools=` and `overlay=` are both optional. The SDK ships with a fully wired `CodingOverlay` (plan/act blocks, git snapshot, compaction note, todos, session memory, loaded skills) and defaults `AuxPrompts` (compaction / reflection / subagent / thread_summary / init_memory) to internal instruction texts — so a bare agent is a working coding agent.

```python
from ness_agent import NessAgent, PromptLayersConfig
from langchain_openai import ChatOpenAI

agent = NessAgent(
    model=ChatOpenAI(model="gpt-4o"),
    prompt=PromptLayersConfig(),   # default L0 from ness_agent.instructions.L0_HARNESS
    # tools=      omitted -> all SDK built-in tools (BUILTIN_TOOLS)
    # overlay=    omitted -> CodingOverlay (plan/act, git, todos, compaction, ...)
    # aux_prompts= omitted -> defaults to ness_agent.instructions.{COMPACTION,REFLECTION,...}
)
session = agent.session(thread_id="proj-1")
# session.toggle_mode() flips plan <-> act using the SDK's default plan/act instruction texts
await session.run("Plan then implement: add a rate limiter on /api/login")
```

### Default overlay

- If `overlay=` is omitted, the agent is configured with `CodingOverlay` (`from ness_agent import CodingOverlay`). It renders:
  - `<plan-mode path="...">...</plan-mode>` when `session.mode == "plan"`, using `ness_agent.instructions.PLAN_MODE` (or `modes.plan_mode_template` if you supply a `ModeConfig`). The coding CLI sets `modes.plans_dir` to the global `plans/<project-slug>/` directory; the SDK default string is `.ness/plans/`.
  - `mode_switch` on the first act turn after a plan->act toggle, using `ness_agent.instructions.ACT_MODE` (or `modes.act_mode_template`)
  - `git`, `compaction`, `todos`, `session_memory`, `loaded_skills`, and `skill_request` sections from the `OverlayContext`
- To **opt out of L3 entirely** pass `overlay=NoOverlay()` (apps that need no working-state overlay, or want to drive everything from the model alone).
- To use a **custom L3**, pass your own `OverlayProvider` (see the four examples below).

### Instruction texts are Python-importable

The default instruction bodies live in the `ness_agent.instructions` package, not as opaque `.md` files:

```python
from ness_agent.instructions import L0_HARNESS, COMPACTION, REFLECTION, SUBAGENT, THREAD_SUMMARY, INIT_MEMORY, PLAN_MODE, ACT_MODE, L1_PROFILE

# Copy and modify, then feed the modified text back in:
my_l0 = L0_HARNESS.replace("NESS", "Acme Assistant")
agent = NessAgent(
    model=...,
    prompt=PromptLayersConfig(l0=my_l0, persona="..."),
    # or override only one task prompt:
    # aux_prompts=AuxPrompts(compaction=my_compaction_instruction),
)
```

### Tools: `BaseTool`, callable, or built-in name

`tools=` accepts a mix of `BaseTool` instances, plain callables (auto-wrapped with `StructuredTool.from_function`), and strings naming SDK built-ins (`"read"`, `"grep"`, `"glob"`, `"shell"`, ...):

```python
from ness_agent import NessAgent, PromptLayersConfig

agent = NessAgent(
    model=...,
    prompt=PromptLayersConfig(),
    tools=["read", "grep", "glob", my_custom_fn],   # mixed list
)
```

---

## MCP-enabled application

`MCPRuntime` is the domain-agnostic connection layer. It accepts resolved
server specifications and exposes discovered MCP tools as ordinary LangChain
tools. It does not read `.ness/mcp.json`, display trust prompts, or persist
OAuth credentials; those policies belong to the embedding application.

```python
from ness_agent import MCPRuntime, MCPServerSpec, NessAgent, PromptLayersConfig


async def answer_with_mcp(model, question: str):
    runtime = MCPRuntime(http_auth_factory=my_optional_auth_factory)
    try:
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
            prompt=PromptLayersConfig(
                l0="Use the connected knowledge source when it can help."
            ),
            tools=list(runtime.tools.values()),
        )
        result = await agent.session(thread_id="knowledge-1").run(question)
        return result.assistant_message
    finally:
        await runtime.stop()
```

An application can build `MCPServerSpec` objects from a database, its own
configuration format, user input, or another service. Provide an
`HTTPAuthFactory` when HTTP connections require app-managed authentication.
The Ness CLI's project config, trust fingerprints, OAuth storage, and terminal
status rendering are adapter features rather than requirements of the SDK.

---

## RAG Application

```python
from pathlib import Path
from ness_agent import (
    NessAgent, NessAgentOptions, MemoryConfig,
    OverlayProvider, OverlayContext,
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


@tool
def vector_search(query: str, top_k: int = 5) -> str:
    """Semantic search over the indexed knowledge base."""
    ...

@tool
def fetch_chunk(doc_id: str, chunk_id: str) -> str:
    """Fetch a full chunk by id for deeper inspection."""
    ...

@tool
def cite_sources(sources: list[str]) -> str:
    """Record the sources used in the answer."""
    ...


class RAGOverlay(OverlayProvider):
    def sections(self, state, ctx: OverlayContext) -> dict[str, str]:
        sections = {}
        retrieval = ctx.metadata.get("retrieval_summary", "")
        if retrieval:
            sections["retrieval_context"] = f"RETRIEVED THIS TURN\n{retrieval}"
        score = ctx.metadata.get("grounding_score", "")
        if score:
            sections["confidence"] = f"Grounding score: {score}"
        open_q = ctx.metadata.get("open_questions", "")
        if open_q:
            sections["open_questions"] = f"Unanswered:\n{open_q}"
        return sections


agent = NessAgent(
    model=ChatOpenAI(model="gpt-4o", temperature=0),
    tools=[vector_search, fetch_chunk, cite_sources],
    prompt={
        "l0": (
            "You are a knowledge assistant. Answer ONLY from retrieved sources. "
            "Always cite doc_id. If context is insufficient, say so — do not invent facts."
        ),
        "persona": "Precise, citation-first research assistant for Acme internal docs.",
        "l2_context": kb_catalog.describe(),     # app-supplied; not auto-loaded
        "l2_header": "KNOWLEDGE BASE",
        "include_git_line": False,
        "include_skill_catalog": False,
    },
    skills_dir=None,                            # skills fully disabled (SDK default)
    memory=MemoryConfig(
        project_memory=Path("./kb/POLICIES.md"),
        user_memory=Path("./users/u-42.md"),
        session_memory_dir=Path("./sessions"),
    ),
    overlay=RAGOverlay(),
    options=NessAgentOptions(
        enable_approval=False,
        context_window=128_000,               # drives compaction usable budget
        reflection_token_ratio=0.3,
    ),
)


async def answer(user_id: str, question: str) -> str:
    session = agent.session(thread_id=f"user-{user_id}")
    results = retriever.retrieve(question)
    session.metadata["retrieval_summary"] = results.summary
    session.metadata["grounding_score"] = results.score
    result = await session.run(question)
    await session.finalize_reflection()
    return result.assistant_message
```

## Deep Research Application

```python
from pathlib import Path
from ness_agent import (
    NessAgent, NessAgentOptions, SubagentConfig,
    OverlayProvider, OverlayContext,
)
from langchain_openai import ChatOpenAI
from my_tools import web_search, fetch_url, save_note, spawn_subagent, build_report


class ResearchOverlay(OverlayProvider):
    """L3: outstanding questions, sources collected, report outline."""
    def sections(self, state, ctx: OverlayContext) -> dict[str, str]:
        sections = {}
        outline = ctx.metadata.get("outline", "")
        if outline:
            sections["outline"] = f"REPORT OUTLINE\n{outline}"
        sources = ctx.metadata.get("sources_collected", [])
        if sources:
            sections["sources"] = "SOURCES\n" + "\n".join(f"- {s}" for s in sources[-15:])
        todos = ctx.todos
        if todos:
            lines = "\n".join(f"- [{t.get('status')}] {t.get('content')}" for t in todos
                              if t.get("status") != "completed")
            if lines: sections["plan"] = f"RESEARCH PLAN\n{lines}"
        return sections


agent = NessAgent(
    model=ChatOpenAI(model="gpt-4o"),
    reflection_model=ChatOpenAI(model="gpt-4o-mini"),
    tools=[web_search, fetch_url, save_note, spawn_subagent, build_report],
    prompt={
        "l0": (
            "You are a research analyst. Work in phases: scope → gather → synthesize. "
            "Cite every claim with URL and access date. Prefer primary sources. "
            "Use spawn_subagent to parallelize independent research threads."
        ),
        "persona": "Thorough, skeptical analyst producing structured reports.",
        "l2_context": f"Research brief: {brief}\nDeadline: {deadline}\nOutput format: markdown",
        "l2_header": "RESEARCH BRIEF",
        "include_git_line": False,
    },
    # The SDK scans exactly this directory — nothing is added implicitly.
    # To also load well-known roots (.agents/.claude/.codex/.cursor skills,
    # project + ~/), pass skills_dirs=merge_skill_dirs(project_root, ...) instead.
    skills_dir=Path("./skills/research"),
    subagents=SubagentConfig(
        prompt_template=RESEARCH_SUBAGENT_PROMPT,
        max_parallel=3,
        default_tools=("web_search", "fetch_url", "save_note"),
        default_timeout_seconds=600,
    ),
    overlay=ResearchOverlay(),
    options=NessAgentOptions(
        enable_approval=True,
        context_window=200_000,
        reflection_token_ratio=0.25,
        auto_save_threads=True,
    ),
)


async def research(topic: str) -> str:
    session = agent.session(thread_id=f"research-{slugify(topic)}")
    session.metadata["outline"] = initial_outline(topic)
    session.metadata["sources_collected"] = []
    # Optional: persist a user event for resumable threads
    # agent.config.thread_store.append_event(session.thread_id, {"kind": "user", "content": topic})
    result = await session.run(topic, mode="act")
    await session.finalize_reflection()
    return result.assistant_message
```

## Video Generation Application

```python
from ness_agent import (
    NessAgent, NessAgentOptions, SubagentConfig,
    OverlayProvider, OverlayContext,
)
from langchain_openai import ChatOpenAI
from my_tools import (
    generate_scene, render_clip, stitch_clips, upload_to_bucket,
    review_storyboard, get_asset_duration,
)


class VideoOverlay(OverlayProvider):
    """L3: storyboard progress, render queue, asset inventory."""
    def sections(self, state, ctx: OverlayContext) -> dict[str, str]:
        sections = {}
        queue = ctx.metadata.get("render_queue", [])
        if queue:
            sections["render_queue"] = "RENDER QUEUE\n" + "\n".join(
                f"- {job['clip_id']}: {job['status']}" for job in queue[:10])
        assets = ctx.metadata.get("assets", {})
        if assets:
            lines = "\n".join(f"- {name}: {info['duration']:.1f}s, {info['resolution']}"
                              for name, info in assets.items())
            sections["assets"] = f"ASSET INVENTORY\n{lines}"
        budget = ctx.metadata.get("render_budget_seconds", 0)
        spent = ctx.metadata.get("render_seconds_used", 0)
        if budget:
            sections["budget"] = f"Render budget: {spent:.1f}/{budget:.1f}s used"
        todos = ctx.todos
        if todos:
            lines = "\n".join(f"- [{t.get('status')}] Scene {t.get('id')}: {t.get('content')}"
                              for t in todos if t.get("status") != "completed")
            if lines: sections["storyboard"] = f"STORYBOARD\n{lines}"
        return sections


agent = NessAgent(
    model=ChatOpenAI(model="gpt-4o"),
    tools=[generate_scene, render_clip, stitch_clips, upload_to_bucket,
           review_storyboard, get_asset_duration],
    prompt={
        "l0": (
            "You are a video director. Break the brief into scenes, generate each, "
            "review, stitch, and upload. Respect the render-second budget. "
            "If a scene fails review, regenerate before stitching. "
            "Never upload until all clips pass quality check."
        ),
        "persona": "Efficient video director optimizing for quality within time budget.",
        "l2_context": f"Project: {project_name}\nStyle: {style_guide}\n"
                      f"Duration target: {target_duration}s\nResolution: {resolution}",
        "l2_header": "VIDEO BRIEF",
        "include_git_line": False,
    },
    modes=None,
    subagents=SubagentConfig(
        prompt_template=VIDEO_REVIEW_SUBAGENT_PROMPT,
        max_parallel=2,
        default_tools=("render_clip", "review_storyboard", "get_asset_duration"),
        default_timeout_seconds=900,
    ),
    overlay=VideoOverlay(),
    options=NessAgentOptions(
        enable_approval=True,
        context_window=200_000,
        reflection_token_ratio=0.0,
        auto_save_threads=True,
    ),
)


async def produce_video(brief: str) -> str:
    session = agent.session(thread_id=f"video-{project_id}")
    session.metadata["render_budget_seconds"] = 300
    session.metadata["render_seconds_used"] = 0
    session.metadata["assets"] = {}
    session.metadata["render_queue"] = []
    result = await session.run(brief)
    return result.assistant_message
```

## Customer Support Application

```python
from pathlib import Path
from ness_agent import (
    AgentSpec, NessAgent, NessAgentOptions, MemoryConfig,
    OverlayProvider, OverlayContext, ApprovalHandler,
)
from langchain_openai import ChatOpenAI
from my_tools import (
    lookup_order, search_kb, create_ticket, escalate_to_human,
    update_account, send_email, fetch_conversation_history,
)


class SupportOverlay(OverlayProvider):
    """L3: customer context, SLA timer, escalation state."""
    def sections(self, state, ctx: OverlayContext) -> dict[str, str]:
        sections = {}
        customer = ctx.metadata.get("customer", {})
        if customer:
            lines = [f"Customer: {customer.get('name', 'unknown')}",
                     f"Tier: {customer.get('tier', 'standard')}",
                     f"Account age: {customer.get('account_age_days', 0)} days"]
            sections["customer"] = "CUSTOMER\n" + "\n".join(lines)
        history = ctx.metadata.get("recent_conversations", "")
        if history:
            sections["history"] = f"RECENT CONVERSATIONS\n{history}"
        sla = ctx.metadata.get("sla_minutes_remaining")
        if sla is not None:
            sections["sla"] = f"SLA: {sla:.0f} minutes remaining"
        escalated = ctx.metadata.get("escalated", False)
        if escalated:
            sections["escalation"] = "ESCALATED to human — follow their guidance only."
        sentiment = ctx.metadata.get("sentiment", "")
        if sentiment:
            sections["sentiment"] = f"Detected sentiment: {sentiment}"
        return sections


class SupportApproval(ApprovalHandler):
    """Auto-approve safe actions; escalate destructive ones to a human queue."""
    SAFE_ACTIONS = {"lookup_order", "search_kb", "fetch_conversation_history", "send_email"}
    async def __call__(self, tool: str, args: dict) -> str:
        if tool in self.SAFE_ACTIONS:
            return "yes"
        await enqueue_for_human_review(tool, args)
        return "no"


# Equivalent to NessAgent(...): build an AgentSpec then resolve
agent = NessAgent.from_spec(AgentSpec(
    model=ChatOpenAI(model="gpt-4o", temperature=0.2),
    tools=[lookup_order, search_kb, create_ticket, escalate_to_human,
           update_account, send_email, fetch_conversation_history],
    prompt={
        "l0": (
            "You are a customer support assistant for Acme Corp. "
            "Be concise, empathetic, and action-oriented. "
            "Always look up the order/account before making changes. "
            "Escalate to human when: customer is upset, issue is billing-related, "
            "or you've attempted 2 fixes without resolution. "
            "Never make account changes without customer confirmation."
        ),
        "persona": "Empathetic, efficient support specialist. Concise responses.",
        "l2_context": "Product: Acme SaaS\nSupport hours: 24/7\nEscalation channel: #support-escalations",
        "l2_header": "SUPPORT CONTEXT",
        "include_git_line": False,
        "include_skill_catalog": True,
    },
    skills_dir=Path("./skills/support"),        # exact: only this dir is scanned
    memory=MemoryConfig(
        project_memory=Path("./support/POLICIES.md"),
        user_memory=Path(f"./customers/{customer_id}.md"),
        session_memory_dir=Path("./support/sessions"),
    ),
    overlay=SupportOverlay(),
    approval_handler=SupportApproval(),
    options=NessAgentOptions(
        enable_approval=True,
        context_window=64_000,
        reflection_token_ratio=0.5,
        session_end_reflection=True,
        auto_save_threads=True,
    ),
))


async def handle_message(customer_id: str, message: str) -> str:
    customer = await load_customer(customer_id)
    history = await load_recent_conversations(customer_id, limit=5)
    session = agent.session(thread_id=f"support-{customer_id}-{ticket_id}")
    session.metadata["customer"] = customer
    session.metadata["recent_conversations"] = history
    session.metadata["sla_minutes_remaining"] = await sla_remaining(ticket_id)
    session.metadata["sentiment"] = analyze_sentiment(message)
    result = await session.run(message)
    await session.finalize_reflection()
    return result.assistant_message
```

---

Same harness across all four: turn loop, compaction, reflection, permissions, skills, tool execution. Apps differ only in tools, prompts, overlay, memory paths, approval, modes, options (including `context_window`), and subagent config.

---

## Compaction persistence split

Compaction has several channels — do not conflate them:

| Channel | Kind / signal | Who writes | When |
|---|---|---|---|
| Durable notice | `compact` | CodingSession (adapter) | `/compact`, pre-act notices, and live agent-turn `SessionEvent("compaction")` rows the adapter durable-logs |
| Durable summary checkpoint | `compaction_llm` | SDK compaction boundary | Successful summary with raw-event source boundary |
| Graph state | `model_context_messages` | boundary/agent nodes | Exact model-facing history, including internal L3 tails |
| Live stream | `SessionEvent("compaction")` | Session | Pre-act checkpoints plus summary success/failure notices |

Compaction never rewrites tool output. Completed history becomes one human
`<compacted-history>` message; the active user/tool turn remains verbatim.

## Reflection defaults (SDK vs CLI)

Reflection is **opt-in** on the bare SDK:

- `NessAgentOptions.reflection_token_ratio` defaults to `0.0` (mid-session
  token-delta trigger off).
- `NessAgentOptions.session_end_reflection` defaults to `False`.

The coding CLI factory typically sets `REFLECTION_TOKEN_RATIO=0.4` from env
(and leaves session-end reflection off unless toggled in `/config`). To enable
in an SDK app:

```python
options=NessAgentOptions(
    reflection_token_ratio=0.3,       # mid-session background gate
    session_end_reflection=True,      # finalize on exit / archive
)
# ...
await session.finalize_reflection()
```

An application can also trigger an incremental pass explicitly, even when both
automatic reflection settings are disabled:

```python
result = await session.run_reflection()
print(result.bullets)
```

Durable audit rows use ThreadStore `kind=reflection`. There is no live
`SessionEvent` for reflection.

## Context / graph invariants

These contracts keep ordinary calls and compaction forks cache-safe:

- **No system messages in state.** `AgentState.messages` holds the conversation
  only. The L0–L2 system prefix is rebuilt each `agent_node` turn and never
  checkpointed.
- **Two histories.** `AgentState.messages` is the clean semantic transcript.
  `model_context_messages` retains exact model-facing L3 reminder tails for
  prefix continuity; those tails never enter durable CLI events or reflection.
- **Boundary compaction.** Summary runs after completed tools and before the
  next agent call with the same bound main model, tools, system, and session.
- **`session.metadata` identity.** `make_nodes` snapshots the metadata dict at
  graph build. In-place mutation of `session.metadata` is visible on later
  turns; reassignment (`session.metadata = {...}`) needs `rebuild_graph()`.
- **Plan write gating.** When `mode == "plan"` and
  `ModeConfig.plan_mode_readonly` is true (default, including when
  `modes is None`), state-changing tools are denied. Set
  `ModeConfig(plan_mode_readonly=False)` to allow writes in plan mode.
- **`recursion_limit`.** LangGraph turn depth comes from
  `NessAgentOptions.recursion_limit` (default `75`), used by `Session.run` /
  `Session.stream`.

### Preview system + L3

```python
preview = await session.preview_context()           # current mode
# preview = await session.preview_context(mode="plan")

print(preview.system_message)   # L0–L2 stable prefix
print(preview.overlay)          # joined L3 sections
print(preview.overlay_sections) # dict of named sections
print(preview.overlay_reminder) # <system-reminder>… wrap
```

Does not call the model or compaction LLM. Useful for inspecting what the
next turn would put in the system message and working-state tail.

## SDK persistence recipe

`Session` is the turn engine. Durable thread CRUD lives on
`agent.config.thread_store` (`ThreadStore`, exported from `ness_agent`).
Resume/archive are not CLI-only — wire them with the primitives below (or use
`CodingSession.resume` / `CodingSession.reset` for the batteries-included
coding path).

### Event-kind ownership

| Kind | Writer |
|------|--------|
| `user`, `compact` | App / `CodingSession` |
| `assistant`, `tool`, `usage`, `approval` | Graph (`nodes.py`) |
| `reflection` | `reflection.py` (durable only; not a `SessionEvent`) |
| `compaction_llm` | SDK compaction boundary |

**Important:**

- Bare `Session.run` / `Session.stream` do **not** auto-persist `user` (or
  `compact`) rows — apps must `append_event`, or use `CodingSession.run_turn`.
- `ThreadStore.list_threads` currently filters to `session-*` prefixes only;
  custom thread IDs will not appear until filters are made explicit in a later
  pass.

```python
from ness_agent import NessAgent, PromptLayersConfig, ThreadStore
from ness_cli.events import events_to_messages  # coding transcript rebuild


# --- small helpers apps can copy --------------------------------------------

async def persist_user_turn(session, text: str, *, images=None) -> int | None:
    """Append a user row before session.run / session.stream."""
    store = session.agent.config.thread_store
    event = {"kind": "user", "content": text}
    if images:
        event["images"] = list(images)
    return store.append_event(session.thread_id, event)


async def persist_assistant_turn(session, text: str) -> int | None:
    store = session.agent.config.thread_store
    return store.append_event(
        session.thread_id, {"kind": "assistant", "content": text}
    )


async def archive_thread(session) -> str:
    """Finalize reflection (if enabled) and archive the current thread."""
    await session.finalize_reflection()
    return session.agent.config.thread_store.archive_thread(session.thread_id)


async def resume_thread(session, thread_id: str, *, vision: bool | None = None) -> bool:
    """Rebuild live graph state from the durable event log.

    Uses Session.reset_checkpointer + Session.bootstrap so the event log is
    the single source of truth (do not reuse a dirty MemorySaver).
    """
    store = session.agent.config.thread_store
    events = store.load_thread_events(thread_id)
    if not events:
        return False
    # Use the effective session view so temporary approval rules are preserved.
    permission_store = session.config.permission_store
    messages = events_to_messages(
        events,
        store.list_subagents(thread_id),
        vision=vision,
        permission_store=permission_store,
    )
    session.thread_id = thread_id
    session.reset_checkpointer()
    session.bootstrap(messages)
    await session.refresh_context_snapshot()
    return True


async def list_recent_threads(session, n: int = 10) -> list[dict]:
    return session.agent.config.thread_store.list_threads(n)


# --- usage ------------------------------------------------------------------

agent = NessAgent(model=..., prompt=PromptLayersConfig())
session = agent.session(thread_id="session-demo-1")

await persist_user_turn(session, "add a rate limiter")
result = await session.run("add a rate limiter")
await persist_assistant_turn(session, result.assistant_message)

# later, in another process:
session2 = agent.session(thread_id="session-fresh")
ok = await resume_thread(session2, "session-demo-1")
assert ok
messages = await session2.get_messages()  # public read — no app.aget_state needed
```

Public reads on `Session`: `get_state()`, `get_messages()`, `get_todos()`,
`refresh_context_snapshot()`.

---

## Hooks (three different concepts)

Ness Agent uses the word “hooks” in three places — do not confuse them:

1. **`HookRunner`** — tool pre/post hooks (`preToolUse` / `postToolUse` only).
   Loaded from `{ness_dir}/hooks.json` by default, and/or registered in-memory:

   ```python
   from ness_agent import Hook, NessAgent, PromptLayersConfig

   def deny_shell(payload: dict) -> tuple[bool, str]:
       if payload.get("tool") == "shell":
           return False, "shell blocked by policy"
       return True, ""

   agent = NessAgent(
       model=...,
       prompt=PromptLayersConfig(),
       hooks=[Hook(event="preToolUse", matcher="*", handler=deny_shell)],
       # hooks_config=Path(".ness/hooks.json"),  # default when omitted
   )
   # or later: agent.config.hook_runner.register(Hook(...))
   ```

2. **`approval_handler` / `question_handler`** — interactive gates on the agent
   for destructive tools and the `question` tool (not JSON hooks).

3. **`on_plan_turn` / `on_interrupt`** — per-`Session` coding callbacks
   (plan autosave / interrupt text). Installed via `agent.session(...)`.

---


## Tracing & cost tracking

OpenTelemetry-style tracing emits one span per turn, LLM call, tool execution, compaction summarisation, and reflection gate. Token usage (input / output / cached / cache hit rate) and estimated USD cost are recorded on every LLM span and aggregated per model.

- **Message content** (opt-in): set ``tracing.capture_messages=True`` to record the full conversation as ``gen_ai.prompt``, ``gen_ai.completion``, and ``gen_ai.tool.call.*`` attributes (JSON-serialised in OpenAI ``{role, content}`` format so Langfuse/Arize render a chat UI). Off by default — messages may contain PII and bloat OTLP payloads. ``tracing.max_message_length`` (default 10 000) truncates tool results.
- **Zero config**: by default no tracer or cost tracker is enabled.
- **Console**: set ``exporter="console"`` to print ``[trace]`` lines to stdout — handy for debugging without a collector.
- **OTLP**: point any OTel-compatible ingest (Tempo, Jaeger, Grafana, **Langfuse**, Honeycomb, Datadog, your own collector) at ``endpoint=`` and you get full traces with zero custom code. Langfuse ships an OTLP endpoint, so no dedicated exporter is required.
- **Cost**: provider-reported cost (when present in ``response_metadata``) wins; otherwise a ``pricing=`` dict estimates from per-1M-token USD rates. Pass your own ``estimate_cost`` callback for full control.

Install the optional tracing extra for the OTLP exporter:
```bash
pip install 'ness-agent[tracing]'
```

### SDK examples

```python
from ness_agent import (
    NessAgent, PromptLayersConfig, TracingConfig, CostTracker, MultiTracer, build_tracer,
)
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o")

# --- minimal: no tracing, no cost tracking --------------------------------
agent = NessAgent(model=model, tools=None, prompt=PromptLayersConfig())

# --- OTLP + cost tracking -------------------------------------------------
agent = NessAgent(
    model=model, tools=None, prompt=PromptLayersConfig(),
    tracing=TracingConfig(
        enabled=True,
        exporter="otlp",
        endpoint="http://localhost:4318/v1/traces",
        pricing={"gpt-4o": (2.50, 10.00, 0.50)},   # (input, output, cache_read_ratio) per 1M tokens
    ),
)

# --- custom cost function -------------------------------------------------
agent = NessAgent(
    model=model, tools=None, prompt=PromptLayersConfig(),
    cost_tracker=CostTracker(estimate_cost=lambda m, u, c, o: 0.001),
)

# --- console exporter (debug) ----------------------------------------------
agent = NessAgent(
    model=model, tools=None, prompt=PromptLayersConfig(),
    tracing=TracingConfig(enabled=True, exporter="console"),
)

# --- custom tracer ---------------------------------------------------------
class MyTracer:
    def start_span(self, name, attributes=None, kind=None):
        class Span:
            def set_attribute(self, k, v): ...
            def add_event(self, n, a=None): ...
            def record_exception(self, e, a=None): ...
            def set_status(self, s, d=None): ...
            def end(self): ...
            def __enter__(self): return self
            def __exit__(self, *a): ...
        return Span()

agent = NessAgent(
    model=model, tools=None, prompt=PromptLayersConfig(),
    tracer=MyTracer(),
)

# --- multiple backends at once ---------------------------------------------
otel_cfg = TracingConfig(enabled=True, exporter="otlp", endpoint="http://localhost:4318/v1/traces")
agent = NessAgent(
    model=model, tools=None, prompt=PromptLayersConfig(),
    tracer=MultiTracer([build_tracer(otel_cfg), MyTracer()]),
    cost_tracker=CostTracker(pricing={"gpt-4o": (2.50, 10.00, 0.50)}),
)
```

### Reading session and live-agent cost after a run

```python
session = agent.session(thread_id="abc")
await session.run("Hello")

session_cost = session.cost_tracker
print(session_cost.for_model("gpt-4o").cost_usd)  # this thread, including restored history
print(session_cost.report())                       # per-session report

live_cost = agent.config.cost_tracker
print(live_cost.total().cost_usd)           # live calls across this agent's sessions
```

### Span names

| Span                        | Attributes                                                                                                                                                                                                        |
|-----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `session.turn`              | `session.thread_id`, `session.mode`, `session.turn_count`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `gen_ai.usage.cost_usd` (when the turn emits a usage event)                                |
| `agent.llm_call`            | `gen_ai.request.model`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `gen_ai.usage.cache_read_tokens`, `gen_ai.usage.cache_hit_rate`, `gen_ai.usage.cost_usd`                                       |
|                             | + `gen_ai.prompt`, `gen_ai.completion` (when ``capture_messages=True``)                                                                                                                                         |
| `tool.<name>`               | `tool.name`, `tool.duration_ms`, `tool.error`, `tool.exit_status`, `tool.args` (when ``capture_tool_args=True``)                                                                                                  |
|                             | + `gen_ai.tool.call.arguments`, `gen_ai.tool.call.result` (when ``capture_messages=True``; result truncated to ``max_message_length``)                                                                          |
| `compaction.summarize`      | trigger, before/after tokens, active suffix count, `session.thread_id`, model, operation                                                                                               |
|                             | + `gen_ai.prompt`, `gen_ai.completion` (when ``capture_messages=True``)                                                                                                                                         |
| `reflection.gate`           | `session.thread_id`, `gen_ai.operation.name`, `reflection.bullets`                                                                                                                                               |
|                             | + `gen_ai.prompt`, `gen_ai.completion` (when ``capture_messages=True``)                                                                                                                                         |

All `gen_ai.*` attribute names follow the OpenTelemetry GenAI semantic conventions so OTel-compatible backends chart token usage natively.

### Langfuse via OTLP

No dedicated exporter is required. Point `TracingConfig.exporter="otlp"` at your Langfuse Public Ingestion endpoint and set `headers={"Authorization": "Basic <base64(public_key:secret_key)>"}` (or whatever auth header your Langfuse project expects). Each LLM call, tool execution, compaction, and reflection appears as a trace in the Langfuse UI with token usage and cost attached.
