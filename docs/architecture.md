# Architecture

Ness Agent is split into a reusable **SDK** and a **coding CLI adapter** (Ness).

| Path | Role |
|------|------|
| `src/ness_agent/` | SDK — LangGraph agent loop, tools (files, search, web, shell, todos, `question`, subagents), permissions, memory, persistence, prompt layers/overlays, MCP, skills, hooks, compaction, reflection, and tracing |
| `src/ness_cli/` | Coding adapter — `build_coding_agent` / `CodingSession`, path resolver, chat model factory, settings/pricing, rollback, and git worktree bootstrap |
| `src/ness_cli/tui/` | Ness TUI entry (`ness` / `ness_cli.tui.main`), streaming, slash commands, and clipboard handling |

**Ness Agent** is the project and Python package. **Ness** is the interactive CLI (`ness` command).

See also: [SDK guide](sdk.md) · [CLI guide](cli.md) · [Configuration](configuration.md)

### Runtime ownership

`NessAgent` represents one project/application runtime. It owns resolved defaults and services that are safe or useful to share across that project's sessions. `NessAgent.session()` takes a selective effective-config snapshot so live threads do not mutate one another.

| Scope | State |
|------|------|
| Agent/project | Thread persistence, memory backend, hooks, skill loader, tool definitions and MCP catalog, persistent permission file/lock, tracer, pricing, and defaults inherited by future sessions |
| Session/thread | Effective main/reflection models and options, temporary permission rules, active MCP tools and binding cache, cost totals, graph/checkpointer, event queue, cancellation token, mode, metadata, and adapter callbacks |
| Turn | User input, optional mode/skill overrides, usage aggregate, streamed events, and active cancellation state |

Live model usage is recorded in the session tracker and propagated once to the agent aggregate. Durable replay updates only the session tracker, so resuming a thread does not look like newly incurred provider spend. Persistent permission choices are shared through `permissions.json`; temporary `session` choices stay on the session-local view. Tool definitions are shared, while deferred MCP activation is session-local.

The Ness CLI keeps one `CodingSession` runtime per live thread. `/new` and `/threads` may therefore leave one turn running while another thread becomes selected. A `/config` model/provider/reasoning change rebuilds the selected runtime and updates agent defaults for future runtimes; already-live sibling threads remain pinned.

### MCP boundary

The SDK's `MCPRuntime` accepts fully resolved stdio or HTTP server specifications and owns connections, session lifecycle, tool discovery, LangChain tool conversion, calls, and structured connection state. It has no project-file, terminal, trust, or credential-storage policy, so it can be embedded in domain-specific or domain-agnostic applications.

The Ness adapter owns `.ness/mcp.json`, Cursor/Claude compatibility, environment interpolation, trust fingerprints, OAuth credential persistence, and CLI presentation. It converts project entries into resolved SDK `MCPServerSpec` values before starting the runtime.

---

## Prompt layers

Ness Agent splits context into four layers to keep prompt caching stable:

1. **L0 harness** (`PromptLayers` / `L0_HARNESS`): NESS identity, universal rules, output format, and tool-calling protocol.
2. **L1 profile** (`build_l1`): persona, stable tool catalog, an always-on one-line skill catalog, `USER.md` preferences, and `.ness/NESS.md` project conventions.
3. **L2 project context**: app-supplied domain/repo structure (`PromptLayersConfig.l2_context`); not auto-loaded by bare `Session`.
4. **L3 working state** (`CodingOverlay` / `render_overlay_delta`): wrapped in `<system-reminder>` tags and appended as an internally tagged tail `HumanMessage`. L3 is retained only in checkpointed model context so later requests preserve the exact wire prefix; it is excluded from the semantic transcript, reflection, and durable CLI events. Fresh user turns receive the full overlay and tool loops receive section deltas. After compaction all historical L3 messages are discarded and one current full overlay is injected. Includes git branch/dirty state, compaction status, todos, session memory, skill hints, and plan/act instructions.

The L1 skill catalog lists every available skill with its path; full skill bodies enter the conversation when the model calls `skill_view` (or `read`s the path). `/skill <name>` stages a one-shot L3 hint for the next turn — it does not inject the body itself (see [Skills in the CLI guide](cli.md#skills)).

---

## Agent modes

Ness Agent binds the **full session tool set in every mode** so the provider prefix cache survives plan ↔ act switches without a graph rebuild. Plan mode is enforced at **runtime**: state-changing tool calls are rejected in the tool executor (the model sees the rejection in state; the CLI does not surface it). **Plan** mode instructions live in the ephemeral L3 `<plan-mode>` overlay; **act** mode has no mode block (L0 + tools + dynamic L3 state only).

- **Act** (Shift+Tab): default execution / build mode — full tool set via L0 and permissions. L3 carries git, todos, compaction, and session memory when present. On the first act turn after a plan→act toggle, L3 prepends a one-shot `MODE SWITCH` note (inside the existing `<system-reminder>`) telling the model to call `todo` first, then address the user's message; it is cleared from state after that single model call so it never repeats.
- **Plan** (Shift+Tab): read-only planning. The agent researches the codebase, may ask clarifying multiple-choice questions via `question` (before any plan prose), then delivers exactly one final plan. Only the terminal plan message is auto-saved under the global `plans/<project-slug>/` directory. Shift+Tab back to act mode to execute.

Plan-mode workflow:

1. **Clarify** — if a decision materially changes the plan, call `question` with MCQ options before drafting (mark the recommended choice; never ask in prose).
2. **Research** — read-only tools first; use `spawn_subagent` only when a few targeted reads are insufficient (see L0 subagents rule).
3. **Plan** — one final message: numbered steps with file paths, verification, and risks; no tool calls in that message.
4. **Act** — Shift+Tab to act/build mode; on the first act turn the agent records todos from the plan via `todo`, then executes (or follows the user's message if they redirect); do not re-plan unless blocked or the user redirects.

Session tool tiers (same set bound in both modes):

- Always-on: `todo`, `question`, `skill_view`
- Core: file (`read`, `write`, `delete`, `edit`), search, web (`web_search`, `fetch_url`), and shell
- Tool discovery: `search_tools`, `add_tools` for loading deferred MCP tools on demand
- Advanced: `spawn_subagent`
- Loaded MCP tools: any `mcp__*` tool activated this session (deferred by default; load via `search_tools`/`add_tools` or `/mcp <server> [tool]`)

---

## Memory

| File | Purpose |
|------|---------|
| `.ness/NESS.md` | Durable project conventions (CLAUDE.md / AGENTS.md style). Human-authored via `/memory add`, manual edit, or agent edit when asked; optional LLM draft via `/memory create`. `/init` creates an empty file. Loaded into L1. May inline existing `@AGENTS.md` / `@CLAUDE.md` files (see below). |
| Global `USER.md` | Cross-repo user preferences (see [Configuration](configuration.md)). Human-authored via `/user`; loaded into L1. |
| `.ness/runtime/sessions/mem_<thread_id>.md` | Episodic per-session scratchpad. Current thread bullets load into L3. Maintained by the reflection gate. |

Reflection runs in the background when new messages since the last run exceed `REFLECTION_TOKEN_RATIO` of the usable context budget. An optional final pass at session exit is controlled by `SESSION_END_REFLECTION` (default off). It uses structured output (via `REFLECTION_MODEL_NAME`) to append up to 2 bullets per run to `.ness/runtime/sessions/mem_<thread_id>.md`. Bullets appear in the L3 system-reminder overlay on subsequent turns. `NESS.md` remains human-authored; the CLI warns at startup when its resolved size exceeds 20,000 characters.

### NESS.md includes

A standalone line in `NESS.md` of the form `@<path>` inlines that file's contents in place at runtime, so a repo that already ships an `AGENTS.md` or `CLAUDE.md` is picked up without duplication:

```markdown
@AGENTS.md
@CLAUDE.md

<extra Ness Agent-specific conventions here>
```

Includes resolve relative to the project root, reject paths that escape it, skip missing files (leaving a `# (missing include: ...)` marker), guard against cycles, and are size-capped. Changes to an included file invalidate the L1 prompt cache. The CLI also warns at startup when the assembled static prefix (L0+L1+L2) exceeds ~7,000 tokens.

---

## Compaction

Compaction is a cache-safe fork of the main conversation. At every boundary before a model call, Ness Agent measures the stable system prefix plus the exact canonical model context. When summary compaction is due it invokes the last successfully completed bound main-model request, with the same system message, provider session, tool definitions, and native message history, then appends one human summary instruction. Failed model/schema attempts never replace that parent binding. There is no tool-output rewriting and no tool-less auxiliary summarizer.

| Pressure | Action |
|----------|--------|
| < 70% | No warning |
| 70-80% | Warn that summary compaction is approaching |
| >= 80% | Summarize completed history |

Compaction also runs earlier when necessary to preserve `COMPACTION_BUFFER_TOKENS` for the instruction and capped summary output. The latest unanswered user turn and its complete assistant/tool trajectory remain verbatim; only completed history is summarized. Old L3 reminders are visible to the cache-safe fork but explicitly excluded from summary semantics. The replacement branch contains one human `<compacted-history>` message, the active suffix, and one newly rendered current L3 reminder.

Image-bearing user messages remain structured in live and replayed canonical history after they are answered. They are removed only when their completed turn is replaced by summary compaction (or when vision is explicitly disabled).

Compaction is a separate graph node between `START`/`tools` and `agent`, so it never runs during tool execution and its state checkpoints before the next model request. Summary failures preserve the original history and may continue below the safety boundary; at the boundary the turn stops rather than using a lossy fallback. `/compact` forces this process at the next model boundary.
