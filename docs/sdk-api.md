# SDK API reference

This reference covers the symbols exported by `ness_agent.__all__` in the 0.2.2 source tree. It is intentionally an API map rather than a second tutorial: start with the [SDK guide](sdk.md) for the shortest working example, then use this page when choosing a seam to own.

> **0.x experimental.** Names and signatures below describe the current public surface; pin a version before relying on it in production.

## Build an agent and run a thread

### `NessAgent`

```python
NessAgent(*, model, prompt, tools: Sequence[BaseTool] | None = None, **agent_spec_fields)
NessAgent.from_spec(spec: AgentSpec) -> NessAgent
agent.session(*, thread_id: str, mode: str | None = None,
              metadata: Mapping[str, Any] | None = None,
              git_available: bool | None = None, vision: bool | None = None,
              on_plan_turn: PlanTurnHandler | None = None,
              on_interrupt: InterruptHandler | None = None,
              model: BaseChatModel | None = None,
              reflection_model: BaseChatModel | None | object = _UNSET) -> Session
agent.configure_default_models(*, model: BaseChatModel,
                               reflection_model: BaseChatModel | None,
                               context_window: int | None) -> None
agent.new_thread_id(prefix: str = "session") -> str
```

The top-level SDK entry point owns project services and resolved defaults, then creates an isolated effective configuration for each `Session`. `model` is a LangChain `BaseChatModel`; `prompt` accepts `PromptLayers`, `PromptLayersConfig`, or a mapping. With `tools=None`, the SDK resolves its built-in tool set. `agent.config` exposes project defaults and the live aggregate cost tracker; `session.config` exposes the effective session fork.

`session()` accepts optional per-thread metadata for the L3 overlay and effective main/reflection model overrides. `vision=True` sends supplied image URLs; `False` drops them to text and emits a warning event; `None` leaves content shape untouched. The two callback hooks live on a session, not on the shared agent, so concurrent threads do not overwrite each other. `configure_default_models()` changes inheritance for sessions created later; it does not rebuild existing sessions.

```python
agent = NessAgent(model=model, prompt=PromptLayersConfig())
session = agent.session(thread_id=agent.new_thread_id())
result = await session.run("inspect the deployment flow")
print(result.assistant_message)
```

Source: `src/ness_agent/agent.py`.

### `AgentSpec` and `NessAgentConfig`

```python
AgentSpec(*, model: BaseChatModel,
          prompt: PromptLayers | PromptLayersConfig | Mapping[str, Any],
          tools: Sequence[BaseTool] | None = None,
          reflection_model: BaseChatModel | None = None,
          options: NessAgentOptions = NessAgentOptions(),
          overlay: OverlayProvider | None = None,
          memory: MemoryConfig = MemoryConfig(),
          memory_store: MemoryBackend | None = None,
          modes: ModeConfig | None = None,
          subagents: SubagentConfig | None = None,
          aux_prompts: AuxPrompts = AuxPrompts(),
          skills_dir: Path | None = None,
          skills_dirs: Sequence[Path] | None = None,
          hooks_config: Path | None = None,
          hooks: Sequence[Hook] | None = None,
          approval_handler: ApprovalHandler | None = None,
          question_handler: QuestionHandler | None = None,
          checkpoint_factory: Callable[[], BaseCheckpointSaver] | None = None,
          tracing: TracingConfig = TracingConfig(),
          cost_tracker: CostTracker | None = None, tracer: Tracer | None = None)

NessAgentConfig.resolve(spec: AgentSpec) -> NessAgentConfig
```

`AgentSpec` is the declarative construction surface. Supply it to `NessAgent.from_spec()` when an application wants to assemble all dependencies in one place. `NessAgentConfig` is the fully wired result: prompt layers, normalized tools, stores, permission and hook runners, skill loader, cost tracker, and tracer. Prefer the spec or `NessAgent(...)` for normal construction; use `resolve()` only when the application deliberately needs the fully resolved dependencies.

`skills_dir` and `skills_dirs` are mutually exclusive. Pass exactly the roots you want scanned (nested `category/skill/SKILL.md` layouts are supported); both `None` disables skills. Use `merge_skill_dirs()` / `default_skill_search_dirs()` to opt into well-known agent skill roots. Compaction always uses the main bound `model` — there is no separate `compaction_model`.

With an absent overlay, resolution installs `CodingOverlay`; use `NoOverlay()` to opt out. An injected memory backend must subclass `MemoryBackend`, and an injected approval handler must subclass `ApprovalHandler`.

### `Session`

```python
await session.run(message: str, *, images: Sequence[str] | None = None,
                  active_skills: Sequence[str] | None = None,
                  mode: str | None = None) -> RunResult

async for event in session.stream(message: str, *, images=None,
                                  active_skills=None, mode=None): ...

session.config -> NessAgentConfig
session.cost_tracker -> CostTracker
session.configure_models(*, model: BaseChatModel,
                         reflection_model: BaseChatModel | None,
                         context_window: int | None,
                         vision: bool | None) -> None
```

`run()` collects one complete turn. It returns assistant text, an aggregate `usage_total` over every model call in the turn, the current todos, and every intermediate event. `stream()` yields those `SessionEvent` records as the graph advances. `mode=` is a one-turn override; otherwise the session’s current `act` or `plan` mode is used.

Each session owns a selective shallow fork of the agent configuration. Mutable runtime state—models and options, temporary permission rules, active MCP tools, and cost totals—is isolated. Project services such as thread persistence, hooks, memory, and the underlying permission/tool catalogs remain shared. `session.config` exposes the effective session view; `agent.config` remains the defaults inherited by future sessions plus the live agent-level cost aggregate.

Important control and inspection methods:

| Method | Purpose |
| --- | --- |
| `set_mode(mode)` / `toggle_mode() -> str` | Change between `"act"` and `"plan"`; a plan → act switch schedules the context checkpoint. |
| `set_name(name) -> bool` | Persist a 1–80 character display name for this session; returns `False` when thread autosave is disabled. |
| `bootstrap(messages)` | Seed prior messages into the next turn once; use for resume or rollback replay. |
| `cancel()` / `is_cancelled()` | Request and inspect cooperative cancellation of the active stream. |
| `request_compact()` | Request compaction on the next turn. |
| `configure_models(...)` | Replace this session's effective main/reflection models and context settings, then rebuild its graph. |
| `active_skills(names)` / `stage_skills(names)` | Replace or append one-shot skills for the next turn. |
| `get_state()`, `get_messages()`, `get_todos()` | Async snapshots of the checkpointed state. |
| `preview_context(mode=None) -> ContextPreview` | Assemble L0–L3 for debugging without running the model. |
| `run_reflection()` | Immediately reflect on the unreflected conversation tail and return a `ReflectionResult`, regardless of automatic-reflection settings. |
| `refresh_context_snapshot()` / `finalize_reflection()` | Refresh pressure metrics or run end-of-session reflection when enabled. |
| `rebuild_graph()` / `reset_checkpointer()` | Recompile the graph; the latter swaps in a fresh checkpointer before replay. |

Source: `src/ness_agent/session.py`.

### Turn records and handler types

| Export | Shape / contract |
| --- | --- |
| `UsageEvent` | `model`, input/uncached/cached/output token counts, `cost_usd`, and `calls`. Represents usage from a model call. |
| `aggregate_usage(events) -> UsageEvent | None` | Sums a turn’s usage events; model becomes `"*"` when they differ and cost is `None` if no event reported one. |
| `SessionEvent` | Frozen `{kind, data}` record. Kinds include assistant deltas/final output, tool start/end, usage, approvals, questions, compaction, errors, warnings, interruptions, and plan turns. |
| `RunResult` | Frozen result from `run()`: `assistant_message`, `usage_total` (aggregate of every model call in the turn), `todos`, and `events`. The former single-call `usage` field was removed in 0.2.0. |
| `ContextPreview` | Frozen debug snapshot: stable `system_message`, raw overlay, named `overlay_sections`, wrapped reminder, and active `mode`. |
| `ReflectionResult` | Frozen result from `run_reflection()`: whether memory changed, new bullets, any error, and the reflected message index. |
| `ApprovalHandler` | Abstract async callable `(tool: str, args: dict) -> str`; return `yes`, `no`, `always`, `session`, or `never`. |
| `QuestionHandler` | `Callable[[list[dict]], Awaitable[list[dict]]]` for model-originated choice questions. |
| `PlanTurnHandler` | `Callable[[str], None]`, called after a successful plan-mode turn. |
| `InterruptHandler` | `Callable[[str], str]`, may replace partial assistant text surfaced after interruption. |

Source: `src/ness_agent/types.py`.

## Runtime options and modes

All are dataclasses, passed through `AgentSpec` or the `NessAgent` keyword construction path.

| Export | Key fields and use |
| --- | --- |
| `NessAgentOptions` | `context_window`, `compaction_token_budget`, `compaction_buffer_tokens`, `compaction_summary_max_tokens`, `enable_approval`, `yolo_mode`, `auto_save_threads`, reflection settings, `format_on_write`, `exa_api_key`, `project_root`, `ness_dir`, interruption marker, and `recursion_limit`. These are runtime knobs, not prompt text. |
| `MemoryConfig` | `disabled`, plus optional paths for project, user, and session memory. Used when no custom `MemoryBackend` is injected. |
| `ModeConfig` | `default` mode, optional `plans_dir`, custom plan/act instruction templates, and `plan_mode_readonly`. |
| `SubagentConfig` | Optional subagent prompt template, `max_parallel`, default tool names, and timeout. It supports the `spawn_subagent` tool. |
| `PermissionRules` | Lists of allowed, denied, and approval-required rule patterns; defaults to `ask=["*"]`. |

Source: `src/ness_agent/options.py`. For the operator-level precedence and `.ness/` layout, see [Configuration](configuration.md).

## Prompt layers and L3 overlays

### L0–L2: `PromptLayersConfig`, `PromptLayers`, `AuxPrompts`

```python
PromptLayersConfig(*, l0=L0_HARNESS, persona="...",
                   include_user_memory=True, include_project_memory=True,
                   include_skill_catalog=True, l2_context=None,
                   l2_header="PROJECT CONTEXT", include_git_line=True)
PromptLayers(config)
PromptLayers.from_dict(mapping) -> PromptLayers
PromptLayersConfig.from_dict(mapping) -> PromptLayersConfig
```

An instruction source may be a string, a path, or a zero-argument callable returning text. `PromptLayers` provides `build_l0()`, `build_l1(...)`, `build_l2(...)`, and `build_stable_prefix(...)`. The last one assembles and caches the stable L0–L2 system prefix from active tools, user/project memory, skill catalog, Git availability, metadata, and deferred MCP summary. L3 is intentionally outside this cache.

`AuxPrompts` is a dataclass of optional instruction sources for the compaction, reflection, subagent, thread-summary, and initial-memory auxiliary calls. Set a field to `None` to disable that auxiliary call’s template.

Source: `src/ness_agent/context/layers.py`. See [Architecture → Prompt layers](architecture.md#prompt-layers) for the caching model.

### L3: `OverlayContext`, `OverlayProvider`, `CodingOverlay`, `NoOverlay`

```python
class OverlayProvider:
    def sections(self, state: AgentState, ctx: OverlayContext) -> dict[str, str]: ...

CodingOverlay(*, plans_dir=".ness/plans/",
              plan_mode_template: str | None = None,
              act_mode_template: str | None = None)
NoOverlay()
render_overlay_delta(sections, previous, *, skip=frozenset()) -> str
wrap_system_reminder(body: str) -> str
```

`OverlayContext` is the immutable per-turn input to a provider: thread id, mode, current messages/todos, session memory, compaction and mode-switch notes, metadata, Git snapshot, requested skills, and accumulated loaded skills. A custom provider must subclass `OverlayProvider`, return stable section names from `sections()`, and leave empty sections falsy. Stable names let the harness send only changed L3 sections during a tool loop.

`CodingOverlay` is the default provider; it renders plan/act instructions, Git state, compaction status, todos, session memory, and skill information. `NoOverlay` always renders an empty mapping. `render_overlay_delta()` compares section dictionaries, and `wrap_system_reminder()` surrounds non-empty L3 content with the SDK’s system-reminder tags.

`AgentState` is the checkpointed `TypedDict` used by the graph. Its public keys include `messages`, `todos`, `mode`, approval state, requested/loaded skills, reflection and compaction state, `force_compact`, input tokens, and `mode_switch`. Treat it as graph state, not a long-lived application schema.

Sources: `src/ness_agent/context/overlay.py`, `context/coding_overlay.py`, and `graph/state.py`.

## Tools, permissions, hooks, and skills

### `ToolRegistry` and `coding_tools`

```python
ToolRegistry(tools: Iterable[BaseTool] | None = None, *, include: Iterable[str] | None = None)
registry.fork_for_session() -> ToolRegistry
coding_tools(*, include: list[str] | None = None) -> ToolRegistry
```

The registry owns known tools and the currently bound active set. Core reads are `active_tools`, `tool_map()`, `tool_names()`, `all_tools()`, and `deferred_tool_names()`. Call `bind_model(model)` to return a tool-bound chat model, and `sync()` after a structural change. `register_dynamic(tools)` adds dynamic tools as known but inactive; `activate_mcp(names)` and `deactivate_mcp(names)` return `(changed, unknown)` lists and adjust the active MCP set.

`fork_for_session()` creates a registry view that shares tool definitions and the deferred MCP catalog while keeping its active MCP names, include filter, binding cache, and local generation independent. Structural catalog additions remain visible to all session views; activating a deferred MCP tool affects only that session.

`set_mcp_catalog()` and `deferred_mcp_summary()` maintain the lightweight deferred-MCP prompt catalog. `is_destructive(name, args)` and `is_read_only(name, args)` expose the registry’s policy classification. `coding_tools()` is the small convenience factory for name-selected SDK tools.

The default tool list includes file read/write/delete/edit/glob, search, web fetch/search, shell, todos, tool discovery, subagents, questions, and skill viewing. It is not an API guarantee for every named tool; configure an explicit tool sequence when a host application needs a narrower contract.

### `PermissionStore`

```python
PermissionStore(*, ness_dir: Path = Path(".ness"), project_root: Path | None = None)
store.fork_for_session() -> PermissionStore
```

The file-backed policy store validates paths under the project root and resolves tool calls to `allow`, `deny`, or `ask` decisions. The main public operations are `check(tool, args)`, `check_with_rule(tool, args)`, `pattern_key(tool, args)`, `default_rule_for(tool, args)`, `persist_rule(...)`, `remove_rule(bucket, index)`, `list_rules()`, and `clear_session_rules()`. Use it when embedding an operator-approved policy rather than bypassing the tool executor.

`fork_for_session()` shares the persistent `permissions.json` path and synchronization lock while copying temporary allow/deny rules. A `session` decision therefore affects one session; `always`/`never` decisions are atomically persisted and become visible to every session on the project.

### `Hook` and `HookRunner`

```python
Hook(event: str, matcher: str = "*", command: str | None = None,
     handler: Callable[[dict], tuple[bool, str]] | None = None,
     blocking: bool = True, timeout: int = 30)
HookRunner(hooks_file: Path | None = None, *, project_root: Path = Path.cwd(),
           hooks: Sequence[Hook] | None = None)
```

Hooks run on `preToolUse` and `postToolUse`. A matcher selects a tool; a callable handler takes precedence over a shell command. `HookRunner.register()`, `clear_registered()`, and `load()` manage definitions. `run(event, payload) -> tuple[bool, str]` returns the combined output and fails a blocking pre-tool hook closed. File hooks are read from the configured JSON file.

### `SkillLoader`, `merge_skill_dirs`, `default_skill_search_dirs`

```python
SkillLoader(skills_dir: Path | None = None, *, skills_dirs: Sequence[Path] | None = None)
loader.load() -> dict[str, dict[str, Any]]
loader.render_catalog(skills) -> str

default_skill_search_dirs(project_root: Path, *,
                          project_rels: Sequence[str] | None = None,
                          global_rels: Sequence[str] | None = None) -> list[Path]
merge_skill_dirs(project_root: Path, skills_dir: Path, *,
                 project_rels: Sequence[str] | None = None,
                 global_rels: Sequence[str] | None = None) -> list[Path]
```

Loads `SKILL.md` files from the configured roots and returns parsed metadata/body records. Prefer `skills_dirs=` for an explicit exhaustive list; `skills_dir=` is the single-root shorthand. Earlier roots win on name collisions. `render_catalog()` creates the compact stable-prefix catalog; full skill bodies remain on demand.

`default_skill_search_dirs()` returns the well-known project-local and user-global agent skill roots (it does not include `.ness/skills`). `merge_skill_dirs()` puts your directory first, then those roots, deduped by resolved path. Pass `project_rels=` / `global_rels=` to restrict either set. The SDK never scans these roots unless the host passes them via `AgentSpec.skills_dirs`.

Source: `src/ness_agent/skills.py`.

## MCP runtime

### `MCPRuntime`, `MCPServerSpec`, `MCPServerState`

```python
MCPServerSpec(*, name: str, transport: Literal["stdio", "http"],
              description: str = "", startup_timeout: float = 20.0,
              command: str | None = None, args: tuple[str, ...] = (),
              cwd: Path | None = None, env: tuple[tuple[str, str], ...] = (),
              url: str | None = None, headers: tuple[tuple[str, str], ...] = (),
              redactions: tuple[str, ...] = ())

MCPRuntime(*, http_auth_factory: HTTPAuthFactory | None = None)
await runtime.start(specs: Iterable[MCPServerSpec]) -> None
await runtime.start_server(spec: MCPServerSpec) -> None
await runtime.stop() -> None
runtime.tools: dict[str, StructuredTool]
runtime.states: dict[str, MCPServerState]
```

`MCPRuntime` connects fully resolved server specifications and exposes discovered tools as LangChain tools. It owns connections, session lifecycle, tool discovery, calls, and structured `MCPServerState` — not project files, trust prompts, terminal output, or credential storage. Those stay with the embedding application (the Ness CLI is one such adapter).

Provide an `HTTPAuthFactory` when HTTP servers need per-spec authentication. Failures are isolated per server. Pass `list(runtime.tools.values())` into `NessAgent(..., tools=...)` or any LangChain-compatible host. The older mixed `ness_agent.mcp.MCPManager` was removed in 0.2.0.

Source: `src/ness_agent/mcp.py`. See [SDK guide → MCP](sdk.md#mcp-in-an-sdk-application).

## Cache-safe summarization

### `summarize`

```python
async def summarize(messages: Sequence[BaseMessage], model: Any, *,
                    instruction: str = ..., max_output_tokens: int = 4096) -> str
```

Summarize an exact parent request with a cache-safe human tail. `model` must be the same already-bound runnable used by the parent conversation — identical system messages and tool definitions are what allow the provider to reuse the cached prefix. Automatic compaction uses this path with the main bound model.

Source: `src/ness_agent/compaction.py`.

## Memory and durable history

### `MemoryBackend` and `MemoryStore`

```python
MemoryStore(config: MemoryConfig, ness_dir: Path | None = None, *,
            project_root: Path | None = None)
```

`MemoryBackend` is the required abstract contract for a custom memory provider. Implement `disabled`, project/user load/append/write methods, session load/append/read/write methods, and `check_health()`. `MemoryStore` is the filesystem implementation: project memory defaults to `NESS.md`, user memory to `USER.md`, and episodic session memory to `runtime/sessions/` under its Ness root. Its project loader supports standalone `@path` includes constrained to the project root.

### `ThreadStore`

```python
ThreadStore(threads_dir: Path | None = None, *, auto_save: bool = True,
            default_model: str = "")
```

SQLite persistence for named session-thread events, subagent records, and rollback checkpoints. Key operations are `set_thread_name()`, `thread_exists()`, `append_event()`, `list_threads()`, `load_thread_events()` (or `load_thread_events_since()`), `copy_thread_prefix()`, `archive_thread()`, `register_subagent()`, `complete_subagent()`, `list_subagents()`, `save_checkpoint()`, `add_modified_path()`, `get_checkpoint()`, `list_user_turns()`, and `truncate_after()`. `list_threads()` returns both the explicit `name` and generated `summary`. When `auto_save=False`, writes no-op. The SDK excludes ordinary event rows for subagent thread ids and rolls their usage into the parent.

The current `threads` schema is intentionally not migrated from older releases. Constructing `ThreadStore` against a database without the required `name` column raises an actionable compatibility error without modifying or deleting the database.

Sources: `src/ness_agent/memory.py` and `persistence.py`.

## Usage, pricing, and tracing

### `TokenUsage`, `CostTracker`, and `PricingDict`

```python
PricingDict = dict[str, tuple[float, float, float]]
CostTracker(pricing: PricingDict | None = None,
            estimate_cost: Callable[[str, int, int, int], float | None] | None = None)
tracker.fork_for_session() -> CostTracker
tracker.add(usage, model_name=None, response_metadata=None) -> TokenUsage | None
tracker.restore(usage, model_name=None, response_metadata=None) -> TokenUsage | None
```

`TokenUsage` is a slots dataclass with model, input/uncached/cached/output/total tokens, `cost_usd`, `cost_source`, cache-hit rate, and calls. `as_dict()` serializes it. `CostTracker.add()` ingests provider usage and prefers provider-reported cost, then a supplied estimator, then a matching substring key in `pricing`. Pricing triples are USD per million tokens: `(input, output, cache_read_ratio)`.

`CostTracker.fork_for_session()` returns an empty tracker with shared pricing. New usage also rolls into the parent tracker; `restore()` loads durable historical usage locally without changing that live aggregate. Read per-thread totals from `session.cost_tracker` and current-process agent totals from `agent.config.cost_tracker`.

Use `for_model(model)`, `total()`, `models()`, or the scalar aggregate properties (`input_tokens`, `output_tokens`, `total_tokens`, `calls`, `cost_usd`, `cache_hit_rate`, `total_cost_usd`) to inspect accumulated data. `report()` returns a text report.

### `TracingConfig`, `Tracer`, `Span`, and implementations

```python
TracingConfig(enabled=False, exporter="none", endpoint=None, headers=None,
              service_name="ness-agent", resource_attrs={},
              capture_tool_args=False, capture_messages=False,
              max_message_length=10000, pricing=None)

Tracer.start_span(name, attributes=None, kind=None) -> Span
build_tracer(config: TracingConfig | None = None) -> Tracer
```

`TracingConfig` controls optional `otlp`, `console`, or `none` exporting, service metadata, sensitive-data capture, truncation length, and pricing. Keep `capture_tool_args` and `capture_messages` disabled unless the destination is appropriate for potentially sensitive tool arguments and conversation content.

`Span` is the minimal context-manager protocol: set attributes, add events, record exceptions, set `OK`/`ERROR` status, and end. `Tracer` is the backend protocol. `NoopTracer`/`NoopSpan` are the zero-overhead defaults. `InMemorySpan(name, attrs=None)` records attributes, events, status, and duration for tests or console tracing. `MultiTracer(tracers)` and `MultiSpan(spans)` fan operations out to several backends while keeping one parent context. `build_tracer()` selects the configured tracer and returns a no-op tracer when tracing is disabled or exporter is `none`.

Sources: `src/ness_agent/tracing/config.py`, `cost.py`, and `tracer.py`.

## Small utilities and workspace context

```python
message_to_text(message: Any) -> str
git_worktree_summary(cwd: Path = Path.cwd()) -> str
get_project_context(max_files: int = 80) -> str
setup_ness_structure(ness_dir: Path) -> list[str]
```

`message_to_text()` extracts usable text from LangChain-style message content, including content blocks. `git_worktree_summary()` returns a compact branch/dirty-path snapshot for an overlay (empty outside a usable Git repository). `get_project_context()` renders a bounded project tree plus manifest snippets. `setup_ness_structure()` creates the project-local Ness layout and default files, returning the paths it created. These helpers are useful when an embedding host wants the same project context primitives as the coding adapter, without importing CLI internals.

Sources: `src/ness_agent/utils.py` and `workspace/`.
