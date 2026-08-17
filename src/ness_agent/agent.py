from __future__ import annotations
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
import uuid
from pathlib import Path
from typing import Any, TYPE_CHECKING
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver

from ness_agent.options import (
    NessAgentOptions, MemoryConfig, ModeConfig, SubagentConfig,
)
from ness_agent.context.layers import PromptLayers, PromptLayersConfig, AuxPrompts
from ness_agent.context.overlay import OverlayProvider
from ness_agent.types import ApprovalHandler, QuestionHandler
from ness_agent.memory import MemoryBackend, MemoryStore
from ness_agent.persistence import ThreadStore
from ness_agent.permissions import PermissionStore
from ness_agent.hooks import Hook, HookRunner
from ness_agent.skills import SkillLoader
from ness_agent.tools import BUILTIN_TOOLS, ToolRegistry
from ness_agent.utils import normalize_tool
from ness_agent.tracing.cost import CostTracker
from ness_agent.tracing.config import TracingConfig
from ness_agent.tracing.tracer import NoopTracer, Tracer, build_tracer
if TYPE_CHECKING:
    from ness_agent.session import Session
    from ness_agent.types import (
        InterruptHandler,
        PlanTurnHandler,
    )

_UNSET = object()


def _require_instance(name: str, value: object, expected: type) -> None:
    if value is not None and not isinstance(value, expected):
        raise TypeError(
            f"{name} must be an instance of {expected.__name__}, "
            f"got {type(value).__qualname__}"
        )

@dataclass(kw_only=True)
class AgentSpec:
    """User-facing agent configuration.

    .. highlight:: python

    Pass directly to :meth:`NessAgent.from_spec` or let
    ``NessAgent(model=..., prompt=..., **kwargs)`` build one internally.
    Every field may be overridden individually; backends are lazily
    resolved into a :class:`NessAgentConfig` by ``NessAgentConfig.resolve``.

    **Required fields**

    ``model``
        The primary chat model that drives the agent loop.
    ``prompt``
        L0–L2 prompt: a :class:`~ness_agent.context.layers.PromptLayers`,
        a :class:`~ness_agent.context.layers.PromptLayersConfig`, or a
        plain dict (matching keys extracted, others ignored).

    **Tools** (optional — defaults to all SDK built-in tools)

    ``tools``
        ``None`` or a sequence of ``BaseTool``, plain callables
        (auto-wrapped), or strings naming built-in tools (e.g.
        ``"read"``, ``"grep"``, ``"shell"``).

    **Optional auxiliary models**

    ``reflection_model``
        Model for background reflection. Falls back to ``model`` when ``None``.

    **Behaviours**

    ``options``
        :class:`NessAgentOptions` — compaction budget, context window,
        approval flag, etc.
    ``overlay``
        ``None`` → default :class:`~ness_agent.context.coding_overlay.CodingOverlay`
        (plan/act, git, todos, compaction, session memory).
        Pass :class:`~ness_agent.context.coding_overlay.NoOverlay` for no L3.
        Custom overlays must subclass
        :class:`~ness_agent.context.overlay.OverlayProvider`.
    ``memory``
        :class:`MemoryConfig` — project / user / session memory paths
        (used when ``memory_store`` is not injected).
    ``memory_store``
        Optional :class:`~ness_agent.memory.MemoryBackend` subclass
        instance. When set, skips constructing the default
        :class:`MemoryStore`.
    ``modes``
        :class:`ModeConfig` for plan/act mode (optional; toggle still works
        with default instruction texts when ``None``).
    ``subagents``
        :class:`SubagentConfig` for the ``spawn_subagent`` tool.
    ``aux_prompts``
        :class:`AuxPrompts` — templates for auxiliary LLM calls.

    **Filesystem paths**

    ``skills_dir``
        Single skills directory (CLI default ``.ness/skills/``), or
        ``None`` to disable skills entirely. Exactly this directory is
        scanned (including nested category layouts) — the SDK never adds
        other roots implicitly. Mutually exclusive with ``skills_dirs``.
    ``skills_dirs``
        Explicit, exhaustive list of skill root directories to scan, or
        ``None`` to disable skills entirely. To also load the well-known
        agent skill roots (``.agents/skills``, ``.claude/skills``,
        ``.codex/skills``, ``.cursor/skills``, and their ``~/``
        equivalents), opt in via
        :func:`~ness_agent.skills.merge_skill_dirs` or
        :func:`~ness_agent.skills.default_skill_search_dirs`. Earlier
        roots win on skill-name collisions.
    ``hooks_config``
        Path to ``hooks.json`` for pre/post tool-use hooks. When ``None``,
        defaults to ``{ness_dir}/hooks.json``.
    ``hooks``
        Optional in-memory :class:`~ness_agent.hooks.Hook` list seeded into
        the runner at resolve (combined with the JSON file).

    **Runtime hooks**

    ``approval_handler``, ``question_handler``
        Callbacks for destructive-tool approval and user questions.

    **Integrations**

    ``checkpoint_factory``
        Callable returning a langgraph ``BaseCheckpointSaver``
        (``None`` → in-memory ``MemorySaver``).
    ``cost_tracker``
        Pricing-aware :class:`CostTracker` (optional). When ``None``,
        resolved from ``tracing.pricing``.
    ``tracer``
        OpenTelemetry-compatible :class:`Tracer` (optional). When
        ``None``, resolved via :func:`build_tracer` from ``tracing``.
    ``tracing``
        :class:`TracingConfig` — toggles tracing, exporter selection,
        capture options, and per-model pricing for cost estimation.
    """

    # required
    model: BaseChatModel
    prompt: PromptLayers | PromptLayersConfig | Mapping[str, Any]

    # tools — defaults to all SDK tools (BUILTIN_TOOLS) when None
    tools: Sequence[BaseTool] | None = None

    # optional auxiliary models
    reflection_model: BaseChatModel | None = None

    # behaviours
    options: NessAgentOptions = field(default_factory=NessAgentOptions)
    overlay: OverlayProvider | None = None
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    memory_store: MemoryBackend | None = None
    modes: ModeConfig | None = None
    subagents: SubagentConfig | None = None
    # Prompt templates for auxiliary model calls
    aux_prompts: AuxPrompts = field(default_factory=AuxPrompts)

    # specs
    skills_dir: Path | None = None
    skills_dirs: Sequence[Path] | None = None
    hooks_config: Path | None = None
    hooks: Sequence[Hook] | None = None

    # runtime hooks
    approval_handler: ApprovalHandler | None = None
    question_handler: QuestionHandler | None = None

    # integrations
    checkpoint_factory: Callable[[], BaseCheckpointSaver] | None = None
    tracing: TracingConfig = field(default_factory=TracingConfig)
    cost_tracker: CostTracker | None = None
    tracer: Tracer | None = None


@dataclass(kw_only=True, eq=False)
class NessAgentConfig:
    """Fully resolved agent config including backends. Prefer AgentSpec / NessAgent(...)."""

    # required
    model: BaseChatModel
    tools: Sequence[BaseTool]
    prompts: PromptLayers

    # optional auxiliary models
    reflection_model: BaseChatModel | None = None

    # behaviors
    options: NessAgentOptions = field(default_factory=NessAgentOptions)
    overlay: OverlayProvider | None = None
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    modes: ModeConfig | None = None
    subagents: SubagentConfig | None = None
    aux_prompts: AuxPrompts = field(default_factory=AuxPrompts)

    # specs
    skills_dir: Path | None = None
    skills_dirs: list[Path] | None = None
    hooks_config: Path | None = None

    # runtime hooks
    approval_handler: ApprovalHandler | None = None
    question_handler: QuestionHandler | None = None

    # integrations
    checkpoint_factory: Callable[[], BaseCheckpointSaver] | None = None
    tracing: TracingConfig = field(default_factory=TracingConfig)

    # agent backends (resolved)
    memory_store: MemoryBackend | None = None
    thread_store: ThreadStore | None = None
    permission_store: PermissionStore | None = None
    hook_runner: HookRunner | None = None
    skill_loader: SkillLoader | None = None
    tool_registry: ToolRegistry | None = None
    cost_tracker: CostTracker
    tracer: Tracer

    def fork_for_session(self) -> "NessAgentConfig":
        """Create an effective config with session-local mutable services.

        Project services remain shared by identity. The returned object has
        the same shape as the resolved agent config so graph construction does
        not need a parallel configuration hierarchy.

        
        """
        return replace(
            self,
            options=replace(self.options),  # mutates the options for a single session
            permission_store=self.permission_store.fork_for_session(), # new permission store; still shares the permission.json + lock
            tool_registry=self.tool_registry.fork_for_session(),  # own MCP activation / include filter; shares tool catalog
            cost_tracker=self.cost_tracker.fork_for_session(),   # Empty per session totals; shares pricing; rolls usage upto the agent aggregate.
        )

    @classmethod
    def resolve(cls, spec: AgentSpec) -> "NessAgentConfig":
        """Resolve a user-facing :class:`AgentSpec` into a ready-to-run config.

        This is called internally by ``NessAgent.__init__`` and
        ``NessAgent.from_spec``. It:

        1. Normalises the ``prompt`` field (dict → ``PromptLayersConfig``).
        2. Resolves ``project_root`` and ``ness_dir`` paths without
           mutating the caller's ``NessAgentOptions`` instance.
        3. Normalises every tool via :func:`normalize_tool` (BaseTool,
           callable, or name string).
        4. Instantiates backend stores (``MemoryStore``, ``ThreadStore``,
           ``PermissionStore``, etc.) so the config is fully wired.
        5. Installs a default :class:`CodingOverlay` when ``overlay`` is
           ``None``, picking up plan/act templates from ``spec.modes`` if
           provided.
        """
        prompt = spec.prompt
        if isinstance(prompt, Mapping):
            prompts = PromptLayers.from_dict(prompt)
        elif isinstance(prompt, PromptLayersConfig):
            prompts = PromptLayers(prompt)
        else:
            prompts = prompt

        # Resolve paths without mutating the caller's options instance —
        # reusing one NessAgentOptions across two NessAgent builds in
        # different roots otherwise "sticks" the first agent's paths.
        raw_options = spec.options
        project_root = (raw_options.project_root or Path.cwd()).resolve()
        ness_dir = (raw_options.ness_dir or (project_root / ".ness")).resolve()
        if raw_options.project_root is None or raw_options.ness_dir is None:
            import dataclasses as _dc
            overrides = {}
            if raw_options.project_root is None:
                overrides["project_root"] = project_root
            if raw_options.ness_dir is None:
                overrides["ness_dir"] = ness_dir
            options = _dc.replace(raw_options, **overrides)
        else:
            options = raw_options

        model_name = (
            getattr(spec.model, "model", None)
            or getattr(spec.model, "model_name", None)
            or ""
        )

        resolved_tools = [
            normalize_tool(t)
            for t in (spec.tools if spec.tools is not None else BUILTIN_TOOLS)
        ]

        # Default overlay: CodingOverlay ships with the SDK
        # Pass overlay=NoOverlay() to opt out of L3 entirely.
        overlay = spec.overlay
        if overlay is None:
            from ness_agent.context.coding_overlay import CodingOverlay
            modes_cfg = spec.modes
            overlay = CodingOverlay(
                plans_dir=(
                    str(modes_cfg.plans_dir) if modes_cfg and modes_cfg.plans_dir
                    else ".ness/plans/"
                ),
                plan_mode_template=modes_cfg.plan_mode_template if modes_cfg else None,
                act_mode_template=modes_cfg.act_mode_template if modes_cfg else None,
            )

        _require_instance("overlay", overlay, OverlayProvider)
        _require_instance("memory_store", spec.memory_store, MemoryBackend)
        _require_instance("approval_handler", spec.approval_handler, ApprovalHandler)

        if spec.skills_dir is not None and spec.skills_dirs is not None:
            raise ValueError(
                "Pass either skills_dir (single root) or skills_dirs "
                "(explicit root list), not both."
            )
        skills_dirs: list[Path] | None = (
            list(spec.skills_dirs)
            if spec.skills_dirs is not None
            else ([spec.skills_dir] if spec.skills_dir is not None else None)
        )

        return cls(
            model=spec.model,
            tools=resolved_tools,
            prompts=prompts,

            reflection_model=spec.reflection_model,

            options=options,
            overlay=overlay,

            memory=spec.memory,
            modes=spec.modes,
            subagents=spec.subagents,
            aux_prompts=spec.aux_prompts,
            skills_dir=skills_dirs[0] if skills_dirs else None,
            skills_dirs=skills_dirs,
            hooks_config=spec.hooks_config if spec.hooks_config is not None else ness_dir / "hooks.json",

            approval_handler=spec.approval_handler,
            question_handler=spec.question_handler,
            checkpoint_factory=spec.checkpoint_factory,
            tracing=spec.tracing,

            memory_store=(
                spec.memory_store
                if spec.memory_store is not None
                else MemoryStore(
                    spec.memory, ness_dir=ness_dir, project_root=project_root
                )
            ),
            thread_store=ThreadStore(
                threads_dir=ness_dir / "threads",
                auto_save=options.auto_save_threads,
                default_model=str(model_name or ""),
            ),
            permission_store=PermissionStore(ness_dir=ness_dir, project_root=project_root),
            hook_runner=HookRunner(
                spec.hooks_config if spec.hooks_config is not None else ness_dir / "hooks.json",
                project_root=project_root,
                hooks=spec.hooks,
            ),
            skill_loader=SkillLoader(skills_dirs=skills_dirs),
            tool_registry=ToolRegistry(resolved_tools),
            cost_tracker=spec.cost_tracker or CostTracker(pricing=spec.tracing.pricing),
            tracer=spec.tracer or build_tracer(spec.tracing),
        )


class NessAgent:
    """Top-level project agent with defaults inherited by per-thread sessions.

    This is the primary entry point for the Ness Agent SDK. Construct
    one with a model and prompt, then call ``.session(thread_id=...)``
    for each conversation thread.

    **Minimal usage** (everything defaults to a working coding agent)::

        from ness_agent import NessAgent, PromptLayersConfig
        from langchain_openai import ChatOpenAI

        agent = NessAgent(model=ChatOpenAI(model="gpt-4o"),
                          prompt=PromptLayersConfig())
        session = agent.session(thread_id="proj-1")
        result = await session.run("add rate limiter on /api/login")

    Tools are optional (all SDK built-ins are loaded when omitted).
    The default :class:`~ness_agent.context.coding_overlay.CodingOverlay`
    provides plan/act mode blocks, git snapshot, compaction notes, todos,
    session memory, and loaded skills out of the box.
    """

    def __init__(self, *, model, prompt, tools: Sequence[BaseTool] | None = None, **kwargs) -> None:
        """Create an agent.

        Parameters
        ----------
        model : BaseChatModel
            The primary chat model that drives the agent loop.
        prompt : PromptLayers | PromptLayersConfig | dict
            L0–L2 prompt configuration. Accepts a ``PromptLayers`` instance,
            a ``PromptLayersConfig`` dataclass, or a plain dict of field->value
            (unknown keys ignored).
        tools : iterable of BaseTool / callable / str, optional
            Tools available to the agent. ``None`` (the default) loads all
            SDK built-in tools (:data:`~ness_agent.tools.BUILTIN_TOOLS`).
            Items may be ``BaseTool`` instances, plain callables
            (auto-wrapped), or strings resolved from the built-in tool map.
        **kwargs
            Any remaining :class:`AgentSpec` field such as ``options``,
            ``overlay``, ``modes``, ``aux_prompts``, ``memory``, etc.
        """
        spec = AgentSpec(model=model, tools=tools, prompt=prompt, **kwargs)
        self._config = NessAgentConfig.resolve(spec)

    @classmethod
    def from_spec(cls, spec: AgentSpec) -> "NessAgent":
        """Build an agent from a pre-built :class:`AgentSpec`.

        Useful when you want to assemble the spec in one place and
        resolve backends later, or when you need to create multiple
        agents from the same spec::

            agent = NessAgent.from_spec(AgentSpec(
                model=ChatOpenAI(model="gpt-4o"),
                prompt=PromptLayersConfig(),
                options=NessAgentOptions(context_window=128_000),
            ))
        """
        agent = object.__new__(cls)
        agent._config = NessAgentConfig.resolve(spec)
        return agent

    @property
    def config(self) -> NessAgentConfig:
        """The fully-resolved :class:`NessAgentConfig` for this agent.

        Contains the resolved backends (``memory_store``, ``thread_store``,
        ``tool_registry``, etc.) plus the user-supplied fields.
        """
        return self._config

    def session(
        self,
        *,
        thread_id: str,
        mode: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        git_available: bool | None = None,
        vision: bool | None = None,
        on_plan_turn: "PlanTurnHandler | None" = None,
        on_interrupt: "InterruptHandler | None" = None,
        model: BaseChatModel | None = None,
        reflection_model: BaseChatModel | None | object = _UNSET,
    ) -> Session:
        """Create a runnable :class:`~ness_agent.session.Session` for one thread.

        Parameters
        ----------
        thread_id : str
            Unique identifier for this conversation thread.
        mode : str, optional
            Initial mode — ``"act"`` or ``"plan"``. Falls back to
            ``config.modes.default`` (or ``"act"``) when ``None``.
        metadata : dict, optional
            Arbitrary key-value pairs surfaced to the L3 overlay provider
            via ``ctx.metadata``. Set ``session.metadata[k] = v`` before
            each ``run()`` call if you want to mutate live.
        git_available : bool, optional
            Whether the project has a git repo. Auto-detected when
            ``None``.
        vision : bool, optional
            Forwards image attachments to the model when ``True``; drops
            to text-only and emits a ``warning`` SessionEvent when ``False``;
            shape-blind (forwards verbatim) when ``None``. See
            :class:`~ness_agent.session.Session`.
        on_plan_turn, on_interrupt
            Per-Session runtime hooks. Stored on the :class:`Session` instance
            (not the shared :class:`NessAgentConfig`) so concurrent threads on
            one agent never clobber each other. See :mod:`ness_agent.types`
            for the handler signatures.
        model, reflection_model
            Optional effective-model overrides applied to this session's
            config fork before its graph is compiled. Omitting either inherits
            the corresponding agent default.
        """
        from ness_agent.session import Session

        cfg = self._config.fork_for_session()
        if model is not None:
            cfg.model = model
        if reflection_model is not _UNSET:
            cfg.reflection_model = reflection_model  # type: ignore[assignment]
        mode = mode or (cfg.modes.default if cfg.modes else "act")
        return Session(
            self,
            thread_id=thread_id,
            mode=mode,
            metadata=dict(metadata or {}),
            git_available=git_available,
            vision=vision,
            on_plan_turn=on_plan_turn,
            on_interrupt=on_interrupt,
            _config=cfg,
        )

    def configure_default_models(
        self,
        *,
        model: BaseChatModel,
        reflection_model: BaseChatModel | None,
        context_window: int | None,
    ) -> None:
        """Update defaults inherited by sessions created in the future."""
        self._config.model = model
        self._config.reflection_model = reflection_model
        self._config.options.context_window = context_window

    def new_thread_id(self, prefix: str = "session") -> str:
        """Generate a fresh thread ID string.

        Parameters
        ----------
        prefix : str
            Prefix for the thread ID (default ``"session"``).

        Returns
        -------
        str
            A short hex string like ``"session-a1b2c3d4"``.
        """
        return f"{prefix}-{uuid.uuid4().hex[:8]}"
