from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from contextvars import ContextVar, Token
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

from ness_agent.context.budget import (
    ContextPressure,
    calculate_context_pressure,
    pressure_note,
    resolve_usable_context_budget,
)
from ness_agent.graph.builder import build_graph
from ness_agent.graph.helpers import _effective_conversation, _incremental_input_tokens
from ness_agent.reflection import ReflectionResult
from ness_agent.session_context import SessionContext, set_session_context, reset_session_context
from ness_agent.tracing.semconv import (
    AGENT_MODE,
    COST_USD,
    INPUT_TOKENS,
    OUTPUT_TOKENS,
    THREAD_ID,
    TURN,
    TURN_COUNT,
    GEN_AI_SYSTEM_VALUE,
    GEN_AI_SYSTEM,
    GEN_AI_OPERATION_NAME,
)
from ness_agent.types import (
    ApprovalHandler,
    ContextPreview,
    InterruptHandler,
    PlanTurnHandler,
    RunResult,
    SessionEvent,
    UsageEvent,
    aggregate_usage,
)

_active_session: ContextVar["Session | None"] = ContextVar(
    "ness_agent_active_session", default=None
)

PLAN_COMPACTION_CHECKPOINT_RATIO = 0.75


def _tool_end_data(msg: Any) -> dict[str, Any]:
    """Build ``tool_end`` SessionEvent data, including tool duration when present."""
    data: dict[str, Any] = {
        "name": getattr(msg, "name", "tool"),
        "content": str(getattr(msg, "content", "")),
        "id": getattr(msg, "tool_call_id", None),
    }
    kwargs = getattr(msg, "additional_kwargs", None) or {}
    if "duration_ms" in kwargs and kwargs["duration_ms"] is not None:
        data["duration_ms"] = int(kwargs["duration_ms"])
    return data


def _messages_from_event(event: dict) -> list[Any]:
    """Extracts the messages from the event data."""
    data = event.get("data") or {}
    output = data.get("output")
    if isinstance(output, dict) and "messages" in output:
        return list(output.get("messages") or [])
    if hasattr(output, "get") and callable(output.get):
        msgs = output.get("messages")
        if msgs is not None:
            return list(msgs)
    if isinstance(output, list):
        return list(output)
    return []


def _ensure_config_event_bridges(cfg: Any) -> None:
    """
    It installs one-time wrapper callbacks on the config object that bridge async events
    (approval requests, questions, usage, compaction) into the active Session's event queue.
    """
    # skip if already installed
    if getattr(cfg, "_event_bridges_installed", False):
        return

    # get the original handlers
    original_approval = cfg.approval_handler
    original_question = cfg.question_handler

    # Wraps approval_handler -> when called, emits an approval_required event
    # into the active session's queue, then delegates to the original.
    if original_approval is not None:
        class _ApprovalHandler(ApprovalHandler):
            async def __call__(self, name: str, args: dict) -> str:
                sess = _active_session.get()
                if sess is not None:
                    sess._add_queue("approval_required", {"tool": name, "args": args})
                return await original_approval(name, args)
        cfg.approval_handler = _ApprovalHandler()

    # Same pattern for question_handler -> emits question_required event.
    if original_question is not None:
        async def _question(questions: list[dict]) -> list[dict]:
            sess = _active_session.get()
            if sess is not None:
                sess._add_queue("question_required", {"questions": questions})
            return await original_question(questions)
        # store the wrapped approval handler so langgraph can use this
        cfg.question_handler = _question

    # Internal usage channel: the agent node reports per-call token/cost
    # usage here; the bridge stores ``_last_usage`` / accumulates
    # ``_turn_usages`` on the active session (feeding ``RunResult.usage_total``
    # / tracing spans) and queues a ``usage`` SessionEvent for the caller.
    # Not a user-facing hook — the same data already reaches consumers via
    # the SessionEvent stream, the durable ``usage`` log, and the cost tracker. Not a user-facing hook — the same data
    # already reaches consumers via the SessionEvent stream, the durable
    # ``usage`` log, and the cost tracker.
    def _usage(event: UsageEvent) -> None:
        sess = _active_session.get()
        if sess is not None:
            sess._last_usage = event
            sess._turn_usages.append(event)
            sess._add_queue(
                "usage",
                {
                    "model": event.model,
                    "input_tokens": event.input_tokens,
                    "uncached_input_tokens": event.uncached_input_tokens,
                    "cached_input_tokens": event.cached_input_tokens,
                    "cache_write_input_tokens": event.cache_write_input_tokens,
                    "output_tokens": event.output_tokens,
                    "cost_usd": event.cost_usd,
                    "calls": event.calls,
                },
            )

    # Compaction channel: the boundary node reports summary outcomes.
    def _compaction(data: dict[str, Any]) -> None:
        sess = _active_session.get()
        if sess is not None:
            sess._add_queue("compaction", dict(data))

    cfg._usage_bridge = _usage
    cfg._compaction_bridge = _compaction
    # mark as installed to avoid re-wrappings
    cfg._event_bridges_installed = True


class Session:
    def __init__(
        self,
        agent,
        *,
        thread_id,
        mode="act",
        metadata=None,
        git_available=None,
        vision: bool | None = None,
        on_plan_turn: PlanTurnHandler | None = None,
        on_interrupt: InterruptHandler | None = None,
        _config: Any | None = None,
    ):
        """Create a new interaction session bound to a single thread.

        Args:
            agent: A :class:`NessAgent` instance whose config drives tools,
                   models, prompts, and permissions.
            thread_id: Unique identifier for this conversation thread.
            mode: Initial mode (``"act"`` or ``"plan"``).
            metadata: Arbitrary key-value pairs surfaced in the system prompt.
            git_available: Whether the project has a git repo.
            vision: Whether image attachments should be forwarded to the model.

                * ``None`` — caller-built :class:`HumanMessage` content is
                  forwarded verbatim (today's default; the SDK is shape-blind).
                * ``True`` — image blocks are sent to the model.
                * ``False`` — image blocks are dropped to text-only and a
                  ``warning`` SessionEvent is emitted so the caller can surface
                  it.

                The model-name heuristic that decides this belongs to the
                adapter; the SDK just honours the flag.
            on_plan_turn: Per-Session hook called at the end of a successful
                plan-mode turn with the assistant text. Fire-and-forget; used by
                the adapter to autosave the plan file. When unset, a
                ``plan_turn`` SessionEvent is emitted instead so the caller
                can still observe the text. Success path only — interrupted
                plan turns flow through ``on_interrupt`` / the ``interrupted``
                SessionEvent instead, so there is exactly one interrupt path.
            on_interrupt: Per-Session hook called with the captured partial
                assistant text on interruption; returns the text to surface on
                the ``interrupted`` SessionEvent (returning ``None``/falsy
                keeps the original partial text). When unset, the SDK still
                synthesises the interruption marker itself.
        """
        self.agent = agent
        self.thread_id = thread_id
        self.mode = mode
        self.metadata = dict(metadata or {})
        self.git_available = git_available
        self._cfg = _config or agent.config.fork_for_session()
        self._force_compact = False
        self._pending_act_checkpoint = False
        self._pending_skills: list[str] = []
        self.turn_count = 0
        self.context_used = 0
        self.context_total = 0
        self._event_queue: asyncio.Queue[SessionEvent] = asyncio.Queue()
        self._last_usage: UsageEvent | None = None
        self._turn_usages: list[UsageEvent] = []

        self.checkpointer = (
            self._cfg.checkpoint_factory() if self._cfg.checkpoint_factory else MemorySaver()
        )
        self._skill_loader = self._cfg.skill_loader

        _ensure_config_event_bridges(self._cfg)

        self._app = self._build_graph()

        # Per-Session runtime hooks.
        # Stored on the Session so concurrent threads on the same NessAgent do
        # not clobber each other via the shared NessAgentConfig.
        self.on_plan_turn = on_plan_turn
        self.on_interrupt = on_interrupt

        # Vision gate
        # explicit bool decides whether image blocks reach the model.
        self._vision = vision

        # Bootstrap messages seeded by bootstrap() and consumed once on the next turn's payload.
        self._pending_bootstrap: list[Any] = []

        # Cancellation flag for the active turn. cancel() sets it 
        # _iter_events polls is_cancelled() between yields and then
        # finalises partial state via _finalize_cancelled_turn. 
        # Reset at the top of each run so a stale trigger cannot bleed into the next turn.
        self._cancel_token: asyncio.Event = asyncio.Event()

    def _add_queue(self, kind: str, data: dict[str, Any] | None = None) -> None:
        # add the event to the queue; non-blocking; of type SessionEvent
        self._event_queue.put_nowait(SessionEvent(kind, dict(data or {})))

    def _drain_queue(self) -> list[SessionEvent]:
        # drain the queue and return a list of session events
        out: list[SessionEvent] = []
        while True:
            try:
                # grab items and adds to the list; non-blocking
                out.append(self._event_queue.get_nowait()) 
            except asyncio.QueueEmpty:
                break
        return out

    def _install_session_runtime(self) -> Token:
        # sets up a runtime context for this session
        # it has everything specifications loaded into it (tools, models, options, permissions, threads etc.)
        cfg = self._cfg
        project_root = (cfg.options.project_root or Path.cwd()).resolve()
        ness_dir = (cfg.options.ness_dir or (project_root / ".ness")).resolve()
        return set_session_context(
            SessionContext(
                permissions=cfg.permission_store,
                options=cfg.options,
                thread_store=cfg.thread_store,
                ness_dir=ness_dir,
                project_root=project_root,
                agent_config=cfg,
                all_skills=self._skill_loader.load(),
            )
        )

    def _build_graph(self):
        return build_graph(
            self._cfg,
            thread_id=self.thread_id,
            mode=self.mode,
            git_available=self.git_available,
            checkpointer=self.checkpointer,
            metadata=self.metadata,
        )

    @property
    def app(self):
        """The compiled langgraph application for this session."""
        return self._app

    @property
    def config(self):
        """The effective, session-owned configuration snapshot."""
        return self._cfg

    @property
    def cost_tracker(self):
        """Usage accumulated for this session/thread."""
        return self._cfg.cost_tracker

    def configure_models(
        self,
        *,
        model: BaseChatModel,
        reflection_model: BaseChatModel | None,
        context_window: int | None,
        vision: bool | None,
    ) -> None:
        """Replace this session's effective models and rebuild its graph."""
        self._cfg.model = model
        self._cfg.reflection_model = reflection_model
        self._cfg.options.context_window = context_window
        self._vision = vision
        self._app = self._build_graph()

    def rebuild_graph(self) -> None:
        """Recompile the langgraph application (e.g. after config changes)."""
        self._app = self._build_graph()

    def reset_checkpointer(self) -> None:
        """Drop all checkpointed graph state and recompile.

        Required before a :meth:`bootstrap` replay (resume / rollback): the
        default ``MemorySaver`` keeps prior turns for this thread in memory,
        so replaying events into a reused saver would resurrect truncated
        messages and duplicate the replayed prefix (``add_messages`` appends
        by fresh id). Swapping in a fresh saver makes the durable event log
        the single source of truth for the rebuilt state.

        Caveat: with a custom ``checkpoint_factory`` backed by a persistent
        store, a new saver instance still points at the same backing store;
        the factory should scope savers per session (or clear the thread
        server-side) for replay-style flows.
        """
        self.checkpointer = (
            self._cfg.checkpoint_factory() if self._cfg.checkpoint_factory else MemorySaver()
        )
        self._app = self._build_graph()

    def bootstrap(self, messages: Sequence[Any]) -> None:
        """Seed the next turn's payload with prior messages.

        Consumed exactly once — the bootstrap list is prepended to the next
        turn's ``messages`` payload alongside the new user message, then
        cleared. This is the safe resume/rollback primitive: it mirrors the
        proven payload-seed path (CLI ``SessionApp._bootstrap``) without
        bypassing the graph entry via direct ``aupdate_state`` writes on a
        fresh checkpointer that has no prior checkpoint.
        """
        self._pending_bootstrap = list(messages)

    def cancel(self) -> None:
        """Request a cooperative break-out of the active turn's stream loop.

        ``_iter_events`` polls ``is_cancelled()`` between yields and, on a set
        token, performs partial-state cleanup via
        :meth:`_finalize_cancelled_turn` before returning normally. The TUI's
        hard-escalation backstop (``asyncio.Task.cancel``) lands as
        ``CancelledError`` and is handled by the same finaliser, shielded.
        """
        self._cancel_token.set()

    def is_cancelled(self) -> bool:
        """Whether :meth:`cancel` was requested for the active turn."""
        return self._cancel_token.is_set()

    def set_mode(self, mode: str) -> None:
        """Switch the session to *mode* (``"act"`` or ``"plan"``).

        When switching from ``"plan"`` to ``"act"``, a pre-flight compaction
        checkpoint is scheduled for the next :meth:`run` or :meth:`stream`.
        """
        if mode == self.mode:
            return
        if mode == "act" and self.mode == "plan":
            self._pending_act_checkpoint = True
        else:
            self._pending_act_checkpoint = False
        self.mode = mode

    def toggle_mode(self) -> str:
        """Flip ``"act"`` <-> ``"plan"`` (CLI Shift+Tab semantics). Returns the new mode."""
        if self.mode == "act":
            self.set_mode("plan")
        else:
            self.set_mode("act")
        return self.mode

    def set_name(self, name: str) -> bool:
        """Set this session's persistent display name."""
        return self._cfg.thread_store.set_thread_name(self.thread_id, name)

    def active_skills(self, names: Sequence[str]) -> None:
        """Replace the pending skill list for the next turn (replace-all)."""
        self._pending_skills = list(names)

    def stage_skills(self, names: Sequence[str]) -> None:
        """Append skill names to the pending list.

        Used by CLI ``/skill`` so multiple stages before one turn accumulate.
        Consumed by :meth:`_build_run_payload` when ``active_skills=`` is omitted.
        """
        pending = list(self._pending_skills)
        seen = set(pending)
        for name in names:
            n = str(name).strip()
            if not n or n in seen:
                continue
            pending.append(n)
            seen.add(n)
        self._pending_skills = pending

    def request_compact(self) -> None:
        """Requests a compaction of the session."""
        self._force_compact = True

    def _consume_force_compact(self) -> bool:
        """Consumes the force compact flag and returns its current value."""
        v, self._force_compact = self._force_compact, False
        return v

    async def run_reflection(self) -> ReflectionResult:
        """Run reflection immediately over the unreflected conversation tail."""
        from ness_agent.reflection import run_session_reflection

        return await run_session_reflection(
            self._app,
            self.thread_id,
            self._cfg.reflection_model or self._cfg.model,
            memory=self._cfg.memory_store,
            persistence=self._cfg.thread_store,
            aux_prompts=self._cfg.aux_prompts,
            tracer=self._cfg.tracer,
            tracing=self._cfg.tracing,
        )

    async def finalize_reflection(self) -> ReflectionResult | None:
        """Run end-of-session reflection when it is enabled."""
        if not self._cfg.options.session_end_reflection:
            return
        return await self.run_reflection()

    async def get_state(self) -> dict[str, Any]:
        """Return the current graph state values for this thread.

        Thin wrapper over ``app.aget_state`` so apps can read todos, messages,
        and other state.
        Returns ``{}`` when the checkpointer has no snapshot yet.
        """
        cfg = {"configurable": {"thread_id": self.thread_id}}
        try:
            snap = await self.app.aget_state(cfg)
        except Exception:
            return {}
        return dict(snap.values or {})

    async def get_messages(self) -> list[Any]:
        """Return the message list from the current graph state."""
        state = await self.get_state()
        return list(state.get("messages", []))

    async def get_todos(self) -> list[dict[str, Any]]:
        """Gets the todos from the current graph state."""
        state = await self.get_state()
        return list(state.get("todos", []))

    async def preview_context(self, *, mode: str | None = None) -> ContextPreview:
        """Preview the L0–L2 system message and prospective L3 overlay.

        Assembles the same stable prefix the next turn would send, plus the
        full L3 overlay for a fresh user turn (all sections joined). Does
        **not** run the model or the compaction summarizer — compaction
        pressure is estimated cheaply for the L3 note only.

        Args:
            mode: Optional mode override for this preview only (``\"act\"`` or
                ``\"plan\"``). Defaults to the session's current mode.
        """
        cfg = self._cfg
        options = cfg.options
        memory = cfg.memory_store

        git_flag = self.git_available
        if git_flag is None:
            from ness_agent.tools.fs import is_git_repo

            root = options.project_root or Path.cwd()
            git_flag = is_git_repo(str(root))

        system_message = str(
            self.build_system_message(git_available=bool(git_flag)).content
        )

        state, pressure, conversation = await self._context_pressure_snapshot()
        if not conversation:
            conversation = list(_effective_conversation(state.get("messages", []), state))

        model_name = getattr(cfg.model, "model", "") or getattr(cfg.model, "model_name", "")
        compaction_note = ""
        if pressure is not None:
            compaction_note = pressure_note(
                pressure,
                had_stored_compaction=bool(state.get("model_context_messages")),
            )

        preview_mode = (mode or self.mode or "act").lower()
        mode_switch = ""
        if self._pending_act_checkpoint and preview_mode == "act":
            mode_switch = "plan->act"

        cwd = options.project_root or Path.cwd()
        git_snapshot = ""
        if git_flag:
            from ness_agent.workspace.git_context import git_worktree_summary

            git_snapshot = await asyncio.to_thread(git_worktree_summary, cwd)

        from ness_agent.context.overlay import OverlayContext, wrap_system_reminder

        overlay_sections: dict[str, str] = {}
        overlay_provider = cfg.overlay
        if overlay_provider is not None:
            overlay_ctx = OverlayContext(
                thread_id=self.thread_id,
                mode=preview_mode,
                messages=conversation,
                todos=list(state.get("todos", [])),
                session_memory=(
                    memory.load_session(self.thread_id) if not memory.disabled else ""
                ),
                compaction_note=compaction_note,
                mode_switch=mode_switch,
                metadata=self.metadata,
                git_snapshot=git_snapshot,
                git_available=bool(git_flag),
                activate_skills=list(self._pending_skills),
                loaded_skills=list(state.get("loaded_skills", [])),
            )
            overlay_sections = {
                name: text
                for name, text in (overlay_provider.sections(state, overlay_ctx) or {}).items()
                if text and str(text).strip()
            }

        overlay = "\n\n".join(overlay_sections.values())
        return ContextPreview(
            system_message=system_message,
            overlay=overlay,
            overlay_sections=dict(overlay_sections),
            overlay_reminder=wrap_system_reminder(overlay),
            mode=preview_mode,
        )

    async def _context_pressure_snapshot(
        self,
    ) -> tuple[dict[str, Any], ContextPressure | None, list[Any]]:
        """Load graph state and compute context pressure in one pass.

        Updates :attr:`context_used` / :attr:`context_total` when messages are
        present. Returns ``(state, None, [])`` when there is no conversation yet.
        """
        state = await self.get_state()
        messages = list(state.get("messages", []))
        model_name = getattr(self._cfg.model, "model", "") or getattr(
            self._cfg.model, "model_name", ""
        )
        if not messages:
            # Empty sessions still need the model budget in the stats line (like 0k/128k).
            self.context_used = 0
            self.context_total = resolve_usable_context_budget(
                model_name,
                self._cfg.options,
            )
            return state, None, []

        conversation = list(_effective_conversation(messages, state))
        system = self.build_system_message()
        known_input = _incremental_input_tokens(
            conversation=conversation,
            stored_context=list(state.get("model_context_messages", [])),
            stored_system=state.get("model_system_message"),
            current_system=system,
            last_input=int(state.get("last_input_tokens", 0) or 0),
        )
        pressure = calculate_context_pressure(
            [system] + conversation,
            known_input_tokens=known_input,
            model_name=model_name,
            options=self._cfg.options,
        )
        self.context_used = pressure.token_count
        self.context_total = pressure.usable_budget
        return state, pressure, conversation

    def build_system_message(
        self, *, git_available: bool | None = None
    ) -> SystemMessage:
        """Build the stable L0–L2 system prefix used by the graph.

        Includes tool catalog, memory, skills, git context, and session
        metadata. Useful for context previews, token budgeting, or custom
        tooling outside a turn.
        """
        cfg = self._cfg
        tools_reg = cfg.tool_registry
        tools_reg.sync()
        memory = cfg.memory_store
        skills_loader = cfg.skill_loader
        all_skills = skills_loader.load()
        available = self.git_available is True if git_available is None else git_available
        return SystemMessage(content=cfg.prompts.build_stable_prefix(
            tools_reg.active_tools,
            user_memory=memory.load_user() if not memory.disabled else "",
            project_memory=memory.load_project() if not memory.disabled else "",
            skill_catalog=skills_loader.render_catalog(all_skills),
            git_available=available,
            metadata=self.metadata,
            tool_catalog_groups=[
                (label, frozenset(group))
                for label, group in tools_reg.tool_catalog_groups()
            ],
            deferred_mcp=tools_reg.deferred_mcp_summary(),
        ))

    async def refresh_context_snapshot(self) -> dict[str, Any]:
        """Refresh token-usage metrics from the current graph state."""
        state, _pressure, _conversation = await self._context_pressure_snapshot()
        return state

    def _user_message(
        self, message: str, images: Sequence[str] | None
    ) -> tuple[HumanMessage, str]:
        """Build a HumanMessage and return ``(message, cleaned_text)``.

        Adapter-owned cleanup (e.g. TUI ``[Image #N]`` placeholders) should
        happen before calling :meth:`run` / :meth:`stream`. When
        ``self._vision is False`` and images were supplied, the blocks are
        dropped to text-only and a ``warning`` SessionEvent is queued for the
        caller. When ``None`` (default), the caller-built content shape is
        forwarded verbatim — the SDK is shape-blind and trusts the adapter's
        gating decision.
        """
        cleaned = (message or "").strip()
        if not images:
            return HumanMessage(content=cleaned), cleaned
        if self._vision is False:
            self._add_queue(
                "warning",
                {"message": "Session vision is disabled; sending text only."},
            )
            return (
                HumanMessage(content=cleaned or "[image omitted — model is text-only]"),
                cleaned,
            )
        return (
            HumanMessage(
                content=[{"type": "text", "text": cleaned or "Please inspect this image."}]
                + [{"type": "image_url", "image_url": {"url": u}} for u in images]
            ),
            cleaned,
        )

    async def _maybe_checkpoint_before_act(self) -> None:
        """Pre-execution compaction checkpoint when switching plan→act.

        Measures context pressure and either force-compacts (hard threshold)
        or emits an advisory ``compaction`` SessionEvent so the caller can
        surface a notice / offer ``/compact`` before execution.
        """
        _state, pressure, conversation = await self._context_pressure_snapshot()
        if pressure is None or pressure.ratio < PLAN_COMPACTION_CHECKPOINT_RATIO:
            return

        info = (
            f"Context ~{pressure.token_count:,} tokens of {pressure.usable_budget:,} budget "
            f"({pressure.ratio:.0%}). Compaction if run: cache-safe summary."
        )

        if pressure.hard_threshold_reached:
            self._force_compact = True
            self._add_queue(
                "compaction",
                {
                    "notice_reason": "pre_act_hard_threshold",
                    "info": info,
                    "forced": True,
                },
            )
            return
        # Soft checkpoint: advisory notice only (no interactive prompt).
        self._add_queue(
            "compaction",
            {
                "notice_reason": "pre_act_checkpoint",
                "info": info,
                "forced": False,
                "advisory": True,
            },
        )

    async def _build_run_payload(
        self,
        user_message: HumanMessage,
        *,
        active_skills: Sequence[str] | None,
        mode_switch: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Build the turn payload and run config.

        Takes a *pre-built* ``user_message`` (constructed by
        :meth:`_user_message`). Any pending bootstrap messages are prepended
        to the payload's ``messages`` and consumed once here.
        """
        skills = list(active_skills if active_skills is not None else self._pending_skills)
        if active_skills is None:
            self._pending_skills = []
        initial = list(self._pending_bootstrap)
        if initial:
            self._pending_bootstrap = []
        payload = {
            "messages": [*initial, user_message],
            "approval_declined": {},
            "mode": self.mode,
            "force_compact": self._consume_force_compact(),
            "activate_skills": skills,
            "mode_switch": mode_switch,
        }
        cfg = {"configurable": {"thread_id": self.thread_id}, "recursion_limit": self._cfg.options.recursion_limit}
        return payload, cfg

    def _dispatch_stream_event(
        self, ev: dict, assistant_text: str
    ) -> list[tuple[SessionEvent, str]]:
        """Map one astream_events chunk to SessionEvent pairs."""
        out: list[tuple[SessionEvent, str]] = []
        ek = ev.get("event")
        name = ev.get("name", "")

        # handle streaming token chunks
        if ek == "on_chat_model_stream":
            chunk = ev.get("data", {}).get("chunk")
            
            if chunk is None:
                return out
            
            data: dict[str, Any] = {}
            ak = getattr(chunk, "additional_kwargs", None) or {}
            
            # get the reasoning content
            rtext = ak.get("reasoning_content") if isinstance(ak, dict) else None
            if isinstance(rtext, str) and rtext:
                data["reasoning"] = rtext
            
            # get the main content of the chunk
            text = getattr(chunk, "content", "")
            if isinstance(text, str) and text:
                assistant_text = assistant_text + text
                data["text"] = text
            
            if data:
                # add the event to the output
                out.append((SessionEvent("assistant_delta", data), assistant_text))
            return out

        # handle the on_chain_end event for the agent node -> emits the
        # authoritative assistant output (final text + tool calls). The agent
        # node runs the model via a non-streaming ainvoke and returns the
        # AIMessage; astream_events surfaces it here as the node's output
        # messages. 
        # on_chain_end (name "agent") carries only the agent's response message
        if ek == "on_chain_end" and name == "agent":
            for msg in _messages_from_event(ev):
                if getattr(msg, "type", None) not in {"ai", "assistant"}:
                    continue
                text = str(getattr(msg, "content", "") or "")
                if text.strip():
                    assistant_text = text
                    out.append(
                        (SessionEvent("assistant_final", {"content": text}), assistant_text)
                    )
                for tc in getattr(msg, "tool_calls", None) or []:
                    # add the tool start event to the output
                    out.append(
                        (
                            SessionEvent(
                                "tool_start",
                                {
                                    "name": tc.get("name", "unknown"),
                                    "args": tc.get("args", {}),
                                    "id": tc.get("id"),
                                },
                            ),
                            assistant_text,
                        )
                    )
            return out

        # handle the on_end_chain event that emits tool_end or the end of tools node
        if ek == "on_chain_end" and name == "tools":
            for msg in _messages_from_event(ev):
                if getattr(msg, "type", None) != "tool" and not isinstance(msg, ToolMessage):
                    continue
                if (getattr(msg, "additional_kwargs", None) or {}).get("hidden"):
                    continue
                out.append(
                    (
                        SessionEvent(
                            "tool_end",
                            _tool_end_data(msg),
                        ),
                        assistant_text,
                    )
                )
            return out

        return out

    async def _finalize_cancelled_turn(self, assistant_text: str, cfg: dict) -> None:
        """Flush partial state after a cooperative or hard cancel.

        Pure graph-mutation (no ``render``): synthesises a *failed*
        ``ToolMessage`` for every pending tool call so the checkpoint stays
        consistent, and when neither partial text nor pending tool calls exist,
        injects an ``AIMessage`` interruption marker so the model does not
        silently resume the abandoned request next turn. Emits an
        ``interrupted`` SessionEvent so the caller can surface it.

        The marker is an ``AIMessage`` (not ``HumanMessage``) to preserve
        strict user/assistant alternation — the last checkpoint message may be
        a ``HumanMessage`` (empty-stream turn) or a ``ToolMessage`` (just
        completed tools), and an ``AIMessage`` cap is valid in both cases,
        while a second ``HumanMessage`` risks back-to-back humans that some
        providers reject.
        """
        recorded_text = bool(assistant_text and assistant_text.strip())

        synthetic: list[Any] = []
        has_pending = False
        try:
            snapshot = await self.app.aget_state(cfg)
        except Exception:
            snapshot = None
        if snapshot is not None:
            messages = list((snapshot.values or {}).get("messages", []))
            # find and get the answered tool call ids
            answered_ids = {
                getattr(m, "tool_call_id", None)
                for m in messages
                if isinstance(m, ToolMessage)
            }
            # Find the last AIMessage and its tool calls
            for msg in reversed(messages):
                if not isinstance(msg, AIMessage):
                    continue
                # iterate over the tool calls
                for tc in (msg.tool_calls or []):
                    call_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                    if not call_id or str(call_id) in answered_ids:
                        continue
                    has_pending = True
                    tool_name = (
                        tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "tool")
                    )
                    # add synthetic tool message
                    synthetic.append(
                        ToolMessage(
                            tool_call_id=str(call_id),
                            name=str(tool_name or "tool"),
                            content="Tool execution interrupted",
                        )
                    )
                break

        if not recorded_text and not has_pending:
            # add interruption marker if there is no assistant text and no pending tool calls
            # Cancel during a pure LLM call before any tokens streamed.
            # Cancel after tools finished but before the model answered again 
            # (last message might be ToolMessage, not partial AI text in assistant_text).
            synthetic.append(
                AIMessage(content=self._cfg.options.interruption_marker)
            )

        if synthetic:
            # update the state with the synthetic messages
            try:
                await self.app.aupdate_state(cfg, {"messages": synthetic})
            except Exception:
                # the checkpoint may be left dirty but the next turn will still proceed.
                pass

        interrupted_surface = assistant_text
        if self.on_interrupt is not None:
            try:
                interrupted_surface = self.on_interrupt(assistant_text) or assistant_text
            except Exception:
                pass

        self._add_queue("interrupted", {"partial_text": interrupted_surface})
        # NOTE: interrupted plan turns are NOT routed through ``on_plan_turn``
        # here — that hook is the success-path contract (see ``_iter_events``).
        # The partial text reaches the caller exactly once, via the
        # ``on_interrupt`` hook above and the ``interrupted`` SessionEvent, so
        # an adapter that archives plan text has a single place to do it.

    async def _iter_events(
        self,
        message: str,
        *,
        images: Sequence[str] | None = None,
        active_skills: Sequence[str] | None = None,
        mode: str | None = None,
    ) -> AsyncIterator[tuple[SessionEvent, str]]:
        """Yield (event, assistant_text_so_far) pairs from the graph stream."""

        # sets up a runtime context for this session
        ctx_token = self._install_session_runtime()
        # reset per-turn usage (last call + turn aggregate)
        self._last_usage = None
        self._turn_usages = []
        # reset the cooperative cancel token — a stale trigger from a prior
        # turn must not abort this one.
        self._cancel_token.clear()
        # drain the queue and return a list of session events
        self._drain_queue()
        # set the active session context var
        token = _active_session.set(self)

        tracer = self._cfg.tracer
        with tracer.start_span(
            TURN,
            attributes={
                THREAD_ID: self.thread_id,
                AGENT_MODE: self.mode,
                GEN_AI_SYSTEM: GEN_AI_SYSTEM_VALUE,
                GEN_AI_OPERATION_NAME: "agent",
            },
        ) as span:
            try:
                # Mode override is documented as "this turn only", so snapshot
                # the prior mode and restore it in the ``finally`` below via a
                # direct assignment (NOT ``set_mode`` — that would schedule a
                # spurious plan->act compaction checkpoint on the next turn).
                prior_mode = self.mode
                mode_overridden = bool(mode and mode != self.mode)
                if mode_overridden:
                    self.set_mode(mode)
                mode_switch = ""
                if self._pending_act_checkpoint and self.mode == "act":
                    self._pending_act_checkpoint = False
                    mode_switch = "plan->act"
                    await self._maybe_checkpoint_before_act()

                # build the user message (vision gate + image-strip).
                try:
                    user_message, _cleaned = self._user_message(message, images)
                except Exception as exc:
                    span.set_status("ERROR", str(exc))
                    yield SessionEvent("error", {"message": str(exc)}), ""
                    return

                cfg = {"configurable": {"thread_id": self.thread_id}, "recursion_limit": self._cfg.options.recursion_limit}

                try:
                    payload, cfg_payload = await self._build_run_payload(
                        user_message,
                        active_skills=active_skills,
                        mode_switch=mode_switch,
                    )
                    cfg = cfg_payload
                except Exception as exc:
                    span.set_status("ERROR", str(exc))
                    yield SessionEvent("error", {"message": str(exc)}), ""
                    return

                for queued in self._drain_queue():
                    yield queued, ""

                assistant_text = ""
                cancelled = False
                try:
                    async for ev in self.app.astream_events(
                        payload, config=cfg, version="v2"
                    ):
                        # first yield queued events like usage, compact, etc.
                        for queued in self._drain_queue():
                            yield queued, assistant_text
                        # then dispatch the stream events
                        for event, assistant_text in self._dispatch_stream_event(
                            ev, assistant_text
                        ):
                            yield event, assistant_text
                        # cooperative cancel: break out between events so the
                        # post-loop cleanup can flush partial state cleanly.
                        if self._cancel_token.is_set():
                            cancelled = True
                            break
                    # also catch a cancel that arrived during the last
                    # event's downstream processing (between the final yield
                    # and the loop's natural exit): without this, a late
                    # cancel lands silently instead of finalizing.
                    if not cancelled and self._cancel_token.is_set():
                        cancelled = True
                    # finally yield any remaining queued events
                    for queued in self._drain_queue():
                        yield queued, assistant_text
                except asyncio.CancelledError:
                    # Hard-escalation path: the cooperative cancel token failed
                    # to break the stream loop within the backstop window.
                    # Best-effort finalisation before re-raising so the task is
                    # still properly cancelled.
                    try:
                        await asyncio.shield(
                            self._finalize_cancelled_turn(assistant_text, cfg)
                        )
                    except asyncio.CancelledError:
                        pass
                    except Exception:
                        pass
                    final_events: list[SessionEvent] = []
                    try:
                        final_events = self._drain_queue()
                    except Exception:
                        final_events = []
                    for queued in final_events:
                        yield queued, assistant_text
                    raise
                except Exception as exc:
                    span.set_status("ERROR", str(exc))
                    yield SessionEvent("error", {"message": str(exc)}), assistant_text
                    return

                if cancelled:
                    await self._finalize_cancelled_turn(assistant_text, cfg)
                    # drain any events queued during finalize (interrupted,
                    # and any warnings) and yield them after the model stream
                    # has ended.
                    for queued in self._drain_queue():
                        yield queued, assistant_text
                elif self.mode == "plan":
                    # Plan-turn emission: when an adapter hook is installed, it
                    # takes the text directly; otherwise emit a ``plan_turn``
                    # SessionEvent so the caller can still observe the text.
                    if assistant_text.strip():
                        if self.on_plan_turn is not None:
                            try:
                                self.on_plan_turn(assistant_text)
                            except Exception:
                                pass
                        else:
                            yield SessionEvent("plan_turn", {"text": assistant_text}), assistant_text
            finally:
                # Restore the session mode when a one-turn override was applied
                # (see ``mode`` kwarg docstring). Direct assignment avoids
                # ``set_mode``'s plan->act checkpoint side effect.
                if mode_overridden and self.mode != prior_mode:
                    self.mode = prior_mode
                span.set_attribute(TURN_COUNT, self.turn_count)
                turn_usage = aggregate_usage(self._turn_usages) or self._last_usage
                if turn_usage is not None:
                    span.set_attribute(INPUT_TOKENS, turn_usage.input_tokens)
                    span.set_attribute(OUTPUT_TOKENS, turn_usage.output_tokens)
                    span.set_attribute(COST_USD, turn_usage.cost_usd or 0)
                # remove the active session object from memory or contextvar
                _active_session.reset(token)
                reset_session_context(ctx_token)

    async def run(
        self,
        message: str,
        *,
        images: Sequence[str] | None = None,
        active_skills: Sequence[str] | None = None,
        mode: str | None = None,
    ) -> RunResult:
        """Send a message and collect the full response as a :class:`RunResult`.

        This is the batched (non-streaming) entry point.  It drains the
        entire event stream, yields a single ``RunResult`` with the
        assistant text, usage stats, todos, and all intermediate events.

        Args:
            message: The user message text.
            images: Optional list of image URLs to attach.
            active_skills: Skill names to activate this turn.
            mode: Override the session mode for this turn only.
        """
        events: list[SessionEvent] = []
        assistant_text = ""
        
        async for event, assistant_text in self._iter_events(
            message, images=images, active_skills=active_skills, mode=mode
        ):
            events.append(event)
        
        self.turn_count += 1
        await self.refresh_context_snapshot()
        todos = await self.get_todos()
        return RunResult(
            assistant_message=assistant_text,
            todos=todos,
            events=events,
            usage_total=aggregate_usage(self._turn_usages),
        )

    async def stream(
        self,
        message: str,
        *,
        images: Sequence[str] | None = None,
        active_skills: Sequence[str] | None = None,
        mode: str | None = None,
    ) -> AsyncIterator[SessionEvent]:
        """Send a message and yield :class:`SessionEvent` objects as they arrive.

        This is the streaming entry point.  Each ``SessionEvent``
        represents a discrete milestone (token delta, tool start/end,
        usage, error, etc.).  The caller is responsible for consuming the
        iterator.

        Args:
            message: The user message text.
            images: Optional list of image URLs to attach.
            active_skills: Skill names to activate this turn.
            mode: Override the session mode for this turn only.
        """
        async for event, _ in self._iter_events(
            message, images=images, active_skills=active_skills, mode=mode
        ):
            yield event
        self.turn_count += 1
        await self.refresh_context_snapshot()
