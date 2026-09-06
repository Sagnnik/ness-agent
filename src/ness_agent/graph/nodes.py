from __future__ import annotations

import asyncio, json, time, warnings
from pathlib import Path
from typing import Any, Literal, Mapping
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    BaseMessage,
    message_to_dict,
)
from langgraph.graph import END
from ness_agent.graph.state import AgentState
from ness_agent.graph.helpers import (
    _effective_conversation,
    _with_working_state_tail,
    _semantic_conversation,
    _active_turn_split,
    _split_active_turn_safely,
    _incremental_input_tokens,
    _is_internal_message,
    _needs_approval,
    _denial_tool_messages,
    _all_calls_denied,
    _reflection_token_delta,
    extract_tool_calls,
    _tool_event,
)
from ness_agent.compaction import _invoke_summary
from ness_agent.context.budget import (
    COMPACTION_ACTIVE_TURN_MAX_TOKENS,
    COMPACTION_ACTIVE_TURN_MIN_TOKENS,
    COMPACTION_ACTIVE_TURN_RATIO,
    CompactionStatus,
    calculate_context_pressure,
    pressure_note,
    resolve_token_count,
    resolve_usable_context_budget,
)
from ness_agent.context.overlay import OverlayContext, render_overlay_delta
from ness_agent.tools.ask import set_question_runtime
from ness_agent.tools.subagents import set_subagent_runtime
from ness_agent.reflection import (
    is_reflection_running,
    consume_reflection_message_index,
    run_reflection_gate,
)
from ness_agent.tools.todo import get_thread_todos, render_todos, set_current_thread, set_thread_todos
from ness_agent.workspace.git_context import git_worktree_summary
from ness_agent.tracing.semconv import (
    CACHE_HIT_RATE,
    CACHE_READ_TOKENS,
    COST_USD,
    GEN_AI_COMPLETION,
    GEN_AI_OPERATION_NAME,
    GEN_AI_PROMPT,
    GEN_AI_SYSTEM,
    GEN_AI_SYSTEM_VALUE,
    GEN_AI_TOOL_CALL_ARGUMENTS,
    GEN_AI_TOOL_CALL_RESULT,
    KIND_CLIENT,
    INPUT_TOKENS,
    LLM_CALL,
    MODEL_NAME,
    OUTPUT_TOKENS,
    TOOL_ARGS,
    TOOL_DURATION_MS,
    TOOL_ERROR,
    TOOL_EXEC,
    TOOL_EXIT_STATUS,
    TOOL_NAME,
    COMPACTION_SUMMARIZE,
    THREAD_ID,
)
from ness_agent.tracing.messages import (
    serialize_completion,
    serialize_messages,
    truncate_for_span,
)
from ness_agent.tracing import TokenUsage
from ness_agent.types import UsageEvent

class NodesRuntime:
    """Mutable container that carry the states between the nodes.
    These states need to outlive single node call but cannot remain in the AgentState."""
    def __init__(self, config, *, thread_id, mode = "act", git_available, metadata = None):
        self.cfg = config
        self.thread_id = thread_id
        self.resolved_mode = (mode or "act").lower()
        self.repo_has_git = git_available == True
        self._last_sections: dict[str, str] = {}
        self.reflection_tasks: set[asyncio.Task] = set()
        self.metadata: Mapping[str, Any] = metadata if metadata is not None else {}
        self.last_bound_model = None


def _normalize_tool_result(result: Any) -> tuple[str | list[dict[str, Any]], str]:
    """Keep model-facing content blocks while producing a safe text representation."""
    if not isinstance(result, list):
        text = str(result)
        return text, text

    content: list[dict[str, Any]] = []
    display_parts: list[str] = []
    for item in result:
        if not isinstance(item, dict):
            text = str(result)
            return text, text
        block = dict(item)
        block_type = block.get("type")
        if block_type in {"image_url", "image", "input_image"}:
            content.append(block)
            display_parts.append("[image]")
        elif block_type in {"text", "input_text", "output_text"} and "text" in block:
            content.append(block)
            display_parts.append(str(block["text"]))
        else:
            text = str(result)
            return text, text
    return content, "\n".join(display_parts)

def make_nodes(config, *, thread_id, mode = "act", git_available = None, metadata = None) -> NodesRuntime:
    rt = NodesRuntime(config, thread_id=thread_id, mode=mode, git_available=git_available, metadata=metadata)
    # objects of the backends created in the NessAgentConfig
    tools_reg = config.tool_registry
    skills_loader = config.skill_loader
    permission_store = config.permission_store
    hooks = config.hook_runner
    cost = config.cost_tracker
    memory = config.memory_store
    persist = config.thread_store
    prompts = config.prompts
    overlay_provider = config.overlay
    aux_prompts = config.aux_prompts
    options = config.options
    tracer = config.tracer
    main_model = config.model
    model_name = getattr(config.model, "model", "") or getattr(config.model, "model_name", "")

    def _build_system_message() -> SystemMessage:
        all_skills = skills_loader.load()
        user_mem = memory.load_user() if not memory.disabled else ""
        proj_mem = memory.load_project() if not memory.disabled else ""
        skill_catalog = skills_loader.render_catalog(all_skills)
        return SystemMessage(content=prompts.build_stable_prefix(
            tools_reg.active_tools,
            user_memory=user_mem,
            project_memory=proj_mem,
            skill_catalog=skill_catalog,
            git_available=rt.repo_has_git,
            metadata=rt.metadata,
            tool_catalog_groups=[(l, frozenset(g)) for l, g in tools_reg.tool_catalog_groups()],
            deferred_mcp=tools_reg.deferred_mcp_summary(),
        ))

    def _summary_instruction(base: str, retained_count: int) -> str:
        if "{messages}" in base:
            raise ValueError(
                "compaction instructions no longer accept {messages}; remove the transcript placeholder"
            )
        retained_rule = (
            f"The final {retained_count} transcript message(s) form a recent suffix that will "
            "be retained verbatim; do not repeat them in the summary. "
            if retained_count
            else "No transcript suffix will be retained verbatim. "
        )
        return (
            base.strip()
            + "\n\nHARNESS COMPACTION RULES\n"
            + "The conversation above is already the transcript; do not ask for it again. "
            + "Do not call tools. Output only the continuation summary. "
            + "Ignore every <system-reminder> and <plan-mode> block as historical semantic content. "
            + retained_rule
            + "If the cut is inside the current user turn, preserve the original request, "
            + "constraints and acceptance criteria, completed work, current operation, files modified, "
            + "commands and process or job IDs, verified results with exit status, partial or unverified "
            + "observations, errors and rejected approaches, and the next concrete step. Never turn a "
            + "partial success-looking line into a verified successful command."
        )

    def _resolve_instruction(source) -> str:
        if callable(source):
            source = source()
        if isinstance(source, Path):
            source = source.read_text(encoding="utf-8")
        return str(source)

    async def context_gate(state: AgentState) -> AgentState:
        """Check context pressure and summarize completed history when needed."""
        tools_reg.sync() # sync the tools registry
        
        # get the messages and build the conversation
        messages = list(state.get("messages", []))
        conversation = _effective_conversation(messages, state)
        
        # build the system message
        current_system = _build_system_message()
        stored_context = list(state.get("model_context_messages", []))
        known_input = _incremental_input_tokens(
            conversation=conversation,
            stored_context=stored_context,
            stored_system=state.get("model_system_message"),
            current_system=current_system,
            last_input=int(state.get("last_input_tokens", 0) or 0),
        )
        
        # calculate the context pressure
        pressure = calculate_context_pressure(
            [current_system] + conversation,
            known_input_tokens=known_input,
            options=options,
        )
        forced = bool(state.get("force_compact")) # /compact or force compact
        # check if the conversation includes a compacted-history summary
        had_stored_compaction = any(
            _is_internal_message(message, "compacted_history") for message in conversation
        )
        status: CompactionStatus = {
            "compacted": False,
            "token_count": pressure.token_count,
            "ratio": pressure.ratio,
            "context_limit": pressure.context_limit,
            "overlay_note": pressure_note(
                pressure,
                had_stored_compaction=had_stored_compaction,
            ),
        }
        updates: AgentState = {"force_compact": False, "compaction_status": status}
        # Go to agent node if no compaction reasons are met
        if not forced and not pressure.should_compact:
            return updates
        
        # Split completed history from the current continuation
        completed, active = _active_turn_split(conversation)
        completed_semantic = _semantic_conversation(completed) 
        active_semantic = _semantic_conversation(active) # removed overlay messages
        usable_context = max(1, pressure.context_limit - options.compaction_buffer_tokens)
        active_turn_budget = min(
            usable_context,
            max(
                COMPACTION_ACTIVE_TURN_MIN_TOKENS,
                min(
                    COMPACTION_ACTIVE_TURN_MAX_TOKENS,
                    int(usable_context * COMPACTION_ACTIVE_TURN_RATIO),
                ),
            ),
        )
        active_old_msgs: list[BaseMessage] = []
        retained_semantic = active_semantic
        if (
            active_semantic
            and resolve_token_count(active_semantic, known_input_tokens=None)
            > active_turn_budget
        ):
            active_old_msgs, retained_semantic = _split_active_turn_safely(
                active_semantic,
                keep_recent_tokens=active_turn_budget,
            )
        summarize_semantic = [*completed_semantic, *active_old_msgs]

        retained_transcript_count = 0
        if retained_semantic:
            first_retained = retained_semantic[0]
            retained_index = next(
                (
                    index
                    for index, message in enumerate(conversation)
                    if message is first_retained
                ),
                len(conversation),
            )
            retained_transcript_count = len(conversation) - retained_index

        # get the current turn (for failure retry suppression)
        active_turn_id = str(getattr(active_semantic[0], "id", "") or "") if active_semantic else ""

        # If compaction already failed once on this same active turn, dont call the LLM again on every tool loop.
        if (
            not forced
            and not pressure.safety_threshold_reached
            and active_turn_id
            and state.get("compaction_failed_turn_id") == active_turn_id
        ):
            status["skip_reason"] = "retry_suppressed"
            return updates
        
        if not summarize_semantic:
            status["skip_reason"] = "no_completed_history"
            status["forced"] = forced
            return updates

        # same system message and model for prefix caching
        fork_system = state.get("model_system_message") or current_system
        parent_model = rt.last_bound_model or tools_reg.bind_model(main_model)
        instruction_source = aux_prompts.compaction
        
        # compaction is disabled if AuxPrompts.compaction is None
        if instruction_source is None:
            error = RuntimeError("compaction is disabled because AuxPrompts.compaction is None")
            if pressure.safety_threshold_reached:
                raise error
            status.update({"skip_reason": "disabled", "error": str(error), "forced": forced})
            return updates

        try:
            attrs = {
                MODEL_NAME: model_name,
                THREAD_ID: thread_id,
                GEN_AI_SYSTEM: GEN_AI_SYSTEM_VALUE,
                GEN_AI_OPERATION_NAME: "chat",
            }
            with tracer.start_span(COMPACTION_SUMMARIZE, attributes=attrs, kind=KIND_CLIENT) as span:
                summary, response, request = await _invoke_summary(
                    [fork_system] + conversation,
                    parent_model,
                    instruction=_summary_instruction(
                        _resolve_instruction(instruction_source),
                        retained_transcript_count,
                    ),
                    max_output_tokens=options.compaction_summary_max_tokens,
                )
                if config.tracing.capture_messages:
                    span.set_attribute(GEN_AI_PROMPT, serialize_messages(request))
                    span.set_attribute(GEN_AI_COMPLETION, serialize_completion(response))
        except Exception as exc:
            status.update({"skip_reason": "failed", "error": str(exc), "forced": forced})
            if active_turn_id:
                updates["compaction_failed_turn_id"] = active_turn_id  # record the failed turn id
            bridge = getattr(config, "_compaction_bridge", None) # get the compaction bridge from session
            if bridge is not None:
                bridge({
                    "skip_reason": "failed",
                    "forced": forced,
                    "info": str(exc),
                    "status": "failed",
                }) # bridge the failed compaction
            if pressure.safety_threshold_reached:
                raise RuntimeError(f"Compaction required but summarization failed: {exc}") from exc # raise an error if the safety threshold is reached
            return updates

        # build the compacted history message - useful when compaction is done between tool loops
        compacted_history = HumanMessage(
            content=(
                "<compacted-history>\n"
                "Harness-generated continuation context; this is not a new user request.\n"
                + summary
                + "\n</compacted-history>"
            ),
            additional_kwargs={"ness_internal": "compacted_history"},
        )
        # combine the summarized history and the active turn
        compacted = [compacted_history, *retained_semantic]
        after_tokens = resolve_token_count([current_system] + compacted, known_input_tokens=None)
        if after_tokens >= pressure.context_limit - options.compaction_buffer_tokens:
            raise RuntimeError(
                "The active turn is too large to fit safely after compaction; start a new turn with a smaller payload."
            )

        # Persist usage, events, and update state after compaction
        usage = None
        if getattr(response, "usage_metadata", None):
            usage = cost.add(response.usage_metadata, model_name, response.response_metadata or {})
            usage_event: dict[str, Any] = {"kind": "usage", "model": model_name, "operation": "compaction"}
            if usage is not None:
                usage_event.update(usage.as_dict())
            persist.append_event(thread_id, usage_event)
            bridge = getattr(config, "_usage_bridge", None)
            if usage is not None and bridge is not None:
                bridge(UsageEvent(
                    model=model_name,
                    input_tokens=usage.input_tokens,
                    uncached_input_tokens=usage.uncached_input_tokens,
                    cached_input_tokens=usage.cached_input_tokens,
                    output_tokens=usage.output_tokens,
                    cost_usd=usage.cost_usd,
                ))

        event = {
            "kind": "compaction_llm",
            "instruction": request[-1].content,
            "response": summary,
            "forced": forced,
            "trigger": "manual" if forced else ("safety" if pressure.safety_threshold_reached else "automatic"),
            "before_tokens": pressure.token_count,
            "after_tokens": after_tokens,
            "active_suffix_messages": len(retained_semantic),
            "active_suffix": json.loads(json.dumps(
                [message_to_dict(message) for message in retained_semantic],
                ensure_ascii=False,
                default=str,
            )),
            "model": model_name,
        }
        persist.append_compaction_checkpoint(
            thread_id, event, active_turn=bool(retained_semantic)
        )

        status.update({
            "compacted": True,
            "trigger": event["trigger"],
            "forced": forced,
            "after_tokens": after_tokens,
            "active_suffix_messages": len(retained_semantic),
            "overlay_note": pressure_note(pressure, compacted=True),
        })
        updates.update({
            "model_context_messages": compacted,
            "model_context_source_count": len(messages),
            "model_system_message": current_system,
            "last_input_tokens": 0,
            "compaction_failed_turn_id": "",
        })
        bridge = getattr(config, "_compaction_bridge", None)
        if bridge is not None:
            bridge({
                "trigger": event["trigger"],
                "forced": forced,
                "status": "success",
                "before_tokens": pressure.token_count,
                "after_tokens": after_tokens,
                "active_suffix_messages": len(retained_semantic),
                "info": "Conversation summarized; recent continuation retained verbatim.",
            })
        return updates

    async def agent_node(state: AgentState) -> AgentState:
        """The main agent node that invokes the LLM and handles the ephemeraloverlay logic."""

        # hot-rebind tools if new tools were loaded since the last turn
        tools_reg.sync()
        messages = list(state.get("messages", []))

        set_current_thread(thread_id)
        set_thread_todos(thread_id, list(state.get("todos", [])))

        system = _build_system_message()

        conversation = _effective_conversation(messages, state)
        compaction_status = dict(state.get("compaction_status") or {})
        overlay_note = compaction_status.get("overlay_note")
        # only if the overlay note is not set by context_gate (edge case)
        if overlay_note is None:
            # ``agent_node`` remains usable on its own for internal callers.
            pressure = calculate_context_pressure([system] + conversation, options=options)
            compaction_note = pressure_note(
                pressure,
                compacted=bool(compaction_status.get("compacted")),
                had_stored_compaction=any(
                    _is_internal_message(message, "compacted_history")
                    for message in conversation
                ),
            )
        else:
            compaction_note = str(overlay_note)

        cwd = options.project_root or Path.cwd()
        git_snapshot = (
            await asyncio.to_thread(git_worktree_summary, cwd)
            if rt.repo_has_git else ""
        )

        # L3 Overlay
        if overlay_provider is not None:
            overlay_context = OverlayContext(
                thread_id=thread_id,
                mode=(state.get("mode") or rt.resolved_mode),
                messages=_semantic_conversation(conversation),
                todos=state.get("todos", []),
                session_memory=memory.load_session(thread_id) if not memory.disabled else "",
                compaction_note=compaction_note,
                mode_switch=state.get("mode_switch") or "",
                metadata=rt.metadata,
                git_snapshot=git_snapshot,
                git_available=rt.repo_has_git,
                activate_skills=list(state.get("activate_skills", [])),
                loaded_skills=list(state.get("loaded_skills", [])),
            )
            sections = overlay_provider.sections(state, overlay_context) or {}
        else:
            sections = {}

        # Fresh turn (last conversation message is a HumanMessage) or post-compaction
        # (model context was rewritten) -> inject the FULL overlay so plan-mode
        # instructions are (re)established.  
        # For Tool loop -> inject only the per-section delta,
        # skipping the static plan_mode block (already on the user message).
        # Plain join only — _with_working_state_tail wraps <system-reminder>.
        is_fresh = bool(conversation) and conversation[-1].type == "human"
        if is_fresh or compaction_status.get("compacted"):
            overlay = "\n\n".join(sections.values())
        else:
            overlay = render_overlay_delta(sections, rt._last_sections, skip=frozenset({"plan_mode"}))
        
        rt._last_sections.clear()
        rt._last_sections.update(sections)

        bound_model = tools_reg.bind_model(main_model)
        # reset some states
        updates: AgentState = {
            "messages": [],
            "approval_declined": {},
            "force_compact": False,
            "activate_skills": [],
            "mode_switch": "",
            "compaction_status": {},
        }
        last_input_tokens: int | None = None
        llm_attrs = {
            MODEL_NAME: model_name,
            GEN_AI_SYSTEM: GEN_AI_SYSTEM_VALUE,
            GEN_AI_OPERATION_NAME: "chat",
        }
        # Build the exact invoke payload once (system + working-state tail injected ephemerally).
        invoke_messages = [system] + _with_working_state_tail(conversation, overlay)
        
        with tracer.start_span(LLM_CALL, attributes=llm_attrs, kind=KIND_CLIENT) as llm_span:

            if config.tracing.capture_messages:
                llm_span.set_attribute(GEN_AI_PROMPT, serialize_messages(invoke_messages))
            
            response: AIMessage = await bound_model.ainvoke(invoke_messages)
            # Compaction must fork from the binding of the last model request
            # that actually completed, not from a failed tool/schema attempt.
            rt.last_bound_model = bound_model
            
            # update the AgentState
            updates["messages"] = [response]
            updates["model_context_messages"] = invoke_messages[1:]
            updates["model_context_source_count"] = len(messages)
            updates["model_system_message"] = system

            if config.tracing.capture_messages:
                llm_span.set_attribute(GEN_AI_COMPLETION, serialize_completion(response))

            # track usage after every API call
            # langchain clients track usage metadata for billing and analytics
            # tracks -> input tokens, output tokens, total_tokens, input_token_details
            # (contains cache_read), output_token_details (contains reasoning tokens)
            if response.usage_metadata:
                usage: TokenUsage | None = cost.add(
                    response.usage_metadata,
                    model_name,
                    response.response_metadata or {},
                )
                if usage is not None:
                    llm_span.set_attribute(INPUT_TOKENS, usage.input_tokens)
                    llm_span.set_attribute(OUTPUT_TOKENS, usage.output_tokens)
                    llm_span.set_attribute(CACHE_READ_TOKENS, usage.cached_input_tokens)
                    llm_span.set_attribute(CACHE_HIT_RATE, usage.cache_hit_rate)
                    if usage.cost_usd is not None:
                        llm_span.set_attribute(COST_USD, usage.cost_usd)
                    last_input_tokens = usage.input_tokens

                usage_event: dict[str, Any] = {"kind": "usage", "model": model_name}
                if usage is not None:
                    usage_event.update(usage.as_dict())
                persist.append_event(thread_id, usage_event)

                usage_bridge = getattr(config, "_usage_bridge", None)
                if usage is not None and usage_bridge is not None:
                    usage_bridge(UsageEvent(
                        model=model_name,
                        input_tokens=usage.input_tokens,
                        uncached_input_tokens=usage.uncached_input_tokens,
                        cached_input_tokens=usage.cached_input_tokens,
                        cache_write_input_tokens=usage.cache_write_input_tokens,
                        output_tokens=usage.output_tokens,
                        cost_usd=usage.cost_usd,
                    ))

        if last_input_tokens is None:
            # Provider usage is unavailable, so retain a fallback count for the
            # next gate.  Count the exact request, including the L3 tail.
            last_input_tokens = await asyncio.to_thread(
                resolve_token_count, invoke_messages, known_input_tokens=None
            )
        updates["last_input_tokens"] = last_input_tokens

        persist.append_event(
            thread_id, {
                "kind": "assistant", 
                "content": str(response.content),
                "tool_calls": response.tool_calls or [],
                "additional_kwargs": {
                    key: value
                    for key, value in response.additional_kwargs.items()
                    if key in {"anthropic_content_blocks", "reasoning_content"}
                },
            }
        )
        _schedule_reflection_if_due(rt, state, messages + [response], model_name)
        ci = consume_reflection_message_index(thread_id)
        if ci is not None: 
            updates["last_reflection_index"] = ci
        
        return updates


    async def approval_gate(state: AgentState) -> AgentState:
        """The approval gate node that handles the approval logic."""
        calls = extract_tool_calls(state["messages"][-1])
        gated = [(n, a, cid) for n, a, cid in calls if _needs_approval(n, a, options, permission_store, tools_reg)]

        if not gated:
            return {"approval_declined": {}}

        ah = config.approval_handler
        denials: dict[str, str] = {}
        for name, args, call_id in gated:
            if ah is None:
                persist.append_event(thread_id, {"kind": "approval", "tool": name, "decision": "no"})
                denials[call_id] = f"Approval required but no handler configured: {name}"
                continue

            decision = await ah(name, args)

            if decision == "always":
                rule = permission_store.default_rule_for(name, args)
                permission_store.persist_rule(rule, "allow")
                persist.append_event(
                    thread_id, {"kind": "approval", "tool": name, "decision": "always", "rule": rule}
                )
                continue
            if decision == "session":
                rule = permission_store.default_rule_for(name, args)
                permission_store.persist_rule(rule, "allow", scope="session")
                persist.append_event(
                    thread_id, {"kind": "approval", "tool": name, "decision": "session", "rule": rule}
                )
                continue
            if decision == "never":
                rule = permission_store.default_rule_for(name, args)
                permission_store.persist_rule(rule, "deny")
                persist.append_event(
                    thread_id, {"kind": "approval", "tool": name, "decision": "never", "rule": rule}
                )
                denials[call_id] = f"Denied by persisted permission rule: {rule}"
                continue
            if decision == "yes":
                persist.append_event(thread_id, {"kind": "approval", "tool": name, "decision": "yes"})
                continue
            if decision == "no":
                persist.append_event(thread_id, {"kind": "approval", "tool": name, "decision": "no"})
                denials[call_id] = f"Denied by user approval: {name}"
                continue
            warnings.warn(
                f"Unknown approval decision {decision!r} from {ah!r} for tool {name!r} — "
                f"treating as denied",
                stacklevel=2,
            )
            persist.append_event(thread_id, {"kind": "approval", "tool": name, "decision": "no"})
            denials[call_id] = f"Denied by user approval: {name}"

        updates: AgentState = {"approval_declined": denials}
        # When every tool_call in the batch is denied, emit ToolMessages here and
        # skip tools_node. Partial denials are left for tools_node so siblings run.
        if _all_calls_denied(calls, denials):
            updates["messages"] = _denial_tool_messages(calls, denials)
        return updates

    async def tools_node(state: AgentState) -> AgentState:
        """The tools node that handles the tool calls and tool results logic."""
        # get the last AIMessage and extract the tool calls
        calls = extract_tool_calls(state["messages"][-1])
        if not calls:
            return {"messages": [], "approval_declined": {}}

        set_subagent_runtime(main_model, thread_id)
        set_question_runtime(config.question_handler)
        set_current_thread(thread_id)
        set_thread_todos(thread_id, list(state.get("todos", [])))

        # store tool results in a list of ToolMessage objects
        results: list[ToolMessage] = []
        cur_mode = (state.get("mode") or rt.resolved_mode).lower()
        newly_loaded_names: set[str] = set()
        # Fresh catalog for loaded_skills eligibility (matches skill_view context).
        all_skills = skills_loader.load()
        denials = state.get("approval_declined") or {}
        if not isinstance(denials, dict):
            denials = {}

        for name, args, call_id in calls:
            if call_id in denials:
                content = denials[call_id]
                results.append(ToolMessage(
                    tool_call_id=call_id,
                    name=name,
                    content=content,
                    additional_kwargs={"duration_ms": 0},
                ))
                persist.append_event(
                    thread_id,
                    _tool_event(name, args, content, 0, call_id=call_id, exit_status="denied"),
                )
                continue

            # Plan-mode write gate: honors ModeConfig.plan_mode_readonly
            # (default True even when modes is unset).
            readonly = (
                True
                if config.modes is None
                else bool(config.modes.plan_mode_readonly)
            )
            if (
                cur_mode == "plan"
                and readonly
                and not tools_reg.is_read_only(name, args)
            ):
                content = "Unavailable in plan mode. Switch to act mode to run state-changing tools."
                results.append(ToolMessage(
                    tool_call_id=call_id,
                    name=name,
                    content=content,
                    additional_kwargs={"hidden": True}
                ))
                persist.append_event(thread_id, _tool_event(name, args, content, 0, call_id=call_id, exit_status="mode_gated"))
                continue

            # if the tool is denied by a permission rule then return the denied messages
            decision, rule = permission_store.check_with_rule(name, args)
            if decision == "deny" and not getattr(options, "yolo_mode", False):
                content = f"Denied by permission rule: {rule}"
                results.append(ToolMessage(
                    tool_call_id=call_id,
                    name=name,
                    content=content,
                    additional_kwargs={"duration_ms": 0},
                ))
                persist.append_event(thread_id, _tool_event(name, args, content, 0, call_id=call_id, exit_status="denied"))
                continue

            # run the preToolUse hook
            ok, msg = hooks.run("preToolUse", {"tool": name, "args": args, "thread_id": thread_id})
            # if the hook vetoed the tool use then return the denied messages
            if not ok:
                results.append(ToolMessage(
                    tool_call_id=call_id,
                    name=name,
                    content=f"Hook veto: {msg}",
                    additional_kwargs={"duration_ms": 0},
                ))
                persist.append_event(thread_id, _tool_event(name, args, msg, 0, call_id=call_id, exit_status="denied"))
                continue

            # invoke the tool
            tmap = tools_reg.tool_map().get(name)
            tool_attrs: dict[str, Any] = {
                TOOL_NAME: name,
                GEN_AI_SYSTEM: GEN_AI_SYSTEM_VALUE,
                GEN_AI_OPERATION_NAME: "execute_tool",
            }
            capture_msgs = config.tracing.capture_messages
            if config.tracing.capture_tool_args:
                tool_attrs[TOOL_ARGS] = str(args)[:500]

            t0 = time.monotonic()
            with tracer.start_span(
                TOOL_EXEC.format(name=name), attributes=tool_attrs, kind=KIND_CLIENT
            ) as tool_span:
                if capture_msgs:
                    # Canonical JSON form parsed by Langfuse/Arize as tool input.
                    tool_span.set_attribute(GEN_AI_TOOL_CALL_ARGUMENTS, json.dumps(args, default=str))
                try:
                    if tmap is None:
                        result = f"Error: unknown tool {name}"
                        tool_span.set_attribute(TOOL_EXIT_STATUS, "unknown_tool")
                    elif getattr(tmap, "is_async", False) or getattr(tmap, "coroutine", None) is not None:
                        result = await tmap.ainvoke(args)
                        tool_span.set_attribute(TOOL_EXIT_STATUS, "ok")
                    else:
                        result = await asyncio.to_thread(tmap.invoke, args)
                        tool_span.set_attribute(TOOL_EXIT_STATUS, "ok")
                    tool_span.set_attribute(TOOL_ERROR, False)
                except Exception as exc:
                    tool_span.record_exception(exc)
                    tool_span.set_attribute(TOOL_ERROR, True)
                    tool_span.set_attribute(TOOL_EXIT_STATUS, "exception")
                    result = f"Error: {exc}"
                content, display_text = _normalize_tool_result(result)
                tool_span.set_attribute(TOOL_DURATION_MS, int((time.monotonic() - t0) * 1000))
                if capture_msgs:
                    # Tool results can be MBs — truncate
                    # to keep OTLP batches under the SDK's ~5MB limit
                    tool_span.set_attribute(
                        GEN_AI_TOOL_CALL_RESULT,
                        truncate_for_span(display_text, config.tracing.max_message_length),
                    )
            dur = int((time.monotonic() - t0) * 1000)

            # run the postToolUse hook
            _ok, hook_msg = hooks.run(
                "postToolUse",
                {
                    "tool": name,
                    "args": args,
                    "result": display_text,
                    "thread_id": thread_id,
                },
            )
            if hook_msg:
                if isinstance(content, list):
                    content = [{"type": "text", "text": hook_msg}, *content]
                else:
                    content = hook_msg + "\n\n" + content if content.strip() else hook_msg
                display_text = (
                    hook_msg + "\n\n" + display_text if display_text.strip() else hook_msg
                )
            results.append(ToolMessage(
                tool_call_id=call_id,
                name=name,
                content=content,
                additional_kwargs={"duration_ms": dur, "display_text": display_text},
            ))
            # append the ToolMessage to the results list and append the event to the event log
            persist.append_event(
                thread_id,
                _tool_event(name, args, display_text, dur, call_id=call_id),
            )

            # Track skills loaded via skill_view for the L3 overlay
            if name == "skill_view" and not display_text.startswith("Error:"):
                sk_name = str(args.get("name", ""))
                if sk_name and sk_name in all_skills:
                    newly_loaded_names.add(sk_name)
        # Merge newly-loaded skills into persistent loaded_skills state
        existing = list(state.get("loaded_skills", []))
        existing_names = {s.get("name", "") for s in existing}
        for sk_name in sorted(newly_loaded_names):
            if sk_name not in existing_names and sk_name in all_skills:
                sk = all_skills[sk_name]
                existing.append({
                    "name": sk.get("name", sk_name),
                    "description": sk.get("description", ""),
                    "path": sk.get("source", ""),
                })
        return {
            "messages": results,
            "todos": get_thread_todos(thread_id),
            "loaded_skills": existing,
            "approval_declined": {},
        }

    async def route_after_agent(state) -> Literal["approval_gate", "tools", "__end__"]:
        calls = extract_tool_calls(state["messages"][-1])
        
        if not calls: 
            return END

        cur = (state.get("mode") or rt.resolved_mode).lower()

        # Plan-mode mutating tools skip the approval gate (still denied in tools_node
        # when plan_mode_readonly is on).
        readonly = (
            True if config.modes is None else bool(config.modes.plan_mode_readonly)
        )
        if (
            cur == "plan"
            and readonly
            and any(not tools_reg.is_read_only(n, a) for n, a, _ in calls)
        ):
            return "tools"

        # if the approval is enabled and the tool needs approval then return the approval gate node
        if (
            options.enable_approval
            and not getattr(options, "yolo_mode", False)
            and any(
                _needs_approval(n, a, options, permission_store, tools_reg)
                for n, a, _ in calls
            )
        ):
            return "approval_gate"
        
        return "tools"

    async def route_after_approval(state) -> Literal["context_gate", "tools"]:
        denials = state.get("approval_declined") or {}
        if not isinstance(denials, dict):
            denials = {}
        if not denials:
            return "tools"
        # Denial ToolMessages may already be appended when every call was
        # rejected, so look up the AIMessage that requested the tools.
        messages = state.get("messages") or []
        ai = next(
            (m for m in reversed(messages) if isinstance(m, AIMessage) or getattr(m, "type", None) == "ai"),
            None,
        )
        if ai is None:
            return "tools"
        calls = extract_tool_calls(ai)
        if _all_calls_denied(calls, denials):
            return "context_gate"
        return "tools"

    rt.agent_node = agent_node
    rt.context_gate = context_gate
    rt.approval_gate = approval_gate
    rt.tools_node = tools_node
    rt.route_after_agent = route_after_agent
    rt.route_after_approval = route_after_approval
    return rt



def _schedule_reflection_if_due(
    rt: NodesRuntime, 
    state: AgentState, 
    messages: list[BaseMessage], 
    model_name: str
) -> None:
    """Schedules a reflection task if the reflection token delta is greater than the reflection token ratio."""
    
    if is_reflection_running(rt.thread_id): 
        return
    since = int(state.get("last_reflection_index", 0) or 0)
    delta = _reflection_token_delta(messages, since)
    ratio = float(rt.cfg.options.reflection_token_ratio or 0)
    if ratio <= 0: 
        return
    
    budget = resolve_usable_context_budget(model_name, rt.cfg.options)
    if delta < int(budget * ratio): 
        return
    todos = list(state.get("todos", [])) # get the todos
    
    async def _bg():
        await run_reflection_gate(
            rt.thread_id,
            messages,
            rt.cfg.reflection_model or rt.cfg.model,
            sum(1 for m in messages if m.type == "human"),
            last_reflection_index=since,
            todos=render_todos(todos),
            memory=rt.cfg.memory_store,
            persistence=rt.cfg.thread_store,
            aux_prompts=rt.cfg.aux_prompts,
            tracer=rt.cfg.tracer,
            tracing=rt.cfg.tracing,
        )
    
    task = asyncio.create_task(_bg(), name=f"reflection-{rt.thread_id}")
    rt.reflection_tasks.add(task)
    task.add_done_callback(rt.reflection_tasks.discard)
