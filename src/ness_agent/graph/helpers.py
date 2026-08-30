from __future__ import annotations

from typing import Any
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.messages import ToolMessage, AIMessage
from ness_agent.context.budget import resolve_token_count
from ness_agent.context.overlay import wrap_system_reminder
from ness_agent.tools import ToolRegistry

def _effective_conversation(messages, state) -> list[BaseMessage]:
    """
    Build the effective message list at every turn
    if compaction exists then we need compacted + raw[source_count:] else just raw system message
    """
    compacted = list(state.get("model_context_messages", []))
    source_count = int(state.get("model_context_source_count", 0) or 0)
    raw = [m for m in messages if m.type != "system"]
    if compacted and 0 <= source_count <= len(raw): 
        return compacted + raw[source_count:]
    return raw

def _with_working_state_tail(messages, overlay) -> list[BaseMessage]:
    """Append an immutable, internally tagged L3 reminder tail.

    It is retained in ``model_context_messages`` for wire-prefix continuity,
    but never written to the clean semantic transcript or durable CLI events.
    """
    if not overlay.strip():
        return list(messages)
    reminder = wrap_system_reminder(overlay)
    if not reminder:
        return list(messages)
    return list(messages) + [
        HumanMessage(
            content=reminder,
            additional_kwargs={"ness_internal": "overlay"},
        )
    ]


def _is_internal_message(message: BaseMessage, kind: str | None = None) -> bool:
    marker = (getattr(message, "additional_kwargs", None) or {}).get("ness_internal")
    return bool(marker) if kind is None else marker == kind


def _semantic_conversation(messages) -> list[BaseMessage]:
    return [m for m in messages if not _is_internal_message(m, "overlay")]


def _incremental_input_tokens(
    *,
    conversation: list[BaseMessage],
    stored_context: list[BaseMessage],
    stored_system: BaseMessage | None,
    current_system: BaseMessage,
    last_input: int,
) -> int | None:
    """Reuse the previous provider input count for an append-only conversation."""
    if last_input <= 0 or not stored_context:
        return None

    if getattr(stored_system, "content", None) != current_system.content:
        return None

    stored_semantic = _semantic_conversation(stored_context)
    current_semantic = _semantic_conversation(conversation)

    if len(current_semantic) < len(stored_semantic):
        return None

    if current_semantic[: len(stored_semantic)] != stored_semantic:
        return None

    tail = current_semantic[len(stored_semantic):]
    return last_input + (
        resolve_token_count(tail, known_input_tokens=None)
        if tail else 0
    )


def _active_turn_split(messages) -> tuple[list[BaseMessage], list[BaseMessage]]:
    """Split completed history from the current continuation.

    A mid-turn compaction may summarize the user message that started the
    turn.  On later tool loops, the latest compacted-history message becomes
    the only durable boundary before the retained active suffix.
    """
    items = list(messages)
    for index in range(len(items) - 1, -1, -1):
        message = items[index]
        if message.type == "human" and not _is_internal_message(message):
            return items[:index], items[index:]
    for index in range(len(items) - 1, -1, -1):
        if _is_internal_message(items[index], "compacted_history"):
            return items[: index + 1], items[index + 1 :]
    return items, []


def _split_active_turn_safely(
    messages: list[BaseMessage],
    *,
    keep_recent_tokens: int,
) -> tuple[list[BaseMessage], list[BaseMessage]]:
    """Split an active turn without separating tool calls from results.

    The token budget is a target.  The retained suffix may exceed it when the
    newest tool-call batch is larger than the target.
    """
    items = list(messages)
    if not items:
        return [], []
    if keep_recent_tokens <= 0:
        raise ValueError("keep_recent_tokens must be positive")

    units: list[list[BaseMessage]] = []
    valid_units: list[bool] = []
    index = 0
    while index < len(items):
        message = items[index]
        if isinstance(message, ToolMessage):
            if units:
                units[-1].append(message)
                valid_units[-1] = False
            else:
                units.append([message])
                valid_units.append(False)
            index += 1
            continue

        unit = [message]
        valid = True
        index += 1
        if isinstance(message, AIMessage) and message.tool_calls:
            expected_call_ids = {
                str(call.get("id") or f"native-{call_index}")
                for call_index, call in enumerate(message.tool_calls)
            }
            result_call_ids: list[str] = []
            while index < len(items) and isinstance(items[index], ToolMessage):
                unit.append(items[index])
                result_call_ids.append(str(items[index].tool_call_id or ""))
                index += 1
            valid = (
                bool(expected_call_ids)
                and len(result_call_ids) == len(expected_call_ids)
                and len(set(result_call_ids)) == len(result_call_ids)
                and set(result_call_ids) == expected_call_ids
            )
        units.append(unit)
        valid_units.append(valid)

    retained_start = len(units)
    retained_tokens = 0
    for unit_index in range(len(units) - 1, -1, -1):
        if not valid_units[unit_index]:
            if retained_start == len(units):
                return items, []
            break
        retained_start = unit_index
        retained_tokens += resolve_token_count(
            units[unit_index], known_input_tokens=None
        )
        if retained_tokens >= keep_recent_tokens:
            break

    old_active_msgs = [message for unit in units[:retained_start] for message in unit]
    new_active_msgs = [message for unit in units[retained_start:] for message in unit]

    if new_active_msgs and isinstance(new_active_msgs[0], ToolMessage):
        return items, []
    return old_active_msgs, new_active_msgs


def _needs_approval(name, args, options, permission_store, tools_reg: ToolRegistry) -> bool:
    """Decide whether to ask the user for approval before running a tool."""
    if not options.enable_approval: 
        return False

    d = permission_store.check(name, args)
    if d in ("allow", "deny"): 
        return False
    
    # check if the tool is destructive
    return tools_reg.is_destructive(name, args)

def _denial_tool_messages(
    calls: list[tuple[str, dict[str, Any], str]],
    denials: dict[str, str],
) -> list[ToolMessage]:
    """Build ToolMessages for call_ids present in ``denials``."""
    return [
        ToolMessage(tool_call_id=call_id, name=name, content=denials[call_id])
        for name, _, call_id in calls
        if call_id in denials
    ]


def _all_calls_denied(
    calls: list[tuple[str, dict[str, Any], str]],
    denials: dict[str, str],
) -> bool:
    return bool(calls) and bool(denials) and all(call_id in denials for _, _, call_id in calls)

def _reflection_token_delta(messages, since_index) -> int:
    """Estimate tokens in messages not yet covered by the last reflection run."""
    rec = list(messages)[max(0, since_index):]
    if not rec: 
        return 0
    return resolve_token_count(rec, known_input_tokens=None)

def _result_status(result) -> str | None:
    for line in result.splitlines()[:8]:
        if line.startswith("status="): 
            # remove the status= prefix and strip the whitespace
            return line.removeprefix("status=").strip() or None
    return None

def _tool_event(name, args, result, duration, *, call_id="", exit_status=None) -> dict:
    if exit_status is None:
        exit_status = _result_status(result) or (
            "error" if result.startswith("Error:") or result.startswith("Hook veto:") else "ok"
        )

    return {
        "kind": "tool",
        "tool": name,
        "args": args,
        "result": result,
        "call_id": call_id,
        "duration_ms": duration,
        "exit": exit_status,
    }

def extract_tool_calls(msg: AIMessage) -> list[tuple[str, dict[str, Any], str]]:
    out = []
    for idx, tc in enumerate(msg.tool_calls or []):
        out.append((tc.get("name", "unknown"), tc.get("args", {}), tc.get("id") or f"native-{idx}"))
    return out
