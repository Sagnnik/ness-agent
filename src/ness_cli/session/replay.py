"""Rebuild SDK messages and usage from durable CLI events."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
    messages_from_dict,
)
from ness_agent import PermissionStore

from ness_cli.session.mentions import expand_documents


def _subagent_text(rows: Sequence[Mapping[str, Any]]) -> str:
    lines: list[str] = []
    for index, row in enumerate(rows, start=1):
        heading = (
            f"[{index}] name={row.get('agent_name', '')} "
            f"status={row.get('status', '')} "
            f"duration_ms={row.get('duration_ms', 0)} "
            f"thread_id={row.get('subagent_thread_id', '')}"
        )
        if row.get("label"):
            heading += f" label={row['label']}"
        lines.extend((heading, str(row.get("output") or "").strip(), ""))
    return "\n".join(lines).strip()


def events_to_messages(
    events: Sequence[Mapping[str, Any]],
    *,
    subagents: Sequence[Mapping[str, Any]] = (),
    vision: bool | None = None,
    permission_store: PermissionStore | None = None,
) -> list[BaseMessage]:
    """Replay the latest compaction checkpoint and its raw suffix."""
    rows = [dict(event) for event in events]
    messages: list[BaseMessage] = []

    latest_summary: Mapping[str, Any] | None = None
    for event in rows:
        if (
            event.get("kind") == "compaction_llm"
            and isinstance(event.get("source_event_seq"), int)
            and str(event.get("response") or "").strip()
        ):
            latest_summary = event

    if latest_summary is not None:
        summary = str(latest_summary["response"]).strip()
        messages.append(
            HumanMessage(
                content=(
                    "<compacted-history>\n"
                    "Harness-generated continuation context; this is not a new "
                    "user request.\n"
                    f"{summary}\n</compacted-history>"
                ),
                additional_kwargs={"ness_internal": "compacted_history"},
            )
        )
        active_suffix = latest_summary.get("active_suffix")
        if isinstance(active_suffix, list) and active_suffix:
            try:
                messages.extend(messages_from_dict(active_suffix))
            except (KeyError, TypeError, ValueError):
                pass
        boundary = int(latest_summary["source_event_seq"])
        rows = rows[boundary + 1 :]

    pending_calls: list[dict[str, Any]] = []
    for event in rows:
        kind = event.get("kind")
        if kind == "user":
            text = str(event.get("content") or "")
            if permission_store is not None:
                text = expand_documents(text, permission_store)
            images = list(event.get("images") or ())
            if images and vision is not False:
                content: list[dict[str, Any]] = [
                    {
                        "type": "text",
                        "text": text or "Please inspect this image.",
                    }
                ]
                content.extend(
                    {
                        "type": "image_url",
                        "image_url": {"url": str(url)},
                    }
                    for url in images
                )
                messages.append(HumanMessage(content=content))
            else:
                messages.append(HumanMessage(content=text))
            continue

        if kind == "assistant":
            tool_calls = [
                {
                    "name": call.get("name"),
                    "args": call.get("args", {}),
                    "id": call.get("id"),
                    "type": call.get("type", "tool_call"),
                }
                for call in event.get("tool_calls") or ()
                if isinstance(call, Mapping)
            ]
            text = str(event.get("content") or "")
            if text or tool_calls:
                messages.append(
                    AIMessage(
                        content=text,
                        tool_calls=tool_calls,
                        additional_kwargs=dict(event.get("additional_kwargs") or {}),
                    )
                )
                pending_calls = tool_calls
            continue

        if kind != "tool":
            continue

        call_id = str(event.get("call_id") or "")
        if not call_id and pending_calls:
            call_id = str(pending_calls.pop(0).get("id") or "")
        tool_name = str(event.get("tool") or "")
        result = str(event.get("result") or "")
        if tool_name == "spawn_subagent" and subagents:
            enriched = _subagent_text(subagents)
            if len(enriched) > len(result):
                result = enriched
        messages.append(
            ToolMessage(
                tool_call_id=call_id,
                name=tool_name,
                content=result,
            )
        )

    return messages


def restore_cost(
    events: Sequence[Mapping[str, Any]],
    cost_tracker: Any,
) -> None:
    """Restore prior usage without adding it to the live agent aggregate."""
    for event in events:
        if event.get("kind") != "usage" or event.get("inherited"):
            continue
        usage = {
            "input_tokens": int(event.get("input_tokens", 0) or 0),
            "output_tokens": int(event.get("output_tokens", 0) or 0),
            "input_token_details": {
                "cache_read": int(event.get("cached_input_tokens", 0) or 0),
                "cache_creation": int(event.get("cache_write_input_tokens", 0) or 0),
            },
        }
        recorded_cost = float(event.get("cost_usd", 0.0) or 0.0)
        metadata = {"cost": recorded_cost} if recorded_cost > 0 else {}
        cost_tracker.restore(usage, str(event.get("model") or ""), metadata)
