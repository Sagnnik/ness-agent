"""Event-to-message reconstruction for resume/rollback.

Adapted from ``cli/session_app.py``'s module-scope helpers so the coding
adapter can rebuild a LangGraph transcript from persisted thread events
without importing the CLI ``config``/``settings`` modules. The function
shapes here are identical to the originals so existing tests
(``tests/test_session.py::ResumeReplayTests``-style) work the same.

Differences vs. the CLI originals:

- ``events_to_messages`` takes an explicit ``vision: bool`` (the heuristic
  lives in :mod:`ness_cli.config`) and a
  :class:`~ness_agent.permissions.PermissionStore` (the ``@file`` mention
  re-expansion needs path validation; the CLI imported the module-level
  ``permissions.PROJECT_ROOT``).
- ``restore_cost_from_events`` takes a target
  :class:`~ness_agent.tracing.cost.CostTracker` rather than the CLI's
  process-global singleton.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
    messages_from_dict,
)

from ness_agent.permissions import PermissionStore
from ness_cli.mentions import expand_documents


def plan_autosave_text(assistant_texts: list[str]) -> str | None:
    """Return the last non-empty assistant text from a plan turn, if any."""
    cleaned = [text.strip() for text in assistant_texts if text.strip()]
    return cleaned[-1] if cleaned else None


def _event_content(content: Any) -> Any:
    if isinstance(content, (str, int, float, bool)) or content is None:
        return content
    return str(content)


def _messages_from_event(event: dict) -> list[BaseMessage]:
    output = event.get("data", {}).get("output")
    if isinstance(output, dict):
        messages = output.get("messages")
        if isinstance(messages, list):
            return messages
    return []


def _subagent_batch_text(subagents: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for index, item in enumerate(subagents, start=1):
        heading = (
            f"[{index}] name={item.get('agent_name', '')} status={item.get('status', '')} "
            f"duration_ms={item.get('duration_ms', 0)} "
            f"thread_id={item.get('subagent_thread_id', '')}"
        )
        label = item.get("label")
        if label:
            heading += f" label={label}"
        lines.extend([heading, str(item.get("output") or "").strip(), ""])
    return "\n".join(lines).strip()


def _enrich_spawn_subagent_result(
    tool_name: str,
    result: str,
    subagents: list[dict[str, Any]],
) -> str:
    if tool_name != "spawn_subagent" or not subagents:
        return result
    enriched = _subagent_batch_text(subagents)
    if len(enriched) > len(result):
        return enriched
    return result


def events_to_messages(
    events: list[dict],
    subagents: list[dict[str, Any]] | None = None,
    *,
    vision: bool | None = None,
    permission_store: PermissionStore | None = None,
) -> list[BaseMessage]:
    """Rebuild the LangGraph transcript from saved events.

    Image-bearing user events carry an ``images`` field (list of data URLs).
    Replay retains those blocks for the live canonical history until summary
    compaction replaces the completed turn. ``vision=False`` remains the one
    explicit text-only gate.

    ``@file`` mentions are re-expanded against current disk on replay
    (resume/rollback) so attached file content always reflects the latest
    state rather than the snapshot at send time.
    """
    subagents = subagents or []
    messages: list[BaseMessage] = []
    # A successful new-format summary is a durable context checkpoint. Seed
    # it, then replay only the raw suffix after the boundary it replaced.
    latest_summary: tuple[int, dict] | None = None
    for seq, event in enumerate(events):
        if (
            event.get("kind") == "compaction_llm"
            and isinstance(event.get("source_event_seq"), int)
            and str(event.get("response") or "").strip()
        ):
            latest_summary = (seq, event)
    if latest_summary is not None:
        _summary_seq, checkpoint = latest_summary
        summary = str(checkpoint["response"]).strip()
        messages.append(HumanMessage(
            content=(
                "<compacted-history>\n"
                "Harness-generated continuation context; this is not a new user request.\n"
                f"{summary}\n</compacted-history>"
            ),
            additional_kwargs={"ness_internal": "compacted_history"},
        ))
        active_suffix = checkpoint.get("active_suffix")
        if isinstance(active_suffix, list) and active_suffix:
            try:
                messages.extend(messages_from_dict(active_suffix))
            except (KeyError, TypeError, ValueError):
                # A malformed optional suffix must not make an otherwise
                # usable legacy transcript impossible to resume.
                pass
        boundary = int(checkpoint["source_event_seq"])
        events = [event for seq, event in enumerate(events) if seq > boundary]
    pending_calls: list[dict[str, Any]] = []

    for event in events:
        kind = event.get("kind")
        if kind == "user":
            content = event.get("content", "")
            text = content if isinstance(content, str) else str(content)
            # Re-expand @file mentions against current disk on replay.
            if permission_store is not None:
                text = expand_documents(text, permission_store)
            images = event.get("images") or []
            if images and vision is not False:
                blocks: list[dict[str, Any]] = [
                    {"type": "text", "text": text or "Please inspect this image."}
                ]
                for url in images:
                    blocks.append({"type": "image_url", "image_url": {"url": url}})
                messages.append(HumanMessage(content=blocks))
            else:
                messages.append(HumanMessage(content=text))
        elif kind == "assistant":
            tool_calls_raw = event.get("tool_calls") or []
            tool_calls = [
                {
                    "name": tc.get("name"),
                    "args": tc.get("args", {}),
                    "id": tc.get("id"),
                    "type": tc.get("type", "tool_call"),
                }
                for tc in tool_calls_raw
            ]
            content = event.get("content")
            text = "" if content is None else str(content)
            if text or tool_calls:
                messages.append(
                    AIMessage(
                        content=text,
                        tool_calls=tool_calls,
                        additional_kwargs=dict(event.get("additional_kwargs") or {}),
                    )
                )
                pending_calls = list(tool_calls)
        elif kind == "tool":
            call_id = str(event.get("call_id") or "")
            if not call_id and pending_calls:
                call_id = str(pending_calls[0].get("id") or "")
                pending_calls = pending_calls[1:]
            tool_name = str(event.get("tool") or "")
            result = _enrich_spawn_subagent_result(
                tool_name,
                str(event.get("result") or ""),
                subagents,
            )
            messages.append(
                ToolMessage(tool_call_id=call_id, name=tool_name, content=result)
            )

    return messages


def restore_cost_from_events(events: list[dict], cost_tracker: Any) -> None:
    """Replay usage events into a CostTracker so totals continue after resume.

    The SDK adapter restores the session's own
    :class:`~ness_agent.tracing.cost.CostTracker` so a fresh resume picks up
    where the persisted events left off. Restoration deliberately bypasses
    the agent's live aggregate: replaying history is not new provider spend.

    Each persisted usage row is fed back through :meth:`CostTracker.restore` with
    the recorded token counts (``cache_read`` via ``input_token_details``) and
    the recorded cost passed as provider metadata, so the replayed totals
    reproduce the persisted ones exactly rather than being re-estimated.
    """
    for event in events:
        if event.get("kind") != "usage" or event.get("inherited"):
            continue
        model = str(event.get("model") or "")
        usage = {
            "input_tokens": int(event.get("input_tokens", 0) or 0),
            "output_tokens": int(event.get("output_tokens", 0) or 0),
            "input_token_details": {
                "cache_read": int(event.get("cached_input_tokens", 0) or 0),
                "cache_creation": int(
                    event.get("cache_write_input_tokens", 0) or 0
                ),
            },
        }
        recorded_cost = float(event.get("cost_usd", 0.0) or 0.0)
        metadata = {"cost": recorded_cost} if recorded_cost > 0 else {}
        restore = getattr(cost_tracker, "restore", cost_tracker.add)
        restore(usage, model, metadata)
