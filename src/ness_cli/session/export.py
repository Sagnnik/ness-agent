"""Portable, self-contained HTML exports for durable Ness sessions."""

from __future__ import annotations

import html
import json
import re
import secrets
import shlex
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class ExportRecord:
    """Stable, presentation-neutral representation of one durable event."""

    seq: int
    timestamp: str
    kind: str
    category: str
    title: str
    preview: str
    content: str
    details: dict[str, Any]


@dataclass(frozen=True)
class ExportResult:
    path: Path
    event_count: int
    jsonl_bytes: int


class ExportError(ValueError):
    """Raised when a complete, safe session export cannot be produced."""


_OMITTED_EVENT_KEYS = {
    "active_suffix",
    "additional_kwargs",
    "content",
    "images",
    "instruction",
    "kind",
    "prompt",
    "response",
    "result",
    "seq",
    "t",
}


def resolve_export_path(raw_args: str, project_root: Path) -> Path:
    """Parse one shell-style path and resolve it from the project root."""
    try:
        parts = shlex.split(raw_args)
    except ValueError as exc:
        raise ExportError(f"Invalid export path: {exc}") from exc
    if len(parts) != 1:
        raise ExportError("Usage: /export <path.html>")
    destination = Path(parts[0]).expanduser()
    if not destination.is_absolute():
        destination = Path(project_root) / destination
    destination = destination.resolve()
    if destination.suffix.lower() != ".html":
        raise ExportError("Export path must end in .html")
    return destination


def normalize_events(
    events: Sequence[Mapping[str, Any]],
    *,
    subagents: Sequence[Mapping[str, Any]] = (),
) -> list[ExportRecord]:
    """Convert durable event payloads to the stable exported transcript shape."""
    normalized: list[ExportRecord] = []
    safe_subagents = [_safe_subagent(item) for item in subagents]
    for fallback_seq, event in enumerate(events):
        seq = _as_int(event.get("seq"), fallback_seq)
        timestamp = str(event.get("t") or "")
        kind = str(event.get("kind") or "event")
        category, title, content, details = _normalize_event(
            kind,
            event,
            subagents=safe_subagents,
        )
        preview_source = content or _preview_from_details(details)
        normalized.append(
            ExportRecord(
                seq=seq,
                timestamp=timestamp,
                kind=kind,
                category=category,
                title=title,
                preview=_one_line(preview_source, 110),
                content=content,
                details=details,
            )
        )
    return normalized


def export_thread_html(
    *,
    thread_store: Any,
    thread_id: str,
    project_root: Path,
    destination: Path,
    generated_at: datetime | None = None,
) -> ExportResult:
    """Write the current durable thread to a new self-contained HTML file."""
    if not bool(getattr(thread_store, "auto_save", False)):
        raise ExportError(
            "Thread autosave is disabled; a complete pre-compaction export cannot be guaranteed."
        )
    if destination.exists():
        raise ExportError(f"Refusing to overwrite existing file: {destination}")

    events = list(thread_store.load_thread_events(thread_id))
    if not events:
        raise ExportError("The current session has no durable events to export.")

    metadata: dict[str, Any] = {}
    for item in thread_store.list_threads(50):
        if str(item.get("thread_id") or "") == thread_id:
            metadata = dict(item)
            break
    subagents = list(thread_store.list_subagents(thread_id))
    records = normalize_events(events, subagents=subagents)
    exported_at = generated_at or datetime.now(timezone.utc)
    jsonl = records_to_jsonl(records)
    document = render_export_html(
        records,
        thread_id=thread_id,
        project_root=Path(project_root),
        metadata=metadata,
        generated_at=exported_at,
        jsonl=jsonl,
        jsonl_name=f"{destination.stem}.jsonl",
    )

    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(document)
    except FileExistsError as exc:
        raise ExportError(
            f"Refusing to overwrite existing file: {destination}"
        ) from exc
    except OSError as exc:
        raise ExportError(f"Could not write export: {exc}") from exc
    return ExportResult(
        path=destination,
        event_count=len(records),
        jsonl_bytes=len(jsonl.encode("utf-8")),
    )


def records_to_jsonl(records: Iterable[ExportRecord]) -> str:
    return (
        "\n".join(
            json.dumps(asdict(record), ensure_ascii=False, separators=(",", ":"))
            for record in records
        )
        + "\n"
    )


def render_export_html(
    records: Sequence[ExportRecord],
    *,
    thread_id: str,
    project_root: Path,
    metadata: Mapping[str, Any],
    generated_at: datetime,
    jsonl: str,
    jsonl_name: str,
) -> str:
    """Render a responsive offline transcript with no external dependencies."""
    title = _session_title(metadata, records, thread_id)
    counts = {
        category: sum(record.category == category for record in records)
        for category in ("message", "tool", "event")
    }
    nonce = secrets.token_urlsafe(18)
    nav = "\n".join(
        _render_nav_item(record, index) for index, record in enumerate(records, 1)
    )
    transcript = "\n".join(
        _render_transcript_item(record, index)
        for index, record in enumerate(records, 1)
    )
    json_payload = _script_json(jsonl)
    download_name = _script_json(jsonl_name)
    generated_text = _format_datetime(generated_at.isoformat())
    started_text = _format_datetime(str(metadata.get("started_at") or "")) or "Unknown"
    model = str(metadata.get("model") or "Unknown")
    total = len(records)

    replacements = {
        "__CSP_NONCE__": nonce,
        "__PAGE_TITLE__": html.escape(title),
        "__THREAD_ID__": html.escape(thread_id),
        "__PROJECT_ROOT__": html.escape(str(project_root)),
        "__MODEL__": html.escape(model),
        "__STARTED_AT__": html.escape(started_text),
        "__GENERATED_AT__": html.escape(generated_text),
        "__TOTAL_COUNT__": str(total),
        "__MESSAGE_COUNT__": str(counts["message"]),
        "__TOOL_COUNT__": str(counts["tool"]),
        "__EVENT_COUNT__": str(counts["event"]),
        "__NAV_ITEMS__": nav,
        "__TRANSCRIPT_ITEMS__": transcript,
        "__JSONL_PAYLOAD__": json_payload,
        "__JSONL_NAME__": download_name,
    }
    marker_pattern = re.compile("|".join(re.escape(marker) for marker in replacements))
    return marker_pattern.sub(
        lambda match: replacements[match.group(0)], _HTML_TEMPLATE
    )


def _normalize_event(
    kind: str,
    event: Mapping[str, Any],
    *,
    subagents: list[dict[str, Any]],
) -> tuple[str, str, str, dict[str, Any]]:
    if kind == "user":
        images = event.get("images") or []
        count = len(images) if isinstance(images, list) else 1
        details: dict[str, Any] = {}
        if count:
            details["attachments"] = [
                {"type": "image", "omitted": True, "label": f"Image {index}"}
                for index in range(1, count + 1)
            ]
        return "message", "User", _text(event.get("content")), details

    if kind == "assistant":
        calls = []
        for call in event.get("tool_calls") or []:
            if not isinstance(call, Mapping):
                continue
            calls.append(
                {
                    "name": str(call.get("name") or "unknown"),
                    "args": _json_safe(call.get("args") or {}),
                    "id": str(call.get("id") or ""),
                }
            )
        details = {"tool_calls": calls} if calls else {}
        kwargs = event.get("additional_kwargs") or {}
        if isinstance(kwargs, Mapping) and kwargs.get("reasoning_content"):
            details["reasoning"] = _text(kwargs.get("reasoning_content"))
        return "message", "Assistant", _text(event.get("content")), details

    if kind == "tool":
        name = str(event.get("tool") or "unknown")
        details = {
            "args": _json_safe(event.get("args") or {}),
            "status": str(event.get("exit") or ""),
            "duration_ms": _as_int(event.get("duration_ms"), 0),
            "call_id": str(event.get("call_id") or ""),
        }
        if name == "spawn_subagent" and subagents:
            details["subagents"] = subagents
        return "tool", f"Tool · {name}", _text(event.get("result")), details

    if kind == "usage":
        details = {
            key: _json_safe(event.get(key))
            for key in (
                "model",
                "operation",
                "input_tokens",
                "cached_input_tokens",
                "output_tokens",
                "cost_usd",
                "inherited",
            )
            if event.get(key) not in (None, "")
        }
        model = str(event.get("model") or "model")
        parts = [model]
        if event.get("input_tokens") is not None:
            parts.append(f"{_as_int(event.get('input_tokens'), 0):,} input")
        if event.get("output_tokens") is not None:
            parts.append(f"{_as_int(event.get('output_tokens'), 0):,} output")
        return "event", "Usage", " · ".join(parts), details

    if kind == "approval":
        tool = str(event.get("tool") or "tool")
        decision = str(event.get("decision") or "unknown")
        details = {"tool": tool, "decision": decision}
        if event.get("rule"):
            details["rule"] = _text(event.get("rule"))
        return "event", "Approval", f"{tool}: {decision}", details

    if kind == "compact":
        return "event", "Compaction", _text(event.get("content")), {}

    if kind == "compaction_llm":
        details = {
            key: _json_safe(event.get(key))
            for key in (
                "trigger",
                "forced",
                "model",
                "before_tokens",
                "after_tokens",
                "active_suffix_messages",
                "source_event_seq",
                "active_user_seq",
            )
            if event.get(key) is not None
        }
        return "event", "Compaction checkpoint", _text(event.get("response")), details

    if kind == "reflection":
        response = event.get("response") or {}
        bullets = (
            response.get("new_bullet_points") if isinstance(response, Mapping) else []
        )
        content = "\n".join(f"• {_text(item)}" for item in bullets or [])
        if not content and event.get("error"):
            content = f"Reflection failed: {_text(event.get('error'))}"
        details = {
            "message_index": _as_int(event.get("message_index"), 0),
            "memory_updated": bool(event.get("memory_updated")),
        }
        return "event", "Reflection", content or "No new session memory", details

    if kind == "goal":
        phase = str(event.get("phase") or "event")
        content = _text(
            event.get("goal") or event.get("message") or event.get("content")
        )
        return "event", f"Goal · {phase}", content, _remaining_details(event)

    return (
        "event",
        kind.replace("_", " ").title(),
        _text(event.get("content")),
        _remaining_details(event),
    )


def _remaining_details(event: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): _json_safe(value)
        for key, value in event.items()
        if key not in _OMITTED_EVENT_KEYS
    }


def _safe_subagent(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: _json_safe(item.get(key))
        for key in (
            "subagent_thread_id",
            "agent_name",
            "label",
            "status",
            "started_at",
            "completed_at",
            "duration_ms",
            "output",
        )
        if item.get(key) not in (None, "")
    }


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(_json_safe(value), ensure_ascii=False, indent=2)


def _as_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _one_line(value: str, limit: int) -> str:
    cleaned = " ".join(value.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 1)].rstrip() + "…"


def _preview_from_details(details: Mapping[str, Any]) -> str:
    if not details:
        return ""
    return json.dumps(details, ensure_ascii=False, default=str)


def _session_title(
    metadata: Mapping[str, Any],
    records: Sequence[ExportRecord],
    thread_id: str,
) -> str:
    explicit = str(metadata.get("name") or "").strip()
    if explicit:
        return explicit
    for record in records:
        if record.kind == "user" and record.content.strip():
            return _one_line(record.content, 72)
    summary = str(metadata.get("summary") or "").strip()
    return summary or thread_id


def _format_datetime(value: str) -> str:
    if not value:
        return ""
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value
    return parsed.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def _time_only(value: str) -> str:
    if not value:
        return ""
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value
    return parsed.astimezone().strftime("%H:%M:%S")


def _icon_for(record: ExportRecord) -> str:
    icons = {
        "user": "person",
        "assistant": "spark",
        "tool": "tool",
        "compaction_llm": "compress",
        "compact": "compress",
        "usage": "meter",
        "approval": "check",
        "reflection": "memory",
        "goal": "target",
    }
    return icons.get(record.kind, "event")


def _render_nav_item(record: ExportRecord, index: int) -> str:
    preview = record.preview or "No text content"
    return (
        f'<a class="nav-item" data-category="{record.category}" '
        f'href="#entry-{record.seq}" data-entry="entry-{record.seq}">'
        f'<span class="nav-icon icon-{_icon_for(record)}" aria-hidden="true"></span>'
        f'<span class="nav-copy"><strong>{html.escape(record.title)}</strong>'
        f"<span>{html.escape(preview)}</span></span>"
        f'<span class="nav-number">{index:02}</span></a>'
    )


def _render_transcript_item(record: ExportRecord, index: int) -> str:
    content = record.content
    if not content and not record.details:
        content = "No additional content"
    body_parts: list[str] = []
    if content:
        body_parts.append(f'<pre class="content-block">{html.escape(content)}</pre>')
    for label, value in _detail_sections(record.details):
        body_parts.append(
            '<section class="detail-section">'
            f"<h3>{html.escape(label)}</h3>"
            f"<pre>{html.escape(value)}</pre>"
            "</section>"
        )
    time_text = _time_only(record.timestamp)
    return (
        f'<article class="entry" id="entry-{record.seq}" data-category="{record.category}">'
        "<details>"
        "<summary>"
        '<span class="chevron" aria-hidden="true"></span>'
        f'<span class="entry-icon icon-{_icon_for(record)}" aria-hidden="true"></span>'
        f'<span class="entry-number">{index:02}</span>'
        f'<span class="entry-title">{html.escape(record.title)}</span>'
        f'<span class="entry-preview">{html.escape(record.preview)}</span>'
        f'<span class="entry-kind">{html.escape(record.kind.replace("_", " "))}</span>'
        f"<time>{html.escape(time_text)}</time>"
        "</summary>"
        f'<div class="entry-body">{"".join(body_parts)}</div>'
        "</details>"
        "</article>"
    )


def _detail_sections(details: Mapping[str, Any]) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    for key, value in details.items():
        label = key.replace("_", " ").title()
        if isinstance(value, str):
            rendered = value
        else:
            rendered = json.dumps(value, ensure_ascii=False, indent=2, default=str)
        sections.append((label, rendered))
    return sections


def _script_json(value: str) -> str:
    """Encode a JS/JSON string while making HTML end tags impossible."""
    return json.dumps(value, ensure_ascii=False).replace("<", "\\u003c")


_HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'nonce-__CSP_NONCE__'; base-uri 'none'; form-action 'none'">
  <title>__PAGE_TITLE__ · Ness session export</title>
  <style>
    :root {
      color-scheme: light;
      --canvas: #f6f7fb;
      --surface: #ffffff;
      --surface-soft: #f0f2f8;
      --surface-hover: #e9edf7;
      --ink: #172036;
      --muted: #697187;
      --faint: #949caf;
      --line: #dfe3ec;
      --line-strong: #c7ccda;
      --accent: #4354a3;
      --accent-soft: #edf0ff;
      --tool: #806625;
      --event: #6e587f;
      --shadow: 0 12px 32px rgba(23, 32, 54, .06);
    }
    :root[data-theme="dark"] {
      color-scheme: dark;
      --canvas: #101217;
      --surface: #171a21;
      --surface-soft: #1d212a;
      --surface-hover: #252a35;
      --ink: #eef1f8;
      --muted: #aab1c2;
      --faint: #777f92;
      --line: #2b303b;
      --line-strong: #3a4150;
      --accent: #aab4ff;
      --accent-soft: #252b49;
      --tool: #d5ba70;
      --event: #d1aee2;
      --shadow: 0 14px 38px rgba(0, 0, 0, .25);
    }
    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body {
      margin: 0;
      min-width: 320px;
      background: var(--canvas);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 14px;
      line-height: 1.5;
    }
    button, a { font: inherit; }
    button { color: inherit; }
    .page { width: min(1240px, calc(100% - 48px)); margin: 0 auto; padding: 42px 0 64px; }
    .topline { display: flex; align-items: flex-start; justify-content: space-between; gap: 24px; }
    .eyebrow { margin: 0 0 10px; color: var(--muted); font-size: 11px; font-weight: 750; letter-spacing: .18em; text-transform: uppercase; }
    h1 { margin: 0; max-width: 820px; font-size: clamp(28px, 4vw, 44px); line-height: 1.08; letter-spacing: -.035em; }
    .meta { display: flex; flex-wrap: wrap; gap: 5px 18px; margin: 14px 0 0; color: var(--muted); font-size: 12px; }
    .meta span { min-width: 0; overflow-wrap: anywhere; }
    .theme-button, .control, .filter {
      min-height: 34px;
      border: 1px solid var(--line-strong);
      border-radius: 999px;
      background: var(--surface);
      cursor: pointer;
      transition: background .18s ease, border-color .18s ease, transform .18s ease;
    }
    .theme-button:hover, .control:hover, .filter:hover { background: var(--surface-hover); border-color: var(--accent); }
    .theme-button:active, .control:active, .filter:active { transform: translateY(1px); }
    .theme-button { flex: 0 0 38px; width: 38px; padding: 0; font-size: 17px; }
    .toolbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 18px;
      margin-top: 26px;
      padding: 10px 12px;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: color-mix(in srgb, var(--surface) 88%, transparent);
      box-shadow: var(--shadow);
    }
    .filters, .actions { display: flex; align-items: center; flex-wrap: wrap; gap: 8px; }
    .toolbar-label { margin-right: 4px; color: var(--muted); font-size: 10px; font-weight: 750; letter-spacing: .17em; text-transform: uppercase; }
    .filter { padding: 6px 13px; color: var(--muted); }
    .filter[aria-pressed="true"] { border-color: var(--accent); background: var(--accent-soft); color: var(--accent); font-weight: 700; }
    .filter .dot { display: inline-block; width: 6px; height: 6px; margin-right: 7px; border-radius: 50%; background: currentColor; vertical-align: 1px; }
    .count { margin-left: 6px; color: var(--faint); font-variant-numeric: tabular-nums; }
    .control { padding: 6px 13px; }
    .layout { display: grid; grid-template-columns: minmax(220px, 310px) minmax(0, 1fr); gap: 34px; margin-top: 34px; }
    .section-label { margin: 0 0 13px; color: var(--muted); font-size: 11px; font-weight: 750; letter-spacing: .17em; text-transform: uppercase; }
    .sidebar { min-width: 0; }
    .sidebar-inner { position: sticky; top: 24px; max-height: calc(100vh - 48px); overflow: hidden; }
    .nav-list { max-height: calc(100vh - 88px); overflow: auto; padding-right: 8px; scrollbar-width: thin; scrollbar-color: var(--line-strong) transparent; }
    .nav-item {
      display: grid;
      grid-template-columns: 18px minmax(0, 1fr) auto;
      gap: 9px;
      align-items: start;
      padding: 8px 6px;
      border-radius: 7px;
      color: inherit;
      text-decoration: none;
    }
    .nav-item:hover, .nav-item:focus-visible { outline: none; background: var(--surface-hover); }
    .nav-copy { min-width: 0; }
    .nav-copy strong, .nav-copy span { display: block; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .nav-copy strong { font-size: 12px; font-weight: 680; }
    .nav-copy span { margin-top: 1px; color: var(--muted); font-size: 11px; }
    .nav-number { color: var(--faint); font-size: 10px; font-variant-numeric: tabular-nums; }
    .nav-icon, .entry-icon { position: relative; width: 16px; height: 16px; color: var(--accent); }
    [class*="icon-"]::before { content: ""; position: absolute; inset: 3px; border: 1.5px solid currentColor; border-radius: 50%; }
    .icon-tool { color: var(--tool); }
    .icon-tool::before { inset: 4px 2px; border-radius: 2px; transform: rotate(-38deg); }
    .icon-spark::before { inset: 2px; border-radius: 2px; transform: rotate(45deg) scale(.72); }
    .icon-compress, .icon-meter, .icon-check, .icon-memory, .icon-target, .icon-event { color: var(--event); }
    .icon-compress::before { inset: 3px 2px; border-radius: 2px; }
    .icon-meter::before { inset: 4px 2px 2px; border-radius: 9px 9px 2px 2px; }
    .icon-check::after { content: ""; position: absolute; left: 4px; top: 4px; width: 7px; height: 4px; border-left: 1.5px solid currentColor; border-bottom: 1.5px solid currentColor; transform: rotate(-45deg); }
    .transcript { min-width: 0; }
    .entry { margin-bottom: 8px; border: 1px solid var(--line); border-radius: 9px; background: var(--surface); overflow: clip; transition: border-color .2s ease, box-shadow .2s ease; }
    .entry:hover { border-color: var(--line-strong); }
    .entry.flash { border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-soft); }
    summary {
      display: grid;
      grid-template-columns: 13px 17px auto auto minmax(80px, 1fr) auto auto;
      gap: 8px;
      align-items: center;
      min-height: 46px;
      padding: 9px 13px;
      cursor: pointer;
      list-style: none;
      background: var(--surface);
    }
    summary::-webkit-details-marker { display: none; }
    summary:hover { background: var(--surface-soft); }
    summary:focus-visible { outline: 2px solid var(--accent); outline-offset: -2px; }
    .chevron { width: 7px; height: 7px; border-right: 1.5px solid var(--muted); border-bottom: 1.5px solid var(--muted); transform: rotate(-45deg); transition: transform .18s ease; }
    details[open] .chevron { transform: rotate(45deg) translate(-2px, -2px); }
    .entry-number { font-weight: 760; font-variant-numeric: tabular-nums; }
    .entry-title { font-weight: 720; white-space: nowrap; }
    .entry-preview { min-width: 0; overflow: hidden; color: var(--muted); text-overflow: ellipsis; white-space: nowrap; }
    .entry-kind { color: var(--accent); font-size: 9px; font-weight: 760; letter-spacing: .12em; text-transform: uppercase; white-space: nowrap; }
    time { color: var(--faint); font-size: 10px; font-variant-numeric: tabular-nums; white-space: nowrap; }
    .entry-body { display: grid; gap: 14px; padding: 6px 48px 20px; border-top: 1px solid var(--line); background: var(--surface); }
    pre { margin: 0; overflow: auto; color: inherit; font: 12px/1.62 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; white-space: pre-wrap; overflow-wrap: anywhere; }
    .content-block { padding-top: 14px; font-family: inherit; font-size: 13px; line-height: 1.65; }
    .detail-section { padding-top: 2px; }
    .detail-section h3 { margin: 0 0 5px; color: var(--muted); font-size: 9px; letter-spacing: .14em; text-transform: uppercase; }
    .detail-section pre { padding: 10px 12px; border-left: 2px solid var(--line-strong); background: var(--surface-soft); }
    [hidden] { display: none !important; }
    .empty { display: none; padding: 64px 20px; color: var(--muted); text-align: center; }
    .empty.visible { display: block; }
    @media (max-width: 820px) {
      .page { width: min(100% - 28px, 720px); padding-top: 26px; }
      .toolbar { align-items: flex-start; flex-direction: column; }
      .layout { grid-template-columns: 1fr; gap: 28px; }
      .sidebar-inner { position: static; max-height: none; }
      .nav-list { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); max-height: 260px; }
      summary { grid-template-columns: 13px 17px auto minmax(60px, 1fr) auto; }
      .entry-kind, summary time { display: none; }
      .entry-body { padding-left: 38px; padding-right: 20px; }
    }
    @media (max-width: 520px) {
      .page { width: min(100% - 20px, 480px); }
      .meta { display: grid; }
      .nav-list { grid-template-columns: 1fr; }
      .filters { width: 100%; }
      .filter { flex: 1; padding-inline: 9px; }
      summary { grid-template-columns: 11px 15px auto minmax(0, 1fr); gap: 7px; padding-inline: 10px; }
      .entry-preview { grid-column: 4; }
      .entry-title { display: none; }
      .entry-body { padding-left: 20px; }
    }
    @media print {
      .theme-button, .toolbar, .sidebar { display: none !important; }
      .page { width: 100%; padding: 0; }
      .layout { display: block; }
      .entry { break-inside: avoid; box-shadow: none; }
      details > .entry-body { display: grid !important; }
    }
  </style>
</head>
<body>
  <main class="page">
    <header>
      <div class="topline">
        <div>
          <p class="eyebrow">Ness session export</p>
          <h1>__PAGE_TITLE__</h1>
          <p class="meta">
            <span>Thread: __THREAD_ID__</span>
            <span>Project: __PROJECT_ROOT__</span>
            <span>Model: __MODEL__</span>
            <span>Started: __STARTED_AT__</span>
            <span>Generated: __GENERATED_AT__</span>
          </p>
        </div>
        <button class="theme-button" id="theme-toggle" type="button" aria-label="Toggle color theme" title="Toggle color theme">☼</button>
      </div>
      <div class="toolbar" aria-label="Transcript controls">
        <div class="filters">
          <span class="toolbar-label">View</span>
          <button class="filter" type="button" data-filter="message" aria-pressed="true"><span class="dot"></span>Messages <span class="count">__MESSAGE_COUNT__</span></button>
          <button class="filter" type="button" data-filter="tool" aria-pressed="true"><span class="dot"></span>Tools <span class="count">__TOOL_COUNT__</span></button>
          <button class="filter" type="button" data-filter="event" aria-pressed="true"><span class="dot"></span>Events <span class="count">__EVENT_COUNT__</span></button>
        </div>
        <div class="actions">
          <button class="control" id="expand-all" type="button">Expand all</button>
          <button class="control" id="download-jsonl" type="button">↓ JSONL</button>
        </div>
      </div>
    </header>
    <div class="layout">
      <aside class="sidebar">
        <div class="sidebar-inner">
          <h2 class="section-label">Session · __TOTAL_COUNT__ entries</h2>
          <nav class="nav-list" aria-label="Session entries">__NAV_ITEMS__</nav>
        </div>
      </aside>
      <section class="transcript" aria-labelledby="transcript-title">
        <h2 class="section-label" id="transcript-title">Transcript</h2>
        <div id="entries">__TRANSCRIPT_ITEMS__</div>
        <p class="empty" id="empty-state">No entries match the active filters.</p>
      </section>
    </div>
  </main>
  <script id="export-jsonl" type="application/json" nonce="__CSP_NONCE__">__JSONL_PAYLOAD__</script>
  <script nonce="__CSP_NONCE__">
    (() => {
      const root = document.documentElement;
      const filterButtons = [...document.querySelectorAll('[data-filter]')];
      const entries = [...document.querySelectorAll('.entry')];
      const navItems = [...document.querySelectorAll('.nav-item')];
      const active = new Set(['message', 'tool', 'event']);
      const empty = document.getElementById('empty-state');
      const expand = document.getElementById('expand-all');
      const applyFilters = () => {
        let visible = 0;
        [...entries, ...navItems].forEach((node) => {
          const show = active.has(node.dataset.category);
          node.hidden = !show;
          if (show && node.classList.contains('entry')) visible += 1;
        });
        empty.classList.toggle('visible', visible === 0);
      };
      filterButtons.forEach((button) => button.addEventListener('click', () => {
        const category = button.dataset.filter;
        if (active.has(category)) active.delete(category); else active.add(category);
        button.setAttribute('aria-pressed', String(active.has(category)));
        applyFilters();
      }));
      expand.addEventListener('click', () => {
        const visible = entries.filter((entry) => !entry.hidden);
        const shouldOpen = visible.some((entry) => !entry.querySelector('details').open);
        visible.forEach((entry) => { entry.querySelector('details').open = shouldOpen; });
        expand.textContent = shouldOpen ? 'Collapse all' : 'Expand all';
      });
      navItems.forEach((item) => item.addEventListener('click', (event) => {
        event.preventDefault();
        const target = document.getElementById(item.dataset.entry);
        if (!active.has(target.dataset.category)) {
          active.add(target.dataset.category);
          const button = document.querySelector(`[data-filter="${target.dataset.category}"]`);
          button.setAttribute('aria-pressed', 'true');
          applyFilters();
        }
        target.querySelector('details').open = true;
        target.scrollIntoView({ behavior: 'smooth', block: 'center' });
        target.classList.add('flash');
        window.setTimeout(() => target.classList.remove('flash'), 1200);
      }));
      document.getElementById('theme-toggle').addEventListener('click', () => {
        root.dataset.theme = root.dataset.theme === 'dark' ? 'light' : 'dark';
      });
      if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
        root.dataset.theme = 'dark';
      }
      document.getElementById('download-jsonl').addEventListener('click', () => {
        const jsonl = JSON.parse(document.getElementById('export-jsonl').textContent);
        const blob = new Blob([jsonl], { type: 'application/x-ndjson;charset=utf-8' });
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.download = __JSONL_NAME__;
        link.click();
        window.setTimeout(() => URL.revokeObjectURL(link.href), 0);
      });
    })();
  </script>
</body>
</html>
"""
