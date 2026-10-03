"""Concise, tool-aware transcript summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


BATCHABLE_TOOL_CALLS = frozenset({"read", "grep", "glob"})
_ERROR_STATUSES = frozenset(
    {"error", "failed", "timeout", "denied", "mode_gated"}
)
_ARG_LIMIT = 140
_SHELL_PANEL_LIMIT = 20_000
_SUBAGENT_PANEL_LIMIT = 8_000


def format_tool_args(name: str, arguments: Any) -> str:
    if not isinstance(arguments, dict):
        return _truncate(arguments, _ARG_LIMIT)
    key = name.lower()
    if key == "read":
        parts = [_short_path(arguments.get("path"))]
        offset = _integer(arguments.get("offset"), 1)
        if offset != 1:
            parts.append(f"from {offset}")
        limit = arguments.get("limit")
        if limit is not None and _integer(limit, 400) != 400:
            parts.append(f"{_integer(limit, 400)} lines")
        return " · ".join(parts)
    if key == "grep":
        parts = [_truncate(arguments.get("pattern"), 56)]
        if arguments.get("glob"):
            parts.append(str(arguments["glob"]))
        path = str(arguments.get("path") or ".")
        if path != ".":
            parts.append(_short_path(path))
        return " · ".join(part for part in parts if part)
    if key == "glob":
        return _truncate(arguments.get("pattern") or arguments.get("glob") or "?", 100)
    if key == "delete" and isinstance(arguments.get("paths"), list):
        paths = arguments["paths"]
        if len(paths) == 1:
            return _short_path(paths[0])
        return f"{len(paths)} files"
    if key in {"write", "delete"}:
        return _short_path(arguments.get("path"))
    if key == "edit":
        suffix = " · all matches" if arguments.get("replace_all") else ""
        return _short_path(arguments.get("path")) + suffix
    if key == "shell":
        return _shell_args(arguments)
    if key in {"web_search", "search_tools"}:
        return _truncate(arguments.get("query"), 100)
    if key == "fetch_url":
        return _truncate(arguments.get("url"), 100)
    if key == "add_tools":
        names = arguments.get("names") or []
        if isinstance(names, str):
            names = [names]
        return _truncate(", ".join(str(item) for item in names), 120)
    if key == "question":
        questions = arguments.get("questions") or []
        if questions and isinstance(questions[0], dict):
            first = _truncate(questions[0].get("prompt"), 90)
            return first if len(questions) == 1 else f"{first} · +{len(questions) - 1}"
        return ""
    if key == "todo":
        todos = arguments.get("todos") or []
        return f"{len(todos)} item{'s' if len(todos) != 1 else ''}"
    if key == "spawn_subagent":
        return _subagent_args(arguments)
    if name.startswith("mcp__"):
        short_name = name.removeprefix("mcp__").replace("__", "/", 1)
        detail = _generic_args(arguments, limit=90)
        return f"{short_name} · {detail}" if detail else short_name
    return _generic_args(arguments, limit=_ARG_LIMIT)


def format_batched_tool_args(name: str, calls: list[dict[str, Any]]) -> str:
    details = [format_tool_args(name, call) for call in calls]
    return "  |  ".join(detail for detail in details if detail)


def is_tool_result_error(content: str) -> bool:
    text = str(content or "").lstrip()
    if text.startswith(("Error:", "Hook veto:")):
        return True
    status = result_status(text)
    if status in _ERROR_STATUSES:
        return True
    if text.startswith("{"):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return False
        return isinstance(payload, dict) and bool(payload.get("error"))
    return False


def should_show_tool_result(
    name: str,
    content: str,
    *,
    exit_status: str | None = None,
) -> bool:
    if name.lower() in {"edit", "write", "shell", "spawn_subagent"}:
        return True
    if is_tool_result_error(content):
        return True
    return bool(exit_status and exit_status not in {"ok", "0"})


def result_status(content: str) -> str | None:
    for line in str(content or "").splitlines()[:8]:
        if line.startswith("status="):
            return line.partition("=")[2].strip() or None
    return None


def extract_diff_section(content: str) -> str | None:
    marker = "\ndiff:\n"
    text = str(content or "")
    start = text.find(marker)
    if start < 0:
        return None
    diff = text[start + len(marker) :].strip()
    return diff or None


def extract_edit_summary(content: str) -> str:
    text = str(content or "").strip()
    marker = "\ndiff:\n"
    start = text.find(marker)
    if start >= 0:
        return text[:start].strip()
    return text.splitlines()[0] if text else ""


def format_shell_output(
    content: str,
    *,
    exit_status: str | None,
) -> tuple[str, str]:
    text = str(content or "")
    status = result_status(text) or exit_status or (
        "error" if is_tool_result_error(text) else "ok"
    )
    exit_code = _header_field(text, "exit_code")
    title = f"shell {status}"
    if exit_code and exit_code != "0" and exit_code not in title:
        title += f" · exit {exit_code}"
    return title, _bounded_tail(_output_section(text), limit=_SHELL_PANEL_LIMIT)


def format_subagent_output(content: str) -> tuple[str, str]:
    text = str(content or "").strip()
    status = result_status(text) or (
        "error" if is_tool_result_error(text) else "ok"
    )
    lines = text.splitlines()
    while lines and (
        not lines[0].strip()
        or lines[0].startswith(
            ("status=", "duration_ms=", "tasks_total=", "tasks_ok=", "tasks_failed=")
        )
    ):
        lines.pop(0)
    return f"subagent {status}", _bounded_tail(
        "\n".join(lines).strip() or text,
        limit=_SUBAGENT_PANEL_LIMIT,
    )


def bounded_diff(diff: str, *, max_lines: int = 240, max_chars: int = 16_000) -> str:
    text = str(diff or "")
    lines = text.splitlines()
    clipped = len(text) > max_chars or len(lines) > max_lines
    kept: list[str] = []
    count = 0
    for line in lines[:max_lines]:
        added = len(line) + (1 if kept else 0)
        if count + added > max_chars:
            break
        kept.append(line)
        count += added
    if clipped or len(kept) < len(lines):
        kept.append(f"... diff truncated, {len(lines)} lines total")
    return "\n".join(kept)


def _shell_args(arguments: dict[str, Any]) -> str:
    action = str(arguments.get("action") or "run").lower()
    if action in {"run", "start"}:
        detail = _truncate(arguments.get("command"), 110)
        return f"{action} · {detail}" if detail else action
    if action in {"read", "kill"}:
        job = str(arguments.get("job_id") or "?")
        return f"{action} · {job}"
    return action


def _subagent_args(arguments: dict[str, Any]) -> str:
    tasks = arguments.get("tasks") or []
    if not isinstance(tasks, list):
        return ""
    summaries = []
    for task in tasks[:3]:
        if not isinstance(task, dict):
            continue
        name = str(task.get("label") or task.get("name") or "agent")
        prompt = _truncate(task.get("prompt"), 60)
        summaries.append(f"{name}: {prompt}" if prompt else name)
    if len(tasks) > 3:
        summaries.append(f"+{len(tasks) - 3}")
    return " | ".join(summaries)


def _generic_args(arguments: dict[str, Any], *, limit: int) -> str:
    parts = []
    for key, value in arguments.items():
        if value in (None, "", [], {}):
            continue
        parts.append(f"{key}={_truncate(value, 48)}")
    return _truncate("  ".join(parts), limit)


def _short_path(value: Any) -> str:
    raw = str(value or "?").strip()
    try:
        return str(Path(raw).expanduser().relative_to(Path.cwd()))
    except ValueError:
        return raw


def _truncate(value: Any, limit: int) -> str:
    cleaned = " ".join(str(value or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 1)] + "…"


def _integer(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _header_field(content: str, field: str) -> str | None:
    prefix = f"{field}="
    for line in content.splitlines()[:8]:
        if line.startswith(prefix):
            return line.removeprefix(prefix).strip() or None
    return None


def _output_section(content: str) -> str:
    lines = content.splitlines()
    for index, line in enumerate(lines):
        if line.strip() == "output:":
            return "\n".join(lines[index + 1 :]).strip()
    return content.strip()


def _bounded_tail(text: str, *, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"... showing last {limit:,} characters\n{text[-limit:]}"
