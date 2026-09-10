"""Selective tool-call summaries for the CLI transcript."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Consecutive read/grep/glob/todo calls are merged onto one line.
SILENT_RESULT_TOOLS = frozenset(
    {
        "read",
        "grep",
        "glob",
        "todo",
    }
)
BATCHABLE_TOOL_CALLS = frozenset(SILENT_RESULT_TOOLS)

_ERROR_STATUSES = frozenset({"error", "failed", "timeout", "denied", "mode_gated"})

_PROMPT_PREVIEW = 160
_DEFAULT_ARG_PREVIEW = 120
_DEFAULT_RESULT_PREVIEW = 320
_READ_DEFAULT_LIMIT = 400  # mirrors tools/fs.py READ_FILE_DEFAULT_LIMIT
_SHELL_DEFAULT_TIMEOUT = 30
_SHELL_DEFAULT_OUTPUT_CHARS = 12_000
_WEB_SEARCH_DEFAULT_MAX_RESULTS = 5
_FETCH_URL_DEFAULT_MAX_CHARACTERS = 12_000
_SEARCH_TOOLS_DEFAULT_LIMIT = 5


def parse_result_status(content: str) -> str | None:
    """Return the status= header value from structured tool output, if present."""
    for line in str(content or "").splitlines()[:8]:
        if line.startswith("status="):
            status = line.removeprefix("status=").strip()
            return status or None
    return None


def is_tool_result_error(content: str) -> bool:
    """True when tool output represents a failure (shown on the CLI transcript)."""
    text = str(content or "").lstrip()
    if not text:
        return False
    if text.startswith("Error:") or text.startswith("Hook veto:"):
        return True
    status = parse_result_status(text)
    if status in _ERROR_STATUSES:
        return True
    if text.startswith("{"):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return False
        if isinstance(payload, dict) and payload.get("error"):
            return True
    return False


def should_show_tool_result(
    name: str,
    content: str,
    *,
    exit_status: str | None = None,
) -> bool:
    """Show tool results on the CLI when informative.

    edit/write always surface (so the summary line precedes the rendered diff)
    and shell always surfaces (its output panel is the primary signal).
    Other tools show only on failure.
    """
    if name in ("edit", "write", "shell"):
        return True
    if is_tool_result_error(content):
        return True
    return exit_status is not None and exit_status not in ("ok",)


def should_show_tool_call(name: str) -> bool:
    return True


def _truncate(text: str, limit: int) -> str:
    cleaned = " ".join(str(text).split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 1)] + "…"


def _truncate_arg(text: str, limit: int) -> str:
    """Truncate tool-call arg previews; suffix with '...' when clipped."""
    cleaned = " ".join(str(text).split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 3)] + "..."


def _short_path(path: str) -> str:
    raw = str(path or "").strip()
    if not raw:
        return "?"
    try:
        resolved = Path(raw).expanduser()
        return str(resolved.relative_to(Path.cwd()))
    except ValueError:
        return raw


def _join_parts(*parts: str) -> str:
    return " | ".join(part for part in parts if part)


def _read_token(args: dict[str, Any]) -> str:
    parts: list[str] = []
    offset = int(args.get("offset", 1) or 1)
    if offset != 1:
        parts.append(str(offset))
    parts.append(_short_path(str(args.get("path", ""))))
    limit = args.get("limit")
    if limit is not None and int(limit) != _READ_DEFAULT_LIMIT:
        parts.append(str(int(limit)))
    return _join_parts(*parts)


def _grep_token(args: dict[str, Any]) -> str:
    parts = [_truncate_arg(str(args.get("pattern", "")), 40)]
    file_filter = str(args.get("glob") or "").strip()
    if file_filter:
        parts.append(file_filter)
    path = _short_path(str(args.get("path", ".")))
    if path and path != ".":
        parts.append(path)
    return _join_parts(*parts)


def _glob_token(args: dict[str, Any]) -> str:
    return str(args.get("pattern", "") or args.get("glob", "") or "?")


def _path_token(args: dict[str, Any]) -> str:
    return _short_path(str(args.get("path", "")))


def _paths_token(args: dict[str, Any]) -> str:
    paths = args.get("paths")
    if not isinstance(paths, list) or not paths:
        return ""
    if len(paths) == 1:
        return _short_path(str(paths[0]))
    return f"{len(paths)} files"


def _edit_token(args: dict[str, Any]) -> str:
    path = _short_path(str(args.get("path", "")))
    return _join_parts(path, "1 edit")


def _shell_token(args: dict[str, Any]) -> str:
    action = str(args.get("action") or "run").strip().lower()
    if action == "run":
        parts = ["run", _truncate_arg(str(args.get("command", "")), _DEFAULT_ARG_PREVIEW)]
        timeout = int(args.get("timeout", _SHELL_DEFAULT_TIMEOUT) or _SHELL_DEFAULT_TIMEOUT)
        if timeout != _SHELL_DEFAULT_TIMEOUT:
            parts.append(str(timeout))
        return _join_parts(*parts)
    if action == "start":
        parts = ["start"]
        name = str(args.get("name") or "").strip()
        if name:
            parts.append(name)
        parts.append(_truncate_arg(str(args.get("command", "")), _DEFAULT_ARG_PREVIEW))
        return _join_parts(*parts)
    if action == "jobs":
        if args.get("include_finished", True) is False:
            return "jobs | active"
        return "jobs"
    if action == "read":
        parts = ["read", str(args.get("job_id", ""))]
        tail = int(args.get("tail_chars", _SHELL_DEFAULT_OUTPUT_CHARS) or _SHELL_DEFAULT_OUTPUT_CHARS)
        if tail != _SHELL_DEFAULT_OUTPUT_CHARS:
            parts.append(str(tail))
        return _join_parts(*parts)
    if action == "kill":
        parts = ["kill", str(args.get("job_id", ""))]
        if args.get("force"):
            parts.append("force")
        return _join_parts(*parts)
    return _join_parts(action, _truncate_arg(str(args.get("command", "")), _DEFAULT_ARG_PREVIEW))


def _web_search_token(args: dict[str, Any]) -> str:
    parts = [_truncate_arg(str(args.get("query", "")), 80)]
    max_results = int(args.get("max_results", _WEB_SEARCH_DEFAULT_MAX_RESULTS) or _WEB_SEARCH_DEFAULT_MAX_RESULTS)
    if max_results != _WEB_SEARCH_DEFAULT_MAX_RESULTS:
        parts.append(str(max_results))
    search_type = str(args.get("search_type") or "auto")
    if search_type != "auto":
        parts.append(search_type)
    domains = args.get("include_domains")
    if isinstance(domains, list) and domains:
        parts.append(_truncate_arg(", ".join(str(domain) for domain in domains), 60))
    return _join_parts(*parts)


def _fetch_url_token(args: dict[str, Any]) -> str:
    parts = [_truncate_arg(str(args.get("url", "")), 80)]
    max_characters = int(
        args.get("max_characters", _FETCH_URL_DEFAULT_MAX_CHARACTERS) or _FETCH_URL_DEFAULT_MAX_CHARACTERS
    )
    if max_characters != _FETCH_URL_DEFAULT_MAX_CHARACTERS:
        parts.append(str(max_characters))
    return _join_parts(*parts)


def _search_tools_token(args: dict[str, Any]) -> str:
    parts = [_truncate_arg(str(args.get("query", "")), 80)]
    limit = int(args.get("limit", _SEARCH_TOOLS_DEFAULT_LIMIT) or _SEARCH_TOOLS_DEFAULT_LIMIT)
    if limit != _SEARCH_TOOLS_DEFAULT_LIMIT:
        parts.append(str(limit))
    return _join_parts(*parts)


def _add_tools_token(args: dict[str, Any]) -> str:
    names = args.get("names") or []
    if isinstance(names, str):
        names = [names]
    if not isinstance(names, list) or not names:
        return ""
    if len(names) == 1:
        return _truncate_arg(str(names[0]), 80)
    if len(names) <= 3:
        return _truncate_arg(", ".join(str(name) for name in names), 120)
    return _truncate_arg(f"{names[0]} +{len(names) - 1}", 80)


def _question_token(args: dict[str, Any]) -> str:
    questions = args.get("questions") or []
    if not isinstance(questions, list) or not questions:
        return ""
    first = questions[0]
    prompt = _truncate_arg(str(first.get("prompt", "") if isinstance(first, dict) else ""), 80)
    if len(questions) == 1:
        return prompt
    return _join_parts(prompt, f"+{len(questions) - 1}")


def _mcp_arg_names(full_name: str) -> list[str]:
    from ness_agent.session_context import try_get_session_context

    ctx = try_get_session_context()
    catalog: dict = {}
    if ctx and ctx.agent_config and ctx.agent_config.tool_registry is not None:
        catalog = ctx.agent_config.tool_registry.mcp_catalog()
    for info in catalog.values():
        for entry in info.get("tools", []):
            if str(entry.get("name") or "") == full_name:
                raw = entry.get("arg_names", [])
                return [str(name) for name in raw] if isinstance(raw, list) else []
    return []


def _mcp_short_label(full_name: str) -> str:
    rest = full_name.removeprefix("mcp__")
    server, _, tool = rest.partition("__")
    return f"{server}/{tool}" if tool else rest


def _mcp_token(full_name: str, args: dict[str, Any]) -> str:
    label = _mcp_short_label(full_name)
    arg_names = _mcp_arg_names(full_name)
    keys = arg_names or [str(key) for key in args]
    shown: list[str] = []
    for key in keys:
        if key not in args:
            continue
        value = args[key]
        if value in (None, "", [], {}):
            continue
        shown.append(f"{key}={_truncate_arg(value, 48)}")
        if len(shown) >= 2:
            break
    if not shown and not arg_names:
        for key, value in args.items():
            if value in (None, "", [], {}):
                continue
            shown.append(f"{key}={_truncate_arg(value, 48)}")
            if len(shown) >= 2:
                break
    return _join_parts(label, *shown) if shown else label


def _spawn_subagent_lines(args: dict[str, Any]) -> list[tuple[str, str]]:
    """Return (name, args_text) rows; args_text may contain newlines."""
    tasks = args.get("tasks")
    if isinstance(tasks, list) and tasks:
        rows: list[tuple[str, str]] = []
        for task in tasks:
            if not isinstance(task, dict):
                continue
            agent = str(task.get("name", "?"))
            prompt = _truncate_arg(str(task.get("prompt", "")), _PROMPT_PREVIEW)
            rows.append((agent, prompt))
        if rows:
            return rows
    return [("?", "")]


def _generic_args_text(args: Any) -> str:
    if not isinstance(args, dict) or not args:
        return ""
    parts: list[str] = []
    for key, value in args.items():
        if value in (None, "", [], {}):
            continue
        parts.append(f"{key}={_truncate_arg(value, 48)}")
    return _truncate_arg("  ".join(parts), _DEFAULT_ARG_PREVIEW)


def format_tool_args(name: str, args: Any) -> str:
    """Single-line args summary for unknown / generic tools."""
    if not isinstance(args, dict):
        return _truncate_arg(str(args), _DEFAULT_ARG_PREVIEW)

    if name == "read":
        return _read_token(args)
    if name == "grep":
        return _grep_token(args)
    if name == "glob":
        return _glob_token(args)
    if name == "spawn_subagent":
        rows = _spawn_subagent_lines(args)
        if len(rows) == 1:
            agent, prompt = rows[0]
            return _join_parts(agent, prompt) if prompt else agent
        return "\n".join(_join_parts(agent, prompt) if prompt else agent for agent, prompt in rows)
    if name == "todo":
        return ""
    if name == "shell":
        return _shell_token(args)
    if name == "write":
        return _path_token(args)
    if name == "edit":
        return _edit_token(args)
    if name == "delete":
        return _paths_token(args)
    if name == "web_search":
        return _web_search_token(args)
    if name == "fetch_url":
        return _fetch_url_token(args)
    if name == "search_tools":
        return _search_tools_token(args)
    if name == "add_tools":
        return _add_tools_token(args)
    if name == "question":
        return _question_token(args)
    if name.startswith("mcp__"):
        return _mcp_token(name, args)
    return _generic_args_text(args)


def format_batched_tool_args(name: str, calls: list[dict[str, Any]]) -> str:
    """Side-by-side summary for a run of identical batchable tool calls."""
    tokens: list[str] = []
    for call in calls:
        args = call.get("args") or {}
        if not isinstance(args, dict):
            continue
        if name == "read":
            tokens.append(_read_token(args))
        elif name == "grep":
            tokens.append(_grep_token(args))
        elif name == "glob":
            tokens.append(_glob_token(args))
        else:
            tokens.append(format_tool_args(name, args))
    return "  ".join(token for token in tokens if token)


def _extract_output_section(content: str) -> str:
    lines = str(content or "").splitlines()
    for index, line in enumerate(lines):
        if line.strip() == "output:":
            body = "\n".join(lines[index + 1 :]).strip()
            if body:
                return body
    return str(content or "").strip()


def _json_error_message(content: str) -> str | None:
    text = str(content or "").lstrip()
    if not text.startswith("{"):
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(payload, dict) and payload.get("error"):
        return str(payload["error"])
    return None


def _spawn_subagent_error_preview(content: str) -> str:
    text = str(content or "").strip()
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("error="):
            return stripped.removeprefix("error=").strip()
        if stripped.startswith("Error:"):
            return stripped
    return text


def format_tool_result_preview(name: str, content: str, *, limit: int = _DEFAULT_RESULT_PREVIEW) -> str:
    """One-line preview for a tool result.

    For edit/write the result body carries a trailing unified diff that is
    rendered separately; only the summary line is previewed here.
    """
    text = str(content or "").strip()
    if not text:
        return ""

    if name in ("edit", "write"):
        return _truncate(extract_edit_summary(text), limit)

    json_error = _json_error_message(text)
    if json_error is not None:
        return _truncate(json_error, limit)

    if name == "shell" and parse_result_status(text) in _ERROR_STATUSES:
        return _truncate(_extract_output_section(text), limit)

    if name == "spawn_subagent":
        return _truncate(_spawn_subagent_error_preview(text), limit)

    if parse_result_status(text) in _ERROR_STATUSES:
        body = _extract_output_section(text)
        if body != text:
            return _truncate(body, limit)

    return _truncate(text.replace("\n", " "), limit)


_DIFF_MARKER = "\ndiff:\n"


def extract_diff_section(content: str) -> str | None:
    """Return the unified diff embedded in an edit/write result, if present."""
    text = str(content or "")
    index = text.find(_DIFF_MARKER)
    if index == -1:
        return None
    body = text[index + len(_DIFF_MARKER):].strip()
    return body or None


def extract_edit_summary(content: str) -> str:
    """Return the summary line of an edit/write result (before the diff)."""
    text = str(content or "").strip()
    index = text.find(_DIFF_MARKER)
    if index != -1:
        return text[:index].strip()
    first = text.splitlines()[0] if text else ""
    return first.strip()


_SHELL_DISPLAY_LIMIT = 8000


def _shell_field(content: str, field: str) -> str | None:
    for line in str(content or "").splitlines()[:8]:
        if line.startswith(f"{field}="):
            return line.split("=", 1)[1].strip() or None
    return None


def format_shell_output(content: str) -> tuple[str, str]:
    """Return (header, body) for a shell tool result.

    ``header`` summarises status/exit; ``body`` is the captured output
    (bounded) suitable for a multi-line panel.
    """
    text = str(content or "")
    status = parse_result_status(text)
    if status is None:
        status = "error" if is_tool_result_error(text) else "ok"
    exit_code = _shell_field(text, "exit_code")
    body = _extract_output_section(text)
    if len(body) > _SHELL_DISPLAY_LIMIT:
        body = "..." + body[-_SHELL_DISPLAY_LIMIT:]
    header = f"status={status}"
    if exit_code and exit_code != "0":
        header += f" exit={exit_code}"
    return header, body


_SUBAGENT_META_PREFIXES = ("status=", "duration_ms=", "tasks_total=", "tasks_ok=", "tasks_failed=")


def _subagent_body_after_meta(text: str) -> str:
    """Drop leading structured metadata lines; keep the rest of the report."""
    lines = str(text or "").splitlines()
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        if not stripped:
            index += 1
            break
        if stripped.startswith("error=") and index == 0:
            # batch call-level error: keep full text
            return text.strip()
        if any(stripped.startswith(prefix) for prefix in _SUBAGENT_META_PREFIXES):
            index += 1
            continue
        break
    body = "\n".join(lines[index:]).strip()
    return body or text.strip()


def format_subagent_output(content: str) -> tuple[str, str]:
    """Return (header, body) for a spawn_subagent tool result panel.

    ``header`` is a short status label; ``body`` is the full report (bounded).
    """
    text = str(content or "").strip()
    status = parse_result_status(text)
    if status is None:
        status = "error" if is_tool_result_error(text) else "ok"
    header = f"subagent {status}"
    if text.startswith("status="):
        body = _subagent_body_after_meta(text)
    else:
        body = text
    if len(body) > _SHELL_DISPLAY_LIMIT:
        body = "..." + body[-_SHELL_DISPLAY_LIMIT:]
    return header, body

