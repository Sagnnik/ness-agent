from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Literal

from langchain_core.tools import tool

from ness_agent.session_context import get_session_context
from ness_agent.tools.shell_processes import ProcessManager

DEFAULT_OUTPUT_CHARS = 12_000
MAX_OUTPUT_CHARS = 200_000
SHELL_STATUSES = {"ok", "failed", "timeout", "cancelled", "running", "killed", "error"}


def _project_root() -> Path:
    return get_session_context().project_root


def _process_manager() -> ProcessManager:
    return get_session_context().get_shell_process_manager()


@tool
def shell(
    command: str = "",
    action: Literal["run", "start", "jobs", "read", "kill"] = "run",
    timeout: float | None = None,
    max_output_chars: int = DEFAULT_OUTPUT_CHARS,
    name: str = "",
    include_finished: bool = True,
    job_id: str = "",
    tail_chars: int = DEFAULT_OUTPUT_CHARS,
    force: bool = False,
    offset: int | None = None,
) -> str:
    """Execute a bash command in the project root (cwd is already the project root).

    Defaults to a synchronous foreground run when `action` is omitted.

    Supported Actions & Required Parameters:
      - 'run' (default): Execute a synchronous foreground command.
        -> Required: `command`
        -> Optional: `timeout` in seconds, `max_output_chars`. Omitted timeout uses host default.
        -> Returns: effective timeout, execution `job_id`, and retained `log_path`.
      - 'start': Run a background process until stopped or the owning session closes.
        -> Required: `command`
        -> Optional: `name`
      - 'jobs': List this session's background executions and runtime states.
        -> Optional: `include_finished`
      - 'read': Read logs for a foreground execution or background job.
        -> Required: `job_id`
        -> Optional: `tail_chars`, `offset` (zero-based character offset).
        -> Omit offset for the tail; set offset=0 to read from the beginning.
        -> Continue paginated reads with the returned `next_offset`.
      - 'kill': Forcefully terminate or gracefully stop a running background job group.
        -> Required: `job_id`
        -> Optional: `force`
    """
    if action == "run":
        return _shell_run(command, timeout=timeout, max_output_chars=max_output_chars)
    started = time.monotonic()
    try:
        if action == "start":
            return _shell_start(command, name=name)
        if action == "jobs":
            return _shell_jobs(include_finished=include_finished)
        if action == "read":
            return _shell_read(job_id, tail_chars=tail_chars, offset=offset)
        if action == "kill":
            return _shell_kill(job_id, force=force)
    except (OSError, ValueError, RuntimeError) as exc:
        return _format_result(
            "error",
            duration_ms=_duration_ms(started),
            output=str(exc),
            output_truncated=False,
            job_id=job_id or None,
        )

    return _format_result(
        "error",
        duration_ms=0,
        output=f"Unknown shell action: {action}",
        output_truncated=False,
    )


def _shell_run(
    command: str,
    timeout: float | None = None,
    max_output_chars: int = DEFAULT_OUTPUT_CHARS,
) -> str:
    """Run Bash with retained logs, cancellable waiting, and explicit limits."""
    started = time.monotonic()
    if not command.strip():
        return _format_result(
            "error",
            duration_ms=0,
            output="Empty shell command.",
            output_truncated=False,
        )
    ctx = get_session_context()
    try:
        effective_timeout, timeout_reason = _resolve_timeout(timeout)
        if ctx.shell_cancel_event.is_set():
            return _format_result(
                "cancelled",
                duration_ms=0,
                output="Command cancelled before launch.",
                output_truncated=False,
            )
        if effective_timeout <= 0:
            return _format_result(
                "timeout",
                duration_ms=0,
                output="Session shell deadline has expired; command was not launched.",
                output_truncated=False,
                timeout_seconds=0,
                timeout_reason="deadline",
            )
        job, output, truncated = _process_manager().run(
            command,
            timeout=effective_timeout,
            max_chars=_clamp_output_chars(max_output_chars),
            cancel_event=ctx.shell_cancel_event,
            deadline=ctx.shell_deadline,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        return _format_result(
            "error",
            duration_ms=_duration_ms(started),
            output=str(exc),
            output_truncated=False,
        )
    if job["status"] == "timeout":
        output = f"Command timed out after {effective_timeout:g}s ({timeout_reason})\n{output}"
    elif job["status"] == "cancelled":
        output = f"Command cancelled\n{output}"
    return _format_result(
        job["status"],
        exit_code=job["exit_code"],
        duration_ms=_duration_ms(started),
        output=output,
        output_truncated=truncated,
        job_id=job["job_id"],
        log_path=job["log_path"],
        timeout_seconds=effective_timeout,
        timeout_reason=timeout_reason,
    )


def _shell_start(command: str, name: str = "") -> str:
    started = time.monotonic()
    job = _process_manager().start(command, name=name)
    return _format_result(
        "running",
        duration_ms=_duration_ms(started),
        output=_render_job(job),
        output_truncated=False,
        job_id=job["job_id"],
    )


def _shell_jobs(include_finished: bool = True) -> str:
    started = time.monotonic()
    jobs = _process_manager().jobs(include_finished=include_finished)
    output = "\n".join(_render_job(job) for job in jobs) if jobs else "No shell jobs."
    return _format_result(
        "ok",
        duration_ms=_duration_ms(started),
        output=output,
        output_truncated=False,
    )


def _shell_read(
    job_id: str, tail_chars: int = DEFAULT_OUTPUT_CHARS, offset: int | None = None
) -> str:
    started = time.monotonic()
    job, output, truncated = _process_manager().read(
        job_id, max_chars=_clamp_output_chars(tail_chars), offset=offset
    )
    body = _render_job(job)
    if output:
        body = f"{body}\n\n{output}"
    return _format_result(
        job["status"],
        exit_code=job["exit_code"],
        duration_ms=_duration_ms(started),
        output=body,
        output_truncated=truncated,
        job_id=job_id,
        log_path=job["log_path"],
        next_offset=offset + len(output) if offset is not None else None,
    )


def _shell_kill(job_id: str, force: bool = False) -> str:
    started = time.monotonic()
    job = _process_manager().kill(job_id, force=force)
    return _format_result(
        job["status"],
        exit_code=job["exit_code"],
        duration_ms=_duration_ms(started),
        output=_render_job(job),
        output_truncated=False,
        job_id=job_id,
    )


def _resolve_timeout(value: float | None) -> tuple[float, str]:
    ctx = get_session_context()
    timeout = ctx.options.shell_default_timeout if value is None else value
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValueError("timeout must be a positive finite number of seconds")
    reason = "default" if value is None else "requested"
    limit = ctx.options.shell_max_timeout
    if limit is not None and timeout > limit:
        timeout, reason = limit, "host_limit"
    if ctx.shell_deadline is not None:
        remaining = max(0.0, ctx.shell_deadline - time.monotonic())
        if remaining < timeout:
            timeout, reason = remaining, "deadline"
    return float(timeout), reason


def _clamp_output_chars(value: int) -> int:
    try:
        chars = int(value)
    except (TypeError, ValueError):
        chars = DEFAULT_OUTPUT_CHARS
    return min(max(chars, 0), MAX_OUTPUT_CHARS)


def _duration_ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


def _format_result(
    status: str,
    *,
    exit_code: int | None = None,
    duration_ms: int,
    output: str,
    output_truncated: bool,
    job_id: str | None = None,
    log_path: str | None = None,
    timeout_seconds: float | None = None,
    timeout_reason: str | None = None,
    next_offset: int | None = None,
) -> str:
    if status not in SHELL_STATUSES:
        status = "error"
    lines = [
        f"status={status}",
        f"exit_code={'' if exit_code is None else exit_code}",
        f"duration_ms={duration_ms}",
        f"cwd={_project_root()}",
        f"output_truncated={'true' if output_truncated else 'false'}",
    ]
    if job_id is not None:
        lines.append(f"job_id={job_id}")
    if log_path is not None:
        lines.append(f"log_path={log_path}")
    if timeout_seconds is not None:
        lines.append(f"timeout_seconds={timeout_seconds:g}")
        lines.append(f"timeout_reason={timeout_reason}")
    if next_offset is not None:
        lines.append(f"next_offset={next_offset}")
    lines.append("output:")
    if output:
        lines.append(output)
    return "\n".join(lines)


def _render_job(job: dict[str, Any]) -> str:
    name = str(job.get("name") or "")
    label = f" name={name}" if name else ""
    exit_code = job.get("exit_code")
    return (
        f"job_id={job.get('job_id')} status={job.get('status')}"
        f" exit_code={'' if exit_code is None else exit_code}"
        f" pid={job.get('pid')} pgid={job.get('pgid')}"
        f" log={job.get('log_path')}{label}"
        f" command={job.get('command')}"
    )
