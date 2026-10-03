from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any

TERM_GRACE_SECONDS = 2.0


class ProcessManager:
    """Own shell executions for one live session, without restart recovery.

    Runtime storage is host-configured SDK state, independent of the project
    file policy. Saved metadata is diagnostic only; it never grants ownership
    of a process or supplies paths used by this manager.
    """

    def __init__(self, *, project_root: Path, runtime_root: Path) -> None:
        self.project_root = project_root.resolve()
        self.runtime_dir = runtime_root.resolve() / uuid.uuid4().hex
        self._jobs: dict[str, dict[str, Any]] = {}
        self._processes: dict[str, subprocess.Popen[bytes]] = {}
        self._lock = threading.RLock()
        self._closed = False

    def _path(self, job_id: str, filename: str) -> Path:
        path = self.runtime_dir / job_id / filename
        if not path.resolve().is_relative_to(self.runtime_dir):
            raise ValueError("Shell runtime path escapes its configured directory.")
        return path

    def _write_metadata(self, job: dict[str, Any]) -> None:
        path = self._path(job["job_id"], "metadata.json")
        temporary = self._path(job["job_id"], "metadata.tmp")
        temporary.write_text(json.dumps(job, indent=2), encoding="utf-8")
        temporary.replace(path)

    def start(
        self, command: str, *, name: str = "", foreground: bool = False
    ) -> dict[str, Any]:
        with self._lock:
            if self._closed:
                raise RuntimeError("Shell process manager is closed.")
            if not command.strip():
                raise ValueError("Empty shell command.")
            if self.runtime_dir.resolve() != self.runtime_dir:
                raise ValueError("Shell runtime directory has been redirected.")
            self.runtime_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            job_id = uuid.uuid4().hex
            job_dir = self.runtime_dir / job_id
            job_dir.mkdir(mode=0o700)
            log_path = self._path(job_id, "output.log")
            job: dict[str, Any] = {
                "job_id": job_id,
                "name": name.strip(),
                "foreground": foreground,
                "command": command,
                "pid": None,
                "pgid": None,
                "status": "starting",
                "start_time": time.time(),
                "end_time": None,
                "exit_code": None,
                "log_path": str(log_path),
            }
            self._jobs[job_id] = job
            proc = None
            try:
                # Persist and open all required storage before launching.
                self._write_metadata(job)
                with log_path.open("xb") as out:
                    proc = subprocess.Popen(
                        ["bash", "-lc", command],
                        cwd=self.project_root,
                        stdin=subprocess.DEVNULL,
                        stdout=out,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                self._processes[job_id] = proc
                # start_new_session makes the child's PID its process-group ID.
                job.update(pid=proc.pid, pgid=proc.pid, status="running")
                self._write_metadata(job)
            except BaseException:
                job.update(status="error", end_time=time.time())
                if proc is not None:
                    self._processes[job_id] = proc
                    job.update(pid=proc.pid, pgid=proc.pid)
                    if _kill_process_group(proc.pid, proc=proc, force=True):
                        self._processes.pop(job_id, None)
                        job["exit_code"] = proc.poll()
                try:
                    self._write_metadata(job)
                except (OSError, ValueError):
                    pass  # Preserve the launch failure after process cleanup.
                raise
            return dict(job)

    def run(
        self,
        command: str,
        *,
        timeout: float,
        max_chars: int,
        cancel_event: threading.Event,
        deadline: float | None = None,
    ) -> tuple[dict[str, Any], str, bool]:
        """Wait for a foreground command without holding the manager lock.

        Captured logs remain readable by ID. Ordinary shell backgrounding may
        outlive the command shell; retained handles let session close clean up
        descendants still in the original process group.
        """
        if cancel_event.is_set():
            raise RuntimeError("Shell execution cancelled before launch.")
        end = time.monotonic() + timeout
        if deadline is not None:
            end = min(end, deadline)
        with self._lock:
            job = self.start(command, foreground=True)
            job_id = job["job_id"]
            proc = self._processes[job_id]
        status = None
        try:
            while True:
                if cancel_event.is_set():
                    status = "cancelled"
                    break
                if proc.poll() is not None:
                    break
                remaining = end - time.monotonic()
                if remaining <= 0:
                    status = "timeout"
                    break
                cancel_event.wait(min(remaining, 0.05))
            if status is not None:
                self.kill(job_id)
            with self._lock:
                # close() may have stopped the command while we were waiting.
                job = self._jobs[job_id]
                if job["status"] not in {"killed", "error"} or status is not None:
                    job["status"] = status or ("ok" if proc.poll() == 0 else "failed")
                job.update(exit_code=proc.poll(), end_time=time.time())
                if not _process_group_alive(proc.pid):
                    self._processes.pop(job_id, None)
                self._write_metadata(job)
                return self.read(job_id, max_chars=max_chars)
        except BaseException:
            # Preserve the original error, but never abandon a launched group.
            try:
                self.kill(job_id, force=True)
            except (OSError, ValueError, RuntimeError):
                pass
            raise

    def _refresh(self, job_id: str) -> None:
        proc = self._processes.get(job_id)
        if proc is None:
            return
        exit_code = proc.poll()
        if exit_code is None or _process_group_alive(proc.pid):
            return
        job = self._jobs[job_id]
        if job["status"] == "running":
            job.update(status="ok" if exit_code == 0 else "failed")
        job.update(exit_code=exit_code, end_time=time.time())
        self._processes.pop(job_id, None)
        self._write_metadata(job)

    def jobs(self, *, include_finished: bool = True) -> list[dict[str, Any]]:
        with self._lock:
            for job_id in self._jobs:
                self._refresh(job_id)
            return [
                dict(job)
                for job in self._jobs.values()
                if not job["foreground"]
                and (include_finished or job["status"] == "running")
            ]

    def get(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            if job_id not in self._jobs:
                raise ValueError(f"Unknown shell job: {job_id}")
            self._refresh(job_id)
            return dict(self._jobs[job_id])

    def read(
        self, job_id: str, *, max_chars: int, offset: int | None = None
    ) -> tuple[dict[str, Any], str, bool]:
        with self._lock:
            job = self.get(job_id)
            path = self._path(job_id, "output.log")
            if offset is None:
                with path.open("rb") as out:
                    output, truncated = _tail_stream(out, max_chars)
            else:
                if offset < 0:
                    raise ValueError("Log offset must be nonnegative.")
                with path.open(
                    "r", encoding="utf-8", errors="replace", newline=""
                ) as out:
                    remaining = offset
                    while remaining:
                        chunk = out.read(min(remaining, 65536))
                        if not chunk:
                            break
                        remaining -= len(chunk)
                    output = out.read(max_chars + 1)
                    truncated = len(output) > max_chars
                    output = output[:max_chars]
            return job, output, truncated

    def kill(self, job_id: str, *, force: bool = False) -> dict[str, Any]:
        with self._lock:
            self.get(job_id)
            job = self._jobs[job_id]
            proc = self._processes.get(job_id)
            if proc is not None:
                killed = _kill_process_group(proc.pid, proc=proc, force=force)
                job.update(
                    status="killed" if killed else "error",
                    exit_code=proc.poll(),
                    end_time=time.time() if killed else None,
                )
                if killed:
                    self._processes.pop(job_id, None)
                self._write_metadata(job)
                if not killed:
                    raise RuntimeError(f"Could not stop shell job {job_id}.")
            return dict(job)

    def close(self) -> None:
        """Stop all owned process groups; repeated calls are safe."""
        with self._lock:
            self._closed = True
            errors = []
            for job_id in list(self._processes):
                try:
                    self.kill(job_id)
                except (OSError, ValueError, RuntimeError) as exc:
                    # A metadata failure must not prevent other jobs' cleanup.
                    errors.append(str(exc))
            if errors:
                raise RuntimeError("Shell cleanup failed: " + "; ".join(errors))


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _process_group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    proc_dir = Path("/proc")
    if not proc_dir.exists():
        return True
    for entry in proc_dir.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text(encoding="utf-8")
        except OSError:
            continue
        fields = stat[stat.rfind(")") + 2 :].split()
        if len(fields) >= 3 and fields[2] == str(pgid) and fields[0] != "Z":
            return True
    return False


def _kill_process_group(
    pgid: int, *, proc: subprocess.Popen[bytes], force: bool
) -> bool:
    signals = [signal.SIGKILL] if force else [signal.SIGTERM, signal.SIGKILL]
    for sig in signals:
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            proc.poll()
            return True
        except OSError:
            return False
        deadline = time.monotonic() + TERM_GRACE_SECONDS
        while time.monotonic() < deadline:
            proc.poll()
            if not _process_group_alive(pgid):
                try:
                    proc.wait(timeout=TERM_GRACE_SECONDS)
                except subprocess.TimeoutExpired:
                    return False
                return True
            time.sleep(0.05)
    return False


def _tail_stream(file: Any, max_chars: int) -> tuple[str, bool]:
    file.seek(0, os.SEEK_END)
    size = file.tell()
    if max_chars <= 0:
        return "", size > 0
    max_bytes = min(size, max(max_chars * 4, 4096))
    file.seek(size - max_bytes)
    text = file.read().decode("utf-8", errors="replace")
    truncated = size > max_bytes or len(text) > max_chars
    if len(text) > max_chars:
        text = text[-max_chars:]
    return text.rstrip("\n"), truncated
