from __future__ import annotations

import json
import shlex
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from ness_agent.tools.shell_processes import ProcessManager, _process_group_alive


@pytest.fixture
def manager(tmp_path):
    project = tmp_path / "app"
    project.mkdir()
    manager = ProcessManager(
        project_root=project, runtime_root=tmp_path / "logs" / "agent" / "ness" / "shells"
    )
    try:
        yield manager
    finally:
        manager.close()


def wait_until(predicate):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    pytest.fail("Timed out waiting for background process")


def test_managers_isolate_jobs_even_with_shared_storage(manager):
    other = ProcessManager(
        project_root=manager.project_root, runtime_root=manager.runtime_dir.parent
    )
    try:
        first = manager.start("sleep 30")
        second = other.start("sleep 30")
        assert [job["job_id"] for job in manager.jobs()] == [first["job_id"]]
        assert [job["job_id"] for job in other.jobs()] == [second["job_id"]]
        with pytest.raises(ValueError, match="Unknown shell job"):
            other.kill(first["job_id"])
        manager.close()
        assert not _process_group_alive(first["pgid"])
        assert other.get(second["job_id"])["status"] == "running"
    finally:
        other.close()


def test_concurrent_starts_preserve_every_job(manager):
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = list(pool.map(manager.start, ["sleep 30"] * 8))
    assert len({job["job_id"] for job in jobs}) == 8
    assert {job["job_id"] for job in manager.jobs()} == {job["job_id"] for job in jobs}
    for job in jobs:
        metadata = json.loads(
            (manager.runtime_dir / job["job_id"] / "metadata.json").read_text()
        )
        assert metadata["pid"] == job["pid"]
    manager.close()
    assert all(not _process_group_alive(job["pgid"]) for job in jobs)


def test_failed_registration_stops_launched_process(manager, monkeypatch):
    original = manager._write_metadata

    def fail_running_record(job):
        if job["status"] == "running":
            raise OSError("registration failed")
        original(job)

    monkeypatch.setattr(manager, "_write_metadata", fail_running_record)
    with pytest.raises(OSError, match="registration failed"):
        manager.start("sleep 30")
    [job] = manager.jobs()
    assert job["pid"] is not None
    assert job["status"] == "error"
    assert job["exit_code"] is not None
    assert not _process_group_alive(job["pgid"])


def test_storage_failure_happens_before_process_launch(manager, monkeypatch):
    def fail_record(job):
        raise OSError("storage unavailable")

    monkeypatch.setattr(manager, "_write_metadata", fail_record)
    with pytest.raises(OSError, match="storage unavailable"):
        manager.start("sleep 30")
    [job] = manager.jobs()
    assert job["pid"] is None


def test_close_stops_children_after_the_command_shell_exits(manager):
    ready = manager.project_root / "child.pid"
    command = shlex.join([
        sys.executable,
        "-c",
        "import pathlib,subprocess; "
        "p=subprocess.Popen(['sleep','30']); "
        "pathlib.Path('child.pid').write_text(str(p.pid))",
    ])
    job = manager.start(command)
    wait_until(ready.exists)
    wait_until(lambda: manager._processes[job["job_id"]].poll() is not None)
    assert manager.get(job["job_id"])["status"] == "running"
    manager.close()
    assert not _process_group_alive(job["pgid"])
    manager.close()
    with pytest.raises(RuntimeError, match="closed"):
        manager.start("sleep 30")


def test_runtime_metadata_cannot_redirect_log_reads(manager, tmp_path):
    job = manager.start("printf own-output")
    wait_until(lambda: manager.get(job["job_id"])["status"] != "running")
    outside = tmp_path / "outside.txt"
    outside.write_text("foreign-output")
    metadata = manager.runtime_dir / job["job_id"] / "metadata.json"
    metadata.write_text(json.dumps({"log_path": str(outside), "pid": 1}))
    _, output, _ = manager.read(job["job_id"], max_chars=100)
    assert output == "own-output"
    log = Path(job["log_path"])
    log.unlink()
    log.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        manager.read(job["job_id"], max_chars=100)


def test_close_cleans_all_jobs_even_if_metadata_write_fails(manager, monkeypatch):
    jobs = [manager.start("sleep 30") for _ in range(2)]

    def fail_record(job):
        raise OSError("storage unavailable")

    monkeypatch.setattr(manager, "_write_metadata", fail_record)
    with pytest.raises(RuntimeError, match="cleanup failed"):
        manager.close()
    assert all(not _process_group_alive(job["pgid"]) for job in jobs)
    manager.close()
