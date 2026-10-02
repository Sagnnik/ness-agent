from __future__ import annotations

import os
import re
import shlex
import sys
import tempfile
import time
import unittest
from pathlib import Path

import ness_agent.tools.shell as shell
from ness_agent.session_context import get_session_context

from tests.sdk_fixtures import SessionContextTestMixin


def _field(result: str, name: str) -> str:
    match = re.search(rf"^{re.escape(name)}=(.*)$", result, re.MULTILINE)
    return match.group(1) if match else ""


def _shell_dir() -> Path:
    return get_session_context().ness_dir / "runtime" / "shells"


def _python_command(code: str) -> str:
    return shlex.join([sys.executable, "-c", code])


class ShellToolTests(SessionContextTestMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._home = os.environ.get("HOME")
        self.install_ctx(Path(self._tmp.name))
        shell._job_processes.clear()

    def tearDown(self) -> None:
        try:
            jobs = shell._load_jobs(_shell_dir())
            for job_id, job in jobs.items():
                if job.get("status") != "running":
                    continue
                pgid = shell._optional_int(job.get("pgid")) or shell._optional_int(job.get("pid"))
                if pgid is not None:
                    shell._kill_process_group(pgid, proc=shell._job_processes.get(job_id), force=True)
        except Exception:
            pass
        for proc in list(shell._job_processes.values()):
            try:
                proc.kill()
            except OSError:
                pass
        shell._job_processes.clear()
        if self._home is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = self._home
        self.uninstall_ctx()
        self._tmp.cleanup()

    def test_shell_run_success_and_no_persistent_cwd(self) -> None:
        (self.root / "sub").mkdir()
        first = shell.shell.invoke({"action": "run", "command": "cd sub && pwd"})
        second = shell.shell.invoke({"action": "run", "command": "pwd"})

        self.assertEqual(_field(first, "status"), "ok")
        self.assertIn(str(self.root / "sub"), first)
        self.assertEqual(_field(second, "status"), "ok")
        self.assertIn(f"output:\n{self.root}", second)
        self.assertNotIn(str(self.root / "sub"), second)

    def test_shell_run_defaults_action_when_omitted(self) -> None:
        """Pi/Claude-shaped calls ({command, timeout}) should work without action."""
        schema = shell.shell.args_schema.model_json_schema()
        self.assertNotIn("action", schema.get("required", []))

        result = shell.shell.invoke({"command": "printf hi", "timeout": 10})

        self.assertEqual(_field(result, "status"), "ok")
        self.assertIn("hi", result)

    def test_shell_run_reports_nonzero_exit(self) -> None:
        result = shell.shell.invoke({"action": "run", "command": "printf nope; exit 7"})

        self.assertEqual(_field(result, "status"), "failed")
        self.assertEqual(_field(result, "exit_code"), "7")
        self.assertIn("nope", result)

    def test_shell_run_timeout_handles_partial_output(self) -> None:
        started = time.monotonic()
        result = shell.shell.invoke(
            {
                "action": "run",
                "command": _python_command(
                    'import sys,time; sys.stdout.write("x"); sys.stdout.flush(); time.sleep(5)'
                ),
                "timeout": 1,
            }
        )
        elapsed = time.monotonic() - started

        self.assertLess(elapsed, 4)
        self.assertEqual(_field(result, "status"), "timeout")
        self.assertIn("Command timed out after 1s", result)
        self.assertIn("x", result)

    def test_shell_run_timeout_kills_child_process_group(self) -> None:
        result = shell.shell.invoke(
            {
                "action": "run",
                "command": _python_command(
                    "import pathlib,subprocess,time; "
                    "p=subprocess.Popen([\"sleep\",\"10\"]); "
                    "pathlib.Path(\"child.pid\").write_text(str(p.pid)); "
                    "time.sleep(10)"
                ),
                "timeout": 1,
            }
        )
        pid = int((self.root / "child.pid").read_text(encoding="utf-8"))

        deadline = time.time() + 3
        while time.time() < deadline and shell._pid_alive(pid):
            time.sleep(0.05)

        self.assertEqual(_field(result, "status"), "timeout")
        self.assertFalse(shell._pid_alive(pid))

    def test_shell_run_truncates_output(self) -> None:
        result = shell.shell.invoke(
            {
                "action": "run",
                "command": _python_command('print("abcdef")'),
                "max_output_chars": 4,
            }
        )

        self.assertEqual(_field(result, "status"), "ok")
        self.assertEqual(_field(result, "output_truncated"), "true")
        self.assertTrue(result.rstrip().endswith("def"))

    def test_background_job_uses_bash_and_can_be_read(self) -> None:
        started = shell.shell.invoke({"action": "start", "command": "[[ 1 -eq 1 ]] && echo ok", "name": "bash-syntax"})
        job_id = _field(started, "job_id")
        read = self._wait_for_job(job_id)

        self.assertEqual(_field(read, "status"), "ok")
        self.assertIn("name=bash-syntax", read)
        self.assertIn("ok", read)

    def test_background_job_uses_same_login_shell_context_as_run(self) -> None:
        os.environ["HOME"] = str(self.root)
        (self.root / ".bash_profile").write_text(
            "export NESS_AGENT_PROFILE_VALUE=profile-loaded\n",
            encoding="utf-8",
        )
        command = 'printf "%s" "$NESS_AGENT_PROFILE_VALUE"'

        foreground = shell.shell.invoke({"action": "run", "command": command})
        started = shell.shell.invoke({"action": "start", "command": command})
        read = self._wait_for_job(_field(started, "job_id"))

        self.assertEqual(_field(foreground, "status"), "ok")
        self.assertIn("profile-loaded", foreground)
        self.assertEqual(_field(read, "status"), "ok")
        self.assertIn("profile-loaded", read)

    def test_shell_jobs_lists_background_jobs(self) -> None:
        started = shell.shell.invoke({"action": "start", "command": "echo listed"})
        job_id = _field(started, "job_id")
        self._wait_for_job(job_id)

        result = shell.shell.invoke({"action": "jobs"})

        self.assertEqual(_field(result, "status"), "ok")
        self.assertIn(f"job_id={job_id}", result)
        self.assertIn("command=echo listed", result)

    def test_shell_kill_terminates_running_job(self) -> None:
        started = shell.shell.invoke({"action": "start", "command": "sleep 10"})
        job_id = _field(started, "job_id")

        killed = shell.shell.invoke({"action": "kill", "job_id": job_id})
        read = shell.shell.invoke({"action": "read", "job_id": job_id})

        self.assertEqual(_field(killed, "status"), "killed")
        self.assertEqual(_field(read, "status"), "killed")

    def test_shell_kill_escalates_after_process_table_loss(self) -> None:
        started = shell.shell.invoke(
            {
                "action": "start",
                "command": _python_command(
                    "import pathlib,signal,time; "
                    "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                    "pathlib.Path(\"ready.pid\").write_text(\"1\"); "
                    "time.sleep(30)"
                ),
            }
        )
        job_id = _field(started, "job_id")
        self._wait_for_path(self.root / "ready.pid")
        jobs = shell._load_jobs(_shell_dir())
        pgid = shell._optional_int(jobs[job_id].get("pgid"))
        self.assertIsNotNone(pgid)
        lost_procs = list(shell._job_processes.values())
        shell._job_processes.clear()

        killed = shell.shell.invoke({"action": "kill", "job_id": job_id})
        for proc in lost_procs:
            proc.wait(timeout=1)

        self.assertEqual(_field(killed, "status"), "killed")
        self.assertFalse(shell._process_group_alive(pgid))

    def test_refresh_keeps_lost_process_table_job_running_when_group_exists(self) -> None:
        started = shell.shell.invoke({"action": "start", "command": "sleep 10"})
        job_id = _field(started, "job_id")
        jobs = shell._load_jobs(_shell_dir())
        pgid = shell._optional_int(jobs[job_id].get("pgid"))
        self.assertIsNotNone(pgid)
        lost_procs = list(shell._job_processes.values())
        shell._job_processes.clear()

        read = shell.shell.invoke({"action": "read", "job_id": job_id})
        killed = shell.shell.invoke({"action": "kill", "job_id": job_id, "force": True})
        for proc in lost_procs:
            proc.wait(timeout=1)

        self.assertEqual(_field(read, "status"), "running")
        self.assertEqual(_field(killed, "status"), "killed")
        self.assertFalse(shell._process_group_alive(pgid))

    def _wait_for_job(self, job_id: str) -> str:
        deadline = time.time() + 5
        last = ""
        while time.time() < deadline:
            last = shell.shell.invoke({"action": "read", "job_id": job_id})
            if _field(last, "status") != "running":
                return last
            time.sleep(0.05)
        self.fail(f"job {job_id} did not finish; last result:\n{last}")

    def _wait_for_path(self, path: Path) -> None:
        deadline = time.time() + 5
        while time.time() < deadline:
            if path.exists():
                return
            time.sleep(0.05)
        self.fail(f"path did not appear: {path}")


if __name__ == "__main__":
    unittest.main()
