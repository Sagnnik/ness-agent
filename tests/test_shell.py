from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import replace
from unittest.mock import patch
from pathlib import Path

import ness_agent.tools.shell as shell
from ness_agent.tools.shell_processes import _pid_alive, _process_group_alive

from tests.sdk_fixtures import SessionContextTestMixin


def _field(result: str, name: str) -> str:
    match = re.search(rf"^{re.escape(name)}=(.*)$", result, re.MULTILINE)
    return match.group(1) if match else ""


def _python_command(code: str) -> str:
    return shlex.join([sys.executable, "-c", code])


class ShellToolTests(SessionContextTestMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._home = os.environ.get("HOME")
        self.install_ctx(Path(self._tmp.name))

    def tearDown(self) -> None:
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
        while time.time() < deadline and _pid_alive(pid):
            time.sleep(0.05)

        self.assertEqual(_field(result, "status"), "timeout")
        self.assertFalse(_pid_alive(pid))

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

    def test_foreground_log_survives_truncation_and_can_be_paginated(self) -> None:
        # Keep external runtime storage within this test's temporary root.
        project = self.root / "app"
        project.mkdir()
        self.ctx.project_root = project
        self.ctx.ness_dir = self.root / "runtime"
        output = "first error\n" + "🙂hé\n" * 40 + "last error"
        result = shell.shell.invoke({
            "command": _python_command(f"import sys; sys.stdout.write({output!r})"),
            "max_output_chars": 8,
        })
        self.assertEqual(_field(result, "output_truncated"), "true")
        self.assertEqual(Path(_field(result, "log_path")).read_text(), output)
        job_id = _field(result, "job_id")
        recovered = ""
        offset = 0
        while True:
            page = shell.shell.invoke({"action": "read", "job_id": job_id, "offset": offset, "tail_chars": 7})
            chunk = page.split("output:\n", 1)[1].split("\n\n", 1)[1]
            recovered += chunk
            offset = int(_field(page, "next_offset"))
            if _field(page, "output_truncated") == "false":
                break
        self.assertEqual(recovered, output)
        self.assertNotIn(job_id, shell.shell.invoke({"action": "jobs"}))
        self.ctx.get_shell_process_manager().close()
        self.assertEqual(Path(_field(result, "log_path")).read_text(), output)

    def test_foreground_cancellation_stops_group_and_preserves_output(self) -> None:
        command = _python_command(
            "import pathlib,subprocess,time; "
            "p=subprocess.Popen(['sleep','30']); "
            "print('before cancellation',flush=True); "
            "pathlib.Path('ready.pid').write_text(str(p.pid)); time.sleep(30)"
        )
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(copy_context().run, shell.shell.invoke, {"command": command})
            try:
                self._wait_for_path(self.root / "ready.pid")
            finally:
                self.ctx.shell_cancel_event.set()
            result = future.result(timeout=5)
        self.assertEqual(_field(result, "status"), "cancelled")
        job = self.ctx.get_shell_process_manager().get(_field(result, "job_id"))
        self.assertFalse(_process_group_alive(job["pgid"]))
        self.assertIn("before cancellation", Path(job["log_path"]).read_text())

    def test_timeouts_report_default_request_and_host_limit(self) -> None:
        self.ctx.options = replace(self.ctx.options, shell_default_timeout=45)
        default = shell.shell.invoke({"command": "true"})
        self.assertEqual(_field(default, "timeout_seconds"), "45")
        requested = shell.shell.invoke({"command": "true", "timeout": 900})
        self.assertEqual(_field(requested, "timeout_seconds"), "900")
        self.ctx.options = replace(self.ctx.options, shell_max_timeout=0.15)
        limited = shell.shell.invoke({"command": "printf partial; sleep 30", "timeout": 900})
        self.assertEqual(_field(limited, "status"), "timeout")
        self.assertEqual(_field(limited, "timeout_seconds"), "0.15")
        self.assertEqual(_field(limited, "timeout_reason"), "host_limit")
        self.assertIn("partial", Path(_field(limited, "log_path")).read_text())

    def test_deadline_limits_commands_and_prevents_expired_launch(self) -> None:
        self.ctx.shell_deadline = time.monotonic() + 0.2
        result = shell.shell.invoke({"command": "sleep 30", "timeout": 900})
        self.assertEqual(_field(result, "status"), "timeout")
        self.assertEqual(_field(result, "timeout_reason"), "deadline")
        self.assertLess(float(_field(result, "timeout_seconds")), 0.21)
        with patch.object(subprocess, "Popen") as spawn:
            result = shell.shell.invoke({"command": "touch should-not-exist"})
            spawn.assert_not_called()
        self.assertEqual(_field(result, "status"), "timeout")
        self.assertFalse((self.root / "should-not-exist").exists())

    def test_invalid_timeouts_are_errors_without_launch(self) -> None:
        for value in (0, -1, float("nan"), float("inf")):
            with self.subTest(value=value):
                result = shell._shell_run("touch should-not-exist", timeout=value)
                self.assertEqual(_field(result, "status"), "error")
        self.assertFalse((self.root / "should-not-exist").exists())

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

    def test_shell_kill_escalates_for_uncooperative_process(self) -> None:
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
        manager = self.ctx.get_shell_process_manager()
        pgid = manager.get(job_id)["pgid"]
        self.assertIsNotNone(pgid)
        killed = shell.shell.invoke({"action": "kill", "job_id": job_id})

        self.assertEqual(_field(killed, "status"), "killed")
        self.assertFalse(_process_group_alive(pgid))

    def test_background_jobs_work_with_external_runtime_directory(self) -> None:
        external = self.root / "logs" / "agent" / "ness"
        project = self.root / "app"
        project.mkdir()
        self.ctx.project_root = project
        self.ctx.permissions.project_root = project
        self.ctx.ness_dir = external
        started = shell.shell.invoke({"action": "start", "command": "echo external; sleep 10"})
        job_id = _field(started, "job_id")
        try:
            self.assertEqual(_field(started, "status"), "running")
            manager = self.ctx.get_shell_process_manager()
            self.assertTrue(Path(manager.get(job_id)["log_path"]).is_relative_to(external))
            self.assertIn(job_id, shell.shell.invoke({"action": "jobs"}))
            read = shell.shell.invoke({"action": "read", "job_id": job_id})
            self.assertEqual(_field(read, "status"), "running")
            with self.assertRaises(PermissionError):
                self.ctx.permissions.validate_path(str(external / "outside.txt"))
        finally:
            killed = shell.shell.invoke({"action": "kill", "job_id": job_id})
        self.assertEqual(_field(killed, "status"), "killed")

    def test_background_job_reports_exit_code_and_bounded_output(self) -> None:
        started = shell.shell.invoke({"action": "start", "command": "printf abcdef; exit 7"})
        job_id = _field(started, "job_id")
        self._wait_for_job(job_id)
        read = shell.shell.invoke({"action": "read", "job_id": job_id, "tail_chars": 3})
        self.assertEqual(_field(read, "status"), "failed")
        self.assertEqual(_field(read, "exit_code"), "7")
        self.assertEqual(_field(read, "output_truncated"), "true")
        self.assertTrue(read.endswith("def"))

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
