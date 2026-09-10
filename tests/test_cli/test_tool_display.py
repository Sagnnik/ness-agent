from __future__ import annotations

from ness_cli.tui.tool_display import (
    format_shell_output,
    format_subagent_output,
    format_tool_args,
)


def test_format_shell_output_marks_validation_errors_as_error() -> None:
    content = (
        "Error: 1 validation error for shell\n"
        "action\n"
        "  Field required [type=missing, input_value={'command': 'git status'}, "
        "input_type=dict]"
    )
    header, body = format_shell_output(content)
    assert header == "status=error"
    assert body.startswith("Error: 1 validation error")


def test_format_shell_output_keeps_structured_ok_status() -> None:
    content = "status=ok\nexit_code=0\nduration_ms=12\ncwd=/tmp\noutput_truncated=false\noutput:\nok"
    header, body = format_shell_output(content)
    assert header == "status=ok"
    assert body == "ok"


def test_format_tool_args_defaults_missing_shell_action_to_run() -> None:
    token = format_tool_args("shell", {"command": "git status --short", "timeout": 10})
    assert "run" in token
    assert "git status --short" in token


def test_format_tool_args_summarizes_bulk_delete() -> None:
    token = format_tool_args("delete", {"paths": ["one.png", "two.png"]})
    assert token == "2 files"


def test_format_subagent_output_single_ok() -> None:
    header, body = format_subagent_output("Found routes in src/api.py")
    assert header == "subagent ok"
    assert body == "Found routes in src/api.py"


def test_format_subagent_output_batch_status() -> None:
    content = (
        "status=ok\n"
        "duration_ms=120\n"
        "tasks_total=2\n"
        "tasks_ok=2\n"
        "tasks_failed=0\n"
        "\n"
        "[1] name=explore status=ok duration_ms=50 thread_id=subagent-explore-aaa\n"
        "found routes\n"
        "\n"
        "[2] name=explore status=ok duration_ms=60 thread_id=subagent-explore-bbb\n"
        "found tests"
    )
    header, body = format_subagent_output(content)
    assert header == "subagent ok"
    assert "[1] name=explore" in body
    assert "found routes" in body
    assert "found tests" in body
    assert not body.startswith("status=")


def test_format_subagent_output_error() -> None:
    header, body = format_subagent_output("Error: subagent explore timeout: timed out after 1s")
    assert header == "subagent error"
    assert "timeout" in body
