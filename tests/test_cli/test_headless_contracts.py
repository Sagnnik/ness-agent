from __future__ import annotations

import asyncio

import pytest
from ness_agent import SessionEvent

from ness_cli.headless import (
    auto_answer_questions,
    merge_prompt_parts,
    run_headless_turn,
)


class ScriptedSession:
    def __init__(self, events=(), error: Exception | None = None):
        self.events = events
        self.error = error

    async def stream(self, request):
        if self.error:
            raise self.error
        for event in self.events:
            yield event


def run(coro):
    return asyncio.run(coro)


def test_real_headless_runtime_saves_resumes_and_closes(isolated_cli_env):
    from tests.test_cli.fakes.headless import exercise_headless

    run(exercise_headless(isolated_cli_env.project))


def test_merge_prompt_parts_preserves_stdin_before_arguments():
    assert merge_prompt_parts(["explain", "this"], "log\n") == "log\n\nexplain this"
    assert merge_prompt_parts([], "  ") is None


def test_auto_answer_prefers_recommended_then_first_then_fallback():
    answers = run(
        auto_answer_questions(
            [
                {
                    "id": "a",
                    "options": [
                        {"id": "1", "label": "one"},
                        {"id": "2", "label": "two", "recommended": True},
                    ],
                },
                {"id": "b", "options": [{"id": "3", "label": "three"}]},
                {"id": "c"},
            ]
        )
    )
    assert [answer["selected"]["id"] for answer in answers] == ["2", "3", "0"]


def test_headless_turn_returns_latest_final_and_warns(capsys):
    session = ScriptedSession(
        [
            SessionEvent(kind="assistant_final", data={"content": "first"}),
            SessionEvent(kind="warning", data={"message": "careful"}),
            SessionEvent(kind="assistant_final", data={"content": "last"}),
        ]
    )
    assert run(run_headless_turn(session, "q")) == ("last", 0)
    assert "careful" in capsys.readouterr().err


def test_headless_turn_maps_error_interrupt_and_exception(capsys):
    error = ScriptedSession([SessionEvent(kind="error", data={"message": "boom"})])
    interrupted = ScriptedSession(
        [SessionEvent(kind="interrupted", data={"partial_text": "half"})]
    )
    failed = ScriptedSession(error=RuntimeError("bad"))
    assert run(run_headless_turn(error, "q")) == ("", 1)
    assert run(run_headless_turn(interrupted, "q")) == ("half", 130)
    assert run(run_headless_turn(failed, "q")) == ("", 1)
    assert "boom" in capsys.readouterr().err


@pytest.mark.parametrize("prior_error,expected_code", [(False, 130), (True, 1)])
def test_keyboard_interrupt_preserves_output_and_prior_failure(
    prior_error, expected_code
):
    class InterruptedSession(ScriptedSession):
        async def stream(self, request):
            async for event in super().stream(request):
                yield event
            raise KeyboardInterrupt

    events = [SessionEvent("assistant_final", {"content": "partial answer"})]
    if prior_error:
        events.insert(0, SessionEvent("error", {"message": "provider failed"}))

    assert run(run_headless_turn(InterruptedSession(events), "q")) == (
        "partial answer",
        expected_code,
    )
