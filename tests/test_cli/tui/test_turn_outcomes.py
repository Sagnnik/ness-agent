from __future__ import annotations

import asyncio
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import Mock

import pytest
from ness_agent import SessionEvent, ThreadStore

from ness_cli.headless import run_headless_turn
from ness_cli.session.goal import GoalCoordinator
from ness_cli.session.persistence import SessionRepository
from ness_cli.session.turn import TurnOutcome
from ness_cli.tui.controller import ThreadRuntime, TuiController
from ness_cli.tui.sink import NullPromptDriver, TuiRenderSink
from ness_cli.tui.transcript.store import TranscriptStore


class Session:
    thread_id = "test-outcome"
    mode = "act"

    def __init__(self, events, error=None):
        self.events = events
        self.error = error
        self.calls = []

    async def stream(self, request):
        self.calls.append(request.message)
        for event in self.events:
            yield event
        if self.error is not None:
            raise self.error

    def cancel(self):
        pass


def controller_for(session):
    sink = TuiRenderSink(
        TranscriptStore(), invalidate=lambda: None, prompts=NullPromptDriver()
    )
    controller = TuiController(SimpleNamespace(), sink, changed=lambda: None)
    controller._selected = ThreadRuntime(session, sink=sink)
    return controller, sink


@pytest.mark.parametrize(
    "tail", [(), (SessionEvent("assistant_final", {"content": "partial answer"}),)]
)
def test_streamed_error_stops_queue_even_if_stream_ends_normally(tail, capsys):
    events = [SessionEvent("error", {"message": "provider unavailable"}), *tail]
    session = Session(events)
    controller, sink = controller_for(session)
    controller.enqueue("queued one")
    controller.enqueue("queued two")

    assert asyncio.run(controller.submit("first prompt")) is False
    assert session.calls == ["first prompt"]
    assert [request.message for request in controller.queued_prompts] == [
        "queued one",
        "queued two",
    ]
    assert "provider unavailable" in sink.store.plain_text()
    assert asyncio.run(run_headless_turn(Session(events), "first prompt"))[1] == 1
    assert "provider unavailable" in capsys.readouterr().err


@pytest.mark.parametrize(
    "events,error,code",
    [
        ([SessionEvent("warning", {"message": "warning only"})], None, 0),
        (
            [
                SessionEvent("interrupted", {"partial_text": "partial"}),
                SessionEvent("assistant_final", {"content": "tail"}),
            ],
            None,
            130,
        ),
        (
            [
                SessionEvent("error", {"message": "failed"}),
                SessionEvent("interrupted", {}),
            ],
            None,
            1,
        ),
        (
            [
                SessionEvent("interrupted", {}),
                SessionEvent("error", {"message": "failed"}),
            ],
            None,
            1,
        ),
        (
            [
                SessionEvent("assistant_final", {"content": "tail"}),
                SessionEvent("error", {"message": "failed"}),
            ],
            None,
            1,
        ),
        (
            [SessionEvent("assistant_delta", {"content": "partial"})],
            RuntimeError("stream exploded"),
            1,
        ),
    ],
)
def test_interactive_and_headless_share_outcome_policy(events, error, code):
    session = Session(events, error)
    controller, _ = controller_for(session)
    controller.enqueue("queued prompt")

    assert asyncio.run(controller.submit("first")) is (code == 0)
    assert asyncio.run(run_headless_turn(Session(events, error), "first"))[1] == code
    assert session.calls == (["first", "queued prompt"] if code == 0 else ["first"])
    assert len(controller.queued_prompts) == (0 if code == 0 else 1)


def test_queue_stops_at_failed_queued_turn_and_retains_remaining_prompt():
    class QueuedSession(Session):
        async def stream(self, request):
            self.calls.append(request.message)
            if request.message == "queued failure":
                yield SessionEvent("error", {"message": "queued failed"})
            yield SessionEvent("assistant_final", {"content": request.message})

    session = QueuedSession([])
    controller, _ = controller_for(session)
    controller.enqueue("queued failure")
    controller.enqueue("unstarted prompt")

    assert asyncio.run(controller.submit("successful first turn")) is False
    assert session.calls == ["successful first turn", "queued failure"]
    assert [request.message for request in controller.queued_prompts] == [
        "unstarted prompt"
    ]


def test_new_submission_can_succeed_after_previous_failure():
    session = Session([SessionEvent("error", {"message": "first failed"})])
    controller, _ = controller_for(session)
    assert asyncio.run(controller.submit("first")) is False
    session.events = [SessionEvent("assistant_final", {"content": "recovered"})]
    assert asyncio.run(controller.submit("retry")) is True


def test_typed_outcome_keeps_all_errors_and_ignores_warning_and_final():
    outcome = TurnOutcome()
    for event in [
        SessionEvent("error", {"message": "first failure"}),
        SessionEvent("warning", {"message": "warning"}),
        SessionEvent("assistant_final", {"content": "tail"}),
        SessionEvent("error", {"message": "second failure"}),
        SessionEvent("interrupted", {}),
    ]:
        outcome = outcome.observe(event)
    assert outcome.status == "failed" and outcome.exit_code == 1
    assert outcome.errors == ("first failure", "second failure")


@pytest.mark.parametrize("failure", ["error_event", "exception", "interrupted"])
def test_goal_controller_never_judges_a_failed_stream(tmp_path, failure):
    repo = SessionRepository(
        ThreadStore(threads_dir=tmp_path / "threads", auto_save=True)
    )
    hooks = SimpleNamespace(load=Mock(), run=Mock())
    judge = SimpleNamespace(with_structured_output=Mock())
    events = [SessionEvent("assistant_final", {"content": "partial answer"})]
    error = None
    if failure == "error_event":
        events.append(SessionEvent("error", {"message": "worker unavailable"}))
    elif failure == "exception":
        error = RuntimeError("worker unavailable")
    else:
        events.append(SessionEvent("interrupted", {}))

    class GoalSession(Session):
        async def run_goal(
            self, goal, *, worker_turn, on_status, judge_model, max_attempts
        ):
            runner = GoalCoordinator(
                repository=repo,
                thread_id=self.thread_id,
                hook_runner=hooks,
                judge_model=judge_model,
                max_attempts=max_attempts,
                instructions_dir=Path(__file__).parents[3]
                / "src"
                / "ness_cli"
                / "instructions",
                is_cancelled=lambda: False,
            )
            return await runner.run(goal, worker_turn=worker_turn, on_status=on_status)

    session = GoalSession(events, error)
    controller, sink = controller_for(session)
    controller._runtime = SimpleNamespace(
        config=SimpleNamespace(settings=SimpleNamespace(goal_max_attempts=3)),
        goal_judge_model=lambda _: judge,
    )

    result = asyncio.run(controller.run_goal("finish objective"))

    assert not result.passed and result.attempts == 1
    assert session.calls == ["finish objective"]
    judge.with_structured_output.assert_not_called()
    hooks.load.assert_not_called()
    hooks.run.assert_not_called()
    event = repo.raw_events(session.thread_id)[-1]
    assert event["phase"] == "worker" and event["pass"] is False
    expected = "interrupted" if failure == "interrupted" else "failed"
    assert event["outcome"] == expected
    assert "partial answer" in sink.store.plain_text()
    assert not controller.busy and not controller.cancelling
    if failure != "interrupted":
        assert event["errors"] == ["worker unavailable"]
        assert "worker unavailable" in result.verdict.unmet[0]
