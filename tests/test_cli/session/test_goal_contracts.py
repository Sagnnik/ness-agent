from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from ness_agent import ThreadStore

from ness_cli.session.goal import (
    GoalCoordinator,
    JudgeVerdict,
    _conversation,
)
from ness_cli.session.persistence import SessionRepository
from ness_cli.session.turn import TurnOutcome


class Hooks:
    def __init__(self, result=(True, "")):
        self.result = result

    def load(self):
        return (
            [] if self.result == (True, "") else [SimpleNamespace(event="goalValidate")]
        )

    def run(self, event, payload):
        return self.result


class ScriptedCoordinator(GoalCoordinator):
    def __init__(self, *args, verdicts, **kwargs):
        super().__init__(*args, **kwargs)
        self.verdicts = iter(verdicts)
        self.judge_calls = []

    async def _judge(self, **kwargs):
        self.judge_calls.append(kwargs)
        return next(self.verdicts)


def make_repository(tmp_path):
    return SessionRepository(
        ThreadStore(threads_dir=tmp_path / "threads", auto_save=True)
    )


def coordinator(tmp_path, verdicts, *, hooks=None, attempts=3, cancelled=lambda: False):
    return ScriptedCoordinator(
        repository=make_repository(tmp_path),
        thread_id="goal-thread",
        hook_runner=hooks or Hooks(),
        judge_model=SimpleNamespace(),
        max_attempts=attempts,
        instructions_dir=Path(__file__).parents[3]
        / "src"
        / "ness_cli"
        / "instructions",
        is_cancelled=cancelled,
        verdicts=verdicts,
    )


def test_goal_repair_loop_is_bounded_and_stops_after_pass(tmp_path):
    runner = coordinator(
        tmp_path,
        [
            JudgeVerdict(False, ("missing piece",), (), "add the missing piece"),
            JudgeVerdict(True, (), ("verified",), ""),
        ],
    )
    prompts = []
    statuses = []

    async def worker(prompt):
        prompts.append(prompt)
        return TurnOutcome()

    result = asyncio.run(
        runner.run(
            "ship it",
            worker_turn=worker,
            on_status=lambda *value: statuses.append(value),
        )
    )
    assert result.passed and result.attempts == 2
    assert prompts[0] == "ship it" and "add the missing piece" in prompts[1]
    assert [phase for phase, _ in statuses] == ["worker", "judge", "worker", "judge"]
    assert [
        event.get("phase")
        for event in runner._repository.raw_events("goal-thread")
        if event.get("kind") == "goal"
    ] == ["start", "judge", "judge"]


def test_goal_validation_failure_overrides_passing_judge(tmp_path):
    runner = coordinator(
        tmp_path,
        [JudgeVerdict(True, (), ("looks good",), "")],
        hooks=Hooks((False, "tests failed")),
        attempts=1,
    )

    async def worker(prompt):
        return TurnOutcome()

    result = asyncio.run(
        runner.run("ship", worker_turn=worker, on_status=lambda *_: None)
    )
    assert not result.passed
    assert "tests failed" in result.verdict.repair_instruction
    assert any(
        "Deterministic validation failed" in item for item in result.verdict.unmet
    )


def test_goal_cancellation_stops_before_judge(tmp_path):
    runner = coordinator(
        tmp_path,
        [JudgeVerdict(True, (), (), "")],
        cancelled=lambda: True,
    )

    async def worker(prompt):
        return TurnOutcome()

    result = asyncio.run(
        runner.run("ship", worker_turn=worker, on_status=lambda *_: None)
    )
    assert not result.passed and result.attempts == 1


def test_goal_conversation_keeps_semantic_events_and_drops_usage():
    transcript = _conversation(
        [
            {"seq": 1, "kind": "user", "content": "ship"},
            {
                "seq": 2,
                "kind": "tool",
                "tool": "write",
                "args": {"path": "a.py"},
                "result": "wrote",
                "exit": "ok",
            },
            {"seq": 3, "kind": "usage", "input_tokens": 5},
        ]
    )
    assert "user:\nship" in transcript and '"tool": "write"' in transcript
    assert "usage" not in transcript


def test_judge_failure_becomes_fail_closed_verdict(tmp_path):
    class BrokenModel:
        def with_structured_output(self, schema):
            return self

        async def ainvoke(self, messages):
            raise RuntimeError("provider failed")

    runner = GoalCoordinator(
        repository=make_repository(tmp_path),
        thread_id="t",
        hook_runner=Hooks(),
        judge_model=BrokenModel(),
        max_attempts=1,
        instructions_dir=Path(__file__).parents[3]
        / "src"
        / "ness_cli"
        / "instructions",
        is_cancelled=lambda: False,
    )
    verdict = asyncio.run(
        runner._judge(goal="g", attempt=1, start_seq=0, validation="none")
    )
    assert not verdict.passed and "structured output failed" in verdict.unmet[0]


def test_failed_worker_cannot_pass_on_a_passing_judge(tmp_path):
    runner = coordinator(tmp_path, [JudgeVerdict(True, (), ("stale evidence",), "")])
    calls = []

    async def worker(prompt):
        calls.append(prompt)
        return TurnOutcome().with_error("provider unavailable")

    result = asyncio.run(
        runner.run("ship", worker_turn=worker, on_status=lambda *_: None)
    )

    assert not result.passed and result.attempts == 1
    assert calls == ["ship"]
    assert not runner.judge_calls
    assert "provider unavailable" in result.verdict.unmet[0]
    events = runner._repository.raw_events("goal-thread")
    assert not any(event.get("phase") == "judge" for event in events)
    assert any(
        event.get("phase") == "worker" and event.get("outcome") == "failed"
        for event in events
    )


class RecordingHooks(Hooks):
    def __init__(self):
        super().__init__((True, "validated"))
        self.load_calls = 0
        self.run_calls = 0

    def load(self):
        self.load_calls += 1
        return super().load()

    def run(self, *args):
        self.run_calls += 1
        return super().run(*args)


@pytest.mark.parametrize(
    "outcome,cancelled,expected",
    [
        (TurnOutcome().with_error("worker unavailable"), False, "failed"),
        (TurnOutcome("interrupted"), False, "interrupted"),
        (TurnOutcome(), True, "interrupted"),
    ],
)
def test_unsuccessful_worker_skips_validation_and_judging(
    tmp_path, outcome, cancelled, expected
):
    hooks = RecordingHooks()
    runner = coordinator(
        tmp_path,
        [JudgeVerdict(True, (), (), "")],
        hooks=hooks,
        cancelled=lambda: cancelled,
    )
    statuses = []
    calls = []

    async def worker(prompt):
        calls.append(prompt)
        return outcome

    result = asyncio.run(
        runner.run(
            "ship", worker_turn=worker, on_status=lambda *args: statuses.append(args)
        )
    )

    assert not result.passed and result.attempts == 1
    assert len(calls) == 1 and not runner.judge_calls
    assert hooks.load_calls == hooks.run_calls == 0
    worker_event = runner._repository.raw_events("goal-thread")[-1]
    assert worker_event["phase"] == "worker" and worker_event["outcome"] == expected
    assert worker_event["pass"] is False
    assert worker_event["errors"] == list(outcome.errors)
    assert statuses[-1][0] == expected


@pytest.mark.parametrize("invalid", [False, True, None, "done"])
def test_worker_must_return_explicit_outcome(tmp_path, invalid):
    runner = coordinator(tmp_path, [JudgeVerdict(True, (), (), "")])

    async def worker(prompt):
        return invalid

    result = asyncio.run(
        runner.run("ship", worker_turn=worker, on_status=lambda *_: None)
    )

    assert not result.passed and not runner.judge_calls
    assert "did not return a TurnOutcome" in result.verdict.unmet[0]


def test_worker_exception_is_recorded_and_stops_goal(tmp_path):
    hooks = RecordingHooks()
    runner = coordinator(tmp_path, [JudgeVerdict(True, (), (), "")], hooks=hooks)

    async def worker(prompt):
        raise RuntimeError("worker exploded")

    result = asyncio.run(
        runner.run("ship", worker_turn=worker, on_status=lambda *_: None)
    )

    assert not result.passed and result.attempts == 1
    assert "worker exploded" in result.verdict.unmet[0]
    assert not runner.judge_calls and not hooks.load_calls
    event = runner._repository.raw_events("goal-thread")[-1]
    assert event["outcome"] == "failed" and event["errors"] == ["worker exploded"]


def test_failure_during_repair_cannot_reuse_previous_attempt_evidence(tmp_path):
    runner = coordinator(
        tmp_path,
        [
            JudgeVerdict(
                False, ("missing test",), ("previous partial evidence",), "add test"
            ),
            JudgeVerdict(True, (), ("stale pass",), ""),
        ],
    )
    outcomes = iter([TurnOutcome(), TurnOutcome().with_error("repair failed")])
    prompts = []

    async def worker(prompt):
        prompts.append(prompt)
        return next(outcomes)

    result = asyncio.run(
        runner.run("ship", worker_turn=worker, on_status=lambda *_: None)
    )

    assert not result.passed and result.attempts == 2
    assert len(prompts) == 2 and "add test" in prompts[1]
    assert len(runner.judge_calls) == 1
    assert result.verdict.evidence == ()
    assert "repair failed" in result.verdict.unmet[0]
    phases = [
        event.get("phase") for event in runner._repository.raw_events("goal-thread")
    ]
    assert phases == ["start", "judge", "worker"]


def test_completed_attempts_still_retry_until_budget_is_exhausted(tmp_path):
    runner = coordinator(
        tmp_path,
        [JudgeVerdict(False, ("unfinished",), (), "try again")] * 2,
        attempts=2,
    )
    calls = []

    async def worker(prompt):
        calls.append(prompt)
        return TurnOutcome()

    result = asyncio.run(
        runner.run("ship", worker_turn=worker, on_status=lambda *_: None)
    )

    assert not result.passed and result.attempts == 2
    assert len(calls) == len(runner.judge_calls) == 2
    assert result.verdict.unmet == ("unfinished",)


@pytest.mark.parametrize("recording_fails", [False, True])
def test_hard_cancellation_is_preserved_and_skips_judge(
    tmp_path, monkeypatch, caplog, recording_fails
):
    runner = coordinator(tmp_path, [JudgeVerdict(True, (), (), "")])
    if recording_fails:
        append = runner._repository.append

        def record(thread_id, event):
            if event.get("phase") == "worker":
                raise OSError("goal record failed")
            return append(thread_id, event)

        monkeypatch.setattr(runner._repository, "append", record)

    async def scenario():
        entered = asyncio.Event()

        async def worker(prompt):
            entered.set()
            await asyncio.Event().wait()

        task = asyncio.create_task(
            runner.run("ship", worker_turn=worker, on_status=lambda *_: None)
        )
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as raised:
            await task
        if recording_fails:
            assert "goal record failed" in raised.value.__notes__[0]

    asyncio.run(scenario())

    assert not runner.judge_calls
    if recording_fails:
        assert "goal record failed" in caplog.text
    else:
        event = runner._repository.raw_events("goal-thread")[-1]
        assert event["phase"] == "worker" and event["outcome"] == "interrupted"
