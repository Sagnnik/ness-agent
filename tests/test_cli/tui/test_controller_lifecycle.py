from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from ness_agent import SessionEvent

from ness_cli.tui.controller import ThreadRuntime, TuiController
from ness_cli.tui.sink import NullPromptDriver, TuiRenderSink
from ness_cli.tui.transcript.store import TranscriptStore


class Timer:
    def __init__(self, callback, args):
        self.callback = callback
        self.args = args
        self.cancelled = False
        self.fired = False

    def cancel(self):
        self.cancelled = True

    def fire(self):
        # Deliberately invoke even a cancelled callback to test stale ownership.
        self.fired = True
        self.callback(*self.args)


def capture_backstops(monkeypatch):
    loop = asyncio.get_running_loop()
    original = loop.call_later
    timers = []

    def call_later(delay, callback, *args, **kwargs):
        if delay != 10.0:
            return original(delay, callback, *args, **kwargs)
        timer = Timer(callback, args)
        timers.append(timer)
        return timer

    monkeypatch.setattr(loop, "call_later", call_later)
    return timers


class Session:
    thread_id = "goal-thread"
    mode = "act"

    def __init__(self):
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.cooperative = True
        self.stream_error = None
        self.goal_error = None
        self.cancel_calls = 0
        self.goal_result = object()

    async def stream(self, request):
        self.started.set()
        await self.release.wait()
        if self.stream_error is not None:
            raise self.stream_error
        yield SessionEvent("assistant_final", {"content": "done"})

    async def run_goal(
        self, goal, *, worker_turn, on_status, judge_model, max_attempts
    ):
        on_status("worker", "attempt 1")
        await worker_turn(goal)
        if self.goal_error is not None:
            raise self.goal_error
        return self.goal_result

    def cancel(self):
        self.cancel_calls += 1
        if self.cooperative:
            self.release.set()


def controller_for(session):
    sink = TuiRenderSink(
        TranscriptStore(), invalidate=lambda: None, prompts=NullPromptDriver()
    )
    controller = TuiController(
        SimpleNamespace(
            config=SimpleNamespace(settings=SimpleNamespace(goal_max_attempts=2)),
            goal_judge_model=lambda _: object(),
        ),
        sink,
        changed=lambda: None,
    )
    selected = ThreadRuntime(session, sink=sink)
    controller._selected = selected
    controller._runtimes[session.thread_id] = selected
    return controller, selected


def assert_idle(controller, selected):
    assert selected.task is None
    assert selected.cancel_backstop is None
    assert selected.started_at is None
    assert not selected.cancelling
    assert not controller.busy
    assert controller.working_elapsed == 0.0


@pytest.mark.parametrize("operation", ["goal", "turn"])
@pytest.mark.parametrize("exit_kind", ["normal", "error", "hard_cancel"])
def test_operations_clear_cancellation_state_on_every_exit(
    monkeypatch, operation, exit_kind
):
    async def scenario():
        timers = capture_backstops(monkeypatch)
        session = Session()
        controller, selected = controller_for(session)
        original = RuntimeError("operation failed")
        if exit_kind == "error":
            if operation == "goal":
                session.goal_error = original
            else:
                session.stream_error = original
        session.cooperative = exit_kind != "hard_cancel"
        run = (
            controller.run_goal("finish goal")
            if operation == "goal"
            else controller.submit("finish turn")
        )
        task = asyncio.create_task(run)
        await session.started.wait()
        assert selected.task is task and controller.busy
        assert selected.started_at is not None
        controller.enqueue("discard this queued prompt")
        assert controller.cancel()
        assert selected.cancelling and not selected.prompts
        timer = timers[-1]

        if exit_kind == "hard_cancel":
            timer.fire()
            with pytest.raises(asyncio.CancelledError):
                await task
        elif exit_kind == "error" and operation == "goal":
            with pytest.raises(RuntimeError) as raised:
                await task
            assert raised.value is original
        else:
            result = await task
            if operation == "goal":
                assert result is session.goal_result
            else:
                assert result is (exit_kind == "normal")

        assert timer.cancelled or timer.fired
        assert_idle(controller, selected)
        assert not controller.cancel()

    asyncio.run(scenario())


def test_old_goal_timer_cannot_cancel_next_turn_or_clear_its_timer(monkeypatch):
    async def scenario():
        timers = capture_backstops(monkeypatch)
        session = Session()
        controller, selected = controller_for(session)
        goal = asyncio.create_task(controller.run_goal("finish goal"))
        await session.started.wait()
        assert controller.cancel()
        old_timer = timers[-1]
        await goal

        session.started.clear()
        session.release.clear()
        session.cooperative = False
        turn = asyncio.create_task(controller.submit("next turn"))
        await session.started.wait()
        try:
            assert controller.cancel()
            current_timer = timers[-1]
            old_timer.fire()
            await asyncio.sleep(0)
            assert not turn.done() and not turn.cancelling()
            assert selected.cancel_backstop is current_timer
            assert not current_timer.cancelled
            assert old_timer.cancelled
        finally:
            session.release.set()
            await asyncio.gather(turn, return_exceptions=True)
        assert_idle(controller, selected)

    asyncio.run(scenario())


def test_replaced_timer_cannot_cancel_the_same_operation_early(monkeypatch):
    async def scenario():
        timers = capture_backstops(monkeypatch)
        session = Session()
        session.cooperative = False
        controller, selected = controller_for(session)
        goal = asyncio.create_task(controller.run_goal("finish goal"))
        await session.started.wait()
        try:
            assert controller.cancel()
            old_timer = timers[-1]
            assert controller.cancel()
            current_timer = timers[-1]
            old_timer.fire()
            await asyncio.sleep(0)
            assert not goal.done() and not goal.cancelling()
            assert selected.cancel_backstop is current_timer
            assert old_timer.cancelled
            current_timer.fire()
            with pytest.raises(asyncio.CancelledError):
                await goal
        finally:
            session.release.set()
            await asyncio.gather(goal, return_exceptions=True)
        assert_idle(controller, selected)

    asyncio.run(scenario())


def test_old_timer_cannot_target_a_new_operation_reusing_the_same_task(monkeypatch):
    async def scenario():
        timers = capture_backstops(monkeypatch)
        session = Session()
        controller, selected = controller_for(session)
        operation_task = asyncio.current_task()

        async def cancel_goal():
            await session.started.wait()
            assert selected.task is operation_task
            assert controller.cancel()

        cancel_task = asyncio.create_task(cancel_goal())
        await controller.run_goal("finish goal")
        await cancel_task
        old_timer = timers[-1]

        session.started.clear()
        session.release.clear()
        session.cooperative = False

        async def check_next_turn():
            try:
                await session.started.wait()
                assert selected.task is operation_task
                assert controller.cancel()
                current_timer = timers[-1]
                old_timer.fire()
                assert not operation_task.cancelling()
                assert selected.cancel_backstop is current_timer
            finally:
                session.release.set()

        check_task = asyncio.create_task(check_next_turn())
        try:
            assert await controller.submit("next turn")
        finally:
            await check_task
        assert_idle(controller, selected)

    asyncio.run(scenario())


def test_controller_close_clears_goal_timer_and_waits_for_completion(monkeypatch):
    async def scenario():
        timers = capture_backstops(monkeypatch)
        session = Session()
        session.cooperative = False
        controller, selected = controller_for(session)
        goal = asyncio.create_task(controller.run_goal("finish goal"))
        await session.started.wait()
        assert controller.cancel()
        timer = timers[-1]
        session.cooperative = True

        await controller.close()

        assert timer.cancelled
        assert await goal is session.goal_result
        assert_idle(controller, selected)
        timer.fire()
        assert not goal.cancelled()

    asyncio.run(scenario())
