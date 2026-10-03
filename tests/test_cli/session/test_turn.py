from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from ness_agent import PermissionStore, SessionEvent, ThreadStore

from ness_cli.headless import run_headless_turn
from ness_cli.session.mutations import WorkspaceMutations
from ness_cli.session.persistence import SessionRepository
from ness_cli.session.plans import PlanCapture, PlanStore
from ness_cli.session.rollback import PendingCheckpoint, RollbackService
from ness_cli.session.turn import TurnRequest, TurnRunner


@pytest.fixture
def turn_env(tmp_path):
    plans = Mock(spec=PlanCapture)
    rollback = Mock(spec=RollbackService)
    rollback.snapshot = AsyncMock(return_value=PendingCheckpoint("tree", "memory"))
    repository = Mock(spec=SessionRepository)
    repository.auto_save = True
    repository.append_user.return_value = 7

    async def stream(message, **kwargs):
        yield SessionEvent(kind="assistant_final", data={"content": "done"})

    session = SimpleNamespace(thread_id="thread", stream=stream)
    permissions = PermissionStore(ness_dir=tmp_path / ".ness", project_root=tmp_path)
    runner = TurnRunner(
        session,
        repository,
        permission_store=permissions,
        rollback=rollback,
        plans=plans,
    )
    return SimpleNamespace(
        runner=runner,
        session=session,
        repository=repository,
        rollback=rollback,
        plans=plans,
    )


async def collect(runner, request=None):
    return [event async for event in runner.stream(request or TurnRequest("hello"))]


@pytest.mark.parametrize(
    "plan_fails,mutations_fail",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_finalizers_run_independently_and_emit_warnings(
    turn_env, plan_fails, mutations_fail
):
    env = turn_env
    if plan_fails:
        env.plans.finish_turn.side_effect = OSError("plan disk full")
    if mutations_fail:
        env.rollback.record_mutations.side_effect = OSError("checkpoint disk full")

    events = asyncio.run(collect(env.runner))

    env.plans.finish_turn.assert_called_once_with()
    env.rollback.record_mutations.assert_called_once_with("thread", 7)
    assert events[0].kind == "assistant_final"
    warnings = [event.data["message"] for event in events if event.kind == "warning"]
    assert len(warnings) == int(plan_fails) + int(mutations_fail)
    if plan_fails:
        assert any(
            "save plan" in message and "plan disk full" in message
            for message in warnings
        )
    if mutations_fail:
        assert any(
            "record workspace mutations" in message
            and "checkpoint disk full" in message
            for message in warnings
        )


def test_turn_error_survives_both_cleanup_failures(turn_env, caplog):
    env = turn_env
    original = RuntimeError("model failed")

    async def stream(message, **kwargs):
        yield SessionEvent(kind="assistant_delta", data={"content": "partial"})
        raise original

    env.session.stream = stream
    env.plans.finish_turn.side_effect = OSError("plan failed")
    env.rollback.record_mutations.side_effect = OSError("bookkeeping failed")

    with caplog.at_level(logging.WARNING), pytest.raises(RuntimeError) as raised:
        asyncio.run(collect(env.runner))

    assert raised.value is original
    assert "plan failed" in caplog.text and "bookkeeping failed" in caplog.text
    assert len(original.__notes__) == 2
    env.plans.finish_turn.assert_called_once()
    env.rollback.record_mutations.assert_called_once_with("thread", 7)


@pytest.mark.parametrize(
    "stage", ["begin", "snapshot", "append", "save", "expand", "compaction"]
)
def test_preparation_and_persistence_errors_still_finalize(
    turn_env, monkeypatch, stage
):
    env = turn_env
    original = OSError(f"{stage} failed")
    if stage == "begin":
        env.plans.begin_turn.side_effect = original
    elif stage == "snapshot":
        env.rollback.snapshot.side_effect = original
    elif stage == "append":
        env.repository.append_user.side_effect = original
    elif stage == "save":
        env.rollback.save.side_effect = original
    elif stage == "expand":
        monkeypatch.setattr(
            "ness_cli.session.turn.expand_documents", Mock(side_effect=original)
        )
    else:

        async def stream(message, **kwargs):
            yield SessionEvent(kind="compaction", data={"info": "compacted"})

        env.session.stream = stream
        env.repository.append_compaction.side_effect = original

    with pytest.raises(OSError) as raised:
        asyncio.run(collect(env.runner))

    assert raised.value is original
    env.plans.finish_turn.assert_called_once()
    expected_seq = None if stage in {"begin", "snapshot", "append"} else 7
    env.rollback.record_mutations.assert_called_once_with("thread", expected_seq)


def test_cancellation_preserves_cancel_and_reports_cleanup_failures(turn_env, caplog):
    env = turn_env
    env.plans.finish_turn.side_effect = OSError("plan failed during cancellation")
    env.rollback.record_mutations.side_effect = OSError(
        "bookkeeping failed during cancellation"
    )

    async def scenario():
        entered = asyncio.Event()

        async def stream(message, **kwargs):
            entered.set()
            await asyncio.Event().wait()
            yield  # pragma: no cover

        env.session.stream = stream
        task = asyncio.create_task(collect(env.runner))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as raised:
            await task
        assert len(raised.value.__notes__) == 2

    with caplog.at_level(logging.WARNING):
        asyncio.run(scenario())

    assert "plan failed during cancellation" in caplog.text
    assert "bookkeeping failed during cancellation" in caplog.text
    env.rollback.record_mutations.assert_called_once_with("thread", 7)


def test_explicit_stream_close_finalizes_without_yielding(turn_env, caplog):
    env = turn_env
    env.plans.finish_turn.side_effect = OSError("plan failed during close")
    env.rollback.record_mutations.side_effect = OSError(
        "bookkeeping failed during close"
    )

    async def scenario():
        stream = env.runner.stream(TurnRequest("hello"))
        assert (await anext(stream)).kind == "assistant_final"
        await stream.aclose()

    with caplog.at_level(logging.WARNING):
        asyncio.run(scenario())

    assert "plan failed during close" in caplog.text
    assert "bookkeeping failed during close" in caplog.text
    env.plans.finish_turn.assert_called_once()
    env.rollback.record_mutations.assert_called_once_with("thread", 7)


def test_autosave_disabled_has_no_checkpoint_sequence(turn_env):
    env = turn_env
    env.repository.auto_save = False
    env.repository.append_user.return_value = None

    asyncio.run(collect(env.runner))

    env.rollback.snapshot.assert_not_awaited()
    env.rollback.save.assert_called_once_with(
        "thread", None, PendingCheckpoint(None, "")
    )
    env.rollback.record_mutations.assert_called_once_with("thread", None)


def test_failed_plan_write_releases_workspace_recorder_and_warns_headless(
    tmp_path, capsys
):
    repo = SessionRepository(
        ThreadStore(threads_dir=tmp_path / "threads", auto_save=True)
    )
    mutations = WorkspaceMutations(tmp_path, repo)
    service = RollbackService(
        project_root=tmp_path,
        repository=repo,
        memory=SimpleNamespace(read_session_raw=lambda _: ""),
        mutations=mutations,
    )
    # A file in place of the plans directory causes a real plan-write failure.
    plans_dir = tmp_path / "plans"
    plans_dir.write_text("blocked")
    plans = PlanCapture(
        PlanStore(plans_dir), thread_id=lambda: "thread", in_plan_mode=lambda: True
    )

    async def stream(message, **kwargs):
        assert not mutations.begin_restore("thread")
        plans.on_plan_turn("saved plan content")
        yield SessionEvent(kind="assistant_final", data={"content": "done"})

    runner = TurnRunner(
        SimpleNamespace(thread_id="thread", stream=stream),
        repo,
        permission_store=PermissionStore(
            ness_dir=tmp_path / ".ness", project_root=tmp_path
        ),
        rollback=service,
        plans=plans,
    )
    assert asyncio.run(run_headless_turn(runner, "hello")) == ("done", 0)
    assert "Failed to save plan" in capsys.readouterr().err
    assert mutations.begin_restore("thread")
    mutations.end_restore()
    assert any(event.kind == "workspace_tracking" for event in repo.events("thread"))


def test_partial_checkpoint_save_failure_releases_workspace_recorder(
    tmp_path, monkeypatch
):
    repo = SessionRepository(
        ThreadStore(threads_dir=tmp_path / "threads", auto_save=True)
    )
    mutations = WorkspaceMutations(tmp_path, repo)
    service = RollbackService(
        project_root=tmp_path,
        repository=repo,
        memory=SimpleNamespace(read_session_raw=lambda _: ""),
        mutations=mutations,
    )
    original = OSError("tracking write failed")
    append = repo.append

    def append_with_tracking_failure(thread_id, event):
        if event["kind"] == "workspace_tracking":
            assert not mutations.begin_restore(thread_id)
            raise original
        return append(thread_id, event)

    monkeypatch.setattr(repo, "append", append_with_tracking_failure)
    plans = Mock(spec=PlanCapture)
    session = SimpleNamespace(thread_id="thread", stream=Mock())
    runner = TurnRunner(
        session,
        repo,
        permission_store=PermissionStore(
            ness_dir=tmp_path / ".ness", project_root=tmp_path
        ),
        rollback=service,
        plans=plans,
    )

    with pytest.raises(OSError) as raised:
        asyncio.run(collect(runner))

    assert raised.value is original
    assert mutations.begin_restore("thread")
    mutations.end_restore()
    assert repo.checkpoint("thread", 0) is not None
    plans.finish_turn.assert_called_once()
    session.stream.assert_not_called()


def test_failed_mutation_recording_still_saves_real_plan(turn_env, tmp_path, capsys):
    env = turn_env
    plans = PlanCapture(
        PlanStore(tmp_path / "plans"),
        thread_id=lambda: "thread",
        in_plan_mode=lambda: True,
    )
    env.runner._plans = plans

    async def stream(message, **kwargs):
        plans.on_plan_turn("actual plan")
        yield SessionEvent(kind="assistant_final", data={"content": "done"})

    env.session.stream = stream
    env.rollback.record_mutations.side_effect = OSError("checkpoint database failed")

    assert asyncio.run(run_headless_turn(env.runner, "hello")) == ("done", 0)
    saved = list((tmp_path / "plans").glob("*.md"))
    assert len(saved) == 1 and saved[0].read_text() == "actual plan\n"
    diagnostic = capsys.readouterr().err
    assert "record workspace mutations" in diagnostic
    assert "checkpoint database failed" in diagnostic


def test_invalid_request_does_not_begin_or_finalize(turn_env):
    with pytest.raises(ValueError, match="cannot be empty"):
        asyncio.run(collect(turn_env.runner, TurnRequest("[Image #1]")))
    turn_env.plans.begin_turn.assert_not_called()
    turn_env.plans.finish_turn.assert_not_called()
    turn_env.rollback.record_mutations.assert_not_called()
