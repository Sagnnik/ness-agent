"""Live interactive session state and turn execution."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from functools import partial
from typing import Callable

from ness_cli.runtime import InteractiveRuntime
from ness_cli.session import CodingSession, ForkResult, TurnOutcome, TurnRequest
from ness_cli.session.goal import GoalResult
from ness_cli.session.rollback import RollbackResult
from ness_cli.tui.sink import (
    SinkApprovalHandler,
    TuiRenderSink,
    sink_question_handler,
)
from ness_cli.tui.turn_renderer import TurnRenderer, replay_event


@dataclass(slots=True)
class ThreadRuntime:
    session: CodingSession
    sink: TuiRenderSink | None = None
    task: asyncio.Task[bool] | None = None
    cancel_backstop: asyncio.TimerHandle | None = None
    cancelling: bool = False
    prompts: deque[TurnRequest] = field(default_factory=deque)
    selected: asyncio.Event = field(default_factory=asyncio.Event)
    waiting_for_input: bool = False
    started_at: float | None = None

    @property
    def busy(self) -> bool:
        return self.task is not None and not self.task.done()


class TuiController:
    """Own the live thread runtimes and select one for display and input."""

    def __init__(
        self,
        runtime: InteractiveRuntime,
        sink: TuiRenderSink | None = None,
        *,
        sink_factory: Callable[[str], TuiRenderSink] | None = None,
        selected_changed: Callable[[TuiRenderSink], None] | None = None,
        changed: Callable[[], None],
    ) -> None:
        self._runtime = runtime
        self._fallback_sink = sink
        self._sink_factory = sink_factory
        self._selected_changed = selected_changed
        self._changed = changed
        self._runtimes: dict[str, ThreadRuntime] = {}
        self._selected: ThreadRuntime | None = None

    @property
    def session(self) -> CodingSession:
        if self._selected is None:
            raise RuntimeError("TUI controller is not initialized")
        return self._selected.session

    @property
    def sink(self) -> TuiRenderSink:
        selected = self._require_selected()
        if selected.sink is None:
            raise RuntimeError("selected thread has no render sink")
        return selected.sink

    @property
    def thread_id(self) -> str:
        return self.session.thread_id

    @property
    def mode(self) -> str:
        return self.session.mode

    @property
    def busy(self) -> bool:
        return self._selected is not None and self._selected.busy

    @property
    def cancelling(self) -> bool:
        return bool(self._selected and self._selected.cancelling)

    @property
    def working_elapsed(self) -> float:
        if self._selected is None or self._selected.started_at is None:
            return 0.0
        return max(0.0, time.monotonic() - self._selected.started_at)

    @property
    def queued_prompts(self) -> tuple[TurnRequest, ...]:
        if self._selected is None:
            return ()
        return tuple(self._selected.prompts)

    async def initialize(self) -> None:
        resumed = bool(self._runtime.resume_thread_id)
        missing: str | None = None
        try:
            session = await self._runtime.initial_session()
        except LookupError:
            missing = self._runtime.resume_thread_id or "unknown"
            session = await self._runtime.new_session()
            resumed = False
        selected = self._create_runtime(session)
        await self._refresh_context(selected)
        if resumed:
            for event in session.history():
                replay_event(self._require_sink(selected), event)
        await self._refresh_todos(selected)
        self._select(selected)
        if missing is not None:
            self.sink.warning(
                f"No saved thread named {missing}; started a new thread."
            )

    async def submit(
        self,
        text: str,
        *,
        images: tuple[str, ...] = (),
        requested_skills: tuple[str, ...] = (),
    ) -> bool:
        message = text.strip()
        if not message:
            return False
        selected = self._require_selected()
        if selected.busy:
            self._require_sink(selected).warning("A turn is already running.")
            return False

        with self._running_operation(selected):
            outcome = await self._stream_turn(
                selected, message, images=images, requested_skills=requested_skills
            )
            while outcome.completed and selected.prompts:
                queued = selected.prompts.popleft()
                outcome = await self._stream_turn(
                    selected,
                    queued.message,
                    images=queued.images,
                    requested_skills=queued.requested_skills,
                )
            return outcome.completed

    @contextmanager
    def _running_operation(self, selected: ThreadRuntime) -> Iterator[None]:
        selected.task = asyncio.current_task()
        selected.cancelling = False
        selected.started_at = time.monotonic()
        try:
            self._changed()
            yield
        finally:
            self._clear_cancel_backstop(selected)
            selected.task = None
            selected.cancelling = False
            selected.started_at = None
            self._changed()

    @staticmethod
    def _clear_cancel_backstop(selected: ThreadRuntime) -> None:
        if selected.cancel_backstop is not None:
            selected.cancel_backstop.cancel()
            selected.cancel_backstop = None

    def enqueue(
        self,
        text: str,
        *,
        images: tuple[str, ...] = (),
        requested_skills: tuple[str, ...] = (),
    ) -> int:
        selected = self._require_selected()
        selected.prompts.append(
            TurnRequest(text.strip(), images=images, requested_skills=requested_skills)
        )
        self._changed()
        return len(selected.prompts)

    async def _stream_turn(
        self,
        selected: ThreadRuntime,
        message: str,
        *,
        images: tuple[str, ...] = (),
        requested_skills: tuple[str, ...] = (),
    ) -> TurnOutcome:
        sink = self._require_sink(selected)
        sink.user_message(message)
        sink.begin_turn()
        renderer = TurnRenderer(sink)
        outcome = TurnOutcome()
        renderer_finished = False
        try:
            async for event in selected.session.stream(
                TurnRequest(message, images=images, requested_skills=requested_skills)
            ):
                outcome = outcome.observe(event)
                renderer.feed(event)
                if (
                    event.kind == "tool_end"
                    and str(event.data.get("name") or "").lower() == "todo"
                ):
                    await self._refresh_todos(selected)
            renderer.finish()
            renderer_finished = True
            await self._refresh_context(selected)
            await self._refresh_todos(selected)
            return outcome
        except asyncio.CancelledError:
            selected.session.cancel()
            raise
        except Exception as exc:
            sink.error(str(exc))
            return outcome.with_error(str(exc) or type(exc).__name__)
        finally:
            if not renderer_finished:
                renderer.close()

    def cancel(self) -> bool:
        selected = self._require_selected()
        if not selected.busy:
            return False
        selected.cancelling = True
        selected.prompts.clear()
        selected.session.cancel()
        self._clear_cancel_backstop(selected)
        task = selected.task
        assert task is not None

        def hard_cancel() -> None:
            # A cancelled/replaced timer must not target a later operation or
            # clear that operation's timer, even when the asyncio task is reused.
            if selected.task is not task or selected.cancel_backstop is not backstop:
                return
            selected.cancel_backstop = None
            if not task.done():
                task.cancel()

        backstop = asyncio.get_running_loop().call_later(10.0, hard_cancel)
        selected.cancel_backstop = backstop
        self._changed()
        return True

    def toggle_mode(self) -> str:
        mode = self.session.toggle_mode()
        sink = self._require_selected().sink or self._fallback_sink
        set_header_mode = getattr(sink, "set_header_mode", None)
        if callable(set_header_mode):
            set_header_mode(mode)
        self._changed()
        return mode

    async def new_thread(self) -> CodingSession:
        source = self._require_selected()
        if not source.busy:
            await source.session.save()
        session = await self._runtime.new_session(mode=source.session.mode)
        target = self._create_runtime(session)
        await self._refresh_context(target)
        self._select(target)
        return session

    async def resume_thread(self, thread_id: str) -> CodingSession:
        if thread_id == self.thread_id:
            return self.session
        live = self._runtimes.get(thread_id)
        if live is not None:
            self._select(live)
            return live.session
        session = await self._runtime.resume_session(thread_id)
        target = self._create_runtime(session)
        await self._refresh_context(target)
        for event in session.history():
            replay_event(self._require_sink(target), event)
        await self._refresh_todos(target)
        self._select(target)
        return session

    def list_threads(self) -> list[dict]:
        """Return persisted threads with live state overlaid."""
        rows = [dict(row) for row in self.session.list_threads()]
        by_id = {str(row.get("thread_id") or ""): row for row in rows}
        for thread_id, runtime in self._runtimes.items():
            row = by_id.get(thread_id)
            usage = runtime.session.usage_summary()
            if row is None:
                row = {
                    "thread_id": thread_id,
                    "label": "(no messages)",
                    "turn_count": usage["turns"],
                    "total_cost_usd": usage["cost_usd"],
                }
                rows.insert(0, row)
                by_id[thread_id] = row
            if runtime.waiting_for_input:
                row["live_status"] = "waiting for input"
            elif runtime.cancelling:
                row["live_status"] = "cancelling"
            elif runtime.busy:
                row["live_status"] = "working"
            else:
                row["live_status"] = "live"
        return rows

    async def begin_interaction(self, thread_id: str) -> None:
        runtime = self._runtimes.get(thread_id)
        if runtime is None:
            raise LookupError(thread_id)
        runtime.waiting_for_input = True
        self._changed()
        try:
            await runtime.selected.wait()
        except BaseException:
            runtime.waiting_for_input = False
            self._changed()
            raise

    def finish_interaction(self, thread_id: str) -> None:
        runtime = self._runtimes.get(thread_id)
        if runtime is None:
            return
        runtime.waiting_for_input = False
        self._changed()

    async def rollback_to(self, user_seq: int) -> RollbackResult:
        self._ensure_idle()
        result = await self.session.rollback_to(user_seq)
        self._changed()
        return result

    async def fork_before(self, user_seq: int) -> ForkResult:
        self._ensure_idle()
        source = self.session
        result = await source.fork_before(user_seq)
        await source.save()
        target = await self._runtime.resume_session(result.thread_id)
        target_runtime = self._create_runtime(target)
        for event in target.history():
            replay_event(self._require_sink(target_runtime), event)
        await self._refresh_todos(target_runtime)
        self._select(target_runtime)
        return result

    async def refresh_todos(self) -> None:
        await self._refresh_todos(self._require_selected())

    async def run_goal(self, goal: str) -> GoalResult:
        self._ensure_idle()
        selected = self._require_selected()
        with self._running_operation(selected):
            return await selected.session.run_goal(
                goal,
                worker_turn=lambda prompt: self._stream_turn(selected, prompt),
                on_status=lambda phase, detail: self._require_sink(selected).notice(
                    "goal", f"{phase}: {detail}"
                ),
                judge_model=self._runtime.goal_judge_model(selected.session.thread_id),
                max_attempts=self._runtime.config.settings.goal_max_attempts,
            )

    async def close(self) -> None:
        active: list[asyncio.Task[bool]] = []
        for runtime in self._runtimes.values():
            self._clear_cancel_backstop(runtime)
            if not runtime.busy:
                continue
            runtime.cancelling = True
            runtime.session.cancel()
            assert runtime.task is not None
            active.append(runtime.task)
        if not active:
            return
        done, pending = await asyncio.wait(active, timeout=2.0)
        del done
        for task in pending:
            task.cancel()
        if pending:
            with suppress(asyncio.CancelledError):
                await asyncio.gather(*pending, return_exceptions=True)

    def _create_runtime(self, session: CodingSession) -> ThreadRuntime:
        existing = self._runtimes.get(session.thread_id)
        if existing is not None:
            return existing
        sink = (
            self._sink_factory(session.thread_id)
            if self._sink_factory is not None
            else self._fallback_sink
        )
        if sink is None:
            raise RuntimeError("TUI controller requires a render sink")
        runtime = ThreadRuntime(session, sink=sink)
        set_header_mode = getattr(sink, "set_header_mode", None)
        if callable(set_header_mode):
            set_header_mode(session.mode)
        self._runtimes[session.thread_id] = runtime
        bind = getattr(session, "install_interaction_handlers", None)
        if callable(bind):
            bind(
                approval_handler=SinkApprovalHandler(sink),
                question_handler=partial(sink_question_handler, sink),
            )
        return runtime

    def _select(self, runtime: ThreadRuntime) -> None:
        if self._selected is runtime:
            return
        if self._selected is not None:
            self._selected.selected.clear()
        self._selected = runtime
        runtime.selected.set()
        sink = self._require_sink(runtime)
        if self._selected_changed is not None:
            self._selected_changed(sink)
        self._changed()

    @staticmethod
    def _require_sink(runtime: ThreadRuntime) -> TuiRenderSink:
        if runtime.sink is None:
            raise RuntimeError("thread has no render sink")
        return runtime.sink

    def _require_selected(self) -> ThreadRuntime:
        if self._selected is None:
            raise RuntimeError("TUI controller is not initialized")
        return self._selected

    def _ensure_idle(self) -> None:
        if self.busy:
            raise RuntimeError("This operation requires an idle session.")

    async def _refresh_todos(self, runtime: ThreadRuntime) -> None:
        get_todos = getattr(runtime.session, "get_todos", None)
        if not callable(get_todos):
            return
        try:
            items = await get_todos()
        except Exception as error:
            self._require_sink(runtime).warning(
                f"Could not refresh todos: {error}"
            )
            return
        self._require_sink(runtime).todos(list(items or ()))

    async def _refresh_context(self, runtime: ThreadRuntime) -> None:
        refresh = getattr(runtime.session, "refresh_context_snapshot", None)
        if not callable(refresh):
            return
        try:
            await refresh()
        except Exception as error:
            self._require_sink(runtime).warning(
                f"Could not refresh context usage: {error}"
            )
