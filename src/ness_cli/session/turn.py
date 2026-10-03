"""Policy for one CLI turn."""

from __future__ import annotations

import logging
import re
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Literal

from ness_agent import PermissionStore, Session, SessionEvent

from ness_cli.session.mentions import expand_documents
from ness_cli.session.plans import PlanCapture
from ness_cli.session.persistence import SessionRepository
from ness_cli.session.rollback import PendingCheckpoint, RollbackService

_IMAGE_PLACEHOLDER = re.compile(r"\[Image #\d+\]")
_logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TurnRequest:
    message: str
    images: tuple[str, ...] = ()
    requested_skills: tuple[str, ...] = ()
    mode: str | None = None


@dataclass(frozen=True, slots=True)
class TurnOutcome:
    """Turn completion policy shared by interactive, headless, and goal callers.

    Errors remain failures even if later events contain a final response or an
    interruption. Warnings do not change the outcome.
    """

    status: Literal["completed", "failed", "interrupted"] = "completed"
    errors: tuple[str, ...] = ()

    @property
    def completed(self) -> bool:
        return self.status == "completed"

    @property
    def exit_code(self) -> int:
        return {"completed": 0, "failed": 1, "interrupted": 130}[self.status]

    def with_error(self, message: str) -> TurnOutcome:
        return TurnOutcome("failed", (*self.errors, message))

    def with_interruption(self) -> TurnOutcome:
        return TurnOutcome("interrupted") if self.completed else self

    def observe(self, event: SessionEvent) -> TurnOutcome:
        if event.kind == "error":
            return self.with_error(str(event.data.get("message") or event.data))
        if event.kind == "interrupted":
            return self.with_interruption()
        return self


class TurnRunner:
    """Persist CLI-owned events around an SDK-owned streamed turn."""

    def __init__(
        self,
        session: Session,
        repository: SessionRepository,
        *,
        permission_store: PermissionStore,
        rollback: RollbackService,
        plans: PlanCapture,
    ) -> None:
        self._session = session
        self._repository = repository
        self._permission_store = permission_store
        self._rollback = rollback
        self._plans = plans

    async def stream(self, request: TurnRequest) -> AsyncIterator[SessionEvent]:
        message = _IMAGE_PLACEHOLDER.sub("", request.message or "").strip()
        images: Sequence[str] = request.images
        if not message and not images:
            raise ValueError("turn message cannot be empty")

        thread_id = self._session.thread_id
        user_seq: int | None = None
        turn_error: BaseException | None = None
        warnings: list[str] = []
        try:
            self._plans.begin_turn()
            checkpoint = (
                await self._rollback.snapshot(thread_id)
                if self._repository.auto_save
                else PendingCheckpoint(None, "")
            )
            user_seq = self._repository.append_user(thread_id, message, images=images)
            self._rollback.save(thread_id, user_seq, checkpoint)
            config = getattr(self._session, "config", None)
            expanded = expand_documents(
                message, self._permission_store,
                yolo_mode=bool(getattr(getattr(config, "options", None), "yolo_mode", False)),
            )

            async for event in self._session.stream(
                expanded,
                images=images,
                requested_skills=request.requested_skills,
                mode=request.mode,
            ):
                if event.kind == "compaction":
                    self._repository.append_compaction(
                        thread_id,
                        event.data,
                    )
                yield event
        except BaseException as exc:
            # Preserve turn errors, cancellation, and async-generator closure.
            turn_error = exc
            raise
        finally:
            finalizers = (
                ("save plan", self._plans.finish_turn),
                (
                    "record workspace mutations",
                    lambda: self._rollback.record_mutations(thread_id, user_seq),
                ),
            )
            for operation, finalize in finalizers:
                try:
                    finalize()
                except Exception as exc:
                    diagnostic = f"Failed to {operation} for thread {thread_id}: {exc}"
                    warnings.append(diagnostic)
                    if turn_error is not None:
                        turn_error.add_note(diagnostic)
                        _logger.warning("%s", diagnostic)

        # Yield only after normal completion. Yielding during aclose() would
        # replace GeneratorExit with 'async generator ignored GeneratorExit'.
        for diagnostic in warnings:
            yield SessionEvent(kind="warning", data={"message": diagnostic})
