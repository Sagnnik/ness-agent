"""Bounded worker and judge loop for the CLI goal command."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from ness_cli.instructions import load_instruction
from ness_cli.session.persistence import SessionRepository
from ness_cli.session.turn import TurnOutcome

WorkerTurn = Callable[[str], Awaitable[TurnOutcome]]
StatusCallback = Callable[[str, str], None]

_EVENT_LIMIT = 2_000
_TRANSCRIPT_LIMIT = 100_000
_logger = logging.getLogger(__name__)


class JudgeStructuredOutput(BaseModel):
    passed: bool = Field(description="Whether the worker met the exact goal.")
    unmet: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    repair_instruction: str = ""


@dataclass(frozen=True, slots=True)
class JudgeVerdict:
    passed: bool
    unmet: tuple[str, ...]
    evidence: tuple[str, ...]
    repair_instruction: str


@dataclass(frozen=True, slots=True)
class GoalResult:
    passed: bool
    attempts: int
    verdict: JudgeVerdict


def _clip(value: str, limit: int = _EVENT_LIMIT) -> str:
    text = value.strip()
    return text if len(text) <= limit else text[: limit - 14] + "\n...[truncated]"


def _format_event(event: dict[str, Any]) -> str:
    kind = str(event.get("kind") or "event")
    seq = event.get("seq")
    prefix = f"[{seq}] " if seq is not None else ""
    if kind in {"user", "assistant"}:
        return f"{prefix}{kind}:\n{_clip(str(event.get('content') or ''))}"
    if kind == "tool":
        payload = {
            "tool": event.get("tool"),
            "args": event.get("args"),
            "result": _clip(str(event.get("result") or "")),
            "exit": event.get("exit"),
        }
        return f"{prefix}tool:\n{_clip(json.dumps(payload, ensure_ascii=False, default=str))}"
    payload = {key: value for key, value in event.items() if key != "t"}
    return f"{prefix}{kind}:\n{_clip(json.dumps(payload, ensure_ascii=False, default=str))}"


def _conversation(events: list[dict[str, Any]]) -> str:
    blocks = [
        _format_event(event)
        for event in events
        if event.get("kind") not in {"usage", "compact", "reflection"}
    ]
    kept: list[str] = []
    size = 0
    for block in reversed(blocks):
        if kept and size + len(block) + 2 > _TRANSCRIPT_LIMIT:
            break
        kept.append(block)
        size += len(block) + 2
    kept.reverse()
    omitted = len(blocks) - len(kept)
    prefix = f"(omitted {omitted} older events)\n\n" if omitted else ""
    return prefix + ("\n\n".join(kept) or "(no conversation events)")


class GoalCoordinator:
    def __init__(
        self,
        *,
        repository: SessionRepository,
        thread_id: str,
        hook_runner: Any,
        judge_model: BaseChatModel,
        max_attempts: int,
        instructions_dir: Path,
        is_cancelled: Callable[[], bool],
    ) -> None:
        self._repository = repository
        self._thread_id = thread_id
        self._hooks = hook_runner
        self._judge_model = judge_model
        self._max_attempts = max(1, max_attempts)
        self._instructions_dir = instructions_dir
        self._is_cancelled = is_cancelled

    def _instruction(self, name: str) -> str:
        return load_instruction(name, instructions_dir=self._instructions_dir)

    async def _judge(
        self,
        *,
        goal: str,
        attempt: int,
        start_seq: int,
        validation: str,
    ) -> JudgeVerdict:
        events = [
            event.as_dict()
            for event in self._repository.events_since(self._thread_id, start_seq)
        ]
        prompt = self._instruction("goal_judge.md").format(
            goal=goal,
            attempt=attempt,
            max_attempts=self._max_attempts,
            validation=validation,
            start_seq=start_seq,
            transcript=_conversation(events),
        )
        try:
            structured = self._judge_model.with_structured_output(JudgeStructuredOutput)
            output: JudgeStructuredOutput = await structured.ainvoke(
                [HumanMessage(content=prompt)]
            )
        except Exception as error:
            return JudgeVerdict(
                False,
                (f"Judge structured output failed: {error}",),
                (),
                "",
            )
        return JudgeVerdict(
            bool(output.passed),
            tuple(str(item) for item in output.unmet),
            tuple(str(item) for item in output.evidence),
            str(output.repair_instruction or "").strip(),
        )

    def _stop_worker(
        self,
        attempt: int,
        outcome: TurnOutcome,
        on_status: StatusCallback,
    ) -> GoalResult:
        reason = (
            "Worker turn was interrupted."
            if outcome.status == "interrupted"
            else "Worker turn failed."
        )
        unmet = tuple(
            f"Worker turn failed: {error}" for error in outcome.errors
        ) or (reason,)
        self._repository.append(
            self._thread_id,
            {
                "kind": "goal",
                "phase": "worker",
                "attempt": attempt,
                "outcome": outcome.status,
                "pass": False,
                "errors": list(outcome.errors),
                "unmet": list(unmet),
            },
        )
        on_status(outcome.status, "; ".join(unmet))
        return GoalResult(False, attempt, JudgeVerdict(False, unmet, (), ""))

    async def run(
        self,
        goal: str,
        *,
        worker_turn: WorkerTurn,
        on_status: StatusCallback,
    ) -> GoalResult:
        start_seq = self._repository.append(
            self._thread_id,
            {
                "kind": "goal",
                "phase": "start",
                "goal": goal,
                "max_attempts": self._max_attempts,
            },
        )
        start_seq = int(start_seq or 0)
        instruction = goal
        verdict = JudgeVerdict(False, ("No attempt completed.",), (), goal)

        for attempt in range(1, self._max_attempts + 1):
            on_status("worker", f"attempt {attempt}/{self._max_attempts}")
            try:
                outcome = await worker_turn(instruction)
            except asyncio.CancelledError as error:
                try:
                    self._stop_worker(attempt, TurnOutcome("interrupted"), on_status)
                except Exception as recording_error:
                    diagnostic = f"Could not record goal interruption: {recording_error}"
                    error.add_note(diagnostic)
                    _logger.warning("%s", diagnostic)
                raise
            except Exception as error:
                outcome = TurnOutcome().with_error(str(error) or type(error).__name__)
            if not isinstance(outcome, TurnOutcome):
                outcome = TurnOutcome().with_error("Worker did not return a TurnOutcome.")
            if self._is_cancelled():
                outcome = outcome.with_interruption()
            if not outcome.completed:
                # Retry only completed attempts with unmet validation/verdicts.
                # Repeating a partial failed turn can repeat workspace mutations.
                return self._stop_worker(attempt, outcome, on_status)

            validation_hooks = [
                hook for hook in self._hooks.load() if hook.event == "goalValidate"
            ]
            if validation_hooks:
                hook_ok, message = self._hooks.run(
                    "goalValidate",
                    {
                        "goal": goal,
                        "attempt": attempt,
                        "thread_id": self._thread_id,
                    },
                )
                validation = (
                    f"PASS: {message or 'configured validation passed'}"
                    if hook_ok
                    else f"FAIL: {message or 'configured validation failed'}"
                )
            else:
                hook_ok = True
                validation = "(no deterministic validation configured)"

            on_status("judge", f"verifying attempt {attempt}")
            verdict = await self._judge(
                goal=goal,
                attempt=attempt,
                start_seq=start_seq,
                validation=validation,
            )
            if not hook_ok:
                verdict = JudgeVerdict(
                    False,
                    (*verdict.unmet, f"Deterministic validation failed: {validation}"),
                    (*verdict.evidence, validation),
                    (
                        f"Deterministic validation failed: {validation}. "
                        + verdict.repair_instruction
                    ).strip(),
                )

            passed = verdict.passed and hook_ok
            self._repository.append(
                self._thread_id,
                {
                    "kind": "goal",
                    "phase": "judge",
                    "attempt": attempt,
                    "pass": passed,
                    "unmet": list(verdict.unmet),
                    "evidence": list(verdict.evidence),
                    "repair_instruction": verdict.repair_instruction,
                    "validation": validation,
                },
            )
            if passed:
                return GoalResult(True, attempt, verdict)
            if attempt < self._max_attempts:
                generic = self._instruction("goal_generic_repair.md")
                repair = verdict.repair_instruction.strip()
                if not repair or repair == generic:
                    repair = (
                        "\n".join(item for item in verdict.unmet if item) or generic
                    )
                instruction = self._instruction("goal_repair.md").format(
                    goal=goal,
                    repair=repair,
                )
        return GoalResult(False, self._max_attempts, verdict)
