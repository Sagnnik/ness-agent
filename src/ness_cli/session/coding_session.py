"""Small CLI-facing wrapper around an SDK session."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from ness_agent import ReflectionResult, Session, SessionEvent

from ness_cli.config import RuntimeConfig
from ness_cli.config.settings import sdk_behavior_values
from ness_cli.paths import NessPaths
from ness_cli.session.events import DurableEvent, UserTurn
from ness_cli.session.plans import PlanCapture, PlanStore
from ness_cli.session.persistence import SessionRepository
from ness_cli.session.mutations import WorkspaceMutations
from ness_cli.session.replay import events_to_messages, restore_cost
from ness_cli.session.rollback import RollbackResult, RollbackService
from ness_cli.session.skill_state import SkillStateStore
from ness_cli.session.turn import TurnRequest, TurnRunner

if TYPE_CHECKING:
    from ness_cli.session.goal import GoalResult, StatusCallback, WorkerTurn


@dataclass(frozen=True, slots=True)
class SessionModels:
    model: BaseChatModel
    reflection_model: BaseChatModel | None
    context_window: int | None
    vision: bool | None
    runtime_config: RuntimeConfig


@dataclass(frozen=True, slots=True)
class SaveResult:
    thread_id: str
    resume_thread_id: str | None
    message: str


@dataclass(frozen=True, slots=True)
class ForkResult:
    thread_id: str
    prompt: str
    copied_events: int


ModelLoader = Callable[[str], SessionModels]


class CodingSession:
    """Expose CLI policy without mirroring the full SDK surface."""

    def __init__(
        self,
        session: Session,
        *,
        paths: NessPaths,
        runtime_config: RuntimeConfig,
        repository: SessionRepository,
        model_loader: ModelLoader,
        vision: bool | None,
    ) -> None:
        self._session = session
        self._paths = paths
        self._runtime_config = runtime_config
        self._repository = repository
        self._model_loader = model_loader
        self._closed = False
        self._reflected_turn_count = -1
        self._vision = vision
        self._cost_restored = False
        self._skill_store = SkillStateStore(paths.skill_state_file)

        loader = getattr(session.config, "skill_loader", None)
        if loader is not None:
            discovered = loader.discover()
            snapshot = self._skill_store.load_or_create(
                session.thread_id,
                discovered,
            )
            session.configure_skills(
                snapshot.skills,
                disabled_skill_ids=snapshot.disabled_skill_ids,
            )

        permission_store = session.config.permission_store
        self._plans = PlanCapture(
            PlanStore(paths.plans_dir),
            thread_id=lambda: self.thread_id,
            in_plan_mode=lambda: self._session.mode == "plan",
        )
        self._rollback = RollbackService(
            project_root=paths.project_root,
            repository=repository,
            memory=session.config.memory_store,
            mutations=(
                WorkspaceMutations.attach(
                    session.config.hook_runner, paths.project_root, repository,
                )
                if getattr(session.config, "hook_runner", None) is not None
                else None
            ),
        )
        self._turns = TurnRunner(
            session,
            repository,
            permission_store=permission_store,
            rollback=self._rollback,
            plans=self._plans,
        )
        session.on_plan_turn = self._plans.on_plan_turn
        session.on_interrupt = self._plans.on_interrupt

    @classmethod
    def from_sdk_session(
        cls,
        session: Session,
        *,
        paths: NessPaths,
        runtime_config: RuntimeConfig,
        repository: SessionRepository,
        model_loader: ModelLoader,
        vision: bool | None,
    ) -> CodingSession:
        return cls(
            session,
            paths=paths,
            runtime_config=runtime_config,
            repository=repository,
            model_loader=model_loader,
            vision=vision,
        )

    @property
    def thread_id(self) -> str:
        return self._session.thread_id

    @property
    def mode(self) -> str:
        return self._session.mode

    @property
    def runtime_config(self) -> RuntimeConfig:
        return self._runtime_config

    @property
    def project_root(self) -> Path:
        return self._paths.project_root

    async def stream(
        self,
        request: TurnRequest | str,
    ) -> AsyncIterator[SessionEvent]:
        if self._closed:
            raise RuntimeError("session is closed")
        turn = request if isinstance(request, TurnRequest) else TurnRequest(request)
        async for event in self._turns.stream(turn):
            yield event

    def run_turn(
        self,
        message: str,
        *,
        images: list[str] | None = None,
        requested_skills: list[str] | None = None,
        mode: str | None = None,
    ) -> AsyncIterator[SessionEvent]:
        return self.stream(
            TurnRequest(
                message=message,
                images=tuple(images or ()),
                requested_skills=tuple(requested_skills or ()),
                mode=mode,
            )
        )

    def install_interaction_handlers(
        self,
        *,
        approval_handler: Any,
        question_handler: Any,
    ) -> None:
        """Bind handlers to this thread without changing sibling sessions."""
        config = self._session.config
        if config.approval_handler is None:
            config.approval_handler = approval_handler
        if config.question_handler is None:
            config.question_handler = question_handler

    async def resume(self) -> bool:
        """Bootstrap this thread from its durable event log."""
        return await self._replay(replay_cost=True)

    async def _replay(self, replay_cost: bool, before_seq: int | None = None) -> bool:
        if self._closed:
            raise RuntimeError("session is closed")
        if before_seq is None:
            rows = self._repository.raw_events(self.thread_id)
        else:
            # Prepare the retained conversation before deleting durable events.
            rows = [
                event.as_dict()
                for event in self._repository.events_since(self.thread_id, 0)
                if event.seq < before_seq
            ]
        if not rows and not self._repository.exists(self.thread_id):
            return False

        messages = events_to_messages(
            rows,
            subagents=self._repository.subagents(self.thread_id),
            vision=self._vision,
            permission_store=self._session.config.permission_store,
            yolo_mode=bool(getattr(
                getattr(self._session.config, "options", None), "yolo_mode", False,
            )),
        )
        if replay_cost and not self._cost_restored:
            restore_cost(rows, self._session.cost_tracker)
            self._cost_restored = True
        self._session.reset_checkpointer()
        self._session.bootstrap(messages)
        refresh_skill_catalog = getattr(
            self._session, "refresh_skill_catalog", None
        )
        if callable(refresh_skill_catalog):
            refresh_skill_catalog()
        await self._session.refresh_context_snapshot()
        return True

    async def rollback_to(self, user_seq: int) -> RollbackResult:
        return await self._rollback.rollback(
            self.thread_id,
            user_seq,
            replay=self._replay,
        )

    def user_turns(self) -> tuple[UserTurn, ...]:
        return self._repository.user_turns(self.thread_id)

    def history(self) -> tuple[DurableEvent, ...]:
        """Return typed durable events for presentation replay."""
        return tuple(self._repository.events(self.thread_id))

    def list_threads(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self._repository.list_threads(limit)
        for row in rows:
            row["label"] = (
                row.get("name")
                or row.get("summary")
                or self._repository.first_user_message(str(row.get("thread_id") or ""))
                or "(no messages)"
            )
        return rows

    def rename(self, name: str) -> bool:
        return self._repository.rename(self.thread_id, name)

    def set_autosave(self, enabled: bool) -> None:
        """Keep this session's persistence policy and reported setting in sync."""
        value = bool(enabled)
        self._session.config.options.auto_save_threads = value
        self._repository.auto_save = value
        self._runtime_config = replace(
            self._runtime_config,
            settings=replace(self._runtime_config.settings, auto_save_threads=value),
        )

    def apply_runtime_config(self, config: RuntimeConfig) -> None:
        """Apply behavior settings and autosave to this session's persistence view."""
        options = self._session.config.options
        values = sdk_behavior_values(config.settings, yolo_mode=options.yolo_mode)
        # Run the SDK constructor's validation before touching live state. Keep
        # the options object because SDK tools may already hold references to it.
        replace(options, **values)
        for name, value in values.items():
            setattr(options, name, value)
        self._runtime_config = config
        self.set_autosave(options.auto_save_threads)

    def usage_report(self) -> str:
        """Return the full SDK usage report for the exit summary."""
        return self._session.cost_tracker.report()

    def usage_summary(self) -> dict[str, Any]:
        tracker = self._session.cost_tracker
        return {
            "turns": int(self._session.turn_count or 0),
            "input_tokens": int(tracker.input_tokens or 0),
            "cached_input_tokens": int(tracker.cached_input_tokens or 0),
            "cache_write_input_tokens": int(
                getattr(tracker, "cache_write_input_tokens", 0) or 0
            ),
            "output_tokens": int(tracker.output_tokens or 0),
            "cost_usd": float(tracker.cost_usd or 0.0),
        }

    def context_summary(self) -> dict[str, int]:
        """Return the latest context-pressure snapshot maintained by the SDK."""

        return {
            "used": int(getattr(self._session, "context_used", 0) or 0),
            "total": int(getattr(self._session, "context_total", 0) or 0),
        }

    async def refresh_context_snapshot(self) -> dict[str, Any]:
        return dict(await self._session.refresh_context_snapshot())

    async def save(self) -> SaveResult:
        """Archive the durable thread without running session-end work."""
        exists = self._repository.exists(self.thread_id)
        message = (
            self._repository.archive(self.thread_id)
            if exists
            else f"No thread to archive: {self.thread_id}"
        )
        resume_thread_id = (
            self.thread_id if exists and self._repository.auto_save else None
        )
        return SaveResult(
            thread_id=self.thread_id,
            resume_thread_id=resume_thread_id,
            message=message,
        )

    async def finalize_and_save(self) -> SaveResult:
        """Run session-end reflection once per turn count, then archive."""
        if self._reflected_turn_count != self._session.turn_count:
            await self._session.finalize_reflection()
            self._reflected_turn_count = self._session.turn_count
        return await self.save()

    async def reload_model(self) -> None:
        models = self._model_loader(self.thread_id)
        self._session.configure_models(
            model=models.model,
            reflection_model=models.reflection_model,
            context_window=models.context_window,
            vision=models.vision,
        )
        self._runtime_config = replace(
            models.runtime_config, settings=self._runtime_config.settings
        )
        self._vision = models.vision

    def set_mode(self, mode: str) -> None:
        self._session.set_mode(mode)

    def toggle_mode(self) -> str:
        return self._session.toggle_mode()

    def cancel(self) -> None:
        self._session.cancel()

    def is_cancelled(self) -> bool:
        return self._session.is_cancelled()

    def request_compact(self) -> None:
        self._repository.append(
            self.thread_id,
            {"kind": "compact", "content": "manual compaction requested"},
        )
        self._session.request_compact()

    def activate_mcp_tools(
        self,
        names: list[str],
    ) -> tuple[list[str], list[str]]:
        """Activate deferred MCP tools for this session only."""
        return self._session.config.tool_registry.activate_mcp(names)

    def active_tool_names(self) -> list[str]:
        return self._session.config.tool_registry.tool_names()

    def skills(self) -> tuple[tuple[dict[str, Any], ...], tuple[str, ...]]:
        loader = self._session.config.skill_loader
        records = []
        disabled = loader.disabled_skill_ids
        for skill in loader.all_skills():
            item = dict(skill)
            item["available"] = loader.skill_id(item) not in disabled
            records.append(item)
        return tuple(records), tuple(loader.errors)

    def update_skill_access(self, disabled_skill_ids: frozenset[str]) -> None:
        previous = self._session.config.skill_loader.disabled_skill_ids
        disabled = self._skill_store.apply(self.thread_id, disabled_skill_ids)
        if disabled != previous:
            self._session.set_skill_access(disabled)

    def project_memory(self) -> tuple[Path, str]:
        memory = self._session.config.memory_store
        return Path(memory.ness_file), memory.load_project()

    def append_project_memory(self, text: str) -> str:
        return self._session.config.memory_store.append_project(text)

    async def create_project_memory(
        self,
        prompt: str,
        *,
        overwrite: bool,
    ) -> str:
        response = await self._session.config.model.ainvoke(
            [HumanMessage(content=prompt)]
        )
        return self._session.config.memory_store.write_project(
            str(response.content),
            overwrite=overwrite,
        )

    def user_memory(self) -> tuple[Path, str]:
        memory = self._session.config.memory_store
        return Path(memory.user_file), memory.load_user()

    def append_user_memory(self, text: str) -> str:
        return self._session.config.memory_store.append_user(text)

    def permission_rules(self) -> str:
        return self._session.config.permission_store.list_rules()

    def add_permission_rule(self, pattern: str, decision: str) -> None:
        self._session.config.permission_store.persist_rule(pattern, decision)

    def remove_permission_rule(self, decision: str, index: int) -> str:
        return self._session.config.permission_store.remove_rule(decision, index)

    def hooks_description(self) -> str:
        return self._session.config.hook_runner.describe()

    async def fork_before(self, user_seq: int) -> ForkResult:
        selected = next(
            (turn for turn in self.user_turns() if turn.seq == user_seq),
            None,
        )
        if selected is None:
            raise ValueError(f"Fork target seq {user_seq} is not a user message")
        checkpoint = self._repository.checkpoint(self.thread_id, user_seq)
        if checkpoint is None:
            raise ValueError(f"No checkpoint for seq {user_seq} in this thread")
        target = f"session-{uuid.uuid4().hex[:8]}"
        events = self._repository.copy_prefix(self.thread_id, target, user_seq)
        self._skill_store.copy_thread(self.thread_id, target)
        await asyncio.to_thread(
            self._session.config.memory_store.write_session_raw,
            target,
            checkpoint.memory_snapshot,
        )
        return ForkResult(target, selected.content, len(events))

    async def export_html(self, destination: Path):
        return await asyncio.to_thread(
            self._repository.export_html,
            self.thread_id,
            project_root=self.project_root,
            destination=destination,
        )

    async def run_goal(
        self,
        goal: str,
        *,
        worker_turn: WorkerTurn,
        on_status: StatusCallback,
        judge_model: BaseChatModel,
        max_attempts: int,
    ) -> GoalResult:
        from ness_cli.session.goal import GoalCoordinator

        coordinator = GoalCoordinator(
            repository=self._repository,
            thread_id=self.thread_id,
            hook_runner=self._session.config.hook_runner,
            judge_model=judge_model,
            max_attempts=max_attempts,
            instructions_dir=self._paths.instructions_dir,
            is_cancelled=self.is_cancelled,
        )
        return await coordinator.run(
            goal,
            worker_turn=worker_turn,
            on_status=on_status,
        )

    async def run_reflection(self) -> ReflectionResult:
        return await self._session.run_reflection()

    async def get_state(self) -> dict[str, Any]:
        return await self._session.get_state()

    async def get_messages(self) -> list[Any]:
        return await self._session.get_messages()

    async def get_todos(self) -> list[dict[str, Any]]:
        return await self._session.get_todos()

    async def close(self) -> None:
        if self._closed:
            return
        save_error = None
        try:
            await self.finalize_and_save()
        except BaseException as exc:
            save_error = exc
            raise
        finally:
            try:
                await self._session.close()
            except BaseException as exc:
                if save_error is None:
                    raise
                save_error.add_note(f"Shell runtime cleanup also failed: {exc}")
            finally:
                self._closed = True
