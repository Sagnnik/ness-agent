"""Shared fixtures for the Ness Agent TUI test suite.

The TUI is wired directly to a ``ness_cli.CodingSession``: TuiApp
owns the TUI-side session state (prompt queue, exit flag, staged skills,
assistant history) and consumes the coding session's SessionEvent stream.
The fakes below mirror the ``CodingSession`` surface the TUI and slash
commands touch, without pulling in the LangGraph app or model factory.
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import pytest

from ness_agent.types import SessionEvent
from ness_agent.reflection import ReflectionResult

from ness_cli.config import settings
from ness_cli.tui.app import TuiApp


@pytest.fixture(autouse=True)
def _isolate_model_provider():
    """TUI tests must not inherit the developer's persisted provider login."""
    previous = settings.model_provider
    settings.model_provider = "openrouter"
    try:
        yield
    finally:
        settings.model_provider = previous


class _FakeThreadStore:
    def __init__(self) -> None:
        self.auto_save = True
        self.events: dict[str, list[dict]] = {}
        self.archived: list[str] = []
        self.names: dict[str, str] = {}

    def load_thread_events(self, thread_id: str) -> list[dict]:
        return list(self.events.get(thread_id, []))

    def list_threads(self, n: int = 10) -> list[dict]:
        rows = []
        for thread_id, events in self.events.items():
            rows.append(
                {
                    "thread_id": thread_id,
                    "started_at": "2026-01-01T00:00:00+00:00",
                    "model": "test-model",
                    "name": self.names.get(thread_id, ""),
                    "summary": "",
                }
            )
        return rows[:n]

    def list_subagents(self, thread_id: str) -> list[dict]:
        return []

    def first_user_message(self, thread_id: str) -> str | None:
        return None

    def thread_exists(self, thread_id: str) -> bool:
        return thread_id in self.events or thread_id in self.names

    def set_thread_name(self, thread_id: str, name: str) -> bool:
        self.names[thread_id] = " ".join(name.split())
        return True

    def list_user_turns(self, thread_id: str) -> list[dict]:
        return []

    def archive_thread(self, thread_id: str) -> str:
        self.archived.append(thread_id)
        return f"Archived thread {thread_id}."


class _FakeMemoryStore:
    def __init__(self) -> None:
        self.ness_file = Path(".ness") / "NESS.md"
        self.user_file = Path(".ness") / "USER.md"
        self._session_raw: dict[str, str] = {}

    @property
    def disabled(self) -> bool:
        return False

    def load_project(self) -> str:
        return ""

    def load_user(self) -> str:
        return ""

    def load_session(self, thread_id: str) -> str:
        return ""

    def append_project(self, text: str) -> str:
        return "Updated .ness/NESS.md"

    def append_user(self, text: str) -> str:
        return "Updated USER.md"

    def append_session_bullets(self, thread_id: str, bullets: list[str]) -> bool:
        return False

    def write_project(self, text: str, overwrite: bool = False) -> str:
        return "Wrote .ness/NESS.md"

    def write_user(self, text: str, overwrite: bool = False) -> str:
        return "Wrote USER.md"

    def read_session_raw(self, thread_id: str) -> str:
        return self._session_raw.get(thread_id, "")

    def write_session_raw(self, thread_id: str, text: str) -> None:
        if text:
            self._session_raw[thread_id] = text
        else:
            self._session_raw.pop(thread_id, None)

    def check_health(self) -> str | None:
        return None


class _FakePerms:
    def list_rules(self) -> str:
        return "(no rules)"

    def persist_rule(self, rule: str, bucket: str, scope: str = "always") -> None:
        return None

    def remove_rule(self, bucket: str, index: int) -> str:
        return f"rule #{index}"

    def clear_session_rules(self) -> None:
        return None


class _FakeHookRunner:
    def describe(self) -> str:
        return "(no hooks)"


class _FakeSkillLoader:
    def __init__(self) -> None:
        self.errors: list[str] = []

    def load(self) -> dict:
        return {}


class _FakeToolRegistry:
    def activate_mcp(self, names) -> tuple[list[str], list[str]]:
        return (list(names), [])

    def tool_names(self) -> list[str]:
        return []


class _FakeCostTracker:
    """Flat-attribute stand-in for the SDK CostTracker scalar surface."""

    def __init__(self) -> None:
        self.input_tokens = 0
        self.uncached_input_tokens = 0
        self.cached_input_tokens = 0
        self.cache_write_input_tokens = 0
        self.output_tokens = 0
        self.cost_usd = 0.0
        self.calls = 0

    def report(self) -> str:
        return "Calls: 0"


class FakeCoding:
    """Stand-in for ``ness_cli.CodingSession``.

    Mirrors the surface TuiApp and the slash commands use: backend stores,
    mode/context properties, the run_turn event stream (scripted via
    :meth:`queue_events`), and thread management. ``run_turn`` yields an
    ``assistant_final`` echo by default so tests can drive a turn without
    scripting events.
    """

    def __init__(self) -> None:
        self.thread_id = f"session-{uuid.uuid4().hex[:8]}"
        self.mode = "act"
        self.turn_count = 0
        self.context_used = 12_400
        self.context_total = 128_000
        self.ness_dir = Path(".ness")
        self.project_root = Path.cwd()
        self.thread_store = _FakeThreadStore()
        self.cost_tracker = _FakeCostTracker()
        self.permission_store = _FakePerms()
        self.memory_store = _FakeMemoryStore()
        self.hook_runner = _FakeHookRunner()
        self.skill_loader = _FakeSkillLoader()
        self.tool_registry = _FakeToolRegistry()
        self.model_name = "test-model"
        self.provider_id = "openrouter"
        self.reasoning_effort = "medium"
        options = SimpleNamespace(
            enable_approval=True,
            yolo_mode=False,
            auto_save_threads=True,
            session_end_reflection=False,
        )
        self.cfg = SimpleNamespace(model=SimpleNamespace(), options=options)
        self.agent = SimpleNamespace(
            config=SimpleNamespace(
                model=SimpleNamespace(),
                options=SimpleNamespace(**vars(options)),
            )
        )
        self._pending_skills: list[str] = []

        self.cancelled = False
        self.resumed: list[str] = []
        self.reset_ids: list[str] = []
        self.rolled_back_seq: int | None = None
        self.compact_requested = False
        self.reflection_runs = 0
        self.reflection_result = ReflectionResult(
            memory_updated=True,
            bullets=("Captured the latest work",),
            message_index=2,
        )
        self.saved = False
        self.reloaded = False
        self._events: list[SessionEvent] = []
        self._todos: list[dict] = []

    # --- scripting ---------------------------------------------------------
    def queue_events(self, *events: SessionEvent) -> None:
        """Script the events the next ``run_turn`` will yield (replacing the
        default echo for that turn)."""
        self._events.extend(events)

    def set_todos(self, todos: list[dict]) -> None:
        self._todos = [dict(todo) for todo in todos]

    # --- the turn -----------------------------------------------------------
    async def run_turn(
        self,
        text: str,
        *,
        images: list[str] | None = None,
        active_skills: list[str] | None = None,
        mode: str | None = None,
    ):
        self.turn_count += 1
        events, self._events = self._events, []
        if not events:
            events = [SessionEvent("assistant_final", {"content": f"echo {text}"})]
        for ev in events:
            yield ev

    # --- control -------------------------------------------------------------
    def cancel(self) -> None:
        self.cancelled = True

    def is_cancelled(self) -> bool:
        return self.cancelled

    def toggle_mode(self) -> str:
        self.mode = "plan" if self.mode == "act" else "act"
        return self.mode

    def request_compact(self) -> None:
        self.compact_requested = True

    def set_name(self, name: str) -> bool:
        return self.thread_store.set_thread_name(self.thread_id, name)

    def active_skills(self, names: list[str]) -> None:
        self._pending_skills = list(names)

    def stage_skills(self, names) -> None:
        pending = list(self._pending_skills)
        seen = set(pending)
        for name in names:
            n = str(name).strip()
            if n and n not in seen:
                pending.append(n)
                seen.add(n)
        self._pending_skills = pending

    def save_thread(self) -> str:
        self.saved = True
        return self.thread_store.archive_thread(self.thread_id)

    def reload_model(self) -> None:
        self.reloaded = True

    async def finalize_reflection(self) -> None:
        return None

    async def run_reflection(self) -> ReflectionResult:
        self.reflection_runs += 1
        return self.reflection_result

    async def refresh_context_snapshot(self) -> dict:
        return {}

    async def get_todos(self) -> list[dict]:
        return [dict(todo) for todo in self._todos]

    # --- thread management ----------------------------------------------------
    async def resume(self, thread_id: str, *, replay_cost: bool = True) -> bool:
        self.resumed.append(thread_id)
        self.thread_id = thread_id
        return True

    async def clone_for_thread(self, thread_id: str) -> "FakeCoding":
        clone = self.new_for_thread(thread_id)
        await clone.resume(thread_id)
        return clone

    def new_for_thread(self, thread_id: str) -> "FakeCoding":
        clone = type(self)()
        for name in (
            "ness_dir",
            "project_root",
            "thread_store",
            "cost_tracker",
            "permission_store",
            "memory_store",
            "hook_runner",
            "skill_loader",
            "tool_registry",
            "cfg",
            "agent",
        ):
            setattr(clone, name, getattr(self, name))
        clone.model_name = self.model_name
        clone.provider_id = self.provider_id
        clone.reasoning_effort = self.reasoning_effort
        clone.mode = self.mode
        clone.thread_id = thread_id
        return clone

    async def reset(self, thread_id: str) -> None:
        self.reset_ids.append(thread_id)
        self.thread_id = thread_id
        self.turn_count = 0

    async def rollback_to(self, user_seq: int) -> str:
        self.rolled_back_seq = user_seq
        return f"Rolled back to turn @ seq {user_seq}."


Dispatcher = Callable[[TuiApp, str], Awaitable[None]]


@pytest.fixture
def make_app() -> Callable[..., TuiApp]:
    """Factory fixture: build a fresh TuiApp backed by a FakeCoding.

    Each call returns a TuiApp with its own TemporaryDirectory history path;
    the tempdir's lifetime is tied to the closure (and thus to the test), so
    no explicit cleanup is required.
    """

    def _factory(command_dispatcher: Dispatcher | None = None) -> TuiApp:
        tmp = TemporaryDirectory()
        coding = FakeCoding()
        kwargs: dict = {"history_path": Path(tmp.name) / "hist"}
        if command_dispatcher is not None:
            kwargs["command_dispatcher"] = command_dispatcher
        app = TuiApp(coding, **kwargs)  # type: ignore[arg-type]
        app._tmpdir = tmp
        return app

    return _factory
