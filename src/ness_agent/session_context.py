from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from pathlib import Path
import threading
from typing import TYPE_CHECKING, Any

from ness_agent.options import NessAgentOptions
from ness_agent.permissions import PermissionStore
from ness_agent.persistence import ThreadStore

if TYPE_CHECKING:
    from ness_agent.agent import NessAgentConfig
    from ness_agent.tools.shell_processes import ProcessManager


@dataclass
class SessionContext:
    permissions: PermissionStore
    options: NessAgentOptions
    thread_store: ThreadStore
    ness_dir: Path
    project_root: Path
    agent_config: NessAgentConfig | None = None
    available_skills: dict[str, Any] | None = None
    vision: bool | None = None
    shell_process_manager: ProcessManager | None = None
    shell_cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)
    shell_deadline: float | None = None
    _shell_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def get_shell_process_manager(self) -> ProcessManager:
        from ness_agent.tools.shell_processes import ProcessManager

        with self._shell_lock:
            if self.shell_process_manager is None:
                self.shell_process_manager = ProcessManager(
                    project_root=self.project_root,
                    runtime_root=self.ness_dir / "runtime" / "shells",
                )
            return self.shell_process_manager


_session_ctx: ContextVar[SessionContext | None] = ContextVar("ness_agent_session_context", default=None)


def set_session_context(ctx: SessionContext | None) -> Token:
    return _session_ctx.set(ctx)


def reset_session_context(token: Token) -> None:
    # remove a specific entry from the contextvar stack by its Token
    _session_ctx.reset(token)


def get_session_context() -> SessionContext:
    ctx = _session_ctx.get()
    if ctx is None:
        raise RuntimeError(
            "Session context is not configured. Create a Session (or call set_session_context) "
            "before invoking tools."
        )
    return ctx


def try_get_session_context() -> SessionContext | None:
    return _session_ctx.get()
