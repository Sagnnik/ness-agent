"""Narrow command dependencies for the interactive CLI."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Awaitable, Callable
from typing import Protocol

from ness_cli.runtime import InteractiveRuntime
from ness_cli.session import CodingSession
from ness_cli.tui.controller import TuiController
from ness_cli.tui.input import ChecklistResult, MenuItem
from ness_cli.tui.sink import TuiRenderSink


class CommandUI(Protocol):
    async def choose(
        self,
        title: str,
        items: list[MenuItem],
        *,
        initial_key: str | None = None,
        hint: str = "Up/Down select · Enter confirm · Esc cancel",
        filterable: bool = False,
        refresh: Callable[[], Awaitable[list[MenuItem]]] | None = None,
        horizontal_keys: frozenset[str] = frozenset(),
    ) -> str | None: ...

    async def choose_checklist(
        self,
        title: str,
        items: list[MenuItem],
        *,
        disabled_keys: frozenset[str],
        initial_key: str | None = None,
    ) -> ChecklistResult | None: ...

    async def request_input(
        self,
        label: str,
        *,
        default: str = "",
        secret: bool = False,
    ) -> str | None: ...

    def request_exit(self) -> None: ...
    def clear_transcript(self) -> None: ...
    async def refresh_transcript(self) -> None: ...
    def prefill_input(self, text: str) -> None: ...


@dataclass(frozen=True, slots=True)
class CommandContext:
    runtime: InteractiveRuntime
    threads: TuiController
    ui: CommandUI

    @property
    def session(self) -> CodingSession:
        return self.threads.session

    @property
    def renderer(self) -> TuiRenderSink:
        return self.threads.sink
