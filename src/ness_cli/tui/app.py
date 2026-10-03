"""Prompt-toolkit application for the Ness interactive terminal UI."""

from __future__ import annotations

import asyncio
import re
import shutil
import sys
from collections.abc import Awaitable, Callable
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path

from ness_agent.utils import preview_diff
from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.clipboard.pyperclip import PyperclipClipboard
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.history import FileHistory
from prompt_toolkit.layout.dimension import Dimension
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from ness_cli.runtime import InteractiveRuntime
from ness_cli.terminal import cell_width, clip_cells
from ness_cli.tui.commands import command_registry
from ness_cli.tui.context import CommandContext
from ness_cli.tui.controller import TuiController
from ness_cli.tui.input import ChecklistResult, FileIndex, MenuItem, MenuState
from ness_cli.tui.layout import build_key_bindings, build_layout
from ness_cli.tui.sink import TuiRenderSink
from ness_cli.tui.theme import GRAY, STYLE
from ness_cli.tui.tool_display import format_tool_args
from ness_cli.tui.transcript import TranscriptStore
from ness_cli.tui.turn_renderer import replay_event


_MENU_DESCRIPTION_COLUMN = 28
_ESCAPE_KEY_FLUSH_TIMEOUT = 0
_KEY_BINDING_TIMEOUT = 0.01


def _package_version() -> str:
    try:
        return package_version("ness-agent")
    except PackageNotFoundError:
        return "dev"


class _ThreadPromptDriver:
    """Keep an interactive tool prompt attached to its source thread."""

    def __init__(self, app: "TuiApp", thread_id: str) -> None:
        self._app = app
        self._thread_id = thread_id

    async def request_input(
        self,
        label: str,
        *,
        default: str = "",
    ) -> str | None:
        return await self._app.request_thread_input(
            self._thread_id,
            label,
            default=default,
        )

    async def choose(
        self,
        title: str,
        items: list[MenuItem],
        *,
        initial_key: str | None = None,
        hint: str = "Up/Down select · Enter confirm · Esc cancel",
    ) -> str | None:
        return await self._app.choose_for_thread(
            self._thread_id,
            title,
            items,
            initial_key=initial_key,
            hint=hint,
        )

    async def approve(self, name: str, args: dict) -> str:
        return await self._app.request_thread_approval(
            self._thread_id,
            name,
            args,
        )

    async def choose_with_note(
        self,
        title: str,
        items: list[MenuItem],
        *,
        initial_key: str | None = None,
        allow_note: bool = True,
    ) -> tuple[str, str] | None:
        return await self._app.request_thread_question(
            self._thread_id,
            title,
            items,
            initial_key=initial_key,
            allow_note=allow_note,
        )


class TuiApp:
    """Connect layout, input, controller, and rendering services."""

    def __init__(self, runtime: InteractiveRuntime) -> None:
        self.runtime = runtime
        runtime.paths.cli_history.parent.mkdir(parents=True, exist_ok=True)
        self.input_buffer = Buffer(
            history=FileHistory(str(runtime.paths.cli_history)),
            multiline=True,
            on_text_changed=self._on_input_changed,
        )
        self.transcript_view = None
        self.transcript_window = None
        self.input_window = None
        self.input_top_rule = None
        self.input_bottom_rule = None
        self.status_window = None
        self.queue_window = None
        self.menu_window = None
        self.stats_window = None
        self.path_window = None
        self._application: Application[None] | None = None
        self._prompt_lock = asyncio.Lock()
        self._prompt_future: asyncio.Future[str | None] | None = None
        self._prompt_label = ""
        self._prompt_default = ""
        self._draft = ""
        self._secret_input = False
        self._turn_tasks: set[asyncio.Task[bool]] = set()
        self._command_task: asyncio.Task[bool] | None = None
        self._picker_lock = asyncio.Lock()
        self._menu: MenuState | None = None
        self._menu_future: asyncio.Future[
            str | tuple[str, str] | ChecklistResult | None
        ] | None = None
        self._picker_draft = ""
        self._picker_refresh_task: asyncio.Task[list[MenuItem]] | None = None
        self._status_task: asyncio.Task[None] | None = None
        self._approval_name = ""
        self._approval_args: dict = {}
        self._approval_scope_key = "yes"
        self._pending_paste: str | None = None
        self._collapsing_paste = False
        self._file_index = FileIndex(runtime.paths.project_root)
        self._pending_images: dict[int, str] = {}
        self._selected_skills: set[str] = set()
        self._image_counter = 0
        self.input_buffer.read_only = Condition(self._input_read_only)

        self.controller = TuiController(
            runtime,
            sink_factory=self._make_thread_sink,
            selected_changed=self._selected_sink_changed,
            changed=self.invalidate,
        )
        self.commands = command_registry()
        self.command_context = CommandContext(
            runtime=runtime,
            threads=self.controller,
            ui=self,
        )

    @property
    def sink(self) -> TuiRenderSink:
        return self.controller.sink

    @property
    def transcript(self) -> TranscriptStore:
        return self.sink.store

    async def run(self) -> None:
        await self.controller.initialize()
        self._application = Application(
            layout=build_layout(self),
            key_bindings=build_key_bindings(self),
            style=STYLE,
            full_screen=True,
            mouse_support=True,
            clipboard=PyperclipClipboard(),
            paste_mode=True,
            min_redraw_interval=0.03,
        )
        self._configure_escape_timeouts(self._application)
        try:
            self._status_task = asyncio.create_task(self._status_clock())
            await self._application.run_async()
        finally:
            self.cancel_prompt()
            if self._status_task is not None:
                self._status_task.cancel()
                await asyncio.gather(self._status_task, return_exceptions=True)
                self._status_task = None
            if self._picker_refresh_task is not None:
                self._picker_refresh_task.cancel()
                await asyncio.gather(
                    self._picker_refresh_task,
                    return_exceptions=True,
                )
                self._picker_refresh_task = None
            if self._command_task is not None and not self._command_task.done():
                self._command_task.cancel()
                await asyncio.gather(
                    self._command_task,
                    return_exceptions=True,
                )
            await self.controller.close()

    @staticmethod
    def _configure_escape_timeouts(application: Application) -> None:
        """Handle a standalone Escape key without prompt-toolkit's default delay."""
        application.ttimeoutlen = _ESCAPE_KEY_FLUSH_TIMEOUT
        application.timeoutlen = _KEY_BINDING_TIMEOUT

    def invalidate(self) -> None:
        if self._application is not None:
            self._application.invalidate()

    def on_transcript_resize(self, width: int, height: int) -> None:
        del height
        self.sink.reflow_assistants(width)
        self.invalidate()

    async def _status_clock(self) -> None:
        try:
            while True:
                await asyncio.sleep(0.1)
                if self.controller.busy:
                    self.invalidate()
        except asyncio.CancelledError:
            return

    @property
    def transcript_focused(self) -> bool:
        return bool(
            self._application is not None
            and self.transcript_view is not None
            and self._application.layout.has_focus(self.transcript_view)
        )

    @property
    def transcript_has_selection(self) -> bool:
        return bool(
            self.transcript_view is not None
            and self.transcript_view.has_selection()
        )

    def focus_transcript(self) -> None:
        if self._application is not None and self.transcript_view is not None:
            self._application.layout.focus(self.transcript_view)
            self.invalidate()

    def refocus_input(self) -> None:
        if self.transcript_view is not None:
            self.transcript_view.clear_selection()
        if self._application is not None and self.input_window is not None:
            self._application.layout.focus(self.input_window)
            self.invalidate()

    def scroll_transcript(self, amount: int) -> None:
        if self.transcript_view is not None:
            self.transcript_view.scroll_by(amount)

    def scroll_transcript_page(self, direction: int) -> None:
        if self.transcript_view is None:
            return
        distance = max(1, self.transcript_view.viewport_rows // 2)
        self.transcript_view.scroll_by(direction * distance)

    def scroll_transcript_to_top(self) -> None:
        if self.transcript_view is not None:
            self.transcript_view.scroll_to_top()

    def follow_transcript_end(self) -> None:
        if self.transcript_view is not None:
            self.transcript_view.follow_end()

    def copy_transcript_selection(self) -> None:
        if self.transcript_view is None or self._application is None:
            return
        selected = self.transcript_view.selected_text()
        if not selected:
            return
        try:
            self._application.clipboard.set_text(selected)
        except Exception as error:
            self.sink.warning(f"Clipboard unavailable: {error}")
            return
        self.refocus_input()

    def insert_from_transcript(self, text: str) -> None:
        self.refocus_input()
        self.input_buffer.insert_text(text)

    def backspace_from_transcript(self) -> None:
        self.refocus_input()
        if self.input_buffer.text:
            self.input_buffer.delete_before_cursor()

    def move_input_cursor_vertical(self, row_delta: int) -> bool:
        """Move through rendered input rows, including rows created by wrapping."""
        if row_delta == 0:
            return True

        buffer = self.input_buffer
        width, _ = self._terminal_size()
        prefix_width = sum(
            cell_width(fragment) for _, fragment in self.prompt_fragments()
        )
        positions = self._input_cursor_positions(
            buffer.text,
            width=max(1, width),
            prefix_width=prefix_width,
        )
        current_row, current_column = positions[buffer.cursor_position]
        target_row = current_row + row_delta
        if target_row < 0:
            return False

        preferred_column = buffer.preferred_column
        if preferred_column is None:
            preferred_column = current_column
        candidates = [
            (index, column)
            for index, (row, column) in enumerate(positions)
            if row == target_row
        ]
        if not candidates:
            return False

        target, _ = min(
            candidates,
            key=lambda candidate: (
                abs(candidate[1] - preferred_column),
                candidate[1] > preferred_column,
            ),
        )
        buffer.cursor_position = target
        buffer.preferred_column = preferred_column
        return True

    @staticmethod
    def _input_cursor_positions(
        text: str,
        *,
        width: int,
        prefix_width: int,
    ) -> list[tuple[int, int]]:
        """Map every buffer cursor offset to its rendered row and cell column."""
        width = max(1, width)
        row, column = divmod(max(0, prefix_width), width)
        positions: list[tuple[int, int]] = []

        for index in range(len(text) + 1):
            character = text[index] if index < len(text) else "\n"
            if character == "\n":
                character_width = 1
            else:
                character_width = cell_width(character, start_column=column, tab_size=8)

            if column + character_width > width:
                row += 1
                column = 0
                if character == "\t":
                    character_width = cell_width(character, tab_size=8)

            positions.append((row, column))
            if index == len(text):
                break
            if character == "\n":
                row += 1
                column = 0
            else:
                column += character_width

        return positions

    def prompt_fragments(self) -> StyleAndTextTuples:
        if self._prompt_future is not None:
            suffix = f" [{self._prompt_default}]" if self._prompt_default else ""
            return [
                ("class:prompt.request", f"{self._prompt_label}{suffix} "),
                ("class:prompt", "> "),
            ]
        if self._menu is not None and self._menu.kind not in {
            "slash",
            "mention",
            "skill",
        }:
            if self._menu.kind == "question" and self._menu.note_active:
                label = "note"
            else:
                label = "filter" if self._menu.filterable else "select"
            return [
                ("class:prompt.request", f"{label} "),
                ("class:prompt", "> "),
            ]
        mode = self.controller.mode
        style = "class:prompt.mode.plan" if mode == "plan" else "class:prompt.mode"
        return [(style, f"{mode} "), ("class:prompt", "> ")]

    @property
    def secret_input(self) -> bool:
        return self._secret_input

    @property
    def menu_open(self) -> bool:
        return self._menu is not None

    @property
    def menu_horizontal_enabled(self) -> bool:
        return bool(self._menu is not None and self._menu.horizontal_enabled)

    @property
    def reasoning_toggle_available(self) -> bool:
        return self._menu is None and self._prompt_future is None

    def _input_read_only(self) -> bool:
        return bool(
            self._menu is not None
            and self._menu.kind not in {"slash", "mention", "skill"}
            and not self._menu.filterable
            and not (
                self._menu.kind == "question" and self._menu.note_active
            )
        )

    def _terminal_size(self) -> tuple[int, int]:
        if self._application is not None:
            try:
                size = self._application.output.get_size()
                if size.columns > 0 and size.rows > 0:
                    return size.columns, size.rows
            except Exception:
                pass
        size = shutil.get_terminal_size(fallback=(100, 24))
        return max(20, size.columns), max(12, size.lines)

    def _menu_limits(self) -> tuple[int, int, int]:
        menu = self._menu
        if menu is None:
            return 0, 0, 0
        _, terminal_rows = self._terminal_size()
        input_rows = self._input_row_count()
        # Two rules, two status rows, the input, and at least five transcript
        # rows stay outside the menu. Detail yields before the option list.
        special_rows = 1 if (
            menu.kind == "approval"
            or (menu.kind == "question" and menu.note_allowed)
        ) else 0
        minimum = 3 + special_rows
        available = max(minimum, terminal_rows - input_rows - 10)
        summary_rows = min(3, len(menu.summary_lines))
        menu_chrome_rows = 0 if menu.kind == "slash" else 2
        fixed = menu_chrome_rows + special_rows + summary_rows
        option_rows = min(12, max(1, len(menu.items)))
        detail_rows = 0
        if menu.detail_lines:
            detail_rows = min(8, len(menu.detail_lines))
            fixed += 1
        while fixed + option_rows + detail_rows > available and detail_rows > 2:
            detail_rows -= 1
        while fixed + option_rows + detail_rows > available and option_rows > 3:
            option_rows -= 1
        while fixed + option_rows + detail_rows > available and summary_rows > 0:
            summary_rows -= 1
            fixed -= 1
        while fixed + option_rows + detail_rows > available and detail_rows > 0:
            detail_rows -= 1
        if not detail_rows and menu.detail_lines:
            fixed -= 1
        while fixed + option_rows > available and option_rows > 1:
            option_rows -= 1
        return option_rows, summary_rows, detail_rows

    def menu_height(self) -> Dimension:
        if not self.menu_open:
            return Dimension.exact(0)
        assert self._menu is not None
        option_rows, summary_rows, detail_rows = self._menu_limits()
        empty_row = 1 if not self._menu.items else 0
        detail_header = 1 if detail_rows else 0
        approval_scope = 1 if self._menu.kind == "approval" else 0
        question_note = 1 if (
            self._menu.kind == "question" and self._menu.note_allowed
        ) else 0
        menu_chrome_rows = 0 if self._menu.kind == "slash" else 2
        return Dimension.exact(
            menu_chrome_rows
            + min(option_rows, len(self._menu.items))
            + empty_row
            + summary_rows
            + detail_header
            + detail_rows
            + approval_scope
            + question_note
        )

    def menu_fragments(self) -> StyleAndTextTuples:
        menu = self._menu
        if menu is None:
            return []
        option_rows, summary_rows, detail_rows = self._menu_limits()
        start, items = menu.visible(option_rows)
        position = ""
        if menu.items:
            before = "↑ " if start else ""
            after = " ↓" if start + len(items) < len(menu.items) else ""
            position = (
                f"  {before}{start + 1}-{start + len(items)}/{len(menu.items)}{after}"
            )
        query = f"  filter: {menu.query}" if menu.filterable and menu.query else ""
        fragments: StyleAndTextTuples = []
        if menu.kind != "slash":
            width, _ = self._terminal_size()
            hint = menu.hint
            for key, symbol in (
                ("Up/Down", "↑/↓"),
                ("Left/Right", "←/→"),
                ("PgUp/PgDn", "⇞/⇟"),
                ("Enter", "↵"),
                ("Tab", "⇥"),
                ("Esc", "⎋"),
            ):
                hint = hint.replace(key, symbol)
            title = f" {menu.title}{position}{query}"
            if cell_width(hint) > width - cell_width(title) - 2:
                hint = "  ".join(re.findall(r"[↑↓←→⇞⇟↵⇥⎋/]+", hint))
            hint = clip_cells(hint, max(0, width - 16))
            title = clip_cells(
                title,
                max(0, width - cell_width(hint) - 2),
            )
            gap = max(1, width - cell_width(title) - cell_width(hint) - 1)
            fragments.append(("class:chrome.menu.title", title))
            fragments.append(("class:chrome.menu.hint", " " * gap))
            for part in re.split(r"([↑↓←→⇞⇟↵⇥⎋/]+)", hint):
                style = (
                    "class:chrome.menu.hint.key"
                    if re.fullmatch(r"[↑↓←→⇞⇟↵⇥⎋/]+", part)
                    else "class:chrome.menu.hint"
                )
                fragments.append((style, part))
            fragments.append(
                ("class:chrome.menu.hint", "\n")
            )
        for line in menu.summary_lines[:summary_rows]:
            fragments.append(("class:chrome.menu.summary", f" {line}\n"))
        if menu.kind == "approval":
            scope = next(
                (
                    item.label
                    for item in menu.items
                    if item.key == self._approval_scope_key
                ),
                "Deny once",
            )
            fragments.append(("class:chrome.menu.scope", f" decision  {scope}\n"))
        elif menu.kind == "question" and menu.note_allowed:
            note = menu.note_text.strip() or "optional"
            state = "editing" if menu.note_active else "saved"
            fragments.append(
                (
                    "class:chrome.menu.note",
                    f" note ({state})  {note}\n",
                )
            )
        for offset, item in enumerate(items):
            selected = start + offset == menu.index
            if menu.kind == "slash":
                fragments.extend(self._slash_menu_row(item, selected=selected))
                fragments.append(("class:chrome.menu.row", "\n"))
                continue
            if menu.kind == "checklist":
                fragments.extend(
                    self._checklist_menu_row(
                        item,
                        selected=selected,
                        disabled=item.key in menu.disabled_keys,
                    )
                )
                fragments.append(("class:chrome.menu.row", "\n"))
                continue
            row_style = (
                "class:chrome.menu.row.current" if selected else "class:chrome.menu.row"
            )
            active_scope = (
                menu.kind == "approval" and item.key == self._approval_scope_key
            )
            marker = "›" if selected else ("•" if active_scope else " ")
            fragments.append((row_style, f" {marker} {item.label}"))
            if item.description:
                fragments.append(
                    (
                        f"{row_style} class:chrome.menu.description",
                        f"  {item.description}",
                    )
                )
            if item.suffix:
                fragments.append(
                    (
                        f"{row_style} class:chrome.menu.suffix",
                        f"  {item.suffix}",
                    )
                )
            fragments.append((row_style, "\n"))
        if not menu.items:
            fragments.append(("class:chrome.menu.empty", "   no matches\n"))
        if menu.detail_lines and detail_rows:
            detail_start, detail = menu.visible_detail(detail_rows)
            detail_end = detail_start + len(detail)
            fragments.append(
                (
                    "class:chrome.menu.detail.title",
                    f" {menu.detail_label}  {detail_start + 1}-{detail_end}/{len(menu.detail_lines)}\n",
                )
            )
            for line in detail:
                fragments.append((self._detail_style(line), f" {line}\n"))
        if menu.kind != "slash":
            fragments.append(("class:chrome.menu.hint", " "))
        return fragments

    def _checklist_menu_row(
        self,
        item: MenuItem,
        *,
        selected: bool,
        disabled: bool,
    ) -> StyleAndTextTuples:
        width, _ = self._terminal_size()
        row_style = (
            "class:chrome.menu.row.current" if selected else "class:chrome.menu.row"
        )
        marker = "›" if selected else " "
        prefix = f" {marker} ["
        checkbox = "X" if disabled else " "
        suffix = "] "
        available = max(0, width - cell_width(prefix + checkbox + suffix))
        label = clip_cells(item.label, available, ellipsis="...")
        remaining = max(0, available - cell_width(label))
        description = ""
        if item.description and remaining > 2:
            description = "  " + clip_cells(
                " ".join(item.description.split()),
                remaining - 2,
                ellipsis="...",
            )

        checkbox_style = (
            "class:chrome.menu.checkbox.checked"
            if disabled
            else "class:chrome.menu.checkbox"
        )
        fragments: StyleAndTextTuples = [
            (row_style, prefix),
            (f"{row_style} {checkbox_style}", checkbox),
            (f"{row_style} class:chrome.menu.checkbox", suffix),
            (f"{row_style} class:chrome.menu.label.current" if selected else row_style, label),
        ]
        if description:
            fragments.append(
                (f"{row_style} class:chrome.menu.description", description)
            )
        return fragments

    def _slash_menu_row(
        self,
        item: MenuItem,
        *,
        selected: bool,
    ) -> StyleAndTextTuples:
        """Render slash completion like the legacy full-width command menu."""
        width, _ = self._terminal_size()
        description_column = min(width, _MENU_DESCRIPTION_COLUMN)
        prefix = clip_cells("-> " if selected else "   ", width, ellipsis="")
        label_limit = max(0, description_column - cell_width(prefix))
        label = clip_cells(item.label, label_limit)
        left = prefix + label
        if not selected:
            row = (
                left
                + " " * max(0, description_column - cell_width(left))
                + item.description
            )
            row = clip_cells(row, width)
            return [
                ("class:chrome.menu.row", row + " " * max(0, width - cell_width(row)))
            ]

        fragments: StyleAndTextTuples = [
            ("class:chrome.menu.row.current", prefix[:1]),
            ("class:chrome.menu.arrow", prefix[1:]),
            ("class:chrome.menu.label.current", label),
        ]
        used = cell_width(prefix) + cell_width(label)
        pad = " " * max(0, description_column - used)
        fragments.append(("class:chrome.menu.row.current", pad))
        description_width = max(0, width - description_column)
        description = clip_cells(item.description, description_width)
        fragments.append(("class:chrome.menu.desc.current", description))
        trailing = max(0, width - description_column - cell_width(description))
        fragments.append(("class:chrome.menu.row.current", " " * trailing))
        return fragments

    @staticmethod
    def _detail_style(line: str) -> str:
        if line.startswith(("+++", "---")):
            return "class:transcript.diff.meta"
        if line.startswith("@@"):
            return "class:transcript.diff.hunk"
        if line.startswith("+"):
            return "class:transcript.diff.add"
        if line.startswith("-"):
            return "class:transcript.diff.del"
        return "class:chrome.menu.detail"

    def queue_height(self) -> Dimension:
        return Dimension.exact(1 if self.controller.queued_prompts else 0)

    def status_height(self) -> Dimension:
        visible = self.controller.cancelling or self.controller.busy
        return Dimension.exact(1 if visible else 0)

    def queue_fragments(self) -> StyleAndTextTuples:
        queued = self.controller.queued_prompts
        if not queued:
            return []
        preview = " ".join(queued[0].message.split())
        width, _ = self._terminal_size()
        maximum = max(8, width - 20)
        if len(preview) > maximum:
            preview = preview[: maximum - 1] + "…"
        return [
            ("class:chrome.queue", f" queued ({len(queued)}) "),
            ("class:chrome.queue.arrow", "» "),
            ("class:chrome.queue.preview", preview),
        ]

    def status_fragments(self) -> StyleAndTextTuples:
        if self.controller.cancelling:
            return [("class:chrome.status.busy", "cancelling")]
        if self.controller.busy:
            spinner = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"[
                int(self.controller.working_elapsed * 10) % 10
            ]
            return [
                (
                    "class:chrome.status.busy",
                    f"{spinner} working {self._format_elapsed(self.controller.working_elapsed)}",
                )
            ]
        return []

    def stats_fragments(self) -> StyleAndTextTuples:
        usage = self.controller.session.usage_summary()
        context_summary = getattr(self.controller.session, "context_summary", None)
        context = context_summary() if callable(context_summary) else {}
        used = int(context.get("used", 0) or 0)
        total = int(context.get("total", 0) or 0)
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        provider = self.runtime.providers.active()
        subscription = provider.billing_label == "subscription"
        cost_value = float(usage.get("cost_usd", 0.0) or 0.0)
        cost = "subscription" if subscription else f"${cost_value:.4f}"
        width, _ = self._terminal_size()
        bar_width = 10 if width >= 72 else (5 if width >= 50 else 4)
        bar = self._context_bar(used, total, bar_width)
        if width >= 72:
            left = f"↑ {input_tokens:,}  ↓ {output_tokens:,}  {cost}"
            right = f"context {used // 1000}k/{total // 1000}k used {bar}"
            gap = max(1, width - cell_width(left) - cell_width(right))
            return [
                ("class:chrome.stats.key", "↑ "),
                ("class:chrome.stats.value", f"{input_tokens:,}"),
                ("class:chrome.stats.key", "  ↓ "),
                ("class:chrome.stats.value", f"{output_tokens:,}  {cost}"),
                ("class:chrome.stats.key", " " * gap),
                ("class:chrome.stats.key", "context "),
                (
                    "class:chrome.stats.value",
                    f"{used // 1000}k/{total // 1000}k used ",
                ),
                ("class:chrome.stats.accent", bar),
            ]
        elif width >= 50:
            short_cost = (
                "sub"
                if subscription
                else (f"${cost_value:.1f}" if cost_value >= 1_000 else cost)
            )
            left = (
                f" ↑{self._compact_number(input_tokens)}"
                f" ↓{self._compact_number(output_tokens)} {short_cost}"
            )
            right = f"ctx {self._compact_number(used)}/{self._compact_number(total)} {bar}"
        else:
            short_cost = "sub" if subscription else f"${cost_value:.2f}"
            left = (
                f" ↑{self._tiny_number(input_tokens)}"
                f" ↓{self._tiny_number(output_tokens)} {short_cost}"
            )
            right = f"ctx {self._tiny_number(used)}/{self._tiny_number(total)} {bar}"
        gap = max(1, width - cell_width(left) - cell_width(right))
        return [
            ("class:chrome.stats", left),
            ("class:chrome.stats", " " * gap),
            ("class:chrome.stats", right.removesuffix(bar)),
            ("class:chrome.stats.bar", bar),
        ]

    def path_fragments(self) -> StyleAndTextTuples:
        project = Path(self.runtime.paths.project_root)
        try:
            left = f"~/{project.relative_to(Path.home())}"
        except ValueError:
            left = str(project)
        model = self.runtime.config.model
        effort = model.reasoning_effort or "no reasoning"
        right = f"{model.model_name.rsplit('/', 1)[-1]} - {effort}"
        width, _ = self._terminal_size()
        right = clip_cells(right, max(8, width // 2))
        left = clip_cells(
            left,
            max(1, width - cell_width(right) - 1),
        )
        gap = max(1, width - cell_width(left) - cell_width(right))
        return [("class:chrome.path", left + " " * gap + right)]

    @staticmethod
    def _format_elapsed(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.1f}s"
        minutes, remainder = divmod(int(seconds), 60)
        return f"{minutes}m {remainder:02d}s"

    @staticmethod
    def _compact_number(value: int) -> str:
        if value >= 1_000_000:
            return f"{value / 1_000_000:.1f}m"
        if value >= 1_000:
            return f"{value / 1_000:.1f}k"
        return str(value)

    @staticmethod
    def _tiny_number(value: int) -> str:
        if value >= 1_000_000:
            return f"{round(value / 1_000_000)}m"
        if value >= 1_000:
            return f"{round(value / 1_000)}k"
        return str(value)

    @staticmethod
    def _context_bar(used: int, total: int, width: int) -> str:
        if total <= 0:
            return "░" * width
        filled = round(min(1.0, used / total) * width)
        return "█" * filled + "░" * (width - filled)

    def input_height(self) -> Dimension:
        return Dimension.exact(min(self._input_max_rows(), self._input_row_count()))

    def _input_max_rows(self) -> int:
        _, terminal_rows = self._terminal_size()
        menu_rows = self.menu_height().preferred if self.menu_open else 0
        available = terminal_rows - int(menu_rows or 0) - 10
        return max(1, min(12, terminal_rows // 4, available))

    def _input_row_count(self) -> int:
        text = self.input_buffer.text
        if not text:
            return 1
        width, _ = self._terminal_size()
        prefix = sum(cell_width(fragment) for _, fragment in self.prompt_fragments())
        rows = 0
        for line_index, line in enumerate(text.split("\n")):
            first_width = max(1, width - prefix) if line_index == 0 else width
            line_width = cell_width(
                line, start_column=prefix if line_index == 0 else 0, tab_size=8
            )
            if line_width <= first_width:
                rows += 1
            else:
                rows += 1 + (line_width - first_width + width - 1) // width
        return max(1, rows)

    def submit_input(self) -> None:
        if self._menu is not None and self._menu.kind == "question":
            self._resolve_question()
            return
        if self._menu is not None and self._menu.kind in {
            "picker",
            "approval",
            "checklist",
        }:
            self._resolve_picker()
            return
        if self._menu is not None and self._menu.kind == "slash":
            selected = self._menu.selected
            if selected is not None:
                self._selected_skills.clear()
                self.input_buffer.reset()
                self._close_menu()
                self._schedule_command(f"/{selected.key}")
            return
        if self._menu is not None and self._menu.kind == "mention":
            self.complete_menu()
            return
        if self._menu is not None and self._menu.kind == "skill":
            self._close_menu()
        if self._prompt_future is not None:
            value = self.input_buffer.text.strip()
            if not value and self._prompt_default:
                value = self._prompt_default
            future = self._prompt_future
            self.input_buffer.reset()
            if not future.done():
                future.set_result(value)
            return

        if self._command_task is not None and not self._command_task.done():
            self.sink.warning("Finish or cancel the open command first.")
            return

        text = self._expand_paste(self.input_buffer.text).strip()
        if not text:
            return
        if text.startswith("/"):
            self._append_history(text)
            self._selected_skills.clear()
            self.input_buffer.reset()
            self._pending_paste = None
            self._close_menu()
            self._schedule_command(text)
            return
        if text.startswith("!"):
            self._append_history(text)
            self._selected_skills.clear()
            self.input_buffer.reset()
            self._pending_paste = None
            sink = self.sink
            self._command_task = asyncio.create_task(
                self._run_shell(text[1:].strip(), sink=sink)
            )
            self._command_task.add_done_callback(self._command_finished)
            return
        images = self._images_for_text(text)
        requested_skills = self._skills_for_text(text)
        self._append_history(text)
        self._selected_skills.clear()
        self.input_buffer.reset()
        self._pending_paste = None
        self._pending_images.clear()
        self._image_counter = 0
        if self.controller.busy:
            position = self.controller.enqueue(
                text, images=tuple(images), requested_skills=requested_skills
            )
            preview = " ".join(text.split())
            if len(preview) > 60:
                preview = preview[:59] + "…"
            self.sink.notice("queue", f"Added prompt #{position}: {preview}")
            return
        task = asyncio.create_task(
            self.controller.submit(
                text, images=tuple(images), requested_skills=requested_skills
            )
        )
        self._turn_tasks.add(task)
        task.add_done_callback(self._turn_finished)

    def toggle_mode(self) -> None:
        if (
            self._prompt_future is None
            and self._menu is None
            and not self.controller.busy
        ):
            self.controller.toggle_mode()

    def toggle_reasoning(self) -> bool:
        return self.sink.toggle_reasoning()

    def interrupt_or_exit(self) -> None:
        if self._menu is not None:
            self.cancel_prompt()
        elif self._prompt_future is not None:
            self.cancel_prompt()
        elif self.controller.busy:
            self.controller.cancel()
        elif self.input_buffer.text:
            self.input_buffer.reset()
            self._selected_skills.clear()
            self._pending_images.clear()
            self._image_counter = 0
            self.invalidate()

    def request_exit(self) -> None:
        if self.controller.busy:
            self.controller.cancel()
            return
        if self._application is not None:
            self._application.exit(result=None)

    def cancel_prompt(self) -> None:
        if self._menu is not None:
            if self._menu.kind in {"picker", "approval", "question", "checklist"}:
                future = self._menu_future
                if future is not None and not future.done():
                    future.set_result(None)
            self._close_menu()
            return
        future = self._prompt_future
        if future is not None and not future.done():
            future.set_result(None)

    async def request_input(
        self,
        label: str,
        *,
        default: str = "",
        secret: bool = False,
    ) -> str | None:
        async with self._prompt_lock:
            return await self._request_input(
                label,
                default=default,
                secret=secret,
            )

    async def request_thread_input(
        self,
        thread_id: str,
        label: str,
        *,
        default: str = "",
    ) -> str | None:
        async with self._prompt_lock:
            await self.controller.begin_interaction(thread_id)
            try:
                return await self._request_input(label, default=default)
            finally:
                self.controller.finish_interaction(thread_id)

    async def _request_input(
        self,
        label: str,
        *,
        default: str = "",
        secret: bool = False,
    ) -> str | None:
        if self._application is None:
            return default or None
        loop = asyncio.get_running_loop()
        self._prompt_future = loop.create_future()
        self._prompt_label = label
        self._prompt_default = default
        self._secret_input = secret
        self._draft = self.input_buffer.text
        self.input_buffer.reset()
        self._application.layout.focus(self.input_window)
        self.invalidate()
        try:
            return await self._prompt_future
        finally:
            self._prompt_future = None
            self._prompt_label = ""
            self._prompt_default = ""
            self._secret_input = False
            self.input_buffer.text = self._draft
            self.input_buffer.cursor_position = len(self._draft)
            self._draft = ""
            self.invalidate()

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
    ) -> str | None:
        async with self._picker_lock:
            answer = await self._choose(
                title,
                items,
                initial_key=initial_key,
                hint=hint,
                filterable=filterable,
                refresh=refresh,
                horizontal_keys=horizontal_keys,
            )
            return answer if isinstance(answer, str) else None

    async def choose_checklist(
        self,
        title: str,
        items: list[MenuItem],
        *,
        disabled_keys: frozenset[str],
        initial_key: str | None = None,
    ) -> ChecklistResult | None:
        async with self._picker_lock:
            answer = await self._choose(
                title,
                items,
                initial_key=initial_key,
                hint="Up/Down select · Left/Right toggle · Enter view · Esc close",
                horizontal_keys=frozenset(item.key for item in items),
                kind="checklist",
                disabled_keys=disabled_keys,
            )
            return answer if isinstance(answer, ChecklistResult) else None

    async def choose_for_thread(
        self,
        thread_id: str,
        title: str,
        items: list[MenuItem],
        *,
        initial_key: str | None = None,
        hint: str = "Up/Down select · Enter confirm · Esc cancel",
    ) -> str | None:
        async with self._picker_lock:
            await self.controller.begin_interaction(thread_id)
            try:
                answer = await self._choose(
                    title,
                    items,
                    initial_key=initial_key,
                    hint=hint,
                )
                return answer if isinstance(answer, str) else None
            finally:
                self.controller.finish_interaction(thread_id)

    async def request_thread_question(
        self,
        thread_id: str,
        title: str,
        items: list[MenuItem],
        *,
        initial_key: str | None = None,
        allow_note: bool = True,
    ) -> tuple[str, str] | None:
        async with self._picker_lock:
            await self.controller.begin_interaction(thread_id)
            try:
                answer = await self._choose(
                    title,
                    items,
                    initial_key=initial_key,
                    hint=(
                        "Up/Down option · Tab note · Enter submit · Esc cancel"
                        if allow_note
                        else "Up/Down option · Enter submit · Esc cancel"
                    ),
                    kind="question",
                    note_allowed=allow_note,
                )
                return answer if isinstance(answer, tuple) else None
            finally:
                self.controller.finish_interaction(thread_id)

    async def _choose(
        self,
        title: str,
        items: list[MenuItem],
        *,
        initial_key: str | None,
        hint: str,
        filterable: bool = False,
        refresh: Callable[[], Awaitable[list[MenuItem]]] | None = None,
        horizontal_keys: frozenset[str] = frozenset(),
        kind: str = "picker",
        summary_lines: list[str] | None = None,
        note_allowed: bool = False,
        disabled_keys: frozenset[str] = frozenset(),
    ) -> str | tuple[str, str] | ChecklistResult | None:
        if not items or self._application is None:
            return None
        loop = asyncio.get_running_loop()
        self._menu_future = loop.create_future()
        index = next(
            (
                item_index
                for item_index, item in enumerate(items)
                if item.key == initial_key
            ),
            0,
        )
        self._picker_draft = self.input_buffer.text
        self._reset_buffer_guarded()
        self._menu = MenuState(
            kind,
            title,
            list(items),
            index=index,
            hint=hint,
            filterable=filterable,
            horizontal_keys=horizontal_keys,
            summary_lines=list(summary_lines or ()),
            note_allowed=note_allowed,
            disabled_keys=set(disabled_keys),
        )
        menu = self._menu
        self._application.layout.focus(self.input_window)
        self.invalidate()
        if refresh is not None:
            self._picker_refresh_task = asyncio.create_task(refresh())
            self._picker_refresh_task.add_done_callback(
                lambda task: self._picker_refreshed(menu, task)
            )
        try:
            return await self._menu_future
        finally:
            if self._picker_refresh_task is not None:
                self._picker_refresh_task.cancel()
                await asyncio.gather(
                    self._picker_refresh_task,
                    return_exceptions=True,
                )
                self._picker_refresh_task = None
            self._menu_future = None
            self._close_menu()
            self.input_buffer.text = self._picker_draft
            self.input_buffer.cursor_position = len(self._picker_draft)
            self._picker_draft = ""
            self.invalidate()

    def _picker_refreshed(
        self,
        menu: MenuState,
        task: asyncio.Task[list[MenuItem]],
    ) -> None:
        if task.cancelled() or self._menu is not menu:
            return
        error = task.exception()
        if error is not None:
            self.sink.warning(f"Model catalog refresh failed: {error}")
            return
        menu.replace_items(task.result())
        self.invalidate()

    async def request_thread_approval(
        self,
        thread_id: str,
        name: str,
        args: dict,
    ) -> str:
        async with self._picker_lock:
            await self.controller.begin_interaction(thread_id)
            try:
                return await self._request_approval(name, args)
            finally:
                self.controller.finish_interaction(thread_id)

    async def _request_approval(self, name: str, args: dict) -> str:
        self._approval_name = name
        self._approval_args = dict(args)
        self._approval_scope_key = "yes"
        actions = [
            MenuItem("yes", "Approve once", "Run this tool call."),
            MenuItem(
                "session",
                "Approve session",
                "Allow matching calls for this session.",
            ),
            MenuItem("always", "Always allow", "Persist an allow rule."),
            MenuItem("no", "Deny once", "Skip this tool call."),
            MenuItem("never", "Never allow", "Persist a deny rule."),
        ]
        if name.lower() in {"edit", "write"}:
            actions.append(MenuItem("diff", "Show diff", "Inspect proposed changes."))
        actions.append(MenuItem("show", "Show args", "Inspect full arguments."))
        summary = format_tool_args(name, args) or "No arguments"
        try:
            answer = await self._choose(
                f"approval needed: {name}",
                actions,
                initial_key="yes",
                hint="Up/Down select · PgUp/PgDn detail · Enter choose · Esc deny",
                kind="approval",
                summary_lines=summary.splitlines()[:3],
            )
            return str(answer or "no")
        finally:
            self._approval_name = ""
            self._approval_args = {}
            self._approval_scope_key = "yes"

    def move_menu(self, amount: int) -> None:
        if self._menu is not None:
            self._menu.move(amount)
            selected = self._menu.selected
            if (
                self._menu.kind == "approval"
                and selected is not None
                and selected.key in {"yes", "session", "always", "no", "never"}
            ):
                self._approval_scope_key = selected.key
            self.invalidate()

    def scroll_menu_detail(self, direction: int) -> bool:
        if self._menu is None or not self._menu.detail_lines:
            return False
        _, _, detail_rows = self._menu_limits()
        self._menu.scroll_detail(
            direction * max(1, detail_rows - 1),
            viewport=max(1, detail_rows),
        )
        self.invalidate()
        return True

    def complete_menu(self) -> None:
        if self._menu is None:
            return
        selected = self._menu.selected
        if selected is None:
            return
        if self._menu.kind == "question":
            self.toggle_question_note()
            return
        if self._menu.kind in {"picker", "approval"}:
            self._resolve_picker()
            return
        if self._menu.kind == "checklist":
            return
        if self._menu.kind in {"mention", "skill"}:
            is_skill = self._menu.kind == "skill"
            prefix = "$" if is_skill else "@"
            span = self._active_reference_span(prefix)
            if span is None:
                self._close_menu()
                return
            start, end, _ = span
            text = self.input_buffer.text
            replacement = f"{prefix}{selected.key} "
            self._close_menu()
            self.input_buffer.text = text[:start] + replacement + text[end:]
            self.input_buffer.cursor_position = start + len(replacement)
            if is_skill:
                self._selected_skills.add(selected.key)
            self._close_menu()
            return
        self._close_menu()
        self.input_buffer.text = f"/{selected.key} "
        self.input_buffer.cursor_position = len(self.input_buffer.text)

    def change_menu_horizontal(self) -> None:
        if self._menu is None:
            return
        if self._menu.kind == "checklist":
            self._menu.toggle_selected()
            self.invalidate()
            return
        self.complete_menu()

    def clear_transcript(self) -> None:
        self.sink.clear()
        self._append_header(self.sink, thread_id=self.controller.thread_id)
        self.follow_transcript_end()

    async def refresh_transcript(self) -> None:
        self.sink.clear()
        self.sink.assistant_history.clear()
        self._append_header(self.sink, thread_id=self.controller.thread_id)
        for event in self.controller.session.history():
            replay_event(self.sink, event)
        await self.controller.refresh_todos()
        self.follow_transcript_end()

    def prefill_input(self, text: str) -> None:
        self.input_buffer.text = text
        self.input_buffer.cursor_position = len(text)
        self.invalidate()

    def _turn_finished(self, task: asyncio.Task[bool]) -> None:
        self._turn_tasks.discard(task)
        if task.cancelled():
            self.invalidate()
            return
        error = task.exception()
        if error is not None:
            self.sink.error(str(error))
        self.invalidate()

    def _schedule_command(self, command_line: str) -> None:
        self._command_task = asyncio.create_task(
            self.commands.dispatch(
                self.command_context,
                command_line,
                busy=self.controller.busy,
            )
        )
        self._command_task.add_done_callback(self._command_finished)

    def _command_finished(self, task: asyncio.Task[bool]) -> None:
        self._command_task = None
        if not task.cancelled() and task.exception() is not None:
            self.sink.error(str(task.exception()))
        self.invalidate()

    def _on_input_changed(self, _buffer: Buffer) -> None:
        if self._collapsing_paste:
            return
        if self._prompt_future is not None:
            return
        if self._menu is not None and self._menu.kind == "question":
            if self._menu.note_active:
                self._menu.note_text = self.input_buffer.text
                self.invalidate()
            return
        if self._menu is not None and self._menu.kind in {
            "picker",
            "approval",
            "checklist",
        }:
            if self._menu.filterable:
                self._menu.apply_filter(self.input_buffer.text)
                self.invalidate()
            return
        text = self.input_buffer.text
        if self._pending_paste is not None and self._paste_marker() not in text:
            self._pending_paste = None
        self._sync_pending_images(text)
        self._selected_skills.intersection_update(self._skills_for_text(text))
        if text.startswith("/") and not any(
            character.isspace() for character in text[1:]
        ):
            matches = self.commands.matches(
                text[1:],
                commands_dir=self.runtime.paths.ness_dir / "commands",
            )
            if matches:
                self._menu = MenuState(
                    "slash",
                    "commands",
                    [
                        MenuItem(
                            spec.name,
                            spec.name,
                            description=spec.summary,
                        )
                        for spec in matches
                    ],
                    hint="Up/Down select · Tab complete · Enter run · Esc close",
                )
            else:
                self._close_menu()
        elif self._menu is not None and self._menu.kind == "slash":
            self._close_menu()
        else:
            skill = (
                self._active_reference_span("$")
                if not text.startswith(("/", "!"))
                else None
            )
            mention = self._active_reference_span("@")
            if skill is not None:
                skills, _ = self.controller.session.skills()
                items = {}
                for metadata in skills:
                    name = str(metadata.get("name") or "")
                    if name and metadata.get("available", True):
                        items.setdefault(
                            name,
                            MenuItem(
                                name,
                                name,
                                description=" ".join(
                                    str(metadata.get("description") or "").split()
                                ),
                            ),
                        )
                self._menu = MenuState(
                    "skill",
                    "skills",
                    list(items.values()),
                    hint="Up/Down select · Tab insert · Enter send · Esc close",
                    filterable=True,
                )
                self._menu.apply_filter(skill[2])
            elif mention is not None:
                _, _, query = mention
                matches = self._file_index.search(query)
                if matches:
                    self._menu = MenuState(
                        "mention",
                        "files",
                        matches,
                        hint="Up/Down select · Tab or Enter complete · Esc close",
                    )
                elif self._menu is not None and self._menu.kind in {"mention", "skill"}:
                    self._close_menu()
            elif self._menu is not None and self._menu.kind in {"mention", "skill"}:
                self._close_menu()
        self.invalidate()

    def _active_reference_span(self, prefix: str) -> tuple[int, int, str] | None:
        cursor = self.input_buffer.cursor_position
        text = self.input_buffer.text
        start = cursor
        while start > 0 and not text[start - 1].isspace():
            start -= 1
        token = text[start:cursor]
        if not token.startswith(prefix):
            return None
        return start, cursor, token[1:]

    def _skills_for_text(self, text: str) -> tuple[str, ...]:
        matches = []
        for name in self._selected_skills:
            match = re.search(r"(?<!\S)\$" + re.escape(name) + r"(?=\s|$)", text)
            if match is not None:
                matches.append((match.start(), name))
        return tuple(name for _, name in sorted(matches))

    def _sync_pending_images(self, text: str) -> None:
        visible = {
            int(match.group(1)) for match in re.finditer(r"\[Image #(\d+)\]", text)
        }
        for number in tuple(self._pending_images):
            if number not in visible:
                self._pending_images.pop(number, None)

    def _images_for_text(self, text: str) -> list[str]:
        images: list[str] = []
        seen: set[int] = set()
        for match in re.finditer(r"\[Image #(\d+)\]", text):
            number = int(match.group(1))
            payload = self._pending_images.get(number)
            if payload and number not in seen:
                images.append(payload)
                seen.add(number)
        return images

    def paste_clipboard_image(self) -> None:
        from ness_cli.tui.input.images import ImageTooLarge, save_clipboard_image

        try:
            result = save_clipboard_image(paths=self.runtime.paths)
        except ImageTooLarge as error:
            self.sink.warning(f"Image paste failed: {error}")
            return
        except Exception as error:
            self.sink.warning(f"Image paste failed: {error}")
            return
        if result is None:
            self.sink.warning("No image found on the clipboard.")
            return
        _, data_url = result
        self._image_counter += 1
        self._pending_images[self._image_counter] = data_url
        self.input_buffer.insert_text(f"[Image #{self._image_counter}] ")

    def insert_paste(self, text: str) -> None:
        """Compact a bracketed multiline paste without changing its payload."""

        if "\n" not in text or self._prompt_future is not None:
            self.input_buffer.insert_text(text)
            return
        if self._menu is not None and self._menu.kind != "skill":
            if self._menu.kind == "question" and self._menu.note_active:
                self.input_buffer.insert_text(text)
                return
            if self._menu.filterable:
                self.input_buffer.insert_text(" ".join(text.split()))
            return
        if self._pending_paste is not None:
            expanded = self._expand_paste(self.input_buffer.text)
            self._pending_paste = None
            self._write_buffer_text(expanded)
        self._pending_paste = text
        self.input_buffer.insert_text(self._paste_marker())

    def _paste_marker(self) -> str:
        if self._pending_paste is None:
            return ""
        return f"[pasted {self._pending_paste.count(chr(10)) + 1} lines]"

    def _expand_paste(self, text: str) -> str:
        if self._pending_paste is None:
            return text
        marker = self._paste_marker()
        return text.replace(marker, self._pending_paste, 1) if marker in text else text

    def _write_buffer_text(self, text: str) -> None:
        self._collapsing_paste = True
        try:
            self.input_buffer.text = text
            self.input_buffer.cursor_position = len(text)
        finally:
            self._collapsing_paste = False

    def _reset_buffer_guarded(self) -> None:
        self._collapsing_paste = True
        try:
            self.input_buffer.reset()
        finally:
            self._collapsing_paste = False

    def _append_history(self, text: str) -> None:
        if text:
            self.input_buffer.history.append_string(text)

    def _resolve_picker(self) -> None:
        if self._menu is None or self._menu.kind not in {
            "picker",
            "approval",
            "checklist",
        }:
            return
        selected = self._menu.selected
        if self._menu.kind == "approval" and selected is not None:
            if selected.key in {"diff", "show"}:
                self._inspect_approval(selected.key)
                return
        future = self._menu_future
        if future is not None and not future.done():
            if self._menu.kind == "checklist" and selected is not None:
                future.set_result(
                    ChecklistResult(
                        selected_key=selected.key,
                        disabled_keys=frozenset(self._menu.disabled_keys),
                    )
                )
            else:
                future.set_result(selected.key if selected is not None else None)

    def toggle_question_note(self) -> None:
        menu = self._menu
        if menu is None or menu.kind != "question" or not menu.note_allowed:
            return
        if menu.note_active:
            menu.note_text = self.input_buffer.text
            menu.note_active = False
            self._reset_buffer_guarded()
        else:
            menu.note_active = True
            self._write_buffer_text(menu.note_text)
        self.invalidate()

    def _resolve_question(self) -> None:
        menu = self._menu
        if menu is None or menu.kind != "question":
            return
        selected = menu.selected
        if selected is None:
            return
        if menu.note_active:
            menu.note_text = self.input_buffer.text
        future = self._menu_future
        if future is not None and not future.done():
            future.set_result((selected.key, menu.note_text.strip()))

    def _inspect_approval(self, action: str) -> None:
        menu = self._menu
        if menu is None or menu.kind != "approval":
            return
        if action == "diff":
            detail = (
                preview_diff(self._approval_name.lower(), self._approval_args)
                or "(no diff)"
            )
            menu.detail_label = "proposed diff"
        else:
            import json

            try:
                detail = json.dumps(
                    self._approval_args,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            except (TypeError, ValueError):
                detail = repr(self._approval_args)
            menu.detail_label = "full arguments"
        menu.detail_lines = detail.splitlines() or ["(empty)"]
        menu.detail_scroll = 0
        self.invalidate()

    def _close_menu(self) -> None:
        self._menu = None
        self.invalidate()

    def _make_thread_sink(self, thread_id: str) -> TuiRenderSink:
        sink = TuiRenderSink(
            TranscriptStore(),
            invalidate=self.invalidate,
            prompts=_ThreadPromptDriver(self, thread_id),
        )
        self._append_header(sink, thread_id=thread_id)
        return sink

    def _selected_sink_changed(self, sink: TuiRenderSink) -> None:
        if self.transcript_view is not None:
            self.transcript_view.store = sink.store
        self.invalidate()

    def _append_header(self, sink: TuiRenderSink, *, thread_id: str) -> None:
        model = self.runtime.config.model
        summary, level = self.runtime.mcp.startup_summary()
        integrations = summary.removeprefix("MCP: ")
        try:
            mode = self.controller.mode
        except RuntimeError:
            mode = "act"
        approval = getattr(self.runtime, "approval_state", None)
        if not approval:
            enabled = bool(
                getattr(getattr(self.runtime, "settings", None), "enable_approval", True)
            )
            approval = "on" if enabled else "off"
        sink.append_header(
            model=model.model_name,
            provider=model.provider_id,
            project=str(self.runtime.paths.project_root),
            thread_id=thread_id,
            version=_package_version(),
            mode=mode,
            approval=str(approval),
            integrations=integrations,
        )
        if level == "warn":
            sink.warning(summary + "  (/mcp for details)")
        for warning in self.runtime.warnings:
            if warning != summary:
                sink.warning(warning)

    _SHELL_OUTPUT_CAP = 20_000
    _SHELL_TIMEOUT = 60.0

    async def _run_shell(self, command: str, *, sink: TuiRenderSink) -> bool:
        if not command:
            sink.error("Usage: !<shell command>")
            return False
        sink.tool_call("shell", {"command": command})
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                cwd=self.runtime.paths.project_root,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            output, _ = await asyncio.wait_for(
                process.communicate(),
                timeout=self._SHELL_TIMEOUT,
            )
        except TimeoutError:
            if "process" in locals():
                process.kill()
                await process.wait()
            sink.warning(
                f"Shell command timed out after {self._SHELL_TIMEOUT:.0f} seconds."
            )
            return False
        except Exception as error:
            sink.error(f"shell: {error}")
            return False
        text = output.decode(errors="replace") if output else "(no output)"
        if len(text) > self._SHELL_OUTPUT_CAP:
            text = (
                text[: self._SHELL_OUTPUT_CAP]
                + f"\n... (truncated, {len(text)} chars total)"
            )
        status = "ok" if process.returncode == 0 else f"exit {process.returncode}"
        sink.tool_result("shell", text.rstrip(), exit_status=status)
        return process.returncode == 0


async def run_app(runtime: InteractiveRuntime) -> None:
    app = TuiApp(runtime)
    await app.run()
    session = app.controller.session
    saved = await session.finalize_and_save()
    report = session.usage_report()
    provider = runtime.providers.active()
    if provider.billing_label == "subscription":
        report = report.replace("Cost: unknown", "Cost: subscription")
    if saved.resume_thread_id:
        report += f"\nResume:  ness --resume {saved.resume_thread_id}"
    Console(file=sys.stdout).print(
        Panel(
            Text(report),
            title="session summary",
            style=GRAY,
            border_style=GRAY,
        )
    )
