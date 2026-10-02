"""Virtualized prompt-toolkit control for a transcript store."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from prompt_toolkit.data_structures import Point
from prompt_toolkit.layout.controls import UIContent, UIControl
from prompt_toolkit.mouse_events import MouseEvent, MouseEventType

from ness_cli.tui.transcript.store import TranscriptStore, VisualPosition


@dataclass(slots=True)
class _ViewState:
    scroll: int = 0
    viewport_rows: int = 1
    follow: bool = True
    selection_anchor: VisualPosition | None = None
    selection_cursor: VisualPosition | None = None
    selecting: bool = False


class TranscriptView(UIControl):
    def __init__(
        self,
        store: TranscriptStore,
        *,
        on_resize: Callable[[int, int], None] | None = None,
        focus: Callable[[], None] | None = None,
        invalidate: Callable[[], None] | None = None,
    ) -> None:
        self._store = store
        self._states: dict[TranscriptStore, _ViewState] = {store: _ViewState()}
        self._on_resize = on_resize
        self._focus = focus or (lambda: None)
        self._invalidate = invalidate or (lambda: None)

    @property
    def store(self) -> TranscriptStore:
        return self._store

    @store.setter
    def store(self, value: TranscriptStore) -> None:
        self._store = value
        self._states.setdefault(value, _ViewState())

    @property
    def vertical_scroll(self) -> int:
        state = self._state
        maximum = self.store.max_scroll(state.viewport_rows)
        if state.follow:
            state.scroll = maximum
        else:
            state.scroll = max(0, min(state.scroll, maximum))
        return state.scroll

    @property
    def viewport_rows(self) -> int:
        return self._state.viewport_rows

    @property
    def is_following(self) -> bool:
        return self._state.follow

    def is_focusable(self) -> bool:
        return True

    def preferred_height(
        self,
        width: int,
        max_available_height: int,
        wrap_lines: bool,
        get_line_prefix,
    ) -> int:
        del width, max_available_height, wrap_lines, get_line_prefix
        return max(1, self.store.total_rows)

    def create_content(self, width: int, height: int) -> UIContent:
        width = max(1, width)
        height = max(1, height)
        self._state.viewport_rows = height
        changed = self.store.set_width(width)
        if changed and self._on_resize is not None:
            self._on_resize(width, height)
        line_count = max(1, self.store.total_rows)
        scroll = self.vertical_scroll
        selection = self.selection_range()

        def get_line(row: int):
            return self.store.row_fragments(row, selection=selection)

        return UIContent(
            get_line=get_line,
            line_count=line_count,
            cursor_position=Point(x=0, y=min(scroll, line_count - 1)),
            show_cursor=False,
        )

    def scroll_by(self, amount: int) -> None:
        state = self._state
        maximum = self.store.max_scroll(state.viewport_rows)
        state.scroll = max(0, min(maximum, self.vertical_scroll + amount))
        state.follow = state.scroll >= maximum
        self._invalidate()

    def scroll_to_top(self) -> None:
        state = self._state
        state.scroll = 0
        state.follow = False
        self._invalidate()

    def follow_end(self) -> None:
        state = self._state
        state.follow = True
        state.scroll = self.store.max_scroll(state.viewport_rows)
        self._invalidate()

    def has_selection(self) -> bool:
        selection = self.selection_range()
        return selection is not None and selection[0] != selection[1]

    def selection_range(
        self,
    ) -> tuple[VisualPosition, VisualPosition] | None:
        state = self._state
        if state.selection_anchor is None or state.selection_cursor is None:
            return None
        return state.selection_anchor, state.selection_cursor

    def selected_text(self) -> str:
        selection = self.selection_range()
        if selection is None or selection[0] == selection[1]:
            return ""
        return self.store.copy_range(*selection)

    def clear_selection(self) -> None:
        state = self._state
        state.selection_anchor = None
        state.selection_cursor = None
        state.selecting = False
        self._invalidate()

    def mouse_handler(self, mouse_event: MouseEvent):
        event_type = mouse_event.event_type
        if event_type == MouseEventType.SCROLL_UP:
            self.scroll_by(-3)
            return None
        if event_type == MouseEventType.SCROLL_DOWN:
            self.scroll_by(3)
            return None

        state = self._state
        if event_type == MouseEventType.MOUSE_DOWN:
            self._focus()
            position = self._event_position(mouse_event)
            state.selection_anchor = position
            state.selection_cursor = position
            state.selecting = True
            self._invalidate()
            return None
        if event_type == MouseEventType.MOUSE_MOVE and state.selecting:
            state.selection_cursor = self._event_position(mouse_event)
            self._invalidate()
            return None
        if event_type == MouseEventType.MOUSE_UP and state.selecting:
            state.selection_cursor = self._event_position(mouse_event)
            state.selecting = False
            self._invalidate()
            return None
        return NotImplemented

    @property
    def _state(self) -> _ViewState:
        return self._states.setdefault(self.store, _ViewState())

    def _event_position(self, event: MouseEvent) -> VisualPosition:
        last_row = max(0, self.store.total_rows - 1)
        row = max(0, min(last_row, event.position.y))
        text = self.store.row_text(row)
        col = max(0, min(len(text), event.position.x))
        return VisualPosition(row, col)
