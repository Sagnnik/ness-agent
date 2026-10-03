"""Mutable transcript data with stable handles for streamed blocks."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass

from prompt_toolkit.formatted_text import StyleAndTextTuples


SELECTION_STYLE = "class:transcript.selection"


@dataclass(frozen=True, slots=True)
class VisualPosition:
    row: int
    col: int


@dataclass(slots=True)
class TranscriptLine:
    style: str
    text: str
    fragments: StyleAndTextTuples | None = None


@dataclass(eq=False, slots=True)
class TranscriptBlock:
    """Stable handle for a contiguous range owned by the store."""

    start: int
    count: int
    attached: bool = True


class TranscriptStore:
    """Store logical lines and expose width-aware visual rows."""

    def __init__(
        self,
        lines: list[TranscriptLine] | None = None,
        *,
        width: int = 80,
    ) -> None:
        self.lines = list(lines or ())
        self.width = max(1, width)
        self.revision = 0
        self._blocks: list[TranscriptBlock] = []
        self._row_counts: list[int] = []
        self._row_offsets: list[int] = [0]
        self._plain_text: str | None = None
        self._rebuild_rows()

    @property
    def total_rows(self) -> int:
        return self._row_offsets[-1]

    def max_scroll(self, viewport_rows: int) -> int:
        return max(0, self.total_rows - max(1, viewport_rows))

    def set_width(self, width: int) -> bool:
        width = max(1, width)
        if width == self.width:
            return False
        self.width = width
        self._rebuild_rows()
        self._changed()
        return True

    def append(self, lines: list[TranscriptLine]) -> None:
        if not lines:
            return
        self.lines.extend(lines)
        self._row_counts.extend(self._row_count(line) for line in lines)
        self._rebuild_offsets_from(len(self.lines) - len(lines))
        self._changed()

    def insert(self, start: int, lines: list[TranscriptLine]) -> None:
        if not lines:
            return
        start = max(0, min(start, len(self.lines)))
        for block in self._blocks:
            if block.start < start < block.start + block.count:
                raise ValueError(
                    "raw transcript insertion splits a tracked block; "
                    "use a tracked-block operation"
                )
        self._shift_blocks(start, old_count=0, new_count=len(lines))
        self.lines[start:start] = lines
        self._row_counts[start:start] = [self._row_count(line) for line in lines]
        self._rebuild_offsets_from(start)
        self._changed()

    def replace(
        self,
        start: int,
        count: int,
        lines: list[TranscriptLine],
    ) -> None:
        start, count = self._validate_untracked_range(start, count)
        self._shift_blocks(start, old_count=count, new_count=len(lines))
        self.lines[start : start + count] = lines
        self._row_counts[start : start + count] = [
            self._row_count(line) for line in lines
        ]
        self._rebuild_offsets_from(start)
        self._changed()

    def delete(self, start: int, count: int) -> None:
        if count <= 0:
            return
        start, count = self._validate_untracked_range(start, count)
        self._shift_blocks(start, old_count=count, new_count=0)
        del self.lines[start : start + count]
        del self._row_counts[start : start + count]
        self._rebuild_offsets_from(start)
        self._changed()

    def append_tracked(self, lines: list[TranscriptLine]) -> TranscriptBlock:
        start = len(self.lines)
        self.append(lines)
        return self._register_block(start, len(lines))

    def insert_tracked(
        self,
        start: int,
        lines: list[TranscriptLine],
    ) -> TranscriptBlock:
        start = max(0, min(start, len(self.lines)))
        self.insert(start, lines)
        return self._register_block(start, len(lines))

    def replace_tracked(
        self,
        block: TranscriptBlock,
        lines: list[TranscriptLine],
    ) -> None:
        self._require_attached(block)
        start = block.start
        old_count = block.count
        old_end = start + old_count
        delta = len(lines) - old_count
        for other in self._blocks:
            if other is not block and other.start >= old_end:
                other.start += delta
        self.lines[start:old_end] = lines
        self._row_counts[start:old_end] = [self._row_count(line) for line in lines]
        block.count = len(lines)
        self._rebuild_offsets_from(start)
        self._changed()

    def delete_tracked(self, block: TranscriptBlock) -> None:
        self._require_attached(block)
        start = block.start
        count = block.count
        end = start + count
        del self.lines[start:end]
        del self._row_counts[start:end]
        for other in self._blocks:
            if other is not block and other.start >= end:
                other.start -= count
        self._detach(block)
        self._rebuild_offsets_from(start)
        self._changed()

    def release_tracked(self, block: TranscriptBlock) -> None:
        self._require_attached(block)
        self._detach(block)

    def move_tracked_to_end(self, block: TranscriptBlock) -> None:
        self._require_attached(block)
        start = block.start
        end = start + block.count
        if end == len(self.lines):
            return
        lines = self.lines[start:end]
        rows = self._row_counts[start:end]
        del self.lines[start:end]
        del self._row_counts[start:end]
        for other in self._blocks:
            if other is not block and other.start >= end:
                other.start -= block.count
        block.start = len(self.lines)
        self.lines.extend(lines)
        self._row_counts.extend(rows)
        self._rebuild_offsets_from(start)
        self._changed()

    def reset(self, lines: list[TranscriptLine] | None = None) -> None:
        for block in self._blocks:
            block.attached = False
        self._blocks.clear()
        self.lines[:] = list(lines or ())
        self._rebuild_rows()
        self._changed()

    def plain_text(self) -> str:
        if self._plain_text is None:
            self._plain_text = "\n".join(line.text for line in self.lines)
        return self._plain_text

    def row_text(self, row: int) -> str:
        line_index, offset = self._line_for_row(row)
        if line_index is None:
            return ""
        text = self.lines[line_index].text
        start = offset * self.width
        return text[start : start + self.width]

    def row_fragments(
        self,
        row: int,
        selection: tuple[VisualPosition, VisualPosition] | None = None,
    ) -> StyleAndTextTuples:
        line_index, offset = self._line_for_row(row)
        if line_index is None:
            return []
        line = self.lines[line_index]
        start = offset * self.width
        end = start + self.width
        fragments = self._slice_fragments(line, start, end)
        if selection is None:
            return fragments
        selected = self._selected_columns(
            row,
            len(line.text[start:end]),
            selection,
        )
        if selected is None:
            return fragments
        return self._apply_selection(fragments, *selected)

    def copy_range(self, start: VisualPosition, end: VisualPosition) -> str:
        start, end = self._ordered(start, end)
        rows: list[str] = []
        for row in range(start.row, end.row + 1):
            text = self.row_text(row)
            left = start.col if row == start.row else 0
            right = end.col if row == end.row else len(text)
            left = max(0, min(len(text), left))
            right = max(0, min(len(text), right))
            rows.append(text[left:right])
        return "\n".join(rows)

    def _slice_fragments(
        self,
        line: TranscriptLine,
        start: int,
        end: int,
    ) -> StyleAndTextTuples:
        output: StyleAndTextTuples = []
        cursor = 0
        for style, text in self._line_fragments(line):
            next_cursor = cursor + len(text)
            if next_cursor > start and cursor < end:
                left = max(0, start - cursor)
                right = min(len(text), end - cursor)
                output.append((style, text[left:right]))
            cursor = next_cursor
        return output or [(line.style or "class:transcript.muted", "")]

    def _selected_columns(
        self,
        row: int,
        row_length: int,
        selection: tuple[VisualPosition, VisualPosition],
    ) -> tuple[int, int] | None:
        start, end = self._ordered(*selection)
        if row < start.row or row > end.row:
            return None
        left = start.col if row == start.row else 0
        right = end.col if row == end.row else row_length
        left = max(0, min(row_length, left))
        right = max(0, min(row_length, right))
        if right <= left:
            return None
        return left, right

    @staticmethod
    def _apply_selection(
        fragments: StyleAndTextTuples,
        left: int,
        right: int,
    ) -> StyleAndTextTuples:
        output: StyleAndTextTuples = []
        cursor = 0
        for style, text in fragments:
            next_cursor = cursor + len(text)
            if next_cursor <= left or cursor >= right:
                output.append((style, text))
            else:
                before = max(0, left - cursor)
                after = max(0, next_cursor - right)
                selected_end = len(text) - after
                if before:
                    output.append((style, text[:before]))
                output.append(
                    (
                        f"{style} {SELECTION_STYLE}".strip(),
                        text[before:selected_end],
                    )
                )
                if after:
                    output.append((style, text[selected_end:]))
            cursor = next_cursor
        return output

    @staticmethod
    def _ordered(
        start: VisualPosition,
        end: VisualPosition,
    ) -> tuple[VisualPosition, VisualPosition]:
        if (end.row, end.col) < (start.row, start.col):
            return end, start
        return start, end

    def _register_block(self, start: int, count: int) -> TranscriptBlock:
        end = start + count
        for other in self._blocks:
            other_end = other.start + other.count
            if start < other_end and other.start < end:
                raise ValueError("tracked transcript blocks cannot overlap")
        block = TranscriptBlock(start=start, count=count)
        self._blocks.append(block)
        return block

    def _require_attached(self, block: TranscriptBlock) -> None:
        if not block.attached or block not in self._blocks:
            raise ValueError("transcript block is detached")

    def _detach(self, block: TranscriptBlock) -> None:
        self._blocks.remove(block)
        block.attached = False

    def _validate_untracked_range(
        self,
        start: int,
        count: int,
    ) -> tuple[int, int]:
        start = max(0, min(start, len(self.lines)))
        count = max(0, min(count, len(self.lines) - start))
        end = start + count
        for block in self._blocks:
            block_end = block.start + block.count
            if count and start < block_end and block.start < end:
                raise ValueError(
                    "raw transcript mutation overlaps a tracked block; "
                    "use a tracked-block operation"
                )
        return start, count

    def _shift_blocks(
        self,
        start: int,
        *,
        old_count: int,
        new_count: int,
    ) -> None:
        old_end = start + old_count
        delta = new_count - old_count
        for block in self._blocks:
            if block.start >= old_end:
                block.start += delta

    def _rebuild_rows(self) -> None:
        self._row_counts = [self._row_count(line) for line in self.lines]
        self._row_offsets = [0] * (len(self._row_counts) + 1)
        self._rebuild_offsets_from(0)

    def _rebuild_offsets_from(self, start: int) -> None:
        if len(self._row_offsets) != len(self._row_counts) + 1:
            self._row_offsets = [0] * (len(self._row_counts) + 1)
            start = 0
        start = max(0, min(start, len(self._row_counts)))
        if start == 0:
            self._row_offsets[0] = 0
        for index in range(start, len(self._row_counts)):
            self._row_offsets[index + 1] = (
                self._row_offsets[index] + self._row_counts[index]
            )

    def _row_count(self, line: TranscriptLine) -> int:
        return max(1, (len(line.text) + self.width - 1) // self.width)

    def _line_for_row(self, row: int) -> tuple[int | None, int]:
        if row < 0 or row >= self.total_rows or not self.lines:
            return None, 0
        index = max(0, bisect_right(self._row_offsets, row) - 1)
        return index, row - self._row_offsets[index]

    @staticmethod
    def _line_fragments(line: TranscriptLine) -> StyleAndTextTuples:
        if line.fragments and "".join(text for _, text in line.fragments) == line.text:
            return line.fragments
        return [(line.style or "class:transcript.muted", line.text)]

    def _changed(self) -> None:
        self.revision += 1
        self._plain_text = None
