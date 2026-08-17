from __future__ import annotations

import textwrap
from typing import Any

from ness_cli.tui.tool_display import (
    BATCHABLE_TOOL_CALLS,
    format_batched_tool_args,
    format_tool_args,
    format_tool_result_preview,
    should_show_tool_call,
    should_show_tool_result,
)
from ness_cli.tui.formatting import USER_STYLE, user_message_lines
from ness_cli.tui.markdown import (
    _REASONING_COLLAPSED_STYLE,
    _diff_transcript_lines,
    _reasoning_block_lines,
    _shell_output_lines,
    _tagged_lines,
    markdown_transcript_lines,
    todos_transcript_lines,
)
from ness_cli.tui.models import TranscriptLine
from ness_cli.tui.stream import Thinking, TuiAssistantStream
from ness_cli.tui.utils import term_height, term_width
from ness_cli.tui.widgets import TranscriptBlock


# Module-level TranscriptLine row builders (``_diff_transcript_lines``,
# ``_shell_output_lines``, ``_tagged_lines``, ``_reasoning_block_lines`` and
# their helpers) live in ness_cli.tui.markdown alongside ``markdown_transcript_lines``
# and ``todos_transcript_lines``, so every "build TranscriptLines from data"
# routine sits in one module. Imported back here for the TranscriptMixin sink
# methods below.


def _header_project() -> str:
    """CWD-with-branch string for the header's Project cell."""
    from ness_cli.tui.utils import display_cwd

    return display_cwd()


def _header_addons_summary(mcp, skill_loader) -> str:
    """Summarize active MCP servers + skills for the header's Add-ons cell.

    Reads the TuiApp-held ProjectMCPManager and the coding session's SkillLoader;
    both are optional so headless/test paths (no MCP, no skills dir) render
    an empty summary instead of failing.
    """
    parts: list[str] = []
    try:
        if mcp is not None:
            server_names = sorted(
                name
                for name, info in mcp.servers.items()
                if info.get("status") != "error"
            )
            n_mcp = len(server_names)
            if n_mcp:
                names = ", ".join(server_names[:3])
                if len(server_names) > 3:
                    names += ", …"
                parts.append(f"{n_mcp} MCPs ({names})" if names else f"{n_mcp} MCPs")
    except Exception:
        pass
    try:
        if skill_loader is not None:
            parts.append(f"{len(skill_loader.load())} Skills")
    except Exception:
        pass
    return ", ".join(parts)


def _header_version() -> str:
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("ness-agent")
        except PackageNotFoundError:
            return "dev"
    except Exception:
        return "dev"


def _table_column_widths(
    headers: list[str], rows: list[list[str]], available: int
) -> list[int]:
    """Fit natural column widths into *available* cells.

    Columns shrink from the widest value first, preserving enough room for
    their heading. Cell content is wrapped by :func:`_table_cell_lines`.
    """
    if not headers:
        return []
    minimums = [max(4, min(len(str(header)), 12)) for header in headers]
    widths = minimums.copy()
    for index, header in enumerate(headers):
        values = [
            str(header),
            *(str(row[index]) if index < len(row) else "" for row in rows),
        ]
        widths[index] = max(
            minimums[index],
            *(len(line) for value in values for line in value.splitlines() or [""]),
        )

    target = max(sum(minimums), available)
    while sum(widths) > target:
        candidates = [
            index for index, width in enumerate(widths) if width > minimums[index]
        ]
        if not candidates:
            break
        widest = max(candidates, key=lambda index: (widths[index], index))
        widths[widest] -= 1
    return widths


def _table_cell_lines(value: Any, width: int) -> list[str]:
    """Wrap one table cell without allowing prompt_toolkit to break the row."""
    output: list[str] = []
    for raw_line in str(value).splitlines() or [""]:
        output.extend(
            textwrap.wrap(
                raw_line,
                width=max(1, width),
                break_long_words=True,
                break_on_hyphens=False,
                replace_whitespace=True,
                drop_whitespace=True,
            )
            or [""]
        )
    return output


class TranscriptMixin:
    """Transcript buffer, render-sink methods, and scroll behavior.

    Composed into ``TuiApp`` (see cli/app.py) and registered as the active
    ``RenderSink`` via ``render.set_sink``. Methods are grouped by concern
    and separated by banner comments inline:

    - Tagged/diff/shell/reasoning line builders       (module scope: ness_cli.tui.markdown)
    - Sink entry points (high-level render calls)   (append_header, append_user)
    - Resize + reflow machinery                     (_on_transcript_render_width...)
    - Tagged render sink methods                     (append_notice, append_warning, ...)
    - Assistant markdown rendering                  (append_assistant, _reasoning_block_for_span)
    - Reasoning slot lifecycle                       (reserve_reasoning_slot, finalize, toggle)
    - Live assistant streaming                       (set_assistant_stream, finalize, clear)
    - Transcript buffer primitives + reset           (_append_transcript, _sync...)
    - Tool calls & results                          (append_tool_calls, append_tool_result)
    - Todos / diff / shell output                   (append_todos, append_diff, append_shell_output)
    - Streaming adapters (start_assistant_stream, thinking)
    - Layout sizing + scroll navigation             (_chrome_height_lines, _scroll_*)
    """

    # ------------------------------------------------------------------ #
    # Sink entry points (high-level render)                               #
    # ------------------------------------------------------------------ #
    def append_header(
        self,
        *,
        mode: str,
        model: str,
        approval: bool,
        yolo: bool = False,
        autosave: bool,
        session_end_reflection: bool,
    ) -> None:
        del (
            autosave,
            session_end_reflection,
        )  # surfaced elsewhere (not in the new header)
        width = self._transcript_render_width or term_width()
        from ness_cli.tui.header import header_lines

        source = {
            "mode": mode,
            "model": model,
            "approval": approval,
            "yolo": yolo,
            "project": _header_project(),
            # getattr chains: bare TranscriptMixin harnesses (tests) have no
            # mcp/coding wired; both summary inputs degrade to empty.
            "addons_summary": _header_addons_summary(
                getattr(self, "mcp", None),
                getattr(getattr(self, "coding", None), "skill_loader", None),
            ),
            "version": _header_version(),
        }
        rows = header_lines(width=width, show_logo=width >= 96, **source)
        # The trailing blank line is part of the tracked block so an in-place
        # replace removes the old blank too (otherwise spacers accumulate).
        if rows:
            block_lines = [*rows, TranscriptLine("class:transcript.muted", "")]
        else:
            # narrow-terminal fallback (<40 cols): a single [session] notice,
            # still tracked so a later resize regenerates the full header.
            block_lines = [
                *_tagged_lines(
                    "session", f"model {model}  approval {'on' if approval else 'off'}"
                ),
                TranscriptLine("class:transcript.muted", ""),
            ]

        if self._header_block is None:
            # First render: always pin to the top. Startup notices may already
            # be in the transcript (appended before run_async); insert above them.
            block = self._transcript_store.insert_tracked(0, block_lines)
            self._header_block = {
                "block": block,
                "width": width,
                "source": source,
            }
        else:
            # Subsequent renders (e.g. /config changed the model/mode): replace
            # the existing top-of-transcript block in place instead of appending
            # a duplicate banner mid-conversation.
            self._transcript_store.replace_tracked(
                self._header_block["block"], block_lines
            )
            self._header_block["width"] = width
            self._header_block["source"] = source
        self._transcript_revision = self._transcript_store.revision
        self._scroll_transcript_to_bottom()
        self.invalidate()

    def append_user(self, text: str) -> None:
        if not text.strip():
            return
        width = self._transcript_render_width or term_width()
        # one blank (non-bg) line separates the gray user band from the next render,
        # matching the muted spacer convention used by every other transcript block
        self._append_transcript(*user_message_lines(text, width=width))
        self._append_transcript(TranscriptLine("class:transcript.muted", ""))
        self._layout_term_width = width

    # ------------------------------------------------------------------ #
    # Resize + reflow machinery                                          #
    # ------------------------------------------------------------------ #
    def _on_transcript_render_width(self, width: int) -> None:
        self._on_transcript_render_size(
            width, self._transcript_viewport_height or self._transcript_viewport_lines()
        )

    def _on_transcript_render_size(self, width: int, height: int) -> None:
        self._transcript_render_width = width
        self._transcript_viewport_height = height
        if width > 0:
            self._transcript_ready.set()
        if self._transcript_store.set_width(width):
            # Re-flow the tracked header block at the new width so the
            # rounded dashboard / logo don't wrap into a half-screen
            # artifact on terminal shrink (and re-tighten on grow).
            self._reflow_header_for_width(width)
            if self._follow_transcript:
                self._scroll_transcript_to_bottom()

    def _reflow_header_for_width(self, width: int) -> None:
        """Regenerate the tracked header block at ``width`` if it changed.

        Mirrors ``_reflow_user_blocks_for_width`` but for the single tracked
        header block: rebuilds its ``TranscriptLine`` rows from the stored
        source kwargs at the new width and ``replace``s it in place. Guards
        keep this a no-op when no header has been rendered yet or the width
        is unchanged / below the render threshold.
        """
        block = self._header_block
        if block is None or width == block["width"]:
            return
        from ness_cli.tui.header import header_lines

        rows = header_lines(width=width, show_logo=width >= 96, **block["source"])
        if rows:
            new_lines = [*rows, TranscriptLine("class:transcript.muted", "")]
        else:
            new_lines = [
                *_tagged_lines(
                    "session",
                    f"model {block['source']['model']}  "
                    f"approval {'on' if block['source']['approval'] else 'off'}",
                ),
                TranscriptLine("class:transcript.muted", ""),
            ]
        self._transcript_store.replace_tracked(block["block"], new_lines)
        block["width"] = width
        self._transcript_revision = self._transcript_store.revision

    def _after_render(self) -> None:
        width = self._transcript_render_width
        if width <= 0:
            return
        if not self._transcript_store.has_user_blocks:
            self._layout_term_width = width
            self._user_fit_checked_upto = 0
            return
        if width != self._layout_term_width:
            self._reflow_user_blocks_for_width(width)
            return
        # Width unchanged: lines before ``_user_fit_checked_upto`` were already
        # validated at this width and user rows are never mutated in place
        # (only append_user creates them, reflow rebuilds them), so only the
        # appended tail can be unvalidated. Scanning just the tail keeps this
        # per-frame check O(new lines) instead of O(transcript).
        start = min(self._user_fit_checked_upto, len(self._lines))
        if self._user_blocks_fit_width(width, start=start):
            self._user_fit_checked_upto = len(self._lines)
            return
        self._reflow_user_blocks_for_width(width)

    def _expected_user_band_width(self, width: int | None = None) -> int:
        from ness_cli.tui.formatting import user_band_width

        return user_band_width(width=width if width is not None else term_width())

    def _user_blocks_fit_width(self, width: int, *, start: int = 0) -> bool:
        expected = self._expected_user_band_width(width)
        index = max(0, min(start, len(self._lines)))
        # Back up to the containing block's first row (the one carrying
        # ``user_source``) so a mid-block ``start`` doesn't skip band rows.
        while (
            0 < index < len(self._lines)
            and self._lines[index].style == USER_STYLE
            and self._lines[index].user_source is None
        ):
            index -= 1
        while index < len(self._lines):
            line = self._lines[index]
            if line.user_source is None:
                index += 1
                continue
            end = index + 1
            while end < len(self._lines) and self._lines[end].style == USER_STYLE:
                end += 1
            for row in self._lines[index:end]:
                if len(row.text) != expected:
                    return False
            index = end
        return True

    def _reflow_user_blocks_for_width(self, width: int) -> None:
        if width == self._layout_term_width and self._user_blocks_fit_width(width):
            self._user_fit_checked_upto = len(self._lines)
            return

        follow = self._follow_transcript
        old_scroll = (
            self._transcript_pane.vertical_scroll if self._transcript_pane else 0
        )

        replacements: list[tuple[int, int, list[TranscriptLine]]] = []
        index = 0
        while index < len(self._lines):
            line = self._lines[index]
            if line.user_source is None:
                index += 1
                continue

            end = index + 1
            while end < len(self._lines) and self._lines[end].style == USER_STYLE:
                end += 1

            replacements.append(
                (index, end - index, user_message_lines(line.user_source, width=width))
            )
            index = end

        # Work backwards so each source range remains valid while the store
        # shifts every tracked block that follows the replaced user band.
        for start, count, new_lines in reversed(replacements):
            self._transcript_store.replace(start, count, new_lines)
        self._transcript_revision = self._transcript_store.revision
        self._layout_term_width = width
        # Every user row was just rebuilt at ``width``: the whole buffer is
        # validated, so the per-frame tail scan resumes from the end.
        self._user_fit_checked_upto = len(self._lines)

        self.invalidate()
        if follow:
            self._scroll_transcript_to_bottom()
        elif self._transcript_pane is not None:
            self._transcript_pane.vertical_scroll = min(
                old_scroll, self._max_transcript_scroll()
            )

    # Tagged render sink methods (notice/warning/error/panel/table) ------------
    def append_notice(self, title: str, *lines: str) -> None:
        self._append_transcript(
            *_tagged_lines(title, *lines), TranscriptLine("class:transcript.muted", "")
        )

    def append_warning(self, text: str) -> None:
        self._append_transcript(
            *_tagged_lines("warning", str(text)),
            TranscriptLine("class:transcript.muted", ""),
        )

    def append_error(self, text: str) -> None:
        self._append_transcript(
            TranscriptLine(
                style="",
                text=f"[error]    {text}",
                fragments=[
                    ("class:transcript.error", "[error]"),
                    ("class:transcript.tag.body", f"    {text}"),
                ],
            ),
            TranscriptLine("class:transcript.muted", ""),
        )

    def append_panel(self, title: str, *lines: str) -> None:
        self._append_transcript(
            *_tagged_lines(title, *lines), TranscriptLine("class:transcript.muted", "")
        )

    def append_table(
        self, title: str, headers: list[str], rows: list[list[str]]
    ) -> None:
        if not headers:
            return
        render_width = self._transcript_render_width or term_width()
        leading = 2
        separator = "  "
        available = max(
            len(headers) * 4,
            render_width - leading - len(separator) * (len(headers) - 1),
        )
        col_widths = _table_column_widths(headers, rows, available)
        header_line = separator.join(
            str(header).upper().ljust(col_widths[index])
            for index, header in enumerate(headers)
        )
        lines_out = [
            TranscriptLine("class:transcript.notice", title),
            TranscriptLine("class:transcript.panel", f"  {header_line}"),
            TranscriptLine(
                "class:transcript.muted",
                "  " + separator.join("─" * width for width in col_widths),
            ),
        ]
        for row in rows:
            cells = [
                _table_cell_lines(
                    row[index] if index < len(row) else "", col_widths[index]
                )
                for index in range(len(headers))
            ]
            for line_index in range(max(len(cell) for cell in cells)):
                line = separator.join(
                    (cell[line_index] if line_index < len(cell) else "").ljust(
                        col_widths[index]
                    )
                    for index, cell in enumerate(cells)
                ).rstrip()
                lines_out.append(
                    TranscriptLine("class:transcript.panel", " " * leading + line)
                )
        lines_out.append(TranscriptLine("class:transcript.muted", ""))
        self._append_transcript(*lines_out)

    # Assistant markdown rendering -----------------------------------------
    def append_assistant(self, text: str) -> None:
        if not text.strip():
            return
        width = self._transcript_render_width or term_width()
        lines = markdown_transcript_lines(text, width=width)
        rendered = [*lines, TranscriptLine("class:transcript.muted", "")]
        if not self._turn_render_active:
            self._append_transcript(*rendered)
            return
        self._release_last_assistant_block()
        self._last_assistant_block = self._transcript_store.append_tracked(rendered)
        self._transcript_revision = self._transcript_store.revision
        self._scroll_transcript_to_bottom()
        self.invalidate()

    def _release_last_assistant_block(self) -> None:
        block = self._last_assistant_block
        if block is not None and block.attached:
            self._transcript_store.release_tracked(block)
        self._last_assistant_block = None

    def _finalize_turn_assistant_order(self) -> None:
        """Keep the turn's final non-empty response below late tool rows."""
        block = self._last_assistant_block
        if block is None or not block.attached:
            self._last_assistant_block = None
            return
        following = self._lines[block.start + block.count :]
        has_later_tool = any(
            line.style.startswith("class:transcript.tool")
            or any(
                style.startswith("class:transcript.tool")
                for style, _text in (line.fragments or [])
            )
            for line in following
        )
        if has_later_tool:
            self._transcript_store.move_tracked_to_end(block)
            self._transcript_revision = self._transcript_store.revision
            self._scroll_transcript_to_bottom()
            self.invalidate()
        self._release_last_assistant_block()

    def begin_turn(self) -> None:
        self._release_last_assistant_block()
        self._turn_render_active = True
        super().begin_turn()

    def finish_turn(self) -> None:
        try:
            self._finalize_turn_assistant_order()
        finally:
            self._turn_render_active = False
            super().finish_turn()

    def _reasoning_block_for_span(
        self, span: dict, *, expanded: bool | None = None
    ) -> list[TranscriptLine]:
        width = self._transcript_render_width or term_width()
        if expanded is None:
            expanded = self._show_reasoning
        return _reasoning_block_lines(
            span.get("text", ""),
            elapsed=float(span.get("elapsed", 0.0)),
            expanded=expanded,
            width=width,
        )

    # Reasoning slot lifecycle ---------------------------------------------
    def reserve_reasoning_slot(
        self, before_stream: TuiAssistantStream | None = None
    ) -> dict:
        """Insert a ``Thinking…`` placeholder above the live assistant stream.

        Called from the turn renderer on the first reasoning chunk of an LLM
        call. The placeholder sits before the assistant stream's tracked block
        (if it already reserved one) or at the current transcript end.
        """
        anchor = len(self._transcript_store.lines)
        if before_stream is not None and before_stream.block is not None:
            anchor = before_stream.block.start
        placeholder = TranscriptLine(
            _REASONING_COLLAPSED_STYLE,
            " Thinking…",
            fragments=[(_REASONING_COLLAPSED_STYLE, " Thinking…")],
        )
        block = self._transcript_store.insert_tracked(anchor, [placeholder])
        span = {"block": block, "text": "", "elapsed": 0.0}
        self._reasoning_spans.append(span)
        self._transcript_revision = self._transcript_store.revision
        self._scroll_transcript_to_bottom()
        self.invalidate()
        return span

    def finalize_reasoning_slot(self, span: dict, text: str, *, elapsed: float) -> None:
        span["text"] = text
        span["elapsed"] = float(elapsed)
        new_lines = self._reasoning_block_for_span(span)
        self._transcript_store.replace_tracked(span["block"], new_lines)
        self._transcript_revision = self._transcript_store.revision
        self._scroll_transcript_to_bottom()
        self.invalidate()

    def toggle_reasoning(self) -> None:
        """Flip ``_show_reasoning`` and re-emit every reasoning span.

        Stable block handles are shifted by ``TranscriptStore`` whenever an
        earlier span changes size, including any active assistant stream.
        """
        self._show_reasoning = not self._show_reasoning
        spans = sorted(self._reasoning_spans, key=lambda s: s["block"].start)
        if not spans:
            self.invalidate()
            return
        for span in spans:
            new_lines = self._reasoning_block_for_span(span)
            self._transcript_store.replace_tracked(span["block"], new_lines)
        self._transcript_revision = self._transcript_store.revision
        self._scroll_transcript_to_bottom()
        self.invalidate()

    def append_reasoning(self, text: str, *, elapsed: float) -> None:
        """Append a finalized reasoning block at the end of the transcript.

        Used on the cancel-finalize path where there is no live assistant
        stream to interleave above; the block simply appends.
        """
        if not text.strip():
            return
        span = {"text": text, "elapsed": float(elapsed)}
        new_lines = self._reasoning_block_for_span(span)
        span["block"] = self._transcript_store.append_tracked(new_lines)
        self._reasoning_spans.append(span)
        self._transcript_revision = self._transcript_store.revision
        self._scroll_transcript_to_bottom()
        self.invalidate()

    @staticmethod
    def _assistant_stream_lines(text: str) -> list[TranscriptLine]:
        # Live-stream stays plain (cheap, smooth incremental paint); the finalized
        # markdown styling is swapped in by finalize_assistant_stream on completion.
        if not text:
            return [TranscriptLine("class:transcript.assistant", "")]
        return [
            TranscriptLine("class:transcript.assistant", part)
            for part in text.split("\n")
        ]

    # Live assistant streaming ----------------------------------------------
    def set_assistant_stream(
        self,
        text: str,
        block: TranscriptBlock | None,
    ) -> TranscriptBlock:
        lines = self._assistant_stream_lines(text)
        if block is None:
            block = self._transcript_store.append_tracked(lines)
        else:
            self._transcript_store.replace_tracked(block, lines)
        self._transcript_revision = self._transcript_store.revision
        self._scroll_transcript_to_bottom()
        self.invalidate()
        return block

    def finalize_assistant_stream(
        self, text: str, block: TranscriptBlock | None
    ) -> None:
        if block is None:
            self.append_assistant(text)
            return
        stripped = text.strip()
        if not stripped:
            self.clear_assistant_stream(block)
            return
        width = self._transcript_render_width or term_width()
        final_lines = markdown_transcript_lines(stripped, width=width)
        self._transcript_store.replace_tracked(
            block, [*final_lines, TranscriptLine("class:transcript.muted", "")]
        )
        if self._turn_render_active:
            self._release_last_assistant_block()
            self._last_assistant_block = block
        else:
            self._transcript_store.release_tracked(block)
        self._transcript_revision = self._transcript_store.revision
        self._scroll_transcript_to_bottom()
        self.invalidate()

    def clear_assistant_stream(self, block: TranscriptBlock | None) -> None:
        if block is None:
            return
        self._transcript_store.delete_tracked(block)
        self._transcript_revision = self._transcript_store.revision
        self._scroll_transcript_to_bottom()
        self.invalidate()

    # Transcript reset / clear ---------------------------------------------
    def _sync_transcript_buffer(
        self, *, scroll: bool = True, invalidate_ui: bool = True
    ) -> None:
        self._transcript_store.reset([])
        self._transcript_revision = self._transcript_store.revision
        if scroll:
            self._scroll_transcript_to_bottom()
        if invalidate_ui:
            self.invalidate()

    def clear_transcript(self) -> None:
        # Reset every index that points into the old transcript before the
        # store revision changes. Session switches, rollback, and /new all
        # share this path.
        self._header_block = None
        self._todos_block = None
        self._reasoning_spans = []
        self._last_assistant_block = None
        self._user_fit_checked_upto = 0
        self._sync_transcript_buffer()

    # Tool calls & results -------------------------------------------------
    def _tool_spacer_line(self) -> TranscriptLine:
        return TranscriptLine("class:transcript.muted", "")

    def _advance_tool_batch(
        self, calls: list[dict[str, Any]], index: int, name: str
    ) -> int:
        if name not in BATCHABLE_TOOL_CALLS:
            return index + 1
        next_index = index + 1
        while (
            next_index < len(calls)
            and str(calls[next_index].get("name") or "?") == name
        ):
            next_index += 1
        return next_index

    def append_tool_calls(self, calls: list[dict[str, Any]]) -> None:
        if not calls:
            return
        lines_out: list[TranscriptLine] = []
        index = 0
        while index < len(calls):
            call = calls[index]
            name = str(call.get("name") or "?")
            args = call.get("args") if isinstance(call.get("args"), dict) else {}

            if not should_show_tool_call(name):
                index = self._advance_tool_batch(calls, index, name)
                continue

            if name in BATCHABLE_TOOL_CALLS:
                batch = [calls[index]]
                next_index = index + 1
                while (
                    next_index < len(calls)
                    and str(calls[next_index].get("name") or "?") == name
                ):
                    batch.append(calls[next_index])
                    next_index += 1
                args_text = format_batched_tool_args(name, batch)
                lines_out.extend(
                    [self._tool_call_line(name, args_text), self._tool_spacer_line()]
                )
                index = next_index
                continue

            if name == "todo":
                index += 1
                continue

            args_text = format_tool_args(name, args)
            parts = args_text.splitlines() or [""]
            for part in parts:
                lines_out.append(self._tool_call_line(name, part))
            lines_out.append(self._tool_spacer_line())
            index += 1
        self._append_transcript(*lines_out)

    def _tool_call_line(self, name: str, args_text: str) -> TranscriptLine:
        prefix = "→ "
        sep = "   " if args_text else ""
        text = f"{prefix}{name}{sep}{args_text}".rstrip()
        fragments = [("class:transcript.tool", f"{prefix}{name}")]
        if args_text:
            fragments.append(("class:transcript.tool.args", f"{sep}{args_text}"))
        return TranscriptLine("", text, fragments=fragments)

    def append_tool_result(
        self, name: str, content: str, *, exit_status: str | None = None
    ) -> None:
        if not should_show_tool_result(name, content, exit_status=exit_status):
            return

        preview = format_tool_result_preview(name, content)
        prefix = (
            f"  [{exit_status}] " if exit_status and exit_status != "ok" else "  └ "
        )
        body = prefix + preview
        self._append_transcript(
            TranscriptLine(
                "class:transcript.tool.result",
                body,
                fragments=[("class:transcript.tool.result", body)],
            )
        )

    def append_usage(self, usage: dict[str, Any]) -> None:
        if not usage:
            self.invalidate()
            return
        input_tokens = int(usage.get("input_tokens") or 0)
        output_tokens = int(usage.get("output_tokens") or 0)
        cached = int(usage.get("cached_input_tokens") or 0)
        cache_write = int(usage.get("cache_write_input_tokens") or 0)
        parts = [f"↑ {input_tokens:,}", f"↓ {output_tokens:,}"]
        if input_tokens:
            parts.append(f"⟳ {cached:,} ({cached / input_tokens:.0%})")
        if cache_write:
            parts.append(f"cache write {cache_write:,}")
        cost = usage.get("cost_usd")
        if cost is not None and float(cost) > 0:
            parts.append(f"${float(cost):.4f}")
        body = "  ".join(parts)
        self._append_transcript(
            TranscriptLine(
                "class:chrome.stats.value",
                body,
                fragments=[("class:chrome.stats.value", body)],
            ),
            TranscriptLine("class:transcript.muted", ""),
        )

    # Todos / diff / shell output ------------------------------------------
    def append_todos(self, todos: list[dict]) -> None:
        if not todos:
            if self._todos_block is not None:
                self._transcript_store.delete_tracked(self._todos_block)
                self._todos_block = None
                self._transcript_revision = self._transcript_store.revision
                self.invalidate()
            return

        width = self._transcript_render_width or term_width()
        lines = [
            *todos_transcript_lines(todos, width=width),
            TranscriptLine("class:transcript.muted", ""),
        ]
        # Active list: always sit at the end of the transcript, not pinned mid-history.
        if self._todos_block is not None:
            self._transcript_store.delete_tracked(self._todos_block)
        self._todos_block = self._transcript_store.append_tracked(lines)
        self._transcript_revision = self._transcript_store.revision
        self._scroll_transcript_to_bottom()
        self.invalidate()

    def append_diff(self, diff_text: str, *, title: str = "diff") -> None:
        self._append_transcript(*_diff_transcript_lines(diff_text, title))

    def append_shell_output(self, content: str) -> None:
        from ness_cli.tui.tool_display import format_shell_output

        header, body = format_shell_output(content)
        title = f"shell {header}".strip()
        body_lines = body.splitlines() if body.strip() else ["(no output)"]
        self._append_transcript(*_shell_output_lines(title, body_lines))

    def append_subagent_output(self, content: str) -> None:
        from ness_cli.tui.tool_display import format_subagent_output

        header, body = format_subagent_output(content)
        body_lines = body.splitlines() if body.strip() else ["(no output)"]
        lines = _shell_output_lines(header, body_lines)
        # Prefer the dedicated subagent summary style for body rows.
        styled = [lines[0]]
        for line in lines[1:-1]:
            styled.append(
                TranscriptLine(
                    "class:transcript.subagent.summary",
                    line.text,
                    fragments=[("class:transcript.subagent.summary", line.text)],
                )
            )
        if lines:
            styled.append(lines[-1])
        self._append_transcript(*styled)

    # Streaming adapters ---------------------------------------------------
    def start_assistant_stream(self) -> TuiAssistantStream:
        return TuiAssistantStream(self)

    def thinking(self, label: str = "thinking") -> Thinking:
        return Thinking(self, label)

    # Transcript buffer helpers (read/append primitives) -------------------
    def _append_transcript(self, *lines: TranscriptLine) -> None:
        if not lines:
            return
        self._transcript_store.append(list(lines))
        self._transcript_revision = self._transcript_store.revision
        self._scroll_transcript_to_bottom()
        self.invalidate()

    # ------------------------------------------------------------------ #
    # Layout sizing + scroll navigation                                   #
    # ------------------------------------------------------------------ #
    def _chrome_height_lines(self) -> int:
        lines = 4 + self._input_row_count()
        if self._working_status_visible():
            lines += 1
        if self._queue_line_visible():
            lines += 1
        if self._form_visible():
            lines += 2
            if self._form_example:
                lines += 1
        if self._menu_header_fragments():
            lines += 1
        lines += self._menu_body_height()
        return lines

    def _transcript_viewport_lines(self) -> int:
        if self._transcript_viewport_height > 0:
            return self._transcript_viewport_height
        return max(1, term_height() - self._chrome_height_lines())

    def _max_transcript_scroll(self) -> int:
        return self._transcript_store.max_scroll(self._transcript_viewport_lines())

    def _set_transcript_scroll(self, value: int) -> None:
        if self._transcript_pane is not None:
            self._transcript_pane.vertical_scroll = max(
                0, min(value, self._max_transcript_scroll())
            )

    def _scroll_transcript_to_bottom(self) -> None:
        if self._follow_transcript and self._transcript_pane is not None:
            self._transcript_pane.vertical_scroll = self._max_transcript_scroll()

    def _scroll_transcript_by(self, delta: int) -> None:
        if self._transcript_pane is None:
            return
        self._transcript_pane.vertical_scroll = max(
            0,
            min(
                self._max_transcript_scroll(),
                self._transcript_pane.vertical_scroll + delta,
            ),
        )
        if delta < 0:
            self._follow_transcript = False
        elif self._transcript_pane.vertical_scroll >= self._max_transcript_scroll():
            self._follow_transcript = True

    def _scroll_transcript_to_top(self) -> None:
        if self._transcript_pane is not None:
            self._transcript_pane.vertical_scroll = 0
            self._follow_transcript = False

    def _resume_transcript_follow(self) -> None:
        self._follow_transcript = True
        self._scroll_transcript_to_bottom()
