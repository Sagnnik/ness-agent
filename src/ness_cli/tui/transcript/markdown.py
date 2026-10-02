"""Width-aware Markdown rendering for the prompt-toolkit transcript."""

from __future__ import annotations

import io
import re
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.theme import Theme

from ness_cli.tui.transcript.store import TranscriptLine


_RICH_THEME = Theme(
    {
        "markdown.h1": "bold #c6a0f6",
        "markdown.h2": "bold #c6a0f6",
        "markdown.h3": "bold #c6a0f6",
        "markdown.h4": "bold #c6a0f6",
        "markdown.h5": "bold #c6a0f6",
        "markdown.h6": "bold #c6a0f6",
        "markdown.code": "bold #87c095",
        "markdown.block_quote": "italic #8a909c",
        "markdown.item.bullet": "#4aa3df",
        "markdown.item.number": "#4aa3df",
        "markdown.link": "underline #a78bfa",
        "markdown.hr": "#5c626d",
        "markdown.strong": "bold",
        "markdown.em": "italic",
    }
)

_ESC = chr(27)
_BEL = chr(7)
_OSC8_RE = re.compile(
    re.escape(_ESC + "]8;")
    + "[^"
    + _ESC
    + _BEL
    + "]*"
    + "(?:"
    + re.escape(_ESC + chr(92))
    + "|"
    + re.escape(_BEL)
    + ")"
)
_CONSOLE = Console(
    file=io.StringIO(),
    theme=_RICH_THEME,
    force_terminal=True,
    color_system="truecolor",
    legacy_windows=False,
    width=80,
)


def markdown_lines(text: str, *, width: int) -> list[TranscriptLine]:
    """Render Markdown without writing to the terminal owned by prompt-toolkit."""
    stripped = str(text or "").strip()
    if not stripped:
        return [TranscriptLine("class:transcript.assistant", "")]
    try:
        ansi = _render_ansi(stripped, width=width)
        rows = _ansi_rows(ansi)
    except Exception:
        return _plain_lines(stripped)

    output: list[TranscriptLine] = []
    for fragments in rows:
        line_text = "".join(part for _, part in fragments)
        output.append(
            TranscriptLine(
                "class:transcript.assistant",
                line_text,
                fragments=fragments if any(style for style, _ in fragments) else None,
            )
        )
    while output and not output[-1].text:
        output.pop()
    return output or _plain_lines(stripped)


def _render_ansi(text: str, *, width: int) -> str:
    _CONSOLE.width = max(8, width)
    with _CONSOLE.capture() as capture:
        _CONSOLE.print(Markdown(text))
    return _OSC8_RE.sub("", capture.get())


def _ansi_rows(ansi: str) -> list[list[tuple[str, str]]]:
    from prompt_toolkit.formatted_text import ANSI, to_formatted_text
    from prompt_toolkit.formatted_text.utils import split_lines

    parsed = [
        (_fragment_style(fragment), _fragment_text(fragment))
        for fragment in to_formatted_text(ANSI(ansi))
    ]
    return [_coalesce(list(row)) for row in split_lines(parsed)]


def _fragment_style(fragment: Any) -> str:
    return str(fragment[0]) if fragment[0] else ""


def _fragment_text(fragment: Any) -> str:
    return str(fragment[1]) if len(fragment) > 1 else ""


def _coalesce(fragments: list[tuple[str, str]]) -> list[tuple[str, str]]:
    output: list[tuple[str, str]] = []
    for style, text in fragments:
        if not text:
            continue
        if output and output[-1][0] == style:
            output[-1] = (style, output[-1][1] + text)
        else:
            output.append((style, text))
    return output


def _plain_lines(text: str) -> list[TranscriptLine]:
    return [
        TranscriptLine("class:transcript.assistant", line)
        for line in text.splitlines()
    ]
