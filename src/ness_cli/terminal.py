"""Terminal-safe rendering helpers."""

from __future__ import annotations

from prompt_toolkit.utils import get_cwidth


def cell_width(
    value: str, *, start_column: int = 0, tab_size: int | None = None
) -> int:
    """Measure terminal cells using prompt-toolkit's Unicode widths.

    By default tabs have prompt-toolkit's zero width. Set ``tab_size`` to a
    positive integer for tab stops relative to ``start_column``. Input uses 8.
    The result measures one line; it does not account for wrapping or newlines.
    """
    if tab_size is None:
        return get_cwidth(value)
    if tab_size <= 0:
        raise ValueError("tab_size must be positive")
    column = start_column
    for character in value:
        column += (
            tab_size - column % tab_size if character == "\t" else get_cwidth(character)
        )
    return column - start_column


def clip_cells(
    value: str,
    width: int,
    *,
    ellipsis: str = "…",
    tab_size: int | None = None,
) -> str:
    """Clip one line by terminal cells, adding ``ellipsis`` only if needed.

    A marker wider than the available space is itself clipped. An empty marker
    cuts text without an ellipsis. With ``tab_size``, tabs become spaces at the
    specified stops; otherwise they retain their zero-width measurement.
    """
    width = max(0, int(width))
    if tab_size is not None:
        if tab_size <= 0:
            raise ValueError("tab_size must be positive")
        parts: list[str] = []
        column = 0
        for character in value:
            size = cell_width(character, start_column=column, tab_size=tab_size)
            parts.append(" " * size if character == "\t" else character)
            column += size
        value = "".join(parts)
    if cell_width(value) <= width:
        return value
    if width == 0:
        return ""
    marker_width = cell_width(ellipsis)
    if marker_width >= width:
        return _cell_prefix(ellipsis, width)
    return _cell_prefix(value, width - marker_width) + ellipsis


def _cell_prefix(value: str, width: int) -> str:
    output: list[str] = []
    used = 0
    for character in value:
        size = cell_width(character)
        if used + size > width:
            break
        output.append(character)
        used += size
    return "".join(output)


def terminal_safe_text(value: object, *, multiline: bool = False) -> str:
    """Escape terminal controls while preserving readable Unicode."""
    result: list[str] = []

    for character in str(value):
        if multiline and character == "\n":
            result.append(character)
        elif character.isprintable():
            result.append(character)
        else:
            codepoint = ord(character)
            if codepoint <= 0xFF:
                result.append(f"\\x{codepoint:02x}")
            elif codepoint <= 0xFFFF:
                result.append(f"\\u{codepoint:04x}")
            else:
                result.append(f"\\U{codepoint:08x}")

    return "".join(result)
