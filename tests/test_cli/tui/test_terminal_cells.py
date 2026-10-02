from types import SimpleNamespace

import pytest
from prompt_toolkit.utils import get_cwidth

from ness_cli.terminal import cell_width, clip_cells
from ness_cli.tui.app import TuiApp
from ness_cli.tui.input.menus import MenuItem
from ness_cli.tui.transcript import render


@pytest.mark.parametrize(
    ("value", "width", "ellipsis", "expected"),
    [
        ("abcdef", -1, "…", ""),
        ("abcdef", 0, "…", ""),
        ("abcdef", 1, "…", "…"),
        ("abcdef", 3, "…", "ab…"),
        ("abcdef", 1, "...", "."),
        ("abcdef", 2, "...", ".."),
        ("abcdef", 3, "...", "..."),
        ("abcdef", 5, "...", "ab..."),
        ("abcdef", 3, "", "abc"),
        ("abc", 3, "...", "abc"),
        ("界界界", 4, "…", "界…"),
        ("界界界", 5, "...", "界..."),
        ("界界", 1, "", ""),
        ("🙂🙂🙂", 5, "…", "🙂🙂…"),
        ("e\u0301abcdef", 2, "…", "e\u0301…"),
        ("e\u0301abcdef", 4, "...", "e\u0301..."),
        ("e\u0301", 1, "…", "e\u0301"),
        ("abcdef", 1, "界", ""),
        ("", 0, "…", ""),
    ],
)
def test_clip_by_cells_and_preserve_each_marker_policy(
    value, width, ellipsis, expected
):
    result = clip_cells(value, width, ellipsis=ellipsis)
    assert result == expected
    assert get_cwidth(result) <= max(0, width)


@pytest.mark.parametrize(
    "value,marker", [("abcde", "…"), ("界界abc", "..."), ("e\u0301界🙂abc", "")]
)
def test_clipping_never_exceeds_available_cells(value, marker):
    for width in range(12):
        result = clip_cells(value, width, ellipsis=marker)
        assert get_cwidth(result) <= width
        if get_cwidth(value) <= width:
            assert result == value


@pytest.mark.parametrize(
    ("value", "start", "tab_size", "expected"),
    [
        ("e\u0301界🙂", 0, None, 5),
        ("a\tb", 0, None, 2),
        ("a\tb", 0, 8, 9),
        ("a\tb", 6, 8, 3),
        ("界\tb", 0, 8, 9),
        ("e\u0301\tb", 0, 8, 9),
        ("a\tb\tc", 0, 4, 9),
    ],
)
def test_cell_width_uses_explicit_tab_stops(value, start, tab_size, expected):
    assert cell_width(value, start_column=start, tab_size=tab_size) == expected


def test_clipping_preserves_default_tabs_and_expands_requested_tabs():
    assert clip_cells("a\tb", 2) == "a\tb"
    assert clip_cells("界\tb", 9, tab_size=8) == "界      b"
    assert clip_cells("界\tb", 8, tab_size=8) == "界     …"
    assert clip_cells("界\tb", 8, tab_size=8, ellipsis="...") == "界   ..."
    assert clip_cells("a\tb", 4, tab_size=4, ellipsis="") == "a   "


@pytest.mark.parametrize("tab_size", [0, -1])
def test_tab_stops_require_a_positive_size(tab_size):
    with pytest.raises(ValueError, match="tab_size must be positive"):
        cell_width("tab\t", tab_size=tab_size)
    with pytest.raises(ValueError, match="tab_size must be positive"):
        clip_cells("tab\t", 20, tab_size=tab_size)


def test_input_measurements_keep_unicode_widths_and_prompt_relative_tab_stops():
    app = object.__new__(TuiApp)
    app.input_buffer = SimpleNamespace(text="a\t界")
    app._terminal_size = lambda: (10, 24)
    app.prompt_fragments = lambda: [("", "> ")]
    assert app._input_row_count() == 1
    app.input_buffer.text += "X"
    assert app._input_row_count() == 2
    app.input_buffer.text = "e\u0301" * 8
    assert app._input_row_count() == 1
    app.input_buffer.text = "界" * 5
    assert app._input_row_count() == 2


@pytest.mark.parametrize(
    ("text", "width", "prefix", "expected"),
    [
        ("e\u0301界\tZ", 16, 3, [(0, 3), (0, 4), (0, 4), (0, 6), (0, 8), (0, 9)]),
        ("\tX", 10, 9, [(1, 0), (1, 8), (1, 9)]),
        ("界X", 10, 9, [(1, 0), (1, 2), (1, 3)]),
    ],
)
def test_cursor_positions_keep_tab_and_wide_character_wrapping(
    text, width, prefix, expected
):
    assert (
        TuiApp._input_cursor_positions(text, width=width, prefix_width=prefix)
        == expected
    )


@pytest.mark.parametrize(
    ("selected", "width", "label"),
    [
        (False, 0, "界" * 30 + "e\u0301"),
        (True, 0, "界e\u0301"),
        (True, 2, "界" * 30 + "e\u0301"),
        (False, 2, "界e\u0301"),
        (True, 3, "界" * 30 + "e\u0301"),
        (False, 3, "界e\u0301"),
        (True, 10, "界e\u0301"),
        (False, 10, "界" * 30 + "e\u0301"),
        (True, 28, "界" * 30 + "e\u0301"),
        (False, 28, "界" * 30 + "e\u0301"),
        (True, 29, "界" * 30 + "e\u0301"),
        (False, 29, "界" * 30 + "e\u0301"),
        (True, 80, "界" * 30 + "e\u0301"),
        (False, 80, "界e\u0301"),
    ],
)
def test_slash_menu_aligns_unicode_labels_by_cells(selected, width, label):
    app = object.__new__(TuiApp)
    app._terminal_size = lambda: (width, 24)
    fragments = app._slash_menu_row(
        MenuItem("key", label, "description"), selected=selected
    )
    text = "".join(part for _, part in fragments)
    assert get_cwidth(text) == width
    if width >= 28 + len("description"):
        assert get_cwidth(text[: text.index("description")]) == 28
    else:
        assert "description" not in text


def test_checklist_keeps_ascii_dots_for_unicode_clipping():
    app = object.__new__(TuiApp)
    app._terminal_size = lambda: (20, 24)
    fragments = app._checklist_menu_row(
        MenuItem("key", "界" * 20), selected=True, disabled=False
    )
    text = "".join(part for _, part in fragments)
    assert text.endswith("...")
    assert get_cwidth(text) <= 20


def test_transcript_fallback_keeps_unicode_ellipsis():
    rows = render.header(
        model="界" * 30,
        provider="test",
        project=".",
        thread_id="thread",
        version="v",
        mode="act",
        approval="on",
        integrations="",
        width=20,
    )
    assert rows[0].text.endswith("…")
    assert get_cwidth(rows[0].text) <= 20
