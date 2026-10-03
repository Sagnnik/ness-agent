from __future__ import annotations

import pytest
from prompt_toolkit.utils import get_cwidth

from ness_cli.tui.header import header_lines
from ness_cli.tui.transcript.store import TranscriptLine


def _text(lines: list[TranscriptLine]) -> str:
    return "\n".join(line.text for line in lines)


def _assert_fragments_match(lines: list[TranscriptLine]) -> None:
    for line in lines:
        if line.fragments is not None:
            assert "".join(text for _, text in line.fragments) == line.text


def _kwargs(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "mode": "act",
        "model": "openrouter/z-ai/glm-5.3-flash",
        "approval": True,
        "project": "/home/sagnnik/projects/liteharness",
        "addons_summary": "1 server, 1 tool (echo)",
        "version": "0.2.3",
    }
    values.update(overrides)
    return values


def test_header_mode_labels_preserve_plan_and_yolo_states():
    plan = header_lines(
        **_kwargs(mode="plan", approval=False, width=100, show_logo=True)
    )
    yolo = header_lines(**_kwargs(approval=False, yolo=True, width=100, show_logo=True))

    assert "Mode    : Plan" in _text(plan)
    assert "auto-approval" not in _text(plan)
    assert "Mode    : Act (yolo)" in _text(yolo)


@pytest.mark.parametrize(
    ("width", "value"),
    [
        (40, "界" * 40),
        (40, "a\tb" * 20),
        (80, "e\u0301" * 80),
        (80, "🙂" * 40),
        (120, "界" * 40),
        (120, "a\tb" * 20),
    ],
)
def test_header_uses_cells_for_unicode_clipping_padding_and_borders(width, value):
    lines = header_lines(
        **_kwargs(
            width=width,
            show_logo=width >= 96,
            model=value,
            project=value,
            addons_summary=value,
            version=value,
        )
    )
    assert all(get_cwidth(line.text) == width for line in lines)
    assert all("\t" not in line.text for line in lines)
    _assert_fragments_match(lines)
    panel_rows = [line.text for line in lines if line.text.count("│") == 3]
    assert len(panel_rows) == 2
    border_columns = [
        [get_cwidth(text[:index]) for index, char in enumerate(text) if char == "│"]
        for text in panel_rows
    ]
    assert border_columns[0] == border_columns[1]
    assert border_columns[0][-1] == width - 1
