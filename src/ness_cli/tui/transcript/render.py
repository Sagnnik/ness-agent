"""Small formatting helpers for transcript content."""

from __future__ import annotations

import io
import textwrap
from collections.abc import Mapping, Sequence
from typing import Any

from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

from ness_cli.terminal import cell_width, clip_cells
from ness_cli.tui.header import header_lines
from ness_cli.tui.tool_display import (
    bounded_diff,
    extract_diff_section,
    extract_edit_summary,
    format_batched_tool_args,
    format_shell_output,
    format_subagent_output,
    format_tool_args,
    is_tool_result_error,
    should_show_tool_result,
)
from ness_cli.tui.transcript.markdown import markdown_lines
from ness_cli.tui.transcript.store import TranscriptLine


def _lines(style: str, text: str) -> list[TranscriptLine]:
    parts = str(text).splitlines() or [""]
    return [TranscriptLine(style, part) for part in parts]


def spacer() -> TranscriptLine:
    return TranscriptLine("class:transcript.muted", "")


def user_message(text: str, *, width: int = 80) -> list[TranscriptLine]:
    """Render user input as a padded, full-width transcript band."""
    stripped = str(text).strip()
    if not stripped:
        return []

    band_width = max(1, int(width))
    horizontal_padding = min(2, max(0, band_width - 1))
    wrap_width = max(1, band_width - horizontal_padding)
    body: list[str] = []
    for paragraph in stripped.splitlines() or [""]:
        if not paragraph.strip():
            body.append("")
            continue
        body.extend(
            textwrap.wrap(
                paragraph,
                width=wrap_width,
                break_long_words=True,
                break_on_hyphens=False,
            )
            or [""]
        )

    def band_row(content: str = "") -> TranscriptLine:
        line = content[:band_width].ljust(band_width)
        return TranscriptLine(
            "class:transcript.user",
            line,
            fragments=[("class:transcript.user", line)],
        )

    output = [band_row()]
    output.extend(band_row((" " * horizontal_padding) + line) for line in body)
    output.extend((band_row(), spacer()))
    return output


def assistant_message(
    text: str,
    *,
    streaming: bool = False,
    width: int = 80,
) -> list[TranscriptLine]:
    style = (
        "class:transcript.assistant.streaming"
        if streaming
        else "class:transcript.assistant"
    )
    output = (
        _lines(style, text.rstrip())
        if streaming
        else markdown_lines(text, width=width)
    )
    output.append(spacer())
    return output


def reasoning_message(
    text: str,
    *,
    elapsed: float,
    expanded: bool = False,
    width: int = 80,
) -> list[TranscriptLine]:
    marker = "-" if expanded else "+"
    output = [
        TranscriptLine(
            "class:transcript.reasoning.label",
            f"{marker} reasoning {elapsed:.1f}s",
        )
    ]
    if expanded and text.strip():
        for line in markdown_lines(text, width=width):
            output.append(
                TranscriptLine(
                    "class:transcript.reasoning",
                    line.text,
                    fragments=line.fragments,
                )
            )
        output.append(spacer())
    return output


def tool_call(name: str, arguments: Mapping[str, Any]) -> list[TranscriptLine]:
    detail = format_tool_args(name, dict(arguments))
    text = f"› {_tool_label(name)}"
    if detail:
        text += f"  {detail}"
    return [TranscriptLine("class:transcript.tool", text)]


def batched_tool_call(
    name: str,
    arguments: list[dict[str, Any]],
) -> list[TranscriptLine]:
    detail = format_batched_tool_args(name, arguments)
    count = f" ×{len(arguments)}" if len(arguments) > 1 else ""
    text = f"› {_tool_label(name)}{count}"
    if detail:
        text += f"  {detail}"
    return [TranscriptLine("class:transcript.tool", text)]


def tool_result(
    name: str,
    content: str,
    *,
    exit_status: str | None,
) -> list[TranscriptLine]:
    if not should_show_tool_result(name, content, exit_status=exit_status):
        return []
    key = name.lower()
    if key in {"edit", "write"}:
        return _edit_result(name, content, exit_status=exit_status)
    if key == "shell":
        title, body = format_shell_output(content, exit_status=exit_status)
        return _panel_result(title, body)
    if key == "spawn_subagent":
        title, body = format_subagent_output(content)
        return _panel_result(title, body)

    failed = is_tool_result_error(content) or bool(
        exit_status and exit_status not in {"ok", "0"}
    )
    style = "class:transcript.error" if failed else "class:transcript.tool.result"
    status = f" · {exit_status}" if exit_status and exit_status != "ok" else ""
    output = [
        TranscriptLine(
            "class:transcript.tool.result.label",
            f"↳ {_tool_label(name)}{status}",
        )
    ]
    cleaned = _bounded_result(content)
    if cleaned:
        output.extend(_lines(style, cleaned))
    output.append(spacer())
    return output


def todos(items: Sequence[Mapping[str, Any]]) -> list[TranscriptLine]:
    if not items:
        return []
    markers = {
        "completed": "[x]",
        "in_progress": "[~]",
        "pending": "[ ]",
        "cancelled": "[-]",
    }
    output = [TranscriptLine("class:transcript.todo.title", "todos")]
    for item in items:
        status = str(item.get("status") or "pending")
        content = str(item.get("content") or "").strip()
        marker = markers.get(status, "[ ]")
        style = (
            "class:transcript.todo.completed"
            if status in {"completed", "cancelled"}
            else "class:transcript.todo"
        )
        output.append(TranscriptLine(style, f"  {marker} {content}".rstrip()))
    output.append(spacer())
    return output


def command_help(
    rows: Sequence[tuple[str, str]], *, width: int
) -> list[TranscriptLine]:
    table = Table(
        box=box.SIMPLE_HEAD,
        show_edge=False,
        padding=(0, 1),
        collapse_padding=True,
    )
    table.add_column("COMMAND")
    table.add_column("DESCRIPTION")
    for command, description in rows:
        table.add_row(Text(command), Text(description))
    console = Console(file=io.StringIO(), width=max(8, width), color_system=None)
    output = [TranscriptLine("class:transcript.notice.title", "commands")]
    for segments in console.render_lines(table, pad=False):
        output.append(
            TranscriptLine(
                "class:transcript.help",
                "".join(segment.text for segment in segments).rstrip(),
            )
        )
    output.extend(
        [
            spacer(),
            TranscriptLine(
                "class:transcript.notice",
                "[notice]    Shift+Tab toggles plan/act mode.",
                fragments=[
                    ("class:transcript.notice.title", "[notice]"),
                    ("class:transcript.notice", "    Shift+Tab toggles plan/act mode."),
                ],
            ),
            spacer(),
        ]
    )
    return output


def skill_detail(
    *,
    name: str,
    source: str,
    alternate_sources: Sequence[str] = (),
    description: str,
    width: int,
) -> list[TranscriptLine]:
    output = [
        TranscriptLine(
            "class:transcript.skill.title",
            f"skill · {name}",
            fragments=[
                ("class:transcript.skill.title", "skill · "),
                ("class:transcript.skill.name", name),
            ],
        ),
        TranscriptLine(
            "class:transcript.skill.value",
            f"source       {source or '(unknown)'}",
            fragments=[
                ("class:transcript.skill.label", "source       "),
                ("class:transcript.skill.value", source or "(unknown)"),
            ],
        ),
    ]
    for alternate in alternate_sources:
        output.append(
            TranscriptLine(
                "class:transcript.skill.value",
                f"also found   {alternate}",
                fragments=[
                    ("class:transcript.skill.label", "also found   "),
                    ("class:transcript.skill.value", alternate),
                ],
            )
        )
    output.extend(
        (
            spacer(),
            TranscriptLine("class:transcript.skill.label", "description"),
        )
    )
    detail = str(description or "").strip() or "(no description)"
    for line in markdown_lines(detail, width=max(8, width - 2)):
        fragments = line.fragments or [(line.style, line.text)]
        output.append(
            TranscriptLine(
                "class:transcript.skill.description",
                "  " + line.text,
                fragments=[
                    ("class:transcript.skill.gutter", "  "),
                    *fragments,
                ],
            )
        )
    output.append(spacer())
    return output


def notice(title: str, lines: Sequence[str]) -> list[TranscriptLine]:
    if title == "notice":
        parts = [part for line in lines for part in str(line).splitlines() or [""]]
        parts = parts or [""]
        tag = "[notice]"
        indent = " " * (len(tag) + 4)
        output: list[TranscriptLine] = []
        for index, body in enumerate(parts):
            if index == 0:
                output.append(
                    TranscriptLine(
                        "",
                        f"{tag}    {body}",
                        fragments=[
                            ("class:transcript.notice.title", tag),
                            ("class:transcript.notice", f"    {body}"),
                        ],
                    )
                )
            else:
                output.append(
                    TranscriptLine("class:transcript.notice", indent + body)
                )
        output.append(spacer())
        return output

    output = [
        TranscriptLine("class:transcript.notice.title", f"info · {title}")
    ]
    for line in lines:
        output.extend(_lines("class:transcript.notice", str(line)))
    output.append(spacer())
    return output


def warning(text: str) -> list[TranscriptLine]:
    return [TranscriptLine("class:transcript.warning", f"warning · {text}"), spacer()]


def error(text: str) -> list[TranscriptLine]:
    return [TranscriptLine("class:transcript.error", f"error · {text}"), spacer()]


def usage(data: Mapping[str, Any]) -> list[TranscriptLine]:
    input_tokens = int(data.get("input_tokens") or 0)
    output_tokens = int(data.get("output_tokens") or 0)
    cached = int(data.get("cached_input_tokens") or 0)
    cache_write = int(data.get("cache_write_input_tokens") or 0)
    cost = data.get("cost_usd")
    parts = [f"↑ {input_tokens:,}", f"↓ {output_tokens:,}"]
    if input_tokens:
        parts.append(f"⟳ {cached:,} ({cached / input_tokens:.0%})")
    if cache_write:
        parts.append(f"cache write {cache_write:,}")
    if cost is not None and float(cost) > 0:
        parts.append(f"${float(cost):.4f}")
    return [
        TranscriptLine("class:transcript.usage", "  ".join(parts)),
        spacer(),
    ]


def header(
    *,
    model: str,
    provider: str,
    project: str,
    thread_id: str,
    version: str,
    mode: str,
    approval: str,
    integrations: str,
    width: int = 80,
) -> list[TranscriptLine]:
    del thread_id
    normalized_approval = str(approval).strip().lower()
    rows = header_lines(
        mode=mode,
        model=f"{provider}/{model}",
        approval=normalized_approval == "on",
        yolo=normalized_approval == "yolo",
        project=project,
        addons_summary=integrations,
        version=version,
        width=width,
        show_logo=width >= 96,
    )
    if not rows:
        approval_label = "yolo" if normalized_approval == "yolo" else normalized_approval
        tag = "[session]"
        body = clip_cells(
            f"    model {provider}/{model}  approval {approval_label}",
            max(0, int(width) - cell_width(tag)),
        )
        text = tag + body
        rows = [
            TranscriptLine(
                "",
                text,
                fragments=[
                    ("class:transcript.tag.session", tag),
                    ("class:transcript.tag.body", body),
                ],
            )
        ]
    return [*rows, spacer()]



def _edit_result(
    name: str,
    content: str,
    *,
    exit_status: str | None,
) -> list[TranscriptLine]:
    failed = is_tool_result_error(content) or bool(
        exit_status and exit_status not in {"ok", "0"}
    )
    summary = extract_edit_summary(content)
    label_style = "class:transcript.error" if failed else "class:transcript.tool.result.label"
    output = [
        TranscriptLine(
            label_style,
            f"↳ {_tool_label(name)}  {summary}".rstrip(),
        )
    ]
    diff = extract_diff_section(content)
    if diff:
        for line in bounded_diff(diff).splitlines():
            output.append(
                TranscriptLine(
                    _diff_style(line),
                    f"  {line}" if line else "",
                )
            )
    output.append(spacer())
    return output


def _panel_result(title: str, content: str) -> list[TranscriptLine]:
    output = [
        TranscriptLine("class:transcript.tool.result.label", f"┌ {title}")
    ]
    lines = content.splitlines() if content else ["(no output)"]
    for line in lines:
        output.append(
            TranscriptLine("class:transcript.tool.result", f"│ {line}")
        )
    output.append(TranscriptLine("class:transcript.tool.result.label", "└"))
    output.append(spacer())
    return output


def _bounded_result(content: str, *, max_lines: int = 40, max_chars: int = 4_000) -> str:
    text = str(content or "").strip()
    lines = text.splitlines()
    output: list[str] = []
    size = 0
    for line in lines[:max_lines]:
        added = len(line) + (1 if output else 0)
        if size + added > max_chars:
            break
        output.append(line)
        size += added
    if len(output) < len(lines):
        output.append(f"... result truncated, {len(lines)} lines total")
    return "\n".join(output)


def _diff_style(line: str) -> str:
    if line.startswith(("+++", "---")):
        return "class:transcript.diff.meta"
    if line.startswith("@@"):
        return "class:transcript.diff.hunk"
    if line.startswith("+"):
        return "class:transcript.diff.add"
    if line.startswith("-"):
        return "class:transcript.diff.del"
    return "class:transcript.diff.context"


def _tool_label(name: str) -> str:
    if name.startswith("mcp__"):
        return name.removeprefix("mcp__").replace("__", "/", 1)
    return name
