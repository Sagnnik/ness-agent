"""Explicit rendering boundary used by the TUI controller."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, replace
from typing import Any, Protocol

from ness_agent import ApprovalHandler

from ness_cli.tui.input import MenuItem
from ness_cli.tui.tool_display import BATCHABLE_TOOL_CALLS
from ness_cli.tui.transcript import (
    TranscriptBlock,
    TranscriptLine,
    TranscriptStore,
)
from ness_cli.tui.transcript import render


class PromptDriver(Protocol):
    async def request_input(
        self,
        label: str,
        *,
        default: str = "",
    ) -> str | None: ...

    async def choose(
        self,
        title: str,
        items: list[MenuItem],
        *,
        initial_key: str | None = None,
        hint: str = "Up/Down select · Enter confirm · Esc cancel",
    ) -> str | None: ...

    async def approve(self, name: str, args: dict) -> str: ...

    async def choose_with_note(
        self,
        title: str,
        items: list[MenuItem],
        *,
        initial_key: str | None = None,
        allow_note: bool = True,
    ) -> tuple[str, str] | None: ...


class RenderSink(Protocol):
    def begin_turn(self) -> None: ...
    def finish_turn(self) -> None: ...
    def user_message(self, text: str) -> None: ...
    def start_assistant(self) -> object: ...
    def assistant_delta(self, handle: object, text: str) -> None: ...
    def assistant_final(self, handle: object, text: str) -> None: ...
    def assistant_message(self, text: str) -> None: ...
    def reasoning(self, handle: object, text: str, *, elapsed: float) -> None: ...
    def append_reasoning(self, text: str, *, elapsed: float) -> None: ...
    def interrupt_assistant(self, handle: object, text: str) -> None: ...
    def tool_call(self, name: str, arguments: dict[str, Any]) -> None: ...
    def tool_result(
        self,
        name: str,
        content: str,
        *,
        exit_status: str | None = None,
    ) -> None: ...
    def todos(self, items: list[dict[str, Any]]) -> None: ...
    def usage(self, data: dict[str, Any]) -> None: ...
    def notice(self, title: str, *lines: str) -> None: ...
    def warning(self, text: str) -> None: ...
    def error(self, text: str) -> None: ...
    def clear(self) -> None: ...
    async def ask_approval(self, name: str, args: dict[str, Any]) -> str: ...
    async def ask_questions(self, questions: list[dict]) -> list[dict]: ...


@dataclass(slots=True)
class _AssistantHandle:
    block: TranscriptBlock
    complete: bool = False


@dataclass(frozen=True, slots=True)
class _ReasoningSource:
    text: str
    elapsed: float


@dataclass(frozen=True, slots=True)
class _HeaderSource:
    model: str
    provider: str
    project: str
    thread_id: str
    version: str
    mode: str
    approval: str
    integrations: str


class TuiRenderSink:
    """Render transcript operations into one store."""

    def __init__(
        self,
        store: TranscriptStore,
        *,
        invalidate,
        prompts: PromptDriver,
    ) -> None:
        self.store = store
        self._invalidate = invalidate
        self._prompts = prompts
        self._turn_assistants: list[_AssistantHandle] = []
        self._turn_active = False
        self._user_sources: dict[TranscriptBlock, str] = {}
        self._assistant_sources: dict[TranscriptBlock, str] = {}
        self._reasoning_sources: dict[TranscriptBlock, _ReasoningSource] = {}
        self._help_sources: dict[TranscriptBlock, tuple[tuple[str, str], ...]] = {}
        self._skill_sources: dict[
            TranscriptBlock, tuple[str, str, tuple[str, ...], str]
        ] = {}
        self._header_source: _HeaderSource | None = None
        self._header_block: TranscriptBlock | None = None
        self._reasoning_expanded = False
        self._todos_block: TranscriptBlock | None = None
        self._tool_batch_name: str | None = None
        self._tool_batch_args: list[dict[str, Any]] = []
        self._tool_batch_block: TranscriptBlock | None = None
        self.assistant_history: list[str] = []

    def append_header(
        self,
        *,
        model: str,
        provider: str,
        project: str,
        thread_id: str,
        version: str,
        mode: str,
        approval: str,
        integrations: str,
    ) -> None:
        source = _HeaderSource(
            model=model,
            provider=provider,
            project=project,
            thread_id=thread_id,
            version=version,
            mode=mode,
            approval=approval,
            integrations=integrations,
        )
        lines = self._render_header(source)
        block = self._header_block
        if block is not None and block.attached:
            self.store.replace_tracked(block, lines)
        else:
            self._header_block = self.store.insert_tracked(0, lines)
        self._header_source = source
        self._changed()

    def set_header_mode(self, mode: str) -> None:
        source = self._header_source
        block = self._header_block
        if source is None or block is None or not block.attached:
            return
        source = replace(source, mode=mode)
        self._header_source = source
        self.store.replace_tracked(block, self._render_header(source))
        self._changed()

    def begin_turn(self) -> None:
        self._close_tool_batch()
        self._turn_assistants = []
        self._turn_active = True

    def finish_turn(self) -> None:
        self._close_tool_batch()
        last = next(
            (
                handle
                for handle in reversed(self._turn_assistants)
                if handle.block.attached
            ),
            None,
        )
        if last is not None:
            self.store.move_tracked_to_end(last.block)
        self._turn_assistants = []
        self._turn_active = False
        self._changed()

    def user_message(self, text: str) -> None:
        self._close_tool_batch()
        stripped = text.strip()
        if not stripped:
            return
        block = self.store.append_tracked(
            render.user_message(stripped, width=self.store.width)
        )
        self._user_sources[block] = stripped
        self._changed()

    def start_assistant(self) -> object:
        handle = _AssistantHandle(
            self.store.append_tracked(
                render.assistant_message(
                    "",
                    streaming=True,
                    width=self.store.width,
                )
            )
        )
        self._turn_assistants.append(handle)
        self._changed()
        return handle

    def assistant_delta(self, handle: object, text: str) -> None:
        stream = self._assistant_handle(handle)
        if stream.complete:
            return
        self.store.replace_tracked(
            stream.block,
            render.assistant_message(
                text,
                streaming=True,
                width=self.store.width,
            ),
        )
        self._changed()

    def assistant_final(self, handle: object, text: str) -> None:
        stream = self._assistant_handle(handle)
        if not text.strip():
            self.store.delete_tracked(stream.block)
            self._assistant_sources.pop(stream.block, None)
            stream.complete = True
            self._changed()
            return
        self.store.replace_tracked(
            stream.block,
            render.assistant_message(text, width=self.store.width),
        )
        self._assistant_sources[stream.block] = text
        self.assistant_history.append(text.strip())
        stream.complete = True
        self._changed()

    def assistant_message(self, text: str) -> None:
        if text.strip():
            self._close_tool_batch()
            lines = render.assistant_message(text, width=self.store.width)
            handle = _AssistantHandle(
                self.store.append_tracked(lines),
                complete=True,
            )
            self._assistant_sources[handle.block] = text
            if self._turn_active:
                self._turn_assistants.append(handle)
            self.assistant_history.append(text.strip())
            self._changed()

    def reasoning(
        self,
        handle: object,
        text: str,
        *,
        elapsed: float,
    ) -> None:
        stream = self._assistant_handle(handle)
        if text.strip() and stream.block.attached:
            self._close_tool_batch()
            block = self.store.insert_tracked(
                stream.block.start,
                render.reasoning_message(
                    text,
                    elapsed=elapsed,
                    expanded=self._reasoning_expanded,
                    width=self.store.width,
                ),
            )
            self._reasoning_sources[block] = _ReasoningSource(text, elapsed)
            self._changed()

    def append_reasoning(self, text: str, *, elapsed: float) -> None:
        if not text.strip():
            return
        self._close_tool_batch()
        block = self.store.append_tracked(
            render.reasoning_message(
                text,
                elapsed=elapsed,
                expanded=self._reasoning_expanded,
                width=self.store.width,
            )
        )
        self._reasoning_sources[block] = _ReasoningSource(text, elapsed)
        self._changed()

    @property
    def reasoning_expanded(self) -> bool:
        return self._reasoning_expanded

    def toggle_reasoning(self) -> bool:
        self._reasoning_expanded = not self._reasoning_expanded
        for block, source in tuple(self._reasoning_sources.items()):
            if not block.attached:
                self._reasoning_sources.pop(block, None)
                continue
            self.store.replace_tracked(
                block,
                render.reasoning_message(
                    source.text,
                    elapsed=source.elapsed,
                    expanded=self._reasoning_expanded,
                    width=self.store.width,
                ),
            )
        self._changed()
        return self._reasoning_expanded

    def interrupt_assistant(self, handle: object, text: str) -> None:
        stream = self._assistant_handle(handle)
        if text.strip():
            final_text = text + " … [interrupted]"
            self.store.replace_tracked(
                stream.block,
                render.assistant_message(final_text, width=self.store.width),
            )
            self._assistant_sources[stream.block] = final_text
            self.assistant_history.append(final_text)
        else:
            self.store.delete_tracked(stream.block)
            self._assistant_sources.pop(stream.block, None)
        stream.complete = True
        self._changed()

    def tool_call(self, name: str, arguments: dict[str, Any]) -> None:
        key = name.lower()
        if (
            key in BATCHABLE_TOOL_CALLS
            and self._tool_batch_name == key
            and self._tool_batch_block is not None
            and self._tool_batch_block.attached
        ):
            self._tool_batch_args.append(dict(arguments))
            self.store.replace_tracked(
                self._tool_batch_block,
                render.batched_tool_call(name, self._tool_batch_args),
            )
            self._changed()
            return

        self._close_tool_batch()
        if key in BATCHABLE_TOOL_CALLS:
            self._tool_batch_name = key
            self._tool_batch_args = [dict(arguments)]
            self._tool_batch_block = self.store.append_tracked(
                render.batched_tool_call(name, self._tool_batch_args)
            )
        else:
            self.store.append(render.tool_call(name, arguments))
        self._changed()

    def tool_result(
        self,
        name: str,
        content: str,
        *,
        exit_status: str | None = None,
    ) -> None:
        self._close_tool_batch()
        lines = render.tool_result(name, content, exit_status=exit_status)
        if lines:
            self.store.append(lines)
            self._changed()

    def todos(self, items: list[dict[str, Any]]) -> None:
        self._close_tool_batch()
        lines = render.todos(items)
        block = self._todos_block
        if block is not None and not block.attached:
            block = None
            self._todos_block = None
        if block is not None:
            self.store.delete_tracked(block)
            self._todos_block = None
        if lines:
            self._todos_block = self.store.append_tracked(lines)
        self._changed()

    def reflow_assistants(self, width: int) -> None:
        self.store.set_width(width)
        changed = False
        if (
            self._header_source is not None
            and self._header_block is not None
            and self._header_block.attached
        ):
            self.store.replace_tracked(
                self._header_block,
                self._render_header(self._header_source),
            )
            changed = True
        for block, source in tuple(self._user_sources.items()):
            if not block.attached:
                self._user_sources.pop(block, None)
                continue
            self.store.replace_tracked(
                block,
                render.user_message(source, width=width),
            )
            changed = True
        for block, source in tuple(self._assistant_sources.items()):
            if not block.attached:
                self._assistant_sources.pop(block, None)
                continue
            self.store.replace_tracked(
                block,
                render.assistant_message(source, width=width),
            )
            changed = True
        for block, source in tuple(self._reasoning_sources.items()):
            if not block.attached:
                self._reasoning_sources.pop(block, None)
                continue
            self.store.replace_tracked(
                block,
                render.reasoning_message(
                    source.text,
                    elapsed=source.elapsed,
                    expanded=self._reasoning_expanded,
                    width=width,
                ),
            )
            changed = True
        for block, rows in tuple(self._help_sources.items()):
            if not block.attached:
                self._help_sources.pop(block, None)
                continue
            self.store.replace_tracked(block, render.command_help(rows, width=width))
            changed = True
        for block, source in tuple(self._skill_sources.items()):
            if not block.attached:
                self._skill_sources.pop(block, None)
                continue
            name, path, alternate_paths, description = source
            self.store.replace_tracked(
                block,
                render.skill_detail(
                    name=name,
                    source=path,
                    alternate_sources=alternate_paths,
                    description=description,
                    width=width,
                ),
            )
            changed = True
        if changed:
            self._changed()

    def command_help(self, rows: tuple[tuple[str, str], ...]) -> None:
        self._close_tool_batch()
        block = self.store.append_tracked(
            render.command_help(rows, width=self.store.width)
        )
        self._help_sources[block] = rows
        self._changed()

    def skill_detail(
        self,
        *,
        name: str,
        source: str,
        alternate_sources: tuple[str, ...] = (),
        description: str,
    ) -> None:
        self._close_tool_batch()
        block = self.store.append_tracked(
            render.skill_detail(
                name=name,
                source=source,
                alternate_sources=alternate_sources,
                description=description,
                width=self.store.width,
            )
        )
        self._skill_sources[block] = (
            name,
            source,
            alternate_sources,
            description,
        )
        self._changed()

    def usage(self, data: dict[str, Any]) -> None:
        self._close_tool_batch()
        self.store.append(render.usage(data))
        self._changed()

    def notice(self, title: str, *lines: str) -> None:
        self._close_tool_batch()
        self.store.append(render.notice(title, lines))
        self._changed()

    def warning(self, text: str) -> None:
        self._close_tool_batch()
        self.store.append(render.warning(text))
        self._changed()

    def error(self, text: str) -> None:
        self._close_tool_batch()
        self.store.append(render.error(text))
        self._changed()

    def clear(self) -> None:
        self.store.reset([])
        self._turn_assistants = []
        self._turn_active = False
        self._user_sources.clear()
        self._assistant_sources.clear()
        self._reasoning_sources.clear()
        self._help_sources.clear()
        self._skill_sources.clear()
        self._header_source = None
        self._header_block = None
        self._reasoning_expanded = False
        self._todos_block = None
        self._tool_batch_name = None
        self._tool_batch_args = []
        self._tool_batch_block = None
        self._changed()

    async def ask_approval(self, name: str, args: dict[str, Any]) -> str:
        approve = getattr(self._prompts, "approve", None)
        if callable(approve):
            return str(await approve(name, args) or "no")
        try:
            preview = json.dumps(args, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError):
            preview = str(args)
        self.notice("approval", f"{name}: {preview}")
        answer = await self._prompts.choose(
            f"approval needed: {name}",
            [
                MenuItem("yes", "Approve once", "Run this tool call."),
                MenuItem(
                    "session",
                    "Approve session",
                    "Allow matching calls for this session.",
                ),
                MenuItem("always", "Always allow", "Persist an allow rule."),
                MenuItem("no", "Deny once", "Skip this tool call."),
                MenuItem("never", "Never allow", "Persist a deny rule."),
            ],
            initial_key="yes",
        )
        return str(answer or "no")

    async def ask_questions(self, questions: list[dict]) -> list[dict]:
        answers: list[dict] = []
        for index, question in enumerate(questions, 1):
            answer = await self._ask_question(index, question)
            answers.append(answer)
            if self._is_cancelled_answer(answer):
                for rest_index, rest in enumerate(
                    questions[index:],
                    index + 1,
                ):
                    answers.append(self._cancelled_answer(rest_index, rest))
                break
        return answers

    async def _ask_question(self, index: int, question: dict) -> dict:
        options = list(question.get("options") or ())
        prompt = str(question.get("prompt") or f"question {index}")
        if options:
            items = [
                MenuItem(
                    str(option.get("id") or option_index),
                    str(option.get("label") or option.get("value") or ""),
                    description=str(option.get("description") or ""),
                    suffix="recommended" if option.get("recommended") else "",
                )
                for option_index, option in enumerate(options)
            ]
            default_key = next(
                (
                    item.key
                    for item, option in zip(items, options)
                    if option.get("recommended")
                ),
                items[0].key,
            )
            note = ""
            choose_with_note = getattr(self._prompts, "choose_with_note", None)
            if callable(choose_with_note):
                choice = await choose_with_note(
                    prompt,
                    items,
                    initial_key=default_key,
                    allow_note=bool(question.get("allow_note", True)),
                )
                if choice is None:
                    return self._cancelled_answer(index, question)
                selected_key, note = choice
            else:
                selected_key = await self._prompts.choose(
                    prompt,
                    items,
                    initial_key=default_key,
                )
                if selected_key is None:
                    return self._cancelled_answer(index, question)
            selected_index = next(
                (
                    option_index
                    for option_index, option in enumerate(options)
                    if str(option.get("id") or option_index) == selected_key
                ),
                -1,
            )
            if selected_index not in range(len(options)):
                return self._cancelled_answer(index, question)
            selected = options[selected_index]
            if not callable(choose_with_note) and question.get("allow_note", True):
                raw_note = await self._prompts.request_input(
                    "note (optional, Enter to skip)"
                )
                if raw_note is None:
                    return self._cancelled_answer(index, question)
                note = raw_note
            return {
                "id": question.get("id", str(index)),
                "selected": {
                    "id": selected.get("id"),
                    "label": selected.get("label"),
                },
                "note": note,
            }

        self.notice(f"question {index}", prompt)
        raw = await self._prompts.request_input("answer")
        if raw is None:
            return self._cancelled_answer(index, question)
        return {
            "id": question.get("id", str(index)),
            "selected": None,
            "note": raw,
        }

    @staticmethod
    def _cancelled_answer(index: int, question: dict) -> dict:
        return {
            "id": question.get("id", str(index)),
            "selected": None,
            "note": "cancelled by user",
        }

    @staticmethod
    def _is_cancelled_answer(answer: dict) -> bool:
        return (
            answer.get("selected") is None
            and answer.get("note") == "cancelled by user"
        )

    def _close_tool_batch(self) -> None:
        block = self._tool_batch_block
        if block is not None and block.attached:
            self.store.release_tracked(block)
        self._tool_batch_name = None
        self._tool_batch_args = []
        self._tool_batch_block = None

    @staticmethod
    def _assistant_handle(handle: object) -> _AssistantHandle:
        if not isinstance(handle, _AssistantHandle):
            raise TypeError("assistant handle belongs to another render sink")
        return handle

    def _render_header(self, source: _HeaderSource) -> list[TranscriptLine]:
        return render.header(
            model=source.model,
            provider=source.provider,
            project=source.project,
            thread_id=source.thread_id,
            version=source.version,
            mode=source.mode,
            approval=source.approval,
            integrations=source.integrations,
            width=self.store.width,
        )

    def _changed(self) -> None:
        self._invalidate()


class SinkApprovalHandler(ApprovalHandler):
    def __init__(self, sink: RenderSink) -> None:
        self._sink = sink

    async def __call__(self, tool: str, args: dict) -> str:
        return await self._sink.ask_approval(tool, args)


async def sink_question_handler(
    sink: RenderSink,
    questions: list[dict],
) -> list[dict]:
    return await sink.ask_questions(questions)


class NullPromptDriver:
    async def request_input(
        self,
        label: str,
        *,
        default: str = "",
    ) -> str | None:
        del label
        await asyncio.sleep(0)
        return default or None

    async def choose(
        self,
        title: str,
        items: list[MenuItem],
        *,
        initial_key: str | None = None,
        hint: str = "Up/Down select · Enter confirm · Esc cancel",
    ) -> str | None:
        del title, items, initial_key, hint
        await asyncio.sleep(0)
        return None

    async def approve(self, name: str, args: dict) -> str:
        del name, args
        await asyncio.sleep(0)
        return "no"

    async def choose_with_note(
        self,
        title: str,
        items: list[MenuItem],
        *,
        initial_key: str | None = None,
        allow_note: bool = True,
    ) -> tuple[str, str] | None:
        del title, items, initial_key, allow_note
        await asyncio.sleep(0)
        return None
