"""Pure adapter from SDK session events to a render sink."""

from __future__ import annotations

import time
from typing import Any

from ness_agent import SessionEvent

from ness_cli.session.events import DurableEvent
from ness_cli.tui.sink import RenderSink


def replay_event(sink: RenderSink, event: DurableEvent) -> None:
    """Render durable history without converting it back to SDK messages."""
    if event.kind == "user":
        sink.user_message(str(event.get("content") or ""))
    elif event.kind == "assistant":
        sink.assistant_message(str(event.get("content") or ""))
    elif event.kind == "reasoning":
        sink.append_reasoning(
            str(event.get("content") or ""),
            elapsed=float(event.get("elapsed") or 0.0),
        )
    elif event.kind == "tool":
        name = str(event.get("tool") or event.get("name") or "tool")
        arguments = event.get("args")
        if isinstance(arguments, dict):
            sink.tool_call(name, arguments)
        sink.tool_result(
            name,
            str(event.get("result") or event.get("content") or ""),
            exit_status=str(event.get("exit") or "") or None,
        )
    elif event.kind == "usage":
        sink.usage(event.as_dict())
    elif event.kind in {"compact", "compaction_llm"}:
        sink.notice("compaction", str(event.get("content") or ""))
    elif event.kind == "reflection":
        sink.notice("reflection", str(event.get("content") or ""))


class TurnRenderer:
    def __init__(self, sink: RenderSink) -> None:
        self._sink = sink
        self._assistant: object | None = None
        self._text: list[str] = []
        self._reasoning: list[str] = []
        self._reasoning_started: float | None = None
        self._usage: dict[str, Any] = {}
        self.interrupted = False

    @property
    def usage(self) -> dict[str, Any]:
        return dict(self._usage)

    def feed(self, event: SessionEvent) -> None:
        data = event.data
        if event.kind == "assistant_delta":
            self._assistant_delta(data)
        elif event.kind == "assistant_final":
            self._assistant_final(data)
        elif event.kind == "tool_start":
            self._sink.tool_call(
                str(data.get("name") or "tool"),
                dict(data.get("args") or {}),
            )
        elif event.kind == "tool_end":
            status = data.get("exit") or data.get("exit_status")
            self._sink.tool_result(
                str(data.get("name") or "tool"),
                str(data.get("content") or ""),
                exit_status=str(status) if status else None,
            )
        elif event.kind == "usage":
            self._accumulate_usage(data)
        elif event.kind == "compaction":
            self._compaction(data)
        elif event.kind == "warning":
            self._sink.warning(str(data.get("message") or data))
        elif event.kind == "error":
            self._sink.error(str(data.get("message") or data))
        elif event.kind == "interrupted":
            self._interrupted(data)

    def finish(self) -> None:
        self._finalize_reasoning()
        if self._assistant is not None:
            text = "".join(self._text)
            self._sink.assistant_final(self._assistant, text)
            self._assistant = None
        self._sink.finish_turn()
        if self._usage and not self.interrupted:
            self._sink.usage(self._usage)

    def close(self) -> None:
        """Release live render handles when a turn exits unexpectedly."""
        self._finalize_reasoning()
        self._sink.finish_turn()

    def _assistant_delta(self, data: dict[str, Any]) -> None:
        if self._assistant is None:
            self._assistant = self._sink.start_assistant()
            self._text = []
            self._reasoning = []
            self._reasoning_started = None
        reasoning = data.get("reasoning")
        if isinstance(reasoning, str) and reasoning:
            if self._reasoning_started is None:
                self._reasoning_started = time.monotonic()
            self._reasoning.append(reasoning)
        text = data.get("text")
        if isinstance(text, str) and text:
            self._text.append(text)
            self._sink.assistant_delta(self._assistant, "".join(self._text))

    def _assistant_final(self, data: dict[str, Any]) -> None:
        authoritative = str(data.get("content") or "")
        if self._assistant is None:
            if authoritative.strip():
                self._sink.assistant_message(authoritative)
            return
        self._finalize_reasoning()
        text = authoritative if authoritative.strip() else "".join(self._text)
        self._sink.assistant_final(self._assistant, text)
        self._assistant = None
        self._text = []

    def _finalize_reasoning(self) -> None:
        if self._assistant is None or not self._reasoning:
            return
        elapsed = time.monotonic() - (self._reasoning_started or time.monotonic())
        self._sink.reasoning(
            self._assistant,
            "".join(self._reasoning),
            elapsed=elapsed,
        )
        self._reasoning = []
        self._reasoning_started = None

    def _accumulate_usage(self, data: dict[str, Any]) -> None:
        self._usage["model"] = data.get("model") or self._usage.get("model")
        for key in (
            "input_tokens",
            "uncached_input_tokens",
            "cached_input_tokens",
            "cache_write_input_tokens",
            "output_tokens",
        ):
            self._usage[key] = int(self._usage.get(key) or 0) + int(data.get(key) or 0)
        if data.get("cost_usd") is not None:
            self._usage["cost_usd"] = float(self._usage.get("cost_usd") or 0.0) + float(
                data["cost_usd"]
            )

    def _compaction(self, data: dict[str, Any]) -> None:
        info = str(data.get("info") or "").strip()
        if data.get("notice_reason") == "pre_act_hard_threshold":
            info = (
                info + " Hard threshold reached; compacting before execution."
            ).strip()
        if info:
            self._sink.notice("compaction", info)

    def _interrupted(self, data: dict[str, Any]) -> None:
        self.interrupted = True
        partial = str(data.get("partial_text") or "").strip()
        if not partial:
            partial = "".join(self._text).strip()
        self._finalize_reasoning()
        if self._assistant is not None:
            self._sink.interrupt_assistant(self._assistant, partial)
            self._assistant = None
        elif partial:
            self._sink.assistant_message(partial + " … [interrupted]")
        self._sink.notice("cancel", "Turn interrupted by user.")
