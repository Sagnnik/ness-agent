"""Per-turn SessionEvent -> render-facade mapping.

One ``TurnRenderer`` per user turn. It consumes the
:class:`~ness_agent.types.SessionEvent` stream produced by
:meth:`ness_cli.CodingSession.run_turn` and drives the ``ness_cli.tui.render``
facade, owning the per-turn render state that used to live in SessionApp's
``astream_events`` dispatch:

- the ``AssistantStream`` lifecycle (opened lazily on the first delta of
  each LLM call, stopped + reasoning-finalised on ``assistant_final``),
- tool call/result routing (todo refreshes, edit/write summary+diff, shell
  output panels),
- per-turn usage accumulation for the footer (the SDK emits one ``usage``
  event per model call; the footer shows the turn total),
- the ``interrupted`` rendering (partial reasoning drain + cancel banner).

Durable persistence, plan autosave, cancel state cleanup, and context
snapshots are NOT done here — the adapter/SDK own those. This module only
renders.
"""

from __future__ import annotations

from typing import Any

from ness_agent.types import SessionEvent

from ness_cli.tui import render
from ness_cli.tui.tool_display import extract_diff_section, extract_edit_summary

#: Suffix appended to recorded assistant text when a turn is interrupted,
#: mirroring the original CLI's convention for /copy.
INTERRUPTED_SUFFIX = " … [interrupted]"


def render_persisted_tool_result(
    name: str,
    content: str,
    *,
    exit_status: str | None = None,
) -> None:
    """Route a tool result the same way live ``tool_end`` events do.

    Used by :class:`TurnRenderer` and resume replay so edit diffs, shell
    panels, and exit markers stay visually faithful.
    """
    if name == "todo":
        render.render_tool_result(name, content, exit_status=exit_status)
        return
    if name in ("edit", "write"):
        summary = extract_edit_summary(content)
        if summary:
            render.render_tool_result(name, summary, exit_status=exit_status)
        diff = extract_diff_section(content)
        if diff:
            render.render_diff(diff, title=f"diff {name}")
        return
    if name == "shell":
        # Non-ok exits (e.g. approval denial) must not use the success-looking
        # shell panel — show the same [denied] / [error] tool-result line.
        if exit_status and exit_status != "ok":
            render.render_tool_result(name, content, exit_status=exit_status)
        else:
            render.render_shell_output(content)
        return
    if name == "spawn_subagent":
        if exit_status and exit_status != "ok":
            render.render_tool_result(name, content, exit_status=exit_status)
        else:
            render.render_subagent_output(content)
        return
    render.render_tool_result(name, content, exit_status=exit_status)


class TurnRenderer:
    """Map one turn's SessionEvent stream onto the render facade."""

    def __init__(self) -> None:
        self._stream: render.AssistantStream | None = None
        self._streamed_any = False
        # Turn-total usage accumulator (one footer per turn, summed over the
        # turn's model calls — matches the original cost-snapshot delta).
        self.usage: dict[str, Any] = {}
        # Completed assistant texts this turn, in order. TuiApp appends
        # these to its assistant_history for /copy.
        self.assistant_texts: list[str] = []
        # Set when an ``interrupted`` event arrived (suppresses the normal
        # footer/todos render at end of turn).
        self.interrupted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def feed(self, ev: SessionEvent) -> None:
        kind = ev.kind
        data = ev.data
        if kind == "assistant_delta":
            self._on_delta(data)
        elif kind == "assistant_final":
            self._on_final(data)
        elif kind == "tool_start":
            render.render_tool_calls(
                [
                    {
                        "name": data.get("name"),
                        "args": data.get("args") or {},
                        "id": data.get("id"),
                        "type": "tool_call",
                    }
                ]
            )
        elif kind == "tool_end":
            self._on_tool_end(data)
        elif kind == "usage":
            self._accumulate_usage(data)
        elif kind == "compaction":
            self._on_compaction(data)
        elif kind == "warning":
            render.render_warning(str(data.get("message") or data))
        elif kind == "error":
            render.render_error(str(data.get("message") or data))
        elif kind == "interrupted":
            self._on_interrupted(data)
        # approval_required / question_required: the config handlers own the
        # interactive UI; the events are informational only.
        # plan_turn: the adapter owns plan autosave; nothing to render.

    # ------------------------------------------------------------------
    # Assistant stream lifecycle
    # ------------------------------------------------------------------

    def _on_delta(self, data: dict[str, Any]) -> None:
        # A fresh stream per LLM call: the SDK emits deltas interleaved with
        # tool execution, so an open stream means "this call still streaming".
        if self._stream is None:
            self._stream = render.AssistantStream()
            self._streamed_any = False
        reasoning = data.get("reasoning")
        if isinstance(reasoning, str) and reasoning:
            self._stream.feed_reasoning(reasoning)
        text = data.get("text")
        if isinstance(text, str) and text:
            self._stream.feed(text)
            self._streamed_any = True

    def _on_final(self, data: dict[str, Any]) -> None:
        self._stop_stream()
        text = str(data.get("content") or "")
        if not text.strip():
            return
        self.assistant_texts.append(text.strip())
        # Non-streaming models (or fully-suppressed streams) never produced
        # deltas; render the final text as a panel so it isn't lost.
        if not self._streamed_any:
            render.render_assistant_panel(text)

    def _stop_stream(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.finalize_reasoning()
            self._stream = None

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    def _on_tool_end(self, data: dict[str, Any]) -> None:
        name = str(data.get("name") or "tool")
        content = str(data.get("content") or "")
        exit_status = data.get("exit") or data.get("exit_status")
        render_persisted_tool_result(
            name,
            content,
            exit_status=str(exit_status) if exit_status else None,
        )

    # ------------------------------------------------------------------
    # Usage / compaction
    # ------------------------------------------------------------------

    def _accumulate_usage(self, data: dict[str, Any]) -> None:
        acc = self.usage
        acc["model"] = data.get("model") or acc.get("model")
        for key in (
            "input_tokens",
            "uncached_input_tokens",
            "cached_input_tokens",
            "cache_write_input_tokens",
            "output_tokens",
        ):
            acc[key] = int(acc.get(key) or 0) + int(data.get(key) or 0)
        cost = data.get("cost_usd")
        if cost:
            acc["cost_usd"] = float(acc.get("cost_usd") or 0.0) + float(cost)

    def _on_compaction(self, data: dict[str, Any]) -> None:
        info = str(data.get("info") or "").strip()
        if data.get("notice_reason") == "pre_act_hard_threshold":
            render.render_notice(
                info + " Hard threshold reached; compacting before execution.",
                title="compaction",
            )
        elif info:
            render.render_notice(info, title="compaction")

    # ------------------------------------------------------------------
    # Interrupt
    # ------------------------------------------------------------------

    def _on_interrupted(self, data: dict[str, Any]) -> None:
        self.interrupted = True
        partial_reasoning: tuple[str | None, float] = (None, 0.0)
        partial_text = ""
        if self._stream is not None:
            # Drain in-flight reasoning before stop() discards live state,
            # mirroring the original cancel finalise order.
            partial_reasoning = self._stream.reasoning_state()
            partial_text = self._stream.text.strip()
            self._stream.stop()
            self._stream = None
        # The SDK's partial_text is authoritative (hook-adjusted); fall back
        # to what the live stream captured when it's empty.
        partial_text = str(data.get("partial_text") or "").strip() or partial_text
        if partial_reasoning[0]:
            render.render_reasoning(
                partial_reasoning[0] + INTERRUPTED_SUFFIX,
                elapsed=partial_reasoning[1],
            )
        if partial_text:
            self.assistant_texts.append(partial_text + INTERRUPTED_SUFFIX)
        render.render_notice("Turn interrupted by user.", title="cancel")
