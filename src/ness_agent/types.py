from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Awaitable

@dataclass
class UsageEvent:
    model: str
    input_tokens: int
    uncached_input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    cost_usd: float | None
    calls: int = 1
    cache_write_input_tokens: int = 0


def aggregate_usage(events: list[UsageEvent]) -> UsageEvent | None:
    """Sum per-call :class:`UsageEvent` values into one turn-level snapshot.

    ``model`` is the sole model when every event agrees, otherwise ``\"*\"``.
    ``cost_usd`` is the sum of known costs, or ``None`` when no event reported cost.
    """
    if not events:
        return None
    models = {e.model for e in events}
    cost_vals = [e.cost_usd for e in events if e.cost_usd is not None]
    return UsageEvent(
        model=next(iter(models)) if len(models) == 1 else "*",
        input_tokens=sum(e.input_tokens for e in events),
        uncached_input_tokens=sum(e.uncached_input_tokens for e in events),
        cached_input_tokens=sum(e.cached_input_tokens for e in events),
        output_tokens=sum(e.output_tokens for e in events),
        cost_usd=sum(cost_vals) if cost_vals else None,
        calls=sum(e.calls for e in events),
        cache_write_input_tokens=sum(e.cache_write_input_tokens for e in events),
    )


@dataclass(frozen=True)
class SessionEvent:
    kind: Literal[
        "assistant_delta",
        "assistant_final",
        "tool_start",
        "tool_end",
        "usage",
        "approval_required",
        "question_required",
        "compaction",
        "error",
        "warning",
        "interrupted",
        "plan_turn",
    ]
    data: dict[str, Any]

@dataclass(frozen=True)
class RunResult:
    assistant_message: str
    todos: list[dict[str, Any]]
    events: list[SessionEvent]
    usage_total: UsageEvent | None = None
    """Sum of every LLM call in the turn. ``None`` when no usage was reported."""


@dataclass(frozen=True)
class ContextPreview:
    """Snapshot of the L0–L2 system prefix and prospective L3 overlay.

    Produced by :meth:`~ness_agent.session.Session.preview_context` without
    running the model or compaction LLM. Useful for debugging prompt shape.
    """

    system_message: str
    """Cached L0 + L1 + L2 stable prefix (what becomes the SystemMessage)."""

    overlay: str
    """Joined L3 section bodies (before ``<system-reminder>`` wrapping)."""

    overlay_sections: dict[str, str]
    """Named L3 sections from the overlay provider (empty sections omitted)."""

    overlay_reminder: str
    """``overlay`` wrapped with :func:`~ness_agent.context.overlay.wrap_system_reminder`."""

    mode: str
    """Mode used for this preview (``\"act\"`` or ``\"plan\"``)."""


class ApprovalHandler(ABC):
    """Abstract handler for tool-use approval decisions.

    Return one of:
      "yes"     - allow this one call
      "no"      - deny this one call
      "always"  - allow and persist a permanent allow rule
      "session" - allow and persist an allow rule for this session only
      "never"   - deny and persist a permanent deny rule
    """

    @abstractmethod
    async def __call__(self, tool: str, args: dict) -> str: ...


QuestionHandler = Callable[[list[dict]], Awaitable[list[dict]]]

# Per-Session runtime hooks (bound on the Session instance, not on the shared
# NessAgentConfig). They let a coding-domain adapter thread its state
# (plan autosave, interruption text)
# back into the domain-agnostic turn loop without the SDK knowing about it.
# Called at the end of a SUCCESSFUL plan-mode turn with the assistant text.
# Used by the adapter to autosave the plan file. 
# Interrupted plan turns do NOT fire this hook — they flow through InterruptHandler / the interrupted SessionEvent so there is exactly one interrupt path.
PlanTurnHandler = Callable[[str], None]

# Called on interruption with the captured partial assistant text; 
# returns the text the adapter wants surfaced on the interrupted SessionEvent
# (None/falsy keeps the original). Default behaviour (when the hook is None)
# is for the SDK to emit the interruption_marker AIMessage itself.
InterruptHandler = Callable[[str], str]
