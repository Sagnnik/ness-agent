from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import BaseMessage
from ness_agent.graph.state import AgentState

@dataclass(frozen=True)
class OverlayContext:
    """Read-only inputs available each turn to ``OverlayProvider.sections()``.

    Assembled by the graph node from the current ``AgentState``, session
    config, and runtime snapshot. Fields are **read-only** — providers
    should not mutate them.

    .. highlight:: python

    ``thread_id`` : str
        Session thread identifier.
    ``mode`` : str
        Current mode — ``"act"`` or ``"plan"``.
    ``messages`` : list[BaseMessage]
        The conversation (after compaction, before the current turn).
    ``todos`` : list[dict]
        Current todo list (each entry has ``id``, ``content``, ``status``).
    ``session_memory`` : str
        Episodic reflection bullets for this thread, one per line.
    ``compaction_note`` : str
        Human-readable compaction/pressure note.
    ``mode_switch`` : str
        Non-empty on the first act turn after a plan→act toggle.
    ``metadata`` : Mapping[str, Any]
        Arbitrary key-value pairs set via ``session.metadata`` before
        each ``run()`` — the hook for domain-specific data (retrieval
        summaries, render queues, SLA timers, …).
    ``git_snapshot`` : str
        Git branch + dirty-status summary (empty when not in a repo).
    ``git_available`` : bool or None
        Whether a git repo was detected at session start.
    ``requested_skills`` : list[str]
        Skill names requested this turn (one-shot, cleared after).
    ``skill_catalog`` : str
        Complete effective catalog when it must be injected this turn.
    """

    thread_id: str
    mode: str
    messages: list[BaseMessage]
    todos: list[dict[str, Any]]
    session_memory: str
    compaction_note: str
    mode_switch: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    git_snapshot: str = ""
    git_available: bool | None = None
    requested_skills: list[str] = field(default_factory=list)
    skill_catalog: str = ""

    @property
    def activate_skills(self) -> list[str]:
        """Compatibility name for requested_skills."""
        return self.requested_skills

class OverlayProvider(ABC):
    """Abstract base for building internal L3 sections injected each turn.

    Subclass and implement :meth:`sections`. Custom overlays must inherit
    this class — duck-typed objects are rejected at agent construction.

    L3 tails are retained in canonical model context for cache continuity,
    but excluded from semantic messages and durable events. Implementations receive the current graph ``AgentState`` and an
    :class:`OverlayContext` snapshot and return a dict of
    ``section_name → rendered_text``.

    Key design rules
    ----------------
    *   Section **names are stable keys** — the harness tracks deltas
        per-section so that unchanged sections can be skipped on
        consecutive tool-loop turns, reducing token waste.
    *   **Empty sections** (falsy text) are skipped entirely by the
        renderer.
    *   The harness calls **every turn**, including tool-loop iterations
        within a single user message. Use ``ctx.messages`` or the
        ``state`` to detect fresh turns vs. tool loops if you need
        different behaviour (e.g. full overlay on fresh, delta on loop).

    Built-in implementations
    ------------------------
    *   :class:`~ness_agent.context.coding_overlay.CodingOverlay` —
        shipped with the SDK as the default. Provides plan/act mode
        blocks, git snapshot, compaction note, todos, session memory,
        skill-load hints. Covers most coding-agent needs.
    *   :class:`~ness_agent.context.coding_overlay.NoOverlay` —
        renders nothing. Pass it as ``overlay=NoOverlay()`` to opt out
        of L3 entirely.
    """

    @abstractmethod
    def sections(self, state: AgentState, ctx: OverlayContext) -> dict[str, str]:
        """Build the L3 sections for this turn.

        Parameters
        ----------
        state : AgentState
            The current langgraph agent state (messages, todos, mode,
            approval flags, …).
        ctx : OverlayContext
            Pre-computed snapshot of runtime context for this turn.

        Returns
        -------
        dict[str, str]
            Ordered mapping of ``section_name → rendered_text``.
            Empty/falsy values are treated as absent.
            Section names must be **consistent across turns** so that
            the delta renderer can compare old and new content.
        """

def render_overlay_delta(
    sections: dict[str, str],
    previous: dict[str, str],
    *,
    skip: frozenset[str] = frozenset(),
) -> str:
    """Compute the delta between the current and previous overlay sections.

    Only sections whose content differs from the previous turn are
    included (unchanged sections are skipped to save tokens). Certain
    sections (e.g. ``plan_mode``) can be force-reset each turn via the
    ``skip`` set.

    Parameters
    ----------
    sections : dict[str, str]
        Current turn's sections (from ``OverlayProvider.sections``).
    previous : dict[str, str]
        Previous turn's sections (stored on ``NodesRuntime._last_sections``).
    skip : frozenset[str]
        Section names to always skip (rendered once on fresh turns,
        excluded on tool-loop deltas).

    Returns
    -------
    str
        The delta text, sections joined by ``\\n\\n``.
    """
    parts = []
    for name, text in sections.items():
        if name in skip: 
            continue
        if text.strip() and text != previous.get(name, ""): 
            parts.append(text)
    return "\n\n".join(parts)

def wrap_system_reminder(body: str) -> str:
    """Wrap a string in ``<system-reminder>…</system-reminder>`` tags.

    Used by the harness to inject the overlay delta into the model's
    context at the end of the last message.
    """
    body = body.strip()
    return f"<system-reminder>\n{body}\n</system-reminder>" if body else ""
