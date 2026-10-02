from __future__ import annotations

from ness_agent.context.overlay import OverlayContext, OverlayProvider
from ness_agent.graph.state import AgentState
from ness_agent.instructions import PLAN_MODE, ACT_MODE


def _render_todos(todos: list[dict] | None) -> str:
    """Render non-completed todos as ``- [status] id: content`` lines."""
    if not todos:
        return ""
    active = [t for t in todos if t.get("status") != "completed"]
    if not active:
        return ""
    return "\n".join(
        f"- [{t.get('status', 'pending')}] {t.get('id', '')}: {t.get('content', '')}"
        for t in active
    )


class CodingOverlay(OverlayProvider):
    """Default L3 overlay for coding agents.

    Renders these sections in insertion order (empty sections are skipped by
    the renderer):

    - ``skill_catalog`` — complete effective catalog when first sent or refreshed
    - ``skill_request`` — one-shot hint to load skills requested this turn
    - ``mode_switch`` — one-shot, on the first act turn after a plan->act toggle
    - ``plan_mode`` — plan-mode only; a ``<plan-mode>`` block with plan instructions
    - ``git`` — git branch + dirty status snapshot
    - ``compaction`` — compaction status and pressure note
    - ``todos`` — only when there are non-completed items
    - ``session_memory`` — distilled episodic bullets for this thread
    """

    def __init__(
        self,
        *,
        plans_dir: str = ".ness/plans/",
        plan_mode_template: str | None = None,
        act_mode_template: str | None = None,
    ) -> None:
        self.plans_dir = plans_dir
        self.plan_mode_template = plan_mode_template
        self.act_mode_template = act_mode_template

    def _resolve_plan_mode(self) -> str:
        return self.plan_mode_template if self.plan_mode_template is not None else PLAN_MODE

    def _resolve_act_mode(self) -> str:
        return self.act_mode_template if self.act_mode_template is not None else ACT_MODE

    def sections(self, state: AgentState, ctx: OverlayContext) -> dict[str, str]:
        """Build the coding-specific L3 overlay for this turn.

        Rendered sections (in insertion order; empty ones are skipped):

        ``skill_catalog``
            Complete effective catalog on the first turn, after an access
            change, and when model context is rebuilt.

        ``skill_request``
            One-shot hint when ``ctx.requested_skills`` is non-empty.
            Cleared after a single turn.

        ``mode_switch``
            One-shot, emitted on the first act turn after a plan→act
            toggle. Contains the act-mode instructions
            (:attr:`act_mode_template` or
            :data:`~ness_agent.instructions.ACT_MODE`).

        ``plan_mode``
            Only rendered when ``ctx.mode == "plan"``. Wraps
            the plan-mode instructions in a ``<plan-mode path="…">``
            block.

        ``git``
            Git branch / dirty-status snapshot from ``ctx.git_snapshot``.

        ``compaction``
            Compaction status + pressure note from
            ``ctx.compaction_note``.

        ``todos``
            Non-completed items from ``ctx.todos``, one per line.

        ``session_memory``
            Episodic reflection bullets from ``ctx.session_memory``.

        Parameters
        ----------
        state : AgentState
            Current langgraph agent state (messages, todos, mode, …).
        ctx : OverlayContext
            Pre-computed snapshot of runtime context for this turn.

        Returns
        -------
        dict[str, str]
            Section name → rendered text mapping, in insertion order.
        """
        sections: dict[str, str] = {}
        mode = (ctx.mode or "act").lower()

        if ctx.skill_catalog.strip():
            sections["skill_catalog"] = ctx.skill_catalog.strip()

        req = list(ctx.requested_skills or [])
        if req:
            names = ", ".join(f'"{n}"' for n in req)
            sections["skill_request"] = (
                "SKILL REQUEST\n"
                f"The user intended to use these skills: {names}. "
                f"Load each with the skill_view tool. "
                f"Call skill_view(name=<skill-name>) to load the full skill content."
            )

        if ctx.mode_switch.strip():
            sections["mode_switch"] = self._resolve_act_mode()

        if mode == "plan":
            tmpl = self._resolve_plan_mode()
            sections["plan_mode"] = f'<plan-mode path="{self.plans_dir}">\n{tmpl}\n</plan-mode>'

        if ctx.git_snapshot.strip():
            sections["git"] = "GIT\n" + ctx.git_snapshot.strip()

        if ctx.compaction_note.strip():
            sections["compaction"] = "COMPACTION\n" + ctx.compaction_note.strip()

        todos = _render_todos(ctx.todos)
        if todos.strip():
            sections["todos"] = "TODOS\n" + todos.strip()

        if ctx.session_memory.strip():
            sections["session_memory"] = "SESSION MEMORY\n" + ctx.session_memory.strip()

        return sections


class NoOverlay(OverlayProvider):
    """An overlay that renders no sections — opt-out for apps that want no L3."""

    def sections(self, state: AgentState, ctx: OverlayContext) -> dict[str, str]:
        return {}
