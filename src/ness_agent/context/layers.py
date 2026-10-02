from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool

from ness_agent.instructions import (
    L0_HARNESS,
    COMPACTION,
    REFLECTION,
    SUBAGENT,
    THREAD_SUMMARY,
    INIT_MEMORY,
)

InstructionSource = str | Path | Callable[[], str]


@dataclass(kw_only=True, eq=False)
class PromptLayersConfig:
    """L0-L2 prompt configuration.

    Pure data container — does not render anything. Hand it to
    :class:`PromptLayers` which assembles the actual system message prefix.
    L3 is **not** here — it is produced each turn by an
    :class:`~ness_agent.context.overlay.OverlayProvider`.

    .. highlight:: python

    **Quick start** — accept every SDK default::

        PromptLayersConfig()

    **Minimal override** — custom L0 only::

        PromptLayersConfig(l0="You're a code reviewer. Be thorough.")

    **From a dict** (non-matching keys ignored)::

        PromptLayersConfig.from_dict({
            "l0": "...",
            "persona": "Expert in Rust async.",
        })

    All fields accept an ``InstructionSource`` — a plain ``str``,
    a ``Path`` to a text file, or a zero-arg callable ``() -> str``.

    **L0 — Harness identity & universal rules**
        ``l0``:
            Harness identity, tool protocol, safety rules. Defaults to
            :data:`ness_agent.instructions.L0_HARNESS`.

    **L1 — Persona & project context**
        ``persona``:
            Role/identity sentence (e.g. ``"Expert in Python backend."``).
        ``include_user_memory``:
            Include ``USER.md`` cross-repo preferences.
        ``include_project_memory``:
            Include ``NESS.md`` project conventions.
        ``include_skill_catalog``:
            Include one-line skill descriptions.

    **L2 — Stable project context**
        ``l2_context``:
            Optional domain/architecture description for the model.
        ``l2_header``:
            Heading for the L2 block (default ``"PROJECT CONTEXT"``).
        ``include_git_line``:
            Append ``"Git repository: yes/no"`` to the L2 block.
    """

    # L0: universal instructions and harness identity
    l0: InstructionSource = L0_HARNESS

    # L1: persona slot
    persona: str = "You are an expert software engineer working inside the user's repository."

    include_user_memory: bool = True
    include_project_memory: bool = True
    include_skill_catalog: bool = True

    # L2: Stable context block
    l2_context: str | InstructionSource | None = None
    l2_header: str = "PROJECT CONTEXT"  # Change it based on project domain
    include_git_line: bool = True       # For coding agent it's True

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "PromptLayersConfig":
        """Build from a dict, ignoring keys that are not recognised fields.

        This is the conversion used internally when ``AgentSpec.prompt``
        is a plain dict. Unknown keys are silently dropped so applications
        can pass extra keys without breaking the config.
        """
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class PromptLayers:
    """L0–L2 prompt assembly engine.

    Combines :class:`PromptLayersConfig` fields into the system message
    prefix that is sent to the model every turn. The result is cached
    by a stable key (persona + tool set + memory hashes + metadata);
    only a structural change (new tool, memory update, …) triggers a
    rebuild. L3 is **not** included — it is assembled each turn by the
    :class:`~ness_agent.context.overlay.OverlayProvider` and injected
    as an ephemeral ``<system-reminder>`` tail.

    Usage::

        layers = PromptLayers(PromptLayersConfig(persona="You are a bot."))
        prefix = layers.build_stable_prefix(
            active_tools,
            user_memory=...,
            project_memory=...,
            git_available=True,
        )
        system_message = SystemMessage(content=prefix)
    """

    def __init__(self, config: PromptLayersConfig) -> None:
        self.config = config
        self._cache: dict = {}

    @classmethod
    def from_dict(cls, d) -> "PromptLayers":
        """Build from a dict (same as ``PromptLayersConfig.from_dict``)."""
        return cls(PromptLayersConfig.from_dict(d))

    def _resolve(self, src: InstructionSource | None) -> str:
        """Resolve an ``InstructionSource`` to a plain string."""
        if src is None:
            return ""
        # callable-> call it and return the result
        if callable(src):
            return src()
        # path-> read the file and return the content
        text = str(src)
        if "\n" not in text and len(text) < 512:
            p = Path(text)
            try:
                if p.exists() and p.is_file():
                    return p.read_text(encoding="utf-8").strip()
            except OSError:
                pass
        # string-> return it as is
        return text 

    def build_l0(self) -> str:
        """Render the L0 context block (harness identity + universal rules)."""
        return self._resolve(self.config.l0)

    def build_l1(
        self, 
        tools: Iterable[BaseTool],
        *, 
        user_memory: str = "",
        project_memory: str = "",
        skill_catalog: str = "",
        tool_catalog_groups: list[tuple[str, frozenset[str]]] | None = None,
        deferred_mcp: str = "",
    ) -> str:
        """Render the L1 context block (persona + tool catalog + memories).

        Parameters
        ----------
        tools : iterable of BaseTool
            The currently active tool instances (used for the tool catalog).
        user_memory : str
            Contents of ``USER.md`` (cross-repo user preferences).
        project_memory : str
            Contents of ``NESS.md`` (project conventions).
        skill_catalog : str
            One-line descriptions of available skills (from ``SkillLoader``).
        tool_catalog_groups : list of (label, frozenset), optional
            Tiered tool grouping labels for the catalog.
        deferred_mcp : str
            Pre-computed deferred-MCP servers blurb (header + per-server
            lines). Appended after the active tool catalog when non-empty.
        """
        persona = self.config.persona
        catalog = _render_tool_catalog(tool_catalog_groups, tools)
        sections = [persona, catalog]

        if deferred_mcp and deferred_mcp.strip():
            sections.append(deferred_mcp.strip())

        if self.config.include_skill_catalog and skill_catalog: 
            sections.append(skill_catalog.strip())
        
        if self.config.include_user_memory and user_memory.strip(): 
            sections.append(_user_memory_section(user_memory))
        
        if self.config.include_project_memory and project_memory.strip(): 
            sections.append(_project_memory_section(project_memory))
        
        return "\n\n".join(s for s in sections if s).strip()

    def build_l2(
        self,
        *,
        git_available: bool | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        """Render the L2 context block (project context + git flag + user data).

        Parameters
        ----------
        git_available : bool, optional
            Whether a git repository is detected. Controls the
            ``"Git repository: yes/no"`` line.
        metadata : dict, optional
            Arbitrary key-value pairs rendered as a
            ``"User added data:"`` list (sorted by key). This is the
            same dict that ``OverlayContext.metadata`` points to —
            metadata set on ``session.metadata`` before each turn
            appears here.
        """
        sections = [self.config.l2_header]

        if self.config.include_git_line and git_available is not None:
            sections.append(f"Git repository: {'yes' if git_available else 'no'}")

        l2 = self.config.l2_context
        if l2:
            sections.append(l2 if isinstance(l2, str) else self._resolve(l2))

        if metadata:
            lines = ["User added data:"]
            for k in sorted(metadata):
                lines.append(f"- {k}: {metadata[k]}")
            sections.append("\n".join(lines))

        return "\n\n".join(s for s in sections if s).strip()

    def build_stable_prefix(
        self,
        tools: Iterable[BaseTool],
        *, 
        user_memory: str,
        project_memory: str,
        skill_catalog: str,
        git_available: bool | None,
        metadata: Mapping[str, Any] | None = None,
        tool_catalog_groups=None, 
        deferred_mcp=""
    ) -> str:
        """Build and cache the full L0 + L1 + L2 prefix.

        This is the main assembly method — it concatenates ``build_l0()``,
        ``build_l1()``, and ``build_l2()`` and caches the result under a
        key derived from tool names, skill catalog, user/project memory
        hashes, tool groups, deferred MCP summary, and metadata. The cache
        is invalidated when any of those inputs change, so the prefix is
        rebuilt automatically on structural updates without manual
        management.

        L3 is deliberately excluded from both the output and the cache
        key — it changes every turn (working state, mode, git diff, …)
        and would bust the cache needlessly.

        Parameters
        ----------
        tools : iterable of BaseTool
            Active tool instances (for the L1 tool catalog).
        user_memory : str
            ``USER.md`` content (cross-repo user preferences).
        project_memory : str
            ``NESS.md`` content (project conventions).
        skill_catalog : str
            One-line skill descriptions from the ``SkillLoader``.
        git_available : bool or None
            Whether a git repository is present.
        metadata : dict, optional
            Key-value pairs rendered in L2 as ``"User added data:"``.
            Affects the cache key so that mid-session metadata mutations
            bust the prefix correctly.
        tool_catalog_groups : list of (label, frozenset), optional
            Tiered tool grouping labels (see ``_render_tool_catalog``).
        deferred_mcp : str
            Pre-computed deferred-MCP servers blurb (rendered into L1).

        Returns
        -------
        str
            The assembled L0–L2 system message prefix.
        """
        key = (
            tuple(sorted(t.name for t in tools)),
            git_available,
            hash(user_memory),
            hash(project_memory),
            hash(skill_catalog) if self.config.include_skill_catalog else 0,
            tool_catalog_groups,
            deferred_mcp,
            _hash_metadata(metadata),
        )
        # check if the cache has the key
        cached = self._cache
        if cached.get("key") == key: 
            return cached["content"]
        
        # if key is different, build the body
        body = "\n\n".join([
            self.build_l0(),
            self.build_l1(
                tools,
                user_memory=user_memory,
                project_memory=project_memory,
                skill_catalog=skill_catalog,
                tool_catalog_groups=tool_catalog_groups,
                deferred_mcp=deferred_mcp,
            ),
            self.build_l2(git_available=git_available, metadata=metadata)
        ]).strip()
        
        # cache the new body and return it
        self._cache = {"key": key, "content": body}
        return body


def _render_tool_catalog(groups, tools) -> str:
    """Render the tool catalog into a string."""
    names = {getattr(t, "name", "") for t in tools if getattr(t, "name", "")}
    lines = []
    if groups:
        for label, tier in groups:
            g = set(names) & set(tier)
            if g: lines.append(f"- {label}: {', '.join(sorted(g))}")
    ungrouped = sorted(n for n in names if not groups or not any(n in g for _, g in groups))
    if ungrouped: lines.append(f"- Other active tools: {', '.join(ungrouped)}")
    return "\n".join(lines) if lines else "- No tools registered"


def _hash_metadata(metadata: Mapping[str, Any] | None) -> int:
    """Stable hash for the metadata slot of the stable-prefix cache key.

    Values may contain unhashable nested structures (lists/dicts), so we
    serialize via JSON with ``sort_keys=True`` and ``default=str`` for any
    non-JSON-native values (paths, datetimes, custom objects, ...).
    """
    if not metadata:
        return 0
    try:
        return hash(json.dumps(metadata, sort_keys=True, default=str))
    except (TypeError, ValueError):
        # If JSON chokes on something truly exotic, fall back to repr so the
        # cache still busts on structural change rather than silently serving
        # a stale prefix.
        return hash(repr(metadata))

def _user_memory_section(text) -> str:
    """Render the user memory section into a string."""
    return (f"User preferences (cross-repo, authored by the user; honor unless they conflict "
            f"with an explicit request in the current turn):\n{text.strip()}")

def _project_memory_section(text) -> str:
    """Render the project memory section into a string."""
    return (f"Project conventions (human-authored, stable; honor unless the current turn "
            f"explicitly overrides):\n{text.strip()}")

@dataclass
class AuxPrompts:
    """Templates for auxiliary model calls (not the main agent loop).

    Each field is an ``InstructionSource`` (``str``, ``Path``, or
    ``() -> str``). When ``None`` the corresponding auxiliary call is
    disabled entirely. Defaults ship with the SDK's built-in instruction
    texts (see :mod:`ness_agent.instructions`).

    .. highlight:: python

    ``compaction``
        Cache-safe tail instruction for conversation summaries. The complete
        conversation is supplied as native messages; no placeholders are used.
    ``reflection``
        Template for background reflection (structured output).
        Receives ``{thread_id}``, ``{messages}``, ``{todos}``,
        ``{current_session_bullets}``, ``{user_message_count}``.
    ``subagent``
        Template for spawned sub-agents.
        Receives ``{agent_name}``, ``{agent_body}``, ``{parent_context}``.
    ``thread_summary``
        Template for session-end thread summarisation.
        Receives ``{events}``.
    ``init_memory``
        Template for initial project memory (``NESS.md``) generation.
        Receives ``{project_context}``.
    """

    compaction: InstructionSource | None = COMPACTION
    reflection: InstructionSource | None = REFLECTION
    subagent: InstructionSource | None = SUBAGENT
    thread_summary: InstructionSource | None = THREAD_SUMMARY
    init_memory: InstructionSource | None = INIT_MEMORY
