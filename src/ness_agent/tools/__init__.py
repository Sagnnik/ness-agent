from __future__ import annotations
from collections.abc import Iterable
from typing import Any
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from ness_agent.tools.ask import question
from ness_agent.tools.discover import add_tools, search_tools
from ness_agent.tools.fs import delete, edit, glob, is_git_repo, read, write
from ness_agent.tools.search import grep
from ness_agent.tools.shell import shell as shell_tool
from ness_agent.tools.skill import skill_view
from ness_agent.tools.subagents import spawn_subagent
from ness_agent.tools.todo import todo
from ness_agent.tools.web import fetch_url, web_search

BUILTIN_TOOLS = [
    read, 
    write, 
    delete, 
    edit, 
    glob, 
    grep, 
    web_search, 
    fetch_url,
    shell_tool,
    todo,
    search_tools,
    add_tools,
    spawn_subagent,
    question,
    skill_view,
]

TOOL_MAP = {tool.name: tool for tool in BUILTIN_TOOLS}
TOOL_NAMES = list(TOOL_MAP)

ALWAYS_ON = {"todo", "question", "skill_view"}
CORE = {"read", "write", "delete", "edit", "grep", "web_search", "fetch_url", "glob", "shell"}
DISCOVERY = {"search_tools", "add_tools"}
ADVANCED = {"spawn_subagent"}
READ_ONLY_TOOLS = {
    "read", 
    "grep", 
    "web_search", 
    "fetch_url", 
    "glob", 
    "todo",
    "search_tools",
    "add_tools",
    "spawn_subagent",
    "question",
    "skill_view",
}

EDIT_TOOLS = frozenset({"write", "delete", "edit"})
DESTRUCTIVE_TOOLS = set(EDIT_TOOLS) | {"shell"}

TOOL_CATALOG_GROUPS = (
    ("Always-on", frozenset(ALWAYS_ON)),
    ("Core", frozenset(CORE)),
    ("Tool discovery", frozenset(DISCOVERY)),
    ("Advanced", frozenset(ADVANCED)),
)
FULL_TOOL_SET = set(ALWAYS_ON) | set(CORE) | set(DISCOVERY) | set(ADVANCED)


class ToolCatalogState:
    """Project-wide tool definitions/catalog shared by registry views."""

    def __init__(self, tools: Iterable[BaseTool]) -> None:
        self.all_tools: list[BaseTool] = list(tools)
        self.tool_map: dict[str, BaseTool] = {t.name: t for t in self.all_tools}
        self.mcp_catalog: dict[str, dict[str, Any]] = {}
        self.generation = 0


class ToolRegistry:
    """Bound tool set with optional MCP hot-rebind.

    Tool definitions and the MCP catalog are shared across sessions on a
    :class:`~ness_agent.agent.NessAgent`; activation and binding caches are
    session-local views.
    """

    def __init__(
        self, 
        tools: Iterable[BaseTool] | None = None,
        *, 
        include: Iterable[str] | None = None,
        _catalog_state: ToolCatalogState | None = None,
        _active_mcp_tools: Iterable[str] | None = None,
    ) -> None:
        """Bind a set of tools, optionally filtering by name.

        Args:
            tools: Tool instances. When ``None``, defaults to :const:`BUILTIN_TOOLS`.
            include: If given, only tools whose names appear in this iterable
                     are activated. The full set remains available for later
                     activation via :meth:`activate_mcp` or :meth:`register_dynamic`.
        """
        initial_tools = list(tools) if tools is not None else list(BUILTIN_TOOLS)
        self._catalog_state = _catalog_state or ToolCatalogState(initial_tools)
        # Compatibility aliases for SDK callers that inspect these attributes.
        self._all_tools = self._catalog_state.all_tools
        self._tool_map = self._catalog_state.tool_map
        self._mcp_catalog = self._catalog_state.mcp_catalog
        self._include: set[str] | None = set(include) if include else None
        self.active_mcp_tools: set[str] = set(_active_mcp_tools or ())
        self._generation = 0
        self.runtime: dict[str, Any] = {}
        self._sync(force=True)

    def fork_for_session(self) -> "ToolRegistry":
        """Return a registry view with shared catalog and local activation."""
        return ToolRegistry(
            include=list(self._include) if self._include is not None else None,
            _catalog_state=self._catalog_state,
            _active_mcp_tools=self.active_mcp_tools,
        )

    def _runtime_generation(self) -> tuple[int, int]:
        return self._catalog_state.generation, self._generation

    def _sync(self, force: bool = False) -> None:
        generation = self._runtime_generation()
        if self.runtime and not force and self.runtime.get("generation") == generation:
            return
        if self._include is not None:
            active = [t for t in self._all_tools if t.name in self._include]
        else:
            # Built-in full set plus this registry's activated MCP tools.
            wanted = FULL_TOOL_SET | self.active_mcp_tools
            active = [t for t in self._all_tools if t.name in wanted]
        active = self._dedupe(active)
        self.runtime["active_tools"] = active
        self.runtime["tool_map"] = {t.name: t for t in active}
        self.runtime["tool_names"] = sorted(self.runtime["tool_map"])
        self.runtime["generation"] = generation

    @property
    def active_tools(self) -> list[BaseTool]:
        """The current list of active tool instances (lazy-synced)."""
        self._sync()
        return self.runtime["active_tools"]

    def tool_map(self) -> dict[str, BaseTool]:
        """Map of tool name → tool instance for active tools (lazy-synced)."""
        self._sync()
        return self.runtime["tool_map"]

    def tool_names(self) -> list[str]:
        """Sorted list of active tool names (lazy-synced)."""
        self._sync()
        return list(self.runtime["tool_names"])

    def all_tools(self) -> list[BaseTool]:
        """Stable snapshot of every registered built-in and dynamic tool."""
        return self._dedupe(self._all_tools)

    def deferred_tool_names(self) -> set[str]:
        """Registered dynamic tools not enabled in the active tool map."""
        active = set(self.tool_names())
        return {tool.name for tool in self.all_tools() if tool.name not in active}

    def bind_model(self, model: BaseChatModel) -> BaseChatModel:
        """Bind the currently active tools to *model* and return it.

        The returned model is ready for use with langgraph.
        """
        self._sync()
        registry_binder = getattr(model, "bind_tool_registry", None)
        if callable(registry_binder):
            return registry_binder(self)
        return model.bind_tools(self.runtime["active_tools"])

    def sync(self) -> None:
        """Force a re-synchronisation of the active tool set."""
        self._sync(force=True)

    def generation(self) -> int:
        """Current generation counter — incremented on every structural change."""
        return self._catalog_state.generation + self._generation

    def bump_generation(self) -> int:
        """Force-increment the generation counter (e.g. after a dynamic update)."""
        self._generation += 1
        return self._generation

    def tool_catalog_groups(self) -> list[tuple[str, set[str]]]:
        """Return tiered tool groupings for prompt rendering.

        Each entry is ``(label, set_of_tool_names)``. Groups with no
        active tools are omitted.
        """
        self._sync()
        groups = [
            (label, set(names) & set(self.runtime["tool_names"]))
            for label, names in TOOL_CATALOG_GROUPS
        ]
        groups.append(("Loaded MCP tools", set(self.active_mcp_tools) & set(self.runtime["tool_names"])))
        return [(l, g) for l, g in groups if g]

    def mcp_catalog(self) -> dict[str, dict[str, Any]]:
        """The full MCP server catalog loaded by the session."""
        return self._mcp_catalog

    def set_mcp_catalog(self, catalog: dict[str, dict[str, Any]] | None) -> None:
        """Replace the MCP catalog (clears previous entries)."""
        self._mcp_catalog.clear()
        self._mcp_catalog.update(catalog or {})
        self._catalog_state.generation += 1

    def deferred_mcp_summary(self) -> str:
        """List MCP servers with deferred tools.

        Server lines + deferred count + description (or a short sample of
        tool names). Full schemas stay deferred — models discover tools via
        ``search_tools`` / ``add_tools``. Returns ``""`` when nothing is
        deferred.
        """
        if not self._mcp_catalog:
            return ""

        _desc_max = 100
        server_lines: list[str] = []
        for server in sorted(self._mcp_catalog):
            info = self._mcp_catalog[server]
            deferred = [
                e
                for e in info.get("tools", [])
                if e.get("name") not in self.active_mcp_tools
            ]
            if not deferred:
                continue
            desc = str(info.get("description") or "").strip().replace("\n", " ")
            if not desc:
                sample = [str(e.get("tool") or "") for e in deferred][:4]
                desc = ", ".join(t for t in sample if t)
            if len(desc) > _desc_max:
                desc = desc[:_desc_max].rstrip() + "..."
            suffix = f": {desc}" if desc else ""
            server_lines.append(
                f"  - mcp__{server}__* ({len(deferred)} tool(s)){suffix}"
            )

        if not server_lines:
            return ""
        header = "- Available MCP servers (use search_tools to find, add_tools to load):"
        return "\n".join([header, *server_lines])

    def register_dynamic(self, tools: Iterable[BaseTool]) -> None:
        """Register dynamically loaded tool instances (e.g. from MCP servers).

        New tools join the pool as *known but inactive* — activation is a
        separate step (:meth:`activate_mcp`, or the model-facing ``add_tools``
        discover tool), so startup can register every MCP tool without binding
        them all into the model's active set.
        """
        for t in tools:
            if t.name not in self._tool_map:
                self._all_tools.append(t)
            self._tool_map[t.name] = t
        self._catalog_state.generation += 1

    def activate_mcp(self, names: Iterable[str]) -> tuple[list[str], list[str]]:
        """Activate MCP tools by name.

        Returns ``(added, unknown)`` — tools successfully activated and
        tools that were not found in the tool map or are not MCP tools.
        """
        added, unknown = [], []
        for name in names:
            if name not in self._tool_map or not name.startswith("mcp__"):
                unknown.append(name)
                continue
            if name not in self.active_mcp_tools:
                self.active_mcp_tools.add(name)
                added.append(name)

        if added:
            # With an explicit include list, activation must extend it; with
            # the default (None) set, ``_sync`` already honours
            # ``active_mcp_tools`` and clobbering it with a small include
            # list would unbind every local tool.
            if self._include is not None:
                self._include |= set(added)
            self.bump_generation()

        return added, unknown

    def deactivate_mcp(self, names: Iterable[str]) -> tuple[list[str], list[str]]:
        """Deactivate MCP tools by name (leave them registered for re-activation).

        Returns ``(removed, unknown)`` — tools successfully deactivated and
        names that were not active MCP tools on this registry.
        """
        removed, unknown = [], []
        for name in names:
            if name not in self.active_mcp_tools:
                if name not in self._tool_map or not name.startswith("mcp__"):
                    unknown.append(name)
                continue
            self.active_mcp_tools.discard(name)
            if self._include is not None:
                self._include.discard(name)
            removed.append(name)

        if removed:
            self.bump_generation()

        return removed, unknown

    def is_destructive(self, name: str, args: dict) -> bool:
        """Return ``True`` if a tool invocation may modify state."""
        if name == "shell":
            action = str(args.get("action") or "run").strip().lower()
            return action in {"run", "start", "kill"}
        return name in DESTRUCTIVE_TOOLS or name.startswith("mcp__")

    def is_read_only(self, name: str, args: dict) -> bool:
        """Return ``True`` if a tool invocation is read-only."""
        if name == "shell":
            action = str(args.get("action") or "run").strip().lower()
            return action in {"jobs", "read"}
        return name in READ_ONLY_TOOLS

    def _dedupe(self, tools):
        seen, out = set(), []
        for t in tools:
            if not t.name or t.name in seen:
                continue
            seen.add(t.name)
            out.append(t)
        return out


def coding_tools(*, include: list[str] | None = None) -> ToolRegistry:
    """Convenience factory for selecting a subset of SDK tools by name.

    Example::

        agent = NessAgent(
            model=...,
            tools=coding_tools(include=["read", "grep", "glob"]),
            prompt=...,
        )

    Args:
        include: Tool names to include. When ``None``, all SDK tools are active.
    """
    return ToolRegistry(BUILTIN_TOOLS, include=include)
