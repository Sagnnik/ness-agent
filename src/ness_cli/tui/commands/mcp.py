"""MCP catalog and deferred-tool activation command."""

from __future__ import annotations

from ness_cli.tui.context import CommandContext

_TOOL_COUNT_WARNING = 30


async def mcp_command(context: CommandContext, args: str) -> None:
    parts = args.split()
    if not parts:
        context.renderer.notice("mcp", *context.runtime.mcp.status().splitlines())
        return

    server = parts[0]
    catalog = context.runtime.mcp.catalog()
    server_info = catalog.get(server)
    if server_info is None:
        context.renderer.error(f"Unknown MCP server: {server}. Use /mcp for status.")
        return

    entries = list(server_info.get("tools") or ())
    if len(parts) > 2:
        context.renderer.error("Usage: /mcp [server [tool]]")
        return
    if len(parts) == 2:
        short_name = parts[1]
        wanted = [
            str(entry.get("name"))
            for entry in entries
            if entry.get("tool") == short_name and entry.get("name")
        ]
        if not wanted:
            context.renderer.error(
                f"Unknown tool '{short_name}' on MCP server '{server}'."
            )
            return
    else:
        wanted = [str(entry.get("name")) for entry in entries if entry.get("name")]

    added, unknown = context.session.activate_mcp_tools(wanted)
    if added:
        context.renderer.notice(
            "mcp",
            f"Loaded {len(added)} tool(s) from {server}: {', '.join(sorted(added))}",
        )
    else:
        context.renderer.notice("mcp", f"No new tools loaded from {server}.")
    if unknown:
        context.renderer.warning(
            f"Skipped unknown tools: {', '.join(sorted(set(unknown)))}"
        )
    total = len(context.session.active_tool_names())
    if total > _TOOL_COUNT_WARNING:
        context.renderer.warning(
            f"{total} tools are loaded; tool-selection accuracy may degrade."
        )
