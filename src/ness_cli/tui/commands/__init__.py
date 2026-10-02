"""Built-in slash command registry."""

from __future__ import annotations

from functools import lru_cache

from ness_cli.tui.commands.general import (
    clear_command,
    copy_command,
    exit_command,
    help_command,
)
from ness_cli.tui.commands.context import (
    hooks_command,
    init_command,
    memory_command,
    permissions_command,
    skill_command,
    user_command,
)
from ness_cli.tui.commands.mcp import mcp_command
from ness_cli.tui.commands.provider import (
    config_command,
    login_command,
    status_command,
)
from ness_cli.tui.commands.registry import CommandRegistry, CommandSpec
from ness_cli.tui.commands.session import (
    compact_command,
    export_command,
    fork_command,
    goal_command,
    new_command,
    reflection_command,
    rename_command,
    rollback_command,
    save_command,
    threads_command,
)


@lru_cache(maxsize=1)
def command_registry() -> CommandRegistry:
    return CommandRegistry(
        (
            CommandSpec(
                "help",
                "Show the command reference",
                "General",
                help_command,
                "/help",
                busy_safe=True,
            ),
            CommandSpec(
                "login",
                "Connect a model provider",
                "Provider",
                login_command,
                "/login [provider]",
            ),
            CommandSpec(
                "config",
                "Edit model and behavior settings",
                "Provider",
                config_command,
                "/config",
            ),
            CommandSpec(
                "status",
                "Show provider and session usage",
                "Provider",
                status_command,
                "/status",
                busy_safe=True,
            ),
            CommandSpec(
                "mcp",
                "Show MCP servers or load deferred tools",
                "Context",
                mcp_command,
                "/mcp [server [tool]]",
                busy_safe=True,
            ),
            CommandSpec(
                "skill",
                "Manage available skills",
                "Context",
                skill_command,
                "/skill",
                busy_safe=True,
            ),
            CommandSpec(
                "memory",
                "Read or update project memory",
                "Context",
                memory_command,
                "/memory [add <note>|create [force]]",
                busy_safe=True,
            ),
            CommandSpec(
                "user",
                "Read or update user preferences",
                "Context",
                user_command,
                "/user [add <preference>]",
                busy_safe=True,
            ),
            CommandSpec(
                "init",
                "Initialize project and global Ness files",
                "Context",
                init_command,
                "/init",
            ),
            CommandSpec(
                "permissions",
                "View or edit tool permission rules",
                "Tools",
                permissions_command,
                "/permissions",
                busy_safe=True,
            ),
            CommandSpec(
                "hooks",
                "List configured hooks",
                "Tools",
                hooks_command,
                "/hooks",
                busy_safe=True,
            ),
            CommandSpec(
                "threads",
                "Switch to a saved session",
                "Session",
                threads_command,
                "/threads",
                busy_safe=True,
            ),
            CommandSpec(
                "rename",
                "Name the current session",
                "Session",
                rename_command,
                "/rename <name>",
                busy_safe=True,
            ),
            CommandSpec(
                "save",
                "Archive the current session",
                "Session",
                save_command,
                "/save",
            ),
            CommandSpec(
                "new",
                "Archive and start a new session",
                "Session",
                new_command,
                "/new",
                busy_safe=True,
            ),
            CommandSpec(
                "compact",
                "Compact on the next model turn",
                "Session",
                compact_command,
                "/compact",
            ),
            CommandSpec(
                "reflection",
                "Reflect on new session history",
                "Session",
                reflection_command,
                "/reflection",
            ),
            CommandSpec(
                "rollback",
                "Restore a prior user turn",
                "Session",
                rollback_command,
                "/rollback [<seq>]",
            ),
            CommandSpec(
                "fork",
                "Fork before a prior user message",
                "Session",
                fork_command,
                "/fork",
            ),
            CommandSpec(
                "export",
                "Export the durable session to HTML",
                "Session",
                export_command,
                "/export <path.html>",
            ),
            CommandSpec(
                "goal",
                "Run a bounded worker and judge loop",
                "Session",
                goal_command,
                "/goal <objective>",
            ),
            CommandSpec(
                "clear",
                "Clear the rendered transcript",
                "Input",
                clear_command,
                "/clear",
            ),
            CommandSpec(
                "copy",
                "Copy an assistant response",
                "Input",
                copy_command,
                "/copy [code|<n>]",
                busy_safe=True,
            ),
            CommandSpec(
                "exit",
                "End the interactive session",
                "General",
                exit_command,
                "/exit",
                aliases=("quit",),
            ),
        )
    )


__all__ = ["CommandRegistry", "CommandSpec", "command_registry"]
