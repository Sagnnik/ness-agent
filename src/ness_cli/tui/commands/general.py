"""General slash commands."""

from __future__ import annotations

import re

from ness_cli.tui.context import CommandContext


async def help_command(context: CommandContext, args: str) -> None:
    del args
    from ness_cli.tui.commands import command_registry

    registry = command_registry()
    context.renderer.command_help(
        tuple((spec.command, spec.summary) for spec in registry.specs)
    )


async def exit_command(context: CommandContext, args: str) -> None:
    del args
    context.ui.request_exit()


async def clear_command(context: CommandContext, args: str) -> None:
    del args
    context.ui.clear_transcript()


async def copy_command(context: CommandContext, args: str) -> None:
    history = context.renderer.assistant_history
    if not history:
        context.renderer.warning("No assistant message to copy.")
        return

    text = history[-1]
    option = args.strip().casefold()
    if option == "code":
        blocks = re.findall(r"```(?:\w+)?\n(.*?)```", text, re.DOTALL)
        if blocks:
            text = blocks[-1]
    elif option.isdigit():
        index = int(option)
        if not 1 <= index <= len(history):
            context.renderer.error(f"Assistant message {index} is not available.")
            return
        text = history[-index]
    elif option:
        context.renderer.error("Usage: /copy [code|<n>]")
        return

    try:
        import pyperclip

        pyperclip.copy(text)
    except Exception:
        context.renderer.notice("clipboard unavailable", text)
        return
    context.renderer.notice("copy", "Copied to clipboard.")
