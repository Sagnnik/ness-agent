"""Commands that operate on the selected CLI session."""

from __future__ import annotations

from ness_cli.tui.context import CommandContext
from ness_cli.tui.input import MenuItem
from ness_cli.session.export import ExportError, resolve_export_path


async def threads_command(context: CommandContext, args: str) -> None:
    if args:
        context.renderer.error("Usage: /threads")
        return
    rows = context.threads.list_threads()
    if not rows:
        context.renderer.notice("threads", "No saved sessions.")
        return
    items = [
        MenuItem(
            str(row.get("thread_id") or ""),
            str(row.get("label") or "(no messages)"),
            description=(
                f"{row.get('turn_count', 0)} turns · "
                f"${float(row.get('total_cost_usd', 0.0)):.4f}"
            ),
            suffix=" · ".join(
                part
                for part in (
                    "current"
                    if row.get("thread_id") == context.session.thread_id
                    else "",
                    str(row.get("live_status") or ""),
                )
                if part
            ),
        )
        for row in rows
    ]
    target = await context.ui.choose(
        "saved threads",
        items,
        initial_key=context.session.thread_id,
    )
    if target and target != context.session.thread_id:
        await context.threads.resume_thread(target)


async def rename_command(context: CommandContext, args: str) -> None:
    name = " ".join(args.split())
    if not name:
        context.renderer.error("Usage: /rename <name>")
        return
    if not context.session.rename(name):
        context.renderer.warning(
            "Thread autosave is disabled, so the name was not saved."
        )
        return
    context.renderer.notice("rename", f"Session renamed to {name}.")


async def save_command(context: CommandContext, args: str) -> None:
    if args:
        context.renderer.error("Usage: /save")
        return
    result = await context.session.save()
    context.renderer.notice("save", result.message)


async def new_command(context: CommandContext, args: str) -> None:
    if args:
        context.renderer.error("Usage: /new")
        return
    await context.threads.new_thread()
    context.renderer.notice("notice", "Started a fresh thread.")


async def compact_command(context: CommandContext, args: str) -> None:
    if args:
        context.renderer.error("Usage: /compact")
        return
    context.session.request_compact()
    context.renderer.notice(
        "compaction",
        "Compaction will run on the next model turn.",
    )


async def reflection_command(context: CommandContext, args: str) -> None:
    if args:
        context.renderer.error("Usage: /reflection")
        return
    result = await context.session.run_reflection()
    if result.error:
        context.renderer.error(result.error)
    elif result.bullets:
        context.renderer.notice(
            "reflection",
            *(f"- {bullet}" for bullet in result.bullets),
        )
    elif result.message_index is None:
        context.renderer.notice("reflection", "Nothing new to reflect on.")
    else:
        context.renderer.notice(
            "reflection",
            "Reflection completed with no new memory bullets.",
        )


async def rollback_command(context: CommandContext, args: str) -> None:
    target = args.strip()
    if target and not target.isdigit():
        context.renderer.error("Usage: /rollback [<seq>]")
        return
    if target:
        seq = int(target)
    else:
        turns = context.session.user_turns()
        if not turns:
            context.renderer.notice(
                "rollback",
                "No user turns are available in this thread.",
            )
            return
        selected = await context.ui.choose(
            "rollback to user message",
            [
                MenuItem(
                    str(turn.seq),
                    " ".join(turn.content.split())[:100] or "(empty)",
                    suffix=f"seq {turn.seq}",
                )
                for turn in turns
            ],
            initial_key=str(turns[-1].seq),
        )
        if selected is None:
            return
        seq = int(selected)

    result = await context.threads.rollback_to(seq)
    if result.ok:
        await context.ui.refresh_transcript()
        context.renderer.notice("rollback", result.message)
    else:
        context.renderer.error(result.message)


async def fork_command(context: CommandContext, args: str) -> None:
    if args:
        context.renderer.error("Usage: /fork")
        return
    turns = context.session.user_turns()
    if not turns:
        context.renderer.notice("fork", "No user turns are available in this thread.")
        return
    selected = await context.ui.choose(
        "fork before user message",
        [
            MenuItem(
                str(turn.seq),
                " ".join(turn.content.split())[:100] or "(empty)",
                suffix=f"seq {turn.seq}",
            )
            for turn in turns
        ],
        initial_key=str(turns[-1].seq),
    )
    if selected is None:
        return
    result = await context.threads.fork_before(int(selected))
    await context.ui.refresh_transcript()
    context.ui.prefill_input(result.prompt)
    context.renderer.notice(
        "fork",
        f"Created {result.thread_id} before seq {selected}.",
        f"Copied {result.copied_events} durable events.",
    )


async def export_command(context: CommandContext, args: str) -> None:
    try:
        destination = resolve_export_path(args, context.session.project_root)
        result = await context.session.export_html(destination)
    except ExportError as error:
        context.renderer.error(str(error))
        return
    context.renderer.notice(
        "export",
        f"Exported {result.event_count} entries to {result.path}",
    )


async def goal_command(context: CommandContext, args: str) -> None:
    goal = args.strip()
    if not goal:
        context.renderer.error("Usage: /goal <objective>")
        return
    result = await context.threads.run_goal(goal)
    title = "goal passed" if result.passed else "goal incomplete"
    lines = [f"Attempts: {result.attempts}"]
    lines.extend(f"Unmet: {item}" for item in result.verdict.unmet)
    lines.extend(f"Evidence: {item}" for item in result.verdict.evidence)
    context.renderer.notice(title, *lines)
