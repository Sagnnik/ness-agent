"""Commands for project instructions, memory, permissions, and hooks."""

from __future__ import annotations

import subprocess
from pathlib import Path

from ness_agent.workspace import setup_ness_structure

from ness_cli.paths import ensure_global_config, ensure_project_runtime
from ness_cli.prompts import build_init_memory_prompt
from ness_cli.tui.context import CommandContext
from ness_cli.tui.input import MenuItem


def _add_text(args: str) -> str | None:
    value = args.strip()
    return value[4:].strip() if value.startswith("add ") else None


def _display_source(source: str, project_root: Path) -> str:
    if not source:
        return ""
    path = Path(source)
    try:
        return str(path.relative_to(project_root))
    except (ValueError, OSError):
        try:
            return str(Path("~") / path.relative_to(Path.home()))
        except (ValueError, OSError):
            return str(path)


def _project_context(project_root: Path, limit: int = 80) -> str:
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        files = result.stdout.splitlines()[:limit] if result.returncode == 0 else []
    except (OSError, subprocess.SubprocessError):
        files = []
    if not files:
        files = [
            str(path.relative_to(project_root))
            for path in project_root.rglob("*")
            if path.is_file() and ".git" not in path.parts and ".ness" not in path.parts
        ][:limit]
    return "Project files:\n" + "\n".join(f"- {path}" for path in files)


async def skill_command(context: CommandContext, args: str) -> None:
    if args.strip():
        context.renderer.error("Usage: /skill. Type $ in the prompt to select a skill.")
        return
    skills, errors = context.session.skills()
    by_skill_id = {
        str(skill.get("skill_id") or skill.get("source_id") or ""): skill
        for skill in skills
    }
    if by_skill_id:
        items = [
            MenuItem(
                skill_id,
                str(skill.get("name") or ""),
                description=" ".join(str(skill.get("description") or "").split()),
                search_terms=tuple(
                    str(source.get("source") or "")
                    for source in skill.get("sources", [])
                    if isinstance(source, dict)
                )
                or (str(skill.get("source") or ""),),
            )
            for skill_id, skill in by_skill_id.items()
        ]
        result = await context.ui.choose_checklist(
            f"skills ({len(items)})",
            items,
            disabled_keys=frozenset(
                skill_id
                for skill_id, skill in by_skill_id.items()
                if not bool(skill.get("available", True))
            ),
        )
        if result is not None:
            context.session.update_skill_access(result.disabled_keys)
            skill = by_skill_id[result.selected_key]
            sources = [
                _display_source(
                    str(source.get("source") or ""),
                    context.session.project_root,
                )
                for source in skill.get("sources", [])
                if isinstance(source, dict)
            ] or [
                _display_source(
                    str(skill.get("source") or ""),
                    context.session.project_root,
                )
            ]
            context.renderer.skill_detail(
                name=str(skill.get("name") or ""),
                source=sources[0],
                alternate_sources=tuple(sources[1:]),
                description=str(skill.get("description") or ""),
            )
    else:
        context.renderer.notice("skills", "No skills found.")
    if errors:
        context.renderer.warning("Skill load warnings:\n" + "\n".join(errors))


async def init_command(context: CommandContext, args: str) -> None:
    if args:
        context.renderer.error("Usage: /init")
        return
    paths = context.runtime.paths
    created = setup_ness_structure(paths.ness_dir)
    created.extend(ensure_global_config(paths))
    created.extend(ensure_project_runtime(paths))
    if created:
        context.renderer.notice("init", "Created " + ", ".join(created))
    else:
        context.renderer.notice("init", ".ness and global configuration already exist.")


async def memory_command(context: CommandContext, args: str) -> None:
    raw = args.strip()
    if raw.startswith("create"):
        if context.threads.busy:
            context.renderer.warning(
                "/memory create is unavailable while a turn is running."
            )
            return
        rest = raw[6:].strip()
        overwrite = rest in {"force", "--force"}
        if rest and not overwrite:
            context.renderer.error("Usage: /memory create [force]")
            return
        prompt = build_init_memory_prompt(
            _project_context(context.session.project_root),
            instructions_dir=context.runtime.paths.instructions_dir,
        )
        result = await context.session.create_project_memory(
            prompt,
            overwrite=overwrite,
        )
        if result.startswith("Error:"):
            context.renderer.error(result)
        else:
            context.renderer.notice("memory", result)
        return

    text = _add_text(args)
    if text is not None:
        if not text:
            context.renderer.error("Usage: /memory add <note>")
            return
        context.renderer.notice("memory", context.session.append_project_memory(text))
        return
    if raw:
        context.renderer.error("Usage: /memory [add <note>|create [force]]")
        return
    path, content = context.session.project_memory()
    context.renderer.notice(str(path), content or "(empty)")


async def user_command(context: CommandContext, args: str) -> None:
    raw = args.strip()
    text = _add_text(args)
    if text is not None:
        if not text:
            context.renderer.error("Usage: /user add <preference>")
            return
        context.renderer.notice("user", context.session.append_user_memory(text))
        return
    if raw:
        context.renderer.error("Usage: /user [add <preference>]")
        return
    path, content = context.session.user_memory()
    context.renderer.notice(str(path), content or "(empty)")


async def permissions_command(context: CommandContext, args: str) -> None:
    parts = args.split()
    if not parts or parts == ["list"]:
        context.renderer.notice(
            "permissions", *context.session.permission_rules().splitlines()
        )
        return
    if len(parts) >= 2 and parts[0] in {"allow", "deny"}:
        context.session.add_permission_rule(" ".join(parts[1:]), parts[0])
        context.renderer.notice("permissions", f"Added {parts[0]} rule.")
        return
    if len(parts) == 3 and parts[0] == "remove" and parts[1] in {"allow", "deny"}:
        try:
            removed = context.session.remove_permission_rule(parts[1], int(parts[2]))
        except ValueError as error:
            context.renderer.error(str(error))
            return
        context.renderer.notice("permissions", f"Removed {removed}")
        return
    context.renderer.error(
        "Usage: /permissions [list|allow <pattern>|deny <pattern>|remove <allow|deny> <index>]"
    )


async def hooks_command(context: CommandContext, args: str) -> None:
    if args:
        context.renderer.error("Usage: /hooks")
        return
    context.renderer.notice("hooks", *context.session.hooks_description().splitlines())
