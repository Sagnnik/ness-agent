"""Single source of truth for slash command metadata and dispatch."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from ness_cli.tui.context import CommandContext

CommandHandler = Callable[[CommandContext, str], Awaitable[None]]
ProjectSignature = tuple[tuple[str, int], ...]
ProjectCacheValue = tuple[ProjectSignature, tuple["CommandSpec", ...]]


@dataclass(frozen=True, slots=True)
class CommandSpec:
    name: str
    summary: str
    group: str
    handler: CommandHandler
    usage: str = ""
    aliases: tuple[str, ...] = ()
    busy_safe: bool = False

    @property
    def command(self) -> str:
        return self.usage or f"/{self.name}"


class CommandRegistry:
    def __init__(self, specs: Iterable[CommandSpec]) -> None:
        self._specs = tuple(specs)
        self._names: dict[str, CommandSpec] = {}
        self._project_cache: dict[Path, ProjectCacheValue] = {}
        for spec in self._specs:
            for name in (spec.name, *spec.aliases):
                normalized = name.casefold()
                if normalized in self._names:
                    raise ValueError(f"duplicate slash command: {name}")
                self._names[normalized] = spec

    @property
    def specs(self) -> tuple[CommandSpec, ...]:
        return self._specs

    def matches(
        self,
        prefix: str,
        *,
        commands_dir: Path | None = None,
    ) -> tuple[CommandSpec, ...]:
        query = prefix.casefold().lstrip("/")
        specs = self._specs
        if commands_dir is not None:
            specs = (*specs, *self._project_specs(commands_dir))
        return tuple(spec for spec in specs if spec.name.startswith(query))

    async def dispatch(
        self,
        context: CommandContext,
        command_line: str,
        *,
        busy: bool,
    ) -> bool:
        raw = command_line.strip()
        if not raw.startswith("/"):
            return False
        name, _, args = raw[1:].partition(" ")
        spec = self._names.get(name.casefold())
        if spec is None:
            spec = next(
                (
                    candidate
                    for candidate in self._project_specs(
                        context.runtime.paths.ness_dir / "commands"
                    )
                    if candidate.name.casefold() == name.casefold()
                ),
                None,
            )
        if spec is None:
            context.renderer.error(f"Unknown command: /{name}. Try /help.")
            return True
        if busy and not spec.busy_safe:
            context.renderer.warning(
                f"/{spec.name} is not available while a turn is running."
            )
            return True
        try:
            await spec.handler(context, args.strip())
        except Exception as error:
            context.renderer.error(str(error))
        return True

    def _project_specs(self, commands_dir: Path) -> tuple[CommandSpec, ...]:
        if not commands_dir.is_dir():
            return ()
        paths = tuple(sorted(commands_dir.glob("*.md")))
        signature = tuple(
            (path.name, path.stat().st_mtime_ns) for path in paths if path.is_file()
        )
        cached = self._project_cache.get(commands_dir)
        if cached is not None and cached[0] == signature:
            return cached[1]

        specs: list[CommandSpec] = []
        for path in paths:
            if path.stem.casefold() in self._names:
                continue
            try:
                template = path.read_text(encoding="utf-8")
            except OSError:
                continue
            if template.startswith("---"):
                _, separator, remainder = template[3:].partition("---")
                if separator:
                    template = remainder
            body = template.strip()
            if not body:
                continue

            async def run_project_command(
                context: CommandContext,
                args: str,
                *,
                prompt: str = body,
            ) -> None:
                message = prompt.replace("{{args}}", args)
                if getattr(context.threads, "busy", False):
                    position = context.threads.enqueue(message)
                    context.renderer.notice(
                        "queue",
                        f"Added project prompt #{position}.",
                    )
                    return
                await context.threads.submit(message)

            specs.append(
                CommandSpec(
                    path.stem,
                    "Project prompt command",
                    "Project",
                    run_project_command,
                    f"/{path.stem} [args]",
                    busy_safe=True,
                )
            )
        result = tuple(specs)
        self._project_cache[commands_dir] = (signature, result)
        return result
