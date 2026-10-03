from __future__ import annotations

import asyncio
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
import typer

_MCP_HELP_FOOTER = """MCP management commands:

ness mcp status [SERVER] — Show configured servers and authentication state.

ness mcp login SERVER — Authenticate an HTTP MCP server.

ness mcp logout SERVER — Remove stored OAuth credentials.

ness mcp import PATH — Import a Cursor- or Claude-compatible MCP config.

Run 'ness mcp --help' for command options."""

app = typer.Typer(add_completion=False, help="Ness Agent CLI")


def _version_callback(value: bool | None) -> None:
    if not value:
        return

    try:
        current = package_version("ness-agent")
    except PackageNotFoundError:
        print("ness (version unknown - package not installed)")
    else:
        print(f"ness {current}")

    raise typer.Exit()


def _read_stdin() -> str:
    if sys.stdin.isatty():
        return ""

    try:
        return sys.stdin.read()
    except OSError:
        return ""


def _build_overrides(
    *,
    model: str | None,
    reflection_model: str | None,
    api_key: str | None,
    base_url: str | None,
    session_id: str | None,
    reasoning_effort: str | None,
):
    from ness_cli.config import ConfigManager
    from ness_cli.config.types import CliOverrides
    from ness_cli.paths import resolve_paths
    from ness_cli.cleanup import cleanup_on_exit
    from ness_cli.providers.registry import ProviderRegistry

    values = {
        "model_name": model,
        "reflection_model_name": reflection_model,
        "api_key": api_key,
        "base_url": base_url,
        "session_id": session_id,
        "reasoning_effort": reasoning_effort,
    }

    active = {key: value for key, value in values.items() if value is not None}

    overrides = CliOverrides(**active)

    if reasoning_effort is not None:
        paths = resolve_paths()
        config = ConfigManager.load(paths, overrides)

        async def validate() -> None:
            providers = ProviderRegistry.load(paths=paths, config=config)
            async with cleanup_on_exit("provider validation", providers.close):
                providers.validate_reasoning_effort(
                    config.current.model.model_name, reasoning_effort
                )

        try:
            asyncio.run(validate())
        except ValueError as error:
            raise typer.BadParameter(str(error)) from error

    return overrides


async def _run_interactive(options) -> None:
    from ness_cli.runtime import open_interactive_runtime
    from ness_cli.tui.app import run_app

    async with open_interactive_runtime(options) as runtime:
        await run_app(runtime)


@app.command(epilog=_MCP_HELP_FOOTER)
def run(
    prompt: list[str] = typer.Argument(
        None, help="One-shot query text (requires --print)"
    ),
    model: str = typer.Option(
        None, "--model", help="Chat model name (overrides MODEL_NAME)"
    ),
    reflection_model: str = typer.Option(
        None, "--reflection-model", help="Reflection model name"
    ),
    api_key: str = typer.Option(None, "--api-key", help="OpenAI-compatible API key"),
    base_url: str = typer.Option(None, "--base-url", help="OpenAI-compatible base URL"),
    session_id: str = typer.Option(
        None, "--openrouter-session-id", help="Stable prompt-cache session id"
    ),
    reasoning_effort: str = typer.Option(
        None,
        "--reasoning-effort",
        help="Reasoning effort supported by the selected provider",
    ),
    worktree: str = typer.Option(
        None, "--worktree", "-w", help="Run inside an isolated git worktree"
    ),
    resume: str = typer.Option(
        None,
        "--resume",
        help="Resume a saved thread id at startup (loads prior conversation into the transcript)",
    ),
    yolo: bool = typer.Option(
        False,
        "--yolo",
        help="Bypass act-mode approvals, deny rules, and native file-path restrictions",
    ),
    print_mode: bool = typer.Option(
        False,
        "--print",
        "-p",
        help="Run the query non-interactively, print the final response, and exit",
    ),
    version: bool | None = typer.Option(
        None,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the installed Ness version and exit",
    ),
) -> None:
    """Start an interactive session or execute one headless turn."""

    del worktree, version

    from ness_cli.runtime import (
        HeadlessOptions,
        InteractiveOptions,
    )

    overrides = _build_overrides(
        model=model,
        reflection_model=reflection_model,
        api_key=api_key,
        base_url=base_url,
        session_id=session_id,
        reasoning_effort=reasoning_effort,
    )

    if print_mode:
        from ness_cli.headless import merge_prompt_parts, run_headless

        query = merge_prompt_parts(prompt, _read_stdin())
        if query is None:
            print(
                "error: --print requires a prompt argument or piped stdin",
                file=sys.stderr,
            )
            raise typer.Exit(2)

        options = HeadlessOptions(
            overrides=overrides,
            resume_thread_id=resume,
            yolo=yolo,
        )

        try:
            code = asyncio.run(run_headless(query, options=options))
        except KeyboardInterrupt:
            code = 130

        raise typer.Exit(code)

    if prompt:
        print(
            "error: a positional prompt requires --print/-p",
            file=sys.stderr,
        )
        raise typer.Exit(2)

    options = InteractiveOptions(
        overrides=overrides,
        resume_thread_id=resume,
        yolo=yolo,
    )

    try:
        asyncio.run(_run_interactive(options))
    except KeyboardInterrupt:
        raise typer.Exit(130) from None
