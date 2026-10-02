"""Standalone ``ness mcp`` management commands."""

from __future__ import annotations

import asyncio
import getpass
import json
import socket
import webbrowser
from pathlib import Path
from typing import Any

import typer

from ness_cli.paths import (
    ensure_global_config,
    ensure_project_runtime,
    resolve_paths,
)
from ness_cli.mcp.importer import (
    MCPImportConflictError,
    execute_mcp_import,
    plan_mcp_import,
    provenance_for_server,
)
from ness_cli.mcp.auth import (
    LoopbackOAuthCallback,
    MCPOAuthService,
    OAuthCredentialClearError,
    parse_oauth_callback_url,
)
from ness_cli.mcp.manager import ProjectMCPManager
from ness_cli.mcp.trust import authorize_mcp_interactively
from ness_cli.terminal import terminal_safe_text

app = typer.Typer(add_completion=False, help="Manage Ness MCP servers")


def _paths():
    paths = resolve_paths()
    ensure_global_config(paths)
    ensure_project_runtime(paths)
    return paths


def _manager_and_service():
    paths = _paths()
    service = MCPOAuthService(
        project_root=paths.project_root,
        config_dir=paths.config_dir,
    )
    manager = ProjectMCPManager(
        paths.ness_dir / "mcp.json",
        project_root=paths.project_root,
        http_auth_factory=service.startup_auth,
    )
    manager.load()
    return paths, manager, service


@app.command("status")
def status(
    server: str | None = typer.Argument(None, help="Optional server name"),
) -> None:
    """Show configured MCP servers, OAuth state, and import provenance."""
    asyncio.run(_status(server))


async def _status(server: str | None) -> None:
    paths, manager, service = _manager_and_service()
    names = [server] if server else sorted(manager.servers)
    if server and server not in manager.servers:
        typer.echo(terminal_safe_text(f"error: unknown MCP server: {server}"), err=True)
        raise typer.Exit(2)
    raw_servers = _read_server_map(paths.ness_dir / "mcp.json")
    if not names:
        typer.echo("No MCP servers configured.")
        return
    for name in names:
        info = manager.servers[name]
        spec = manager.server_spec(name)
        line = f"{name}: {info.get('status', 'unknown')}"
        if spec and spec.transport == "http":
            storage = service.storage_for(spec)
            logged_in = await storage.has_credentials()
            backend = await storage.backend_name()
            auth_mode = "static" if spec.oauth and spec.oauth.client_id else "dynamic"
            if spec.oauth is None:
                auth_mode = "none/detected at login"
            line += f"; oauth={auth_mode}; logged_in={'yes' if logged_in else 'no'}; storage={backend}"
        provenance = None
        raw_entry = raw_servers.get(name)
        if isinstance(raw_entry, dict):
            provenance = provenance_for_server(
                config_dir=paths.config_dir,
                project_root=paths.project_root,
                name=name,
                entry=raw_entry,
            )
        if provenance:
            state = (
                "modified since import" if provenance.get("modified") else "unchanged"
            )
            line += f"; imported from {provenance.get('source_path')} ({state})"
        typer.echo(terminal_safe_text(line))
    for warning in service.warnings:
        typer.echo(terminal_safe_text(f"warning: {warning}"), err=True)


@app.command("login")
def login(
    server: str = typer.Argument(..., help="HTTP MCP server name"),
    callback_port: int | None = typer.Option(
        None, "--callback-port", help="Override OAuth callback port"
    ),
    no_open: bool = typer.Option(
        False, "--no-open", help="Print the URL instead of opening a browser"
    ),
    manual_callback: bool = typer.Option(
        False, "--manual-callback", help="Paste the callback URL (for SSH)"
    ),
    timeout: float = typer.Option(
        300.0, "--timeout", min=1.0, help="OAuth flow timeout in seconds"
    ),
) -> None:
    """Authenticate one HTTP MCP server explicitly."""
    asyncio.run(
        _login(
            server,
            callback_port=callback_port,
            no_open=no_open,
            manual_callback=manual_callback,
            timeout=timeout,
        )
    )


async def _login(
    server: str,
    *,
    callback_port: int | None,
    no_open: bool,
    manual_callback: bool,
    timeout: float,
) -> None:
    paths, manager, service = _manager_and_service()
    spec = manager.server_spec(server)
    if spec is None:
        detail = manager.servers.get(server, {}).get("error")
        message = f"unknown or invalid MCP server: {server}"
        if detail:
            message += f" ({detail})"
        typer.echo(terminal_safe_text(f"error: {message}"), err=True)
        raise typer.Exit(2)
    if spec.transport != "http":
        typer.echo(
            "error: OAuth login is supported only for HTTP MCP servers", err=True
        )
        raise typer.Exit(2)
    if any(key.lower() == "authorization" for key, _ in spec.headers):
        typer.echo(
            "error: remove the explicit Authorization header before OAuth login",
            err=True,
        )
        raise typer.Exit(2)
    if callback_port is not None and not 1 <= callback_port <= 65535:
        typer.echo("error: callback port must be from 1 to 65535", err=True)
        raise typer.Exit(2)
    if not authorize_mcp_interactively(manager, config_dir=paths.config_dir):
        typer.echo("MCP configuration was not trusted; login cancelled.", err=True)
        raise typer.Exit(1)

    configured_port = spec.oauth.callback_port if spec.oauth else None
    requested_port = callback_port or configured_port or 0
    loopback: LoopbackOAuthCallback | None = None
    if manual_callback:
        port = requested_port or _available_loopback_port()
        redirect_uri = f"http://localhost:{port}/callback"

        async def callback_handler():
            value = await asyncio.to_thread(
                getpass.getpass,
                "Paste the full OAuth callback URL (input hidden): ",
            )
            return parse_oauth_callback_url(value.strip())
    else:
        loopback = LoopbackOAuthCallback(port=requested_port, timeout=timeout)
        try:
            await loopback.start()
        except RuntimeError as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(1) from exc
        redirect_uri = loopback.redirect_uri
        callback_handler = loopback.callback_handler

    async def redirect_handler(url: str) -> None:
        if manual_callback or no_open:
            typer.echo(f"Open this URL to authenticate:\n{url}")
            return
        opened = await asyncio.to_thread(webbrowser.open, url, 2)
        if not opened:
            typer.echo(f"Browser could not be opened. Open this URL:\n{url}")

    async def auth_factory(target_spec):
        return service.interactive_auth(
            target_spec,
            redirect_uri=redirect_uri,
            redirect_handler=redirect_handler,
            callback_handler=callback_handler,
            timeout=timeout,
        )

    manager.http_auth_factory = auth_factory
    login_spec = spec.with_startup_timeout(timeout + 30)
    try:
        await manager.start_server(server, login_spec)
        tools = manager.servers.get(server, {}).get("tools", [])
        typer.echo(
            terminal_safe_text(
                f"Authenticated {server}; verified {len(tools)} MCP tool(s)."
            )
        )
    except Exception as exc:
        typer.echo(f"error: OAuth login failed: {type(exc).__name__}", err=True)
        raise typer.Exit(1) from exc
    finally:
        await manager.stop()
        if loopback is not None:
            await loopback.close()
    for warning in service.warnings:
        typer.echo(terminal_safe_text(f"warning: {warning}"), err=True)


@app.command("logout")
def logout(server: str = typer.Argument(..., help="HTTP MCP server name")) -> None:
    """Remove locally stored OAuth credentials for one server."""
    asyncio.run(_logout(server))


async def _logout(server: str) -> None:
    _, manager, service = _manager_and_service()
    spec = manager.server_spec(server)
    if spec is None or spec.transport != "http":
        typer.echo(
            terminal_safe_text(f"error: unknown HTTP MCP server: {server}"),
            err=True,
        )
        raise typer.Exit(2)
    try:
        await service.storage_for(spec).clear()
    except OAuthCredentialClearError as exc:
        typer.echo(f"error: {terminal_safe_text(exc)}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(
        terminal_safe_text(
            f"Removed local OAuth credentials for {server}. "
            "Provider-side tokens were not revoked."
        )
    )


@app.command("import")
def import_config(
    source: Path = typer.Argument(
        ..., exists=False, help="Cursor/Claude-compatible MCP JSON"
    ),
    servers: list[str] | None = typer.Option(
        None, "--server", help="Import only this server (repeatable)"
    ),
    replace: list[str] | None = typer.Option(
        None, "--replace", help="Replace this conflicting server (repeatable)"
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview without writing"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the import confirmation"),
) -> None:
    """Import explicitly selected servers into NESS_DIR/mcp.json."""
    paths = _paths()
    plan = plan_mcp_import(
        source,
        paths.ness_dir / "mcp.json",
        project_root=paths.project_root,
        selected=set(servers) if servers else None,
        replace=set(replace or ()),
    )
    typer.echo(plan.render())
    if not plan.valid:
        raise typer.Exit(2)
    if dry_run:
        typer.echo("Dry run: no files changed.")
        return
    if not plan.changes:
        typer.echo("No changes to import.")
        return
    if not yes and not typer.confirm(
        "Apply this MCP import?", default=False, abort=False
    ):
        typer.echo("Import cancelled.")
        raise typer.Exit(1)
    try:
        warnings = execute_mcp_import(plan, config_dir=paths.config_dir)
    except MCPImportConflictError as exc:
        typer.echo(f"error: {terminal_safe_text(exc)}", err=True)
        raise typer.Exit(2) from exc
    typer.echo(
        f"Imported {len(plan.changes)} server(s). Execution trust was not granted."
    )
    for warning in warnings:
        typer.echo(terminal_safe_text(f"warning: {warning}"), err=True)


def _available_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _read_server_map(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(document, dict):
        return {}
    value = document.get("mcpServers", document.get("servers", {}))
    return value if isinstance(value, dict) else {}
