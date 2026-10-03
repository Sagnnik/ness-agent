"""Trust policy for executable project MCP configuration."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import typer

from ness_cli.config.store import ConfigStore
from ness_cli.mcp.manager import ProjectMCPManager
from ness_cli.mcp.models import MCPTrustPreview
from ness_cli.terminal import terminal_safe_text

_TRUST_KEY = "mcp_trust"


def is_mcp_trusted(manager: ProjectMCPManager, *, config_dir: Path) -> bool:
    preview = manager.load()
    if not preview.has_runnable_servers:
        return True
    entries = ConfigStore(config_dir).load_config().get(_TRUST_KEY, {})
    if not isinstance(entries, dict):
        return False
    entry = entries.get(str(manager.project_root.resolve()))
    return (
        isinstance(entry, dict)
        and entry.get("config_path") == str(preview.config_path)
        and entry.get("fingerprint") == preview.fingerprint
    )


def authorize_mcp_interactively(
    manager: ProjectMCPManager,
    *,
    config_dir: Path,
) -> bool:
    preview = manager.load()
    if not preview.has_runnable_servers or is_mcp_trusted(
        manager, config_dir=config_dir
    ):
        return True
    typer.echo(
        terminal_safe_text(
            f"MCP configuration requests permission: {preview.config_path}"
        )
    )
    for summary in preview.servers:
        typer.echo(terminal_safe_text(f"  - {summary}"))
    if not typer.confirm(
        "Allow these MCP servers for this exact configuration?",
        default=False,
        abort=False,
    ):
        manager.mark_untrusted()
        return False
    persist_mcp_trust(manager, preview, config_dir=config_dir)
    return True


def persist_mcp_trust(
    manager: ProjectMCPManager,
    preview: MCPTrustPreview,
    *,
    config_dir: Path,
) -> None:
    store = ConfigStore(config_dir)

    def update(document: dict[str, Any]) -> dict[str, Any]:
        raw = document.get(_TRUST_KEY, {})
        entries = dict(raw) if isinstance(raw, dict) else {}
        entries[str(manager.project_root.resolve())] = {
            "config_path": str(preview.config_path),
            "fingerprint": preview.fingerprint,
            "trusted_at": datetime.now(timezone.utc).isoformat(),
        }
        document[_TRUST_KEY] = entries
        return document

    store.mutate_config(update)
