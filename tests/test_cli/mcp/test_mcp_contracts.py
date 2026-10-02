from __future__ import annotations

import json
from pathlib import Path

import pytest

from ness_cli.mcp.config import ProjectMCPConfig
from ness_cli.mcp.importer import (
    MCPImportConflictError,
    execute_mcp_import,
    plan_mcp_import,
    provenance_for_server,
)
from ness_cli.mcp.manager import ProjectMCPManager
from ness_cli.mcp.trust import authorize_mcp_interactively, is_mcp_trusted


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_config_normalizes_stdio_and_expands_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("MCP_TOKEN", "secret")
    path = tmp_path / "mcp.json"
    write_json(path, {"mcpServers": {"local": {"command": ["python", "-m", "server"], "env": {"TOKEN": "${MCP_TOKEN}"}}}})
    config = ProjectMCPConfig(path, project_root=tmp_path)
    spec = config.specs["local"].connection
    assert spec.transport == "stdio"
    assert spec.command == "python"
    assert spec.args == ("-m", "server")
    assert ("TOKEN", "secret") in spec.env
    assert "secret" not in " ".join(config.trust_preview.servers)


def test_config_fingerprint_changes_without_exposing_credentials(tmp_path):
    path = tmp_path / "mcp.json"
    write_json(path, {"mcpServers": {"x": {"url": "https://example.com/mcp", "headers": {"Authorization": "Bearer one"}}}})
    first = ProjectMCPConfig(path, project_root=tmp_path).load()
    write_json(path, {"mcpServers": {"x": {"url": "https://example.com/mcp", "headers": {"Authorization": "Bearer two"}}}})
    second = ProjectMCPConfig(path, project_root=tmp_path).load()
    assert first.fingerprint != second.fingerprint
    assert "Bearer" not in " ".join(first.servers)


def test_import_is_transactional_and_records_provenance(tmp_path):
    source = tmp_path / "source.json"
    destination = tmp_path / ".ness" / "mcp.json"
    config_dir = tmp_path / "config"
    entry = {"command": "python"}
    write_json(source, {"servers": {"local": entry}})
    plan = plan_mcp_import(source, destination, project_root=tmp_path)
    assert plan.valid and plan.entries[0].action == "add" and not destination.exists()
    assert execute_mcp_import(plan, config_dir=config_dir) == []
    assert json.loads(destination.read_text())["mcpServers"]["local"] == entry
    provenance = provenance_for_server(config_dir=config_dir, project_root=tmp_path, name="local", entry=entry)
    assert provenance and not provenance["modified"]


def test_import_conflict_requires_explicit_replace(tmp_path):
    source = tmp_path / "source.json"
    destination = tmp_path / "destination.json"
    write_json(source, {"mcpServers": {"x": {"command": "new"}}})
    write_json(destination, {"mcpServers": {"x": {"command": "old"}}})
    blocked = plan_mcp_import(source, destination, project_root=tmp_path)
    assert not blocked.valid and blocked.entries[0].action == "conflict"
    allowed = plan_mcp_import(source, destination, project_root=tmp_path, replace={"x"})
    assert allowed.valid and allowed.entries[0].action == "replace"


def test_import_detects_destination_change_before_write(tmp_path):
    source = tmp_path / "source.json"
    destination = tmp_path / "destination.json"
    write_json(source, {"mcpServers": {"x": {"command": "new"}}})
    write_json(destination, {"mcpServers": {}})
    plan = plan_mcp_import(source, destination, project_root=tmp_path)
    write_json(destination, {"mcpServers": {"manual": {"command": "keep"}}})
    with pytest.raises(MCPImportConflictError, match="changed after confirmation"):
        execute_mcp_import(plan, config_dir=tmp_path / "config")
    assert "manual" in destination.read_text()


def test_import_render_redacts_literal_secrets(tmp_path):
    source = tmp_path / "source.json"
    write_json(source, {"mcpServers": {"x": {"url": "https://example.com/mcp?token=value", "headers": {"Authorization": "Bearer secret"}}}})
    rendered = plan_mcp_import(source, tmp_path / "dest.json", project_root=tmp_path).render()
    assert "Bearer secret" not in rendered and "token=value" not in rendered
    assert "literal credential" in rendered


def test_trust_is_bound_to_exact_config_and_denial_is_not_persisted(tmp_path, monkeypatch):
    path = tmp_path / "mcp.json"
    config_dir = tmp_path / "config"
    write_json(path, {"mcpServers": {"x": {"command": "python"}}})
    denied = ProjectMCPManager(path, project_root=tmp_path)
    monkeypatch.setattr("ness_cli.mcp.trust.typer.confirm", lambda *a, **k: False)
    assert not authorize_mcp_interactively(denied, config_dir=config_dir)
    assert denied.servers["x"]["status"] == "pending_trust"
    assert not is_mcp_trusted(denied, config_dir=config_dir)

    approved = ProjectMCPManager(path, project_root=tmp_path)
    monkeypatch.setattr("ness_cli.mcp.trust.typer.confirm", lambda *a, **k: True)
    assert authorize_mcp_interactively(approved, config_dir=config_dir)
    assert is_mcp_trusted(approved, config_dir=config_dir)
    write_json(path, {"mcpServers": {"x": {"command": "uv"}}})
    assert not is_mcp_trusted(ProjectMCPManager(path, project_root=tmp_path), config_dir=config_dir)
