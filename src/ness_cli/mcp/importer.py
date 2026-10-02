"""Transactional import of explicitly selected MCP JSON configurations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ness_agent.mcp import redact_url
from ness_cli.config.store import ConfigStore, atomic_write_json, locked_path
from ness_cli.mcp.config import validate_import_entry
from ness_cli.terminal import terminal_safe_text

_IMPORTS_KEY = "mcp_imports"


class MCPImportConflictError(RuntimeError):
    """The import destination changed after the plan was created."""


@dataclass(frozen=True)
class ImportEntry:
    name: str
    action: str
    summary: str
    entry: dict[str, Any]
    warnings: tuple[str, ...] = ()


@dataclass
class MCPImportPlan:
    source: Path
    destination: Path
    project_root: Path
    source_digest: str
    destination_digest: str | None = None
    entries: list[ImportEntry] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    destination_document: dict[str, Any] | None = None
    destination_key: str = "mcpServers"

    @property
    def valid(self) -> bool:
        return not self.errors

    @property
    def changes(self) -> list[ImportEntry]:
        return [entry for entry in self.entries if entry.action in {"add", "replace"}]

    def render(self) -> str:
        lines = [f"Source: {self.source}", f"Destination: {self.destination}"]
        for entry in self.entries:
            lines.append(f"- {entry.name}: {entry.action} — {entry.summary}")
            lines.extend(f"  warning: {warning}" for warning in entry.warnings)
        lines.extend(f"Warning: {warning}" for warning in self.warnings)
        lines.extend(f"Error: {error}" for error in self.errors)
        return terminal_safe_text("\n".join(lines), multiline=True)


def plan_mcp_import(
    source: Path,
    destination: Path,
    *,
    project_root: Path,
    selected: set[str] | None = None,
    replace: set[str] | None = None,
) -> MCPImportPlan:
    source = source.expanduser().resolve()
    destination = destination.resolve()
    replace = set(replace or ())
    try:
        source_bytes = source.read_bytes()
        source_doc = json.loads(source_bytes)
    except OSError as exc:
        return MCPImportPlan(
            source,
            destination,
            project_root.resolve(),
            "",
            errors=[f"cannot read source: {exc}"],
        )
    except json.JSONDecodeError as exc:
        return MCPImportPlan(
            source,
            destination,
            project_root.resolve(),
            "",
            errors=[
                f"source contains invalid JSON at line {exc.lineno}, column {exc.colno}"
            ],
        )

    plan = MCPImportPlan(
        source=source,
        destination=destination,
        project_root=project_root.resolve(),
        source_digest=hashlib.sha256(source_bytes).hexdigest(),
    )
    if not isinstance(source_doc, dict):
        plan.errors.append("source root must be a JSON object")
        return plan
    if "mcpServers" in source_doc and "servers" in source_doc:
        plan.warnings.append("source contains both keys; using mcpServers")
    source_servers = source_doc.get("mcpServers", source_doc.get("servers"))
    if not isinstance(source_servers, dict):
        plan.errors.append(
            "source must contain an object-valued mcpServers or servers key"
        )
        return plan

    if selected is not None:
        missing = sorted(selected - set(source_servers))
        if missing:
            plan.errors.append(
                f"source does not contain requested server(s): {', '.join(missing)}"
            )
        names = [name for name in source_servers if name in selected]
    else:
        names = list(source_servers)

    (
        destination_doc,
        destination_key,
        destination_error,
        destination_digest,
    ) = _load_destination(destination)
    if destination_error:
        plan.errors.append(destination_error)
        return plan
    plan.destination_document = destination_doc
    plan.destination_key = destination_key
    plan.destination_digest = destination_digest
    existing = destination_doc[destination_key]
    assert isinstance(existing, dict)

    conflicts: set[str] = set()
    for name in names:
        raw = source_servers[name]
        if not isinstance(name, str) or not name.strip():
            plan.errors.append("source server names must be non-empty strings")
            continue
        errors, warnings = validate_import_entry(raw)
        if errors:
            plan.errors.extend(f"{name}: {error}" for error in errors)
            continue
        assert isinstance(raw, dict)
        if name not in existing:
            action = "add"
        elif _entry_digest(existing[name]) == _entry_digest(raw):
            action = "unchanged"
        else:
            action = "replace" if name in replace else "conflict"
            conflicts.add(name)
            if name not in replace:
                plan.errors.append(
                    f"{name}: destination differs; rerun with --replace {name}"
                )
        plan.entries.append(
            ImportEntry(
                name=name,
                action=action,
                summary=_entry_summary(raw),
                entry=dict(raw),
                warnings=tuple(warnings),
            )
        )

    invalid_replace = sorted(replace - conflicts)
    if invalid_replace:
        plan.errors.append(
            "--replace named server(s) that are not selected conflicts: "
            + ", ".join(invalid_replace)
        )
    return plan


def execute_mcp_import(plan: MCPImportPlan, *, config_dir: Path) -> list[str]:
    if not plan.valid or plan.destination_document is None:
        raise ValueError("cannot execute an invalid MCP import plan")
    if not plan.changes:
        return []
    document = dict(plan.destination_document)
    server_map = dict(document[plan.destination_key])
    for item in plan.changes:
        server_map[item.name] = item.entry
    document[plan.destination_key] = server_map
    with locked_path(plan.destination):
        try:
            current_digest = _file_digest(plan.destination)
        except OSError as exc:
            raise MCPImportConflictError(
                "MCP destination could not be re-read; import was not applied"
            ) from exc
        if current_digest != plan.destination_digest:
            raise MCPImportConflictError(
                "MCP destination changed after confirmation; import was not applied"
            )
        atomic_write_json(plan.destination, document)

    warnings: list[str] = []
    try:
        store = ConfigStore(config_dir)
        configs = store.load_config()
        all_projects = configs.get(_IMPORTS_KEY, {})
        all_projects = dict(all_projects) if isinstance(all_projects, dict) else {}
        project_key = str(plan.project_root)
        project_entries = all_projects.get(project_key, {})
        project_entries = (
            dict(project_entries) if isinstance(project_entries, dict) else {}
        )
        now = datetime.now(timezone.utc).isoformat()
        for item in plan.changes:
            project_entries[item.name] = {
                "source_path": str(plan.source),
                "source_digest": plan.source_digest,
                "entry_digest": _entry_digest(item.entry),
                "imported_at": now,
            }
        all_projects[project_key] = project_entries
        store.write_config(_IMPORTS_KEY, all_projects)
    except OSError as exc:
        warnings.append(f"config imported, but provenance could not be saved: {exc}")
    return warnings


def provenance_for_server(
    *,
    config_dir: Path,
    project_root: Path,
    name: str,
    entry: dict[str, Any],
) -> dict[str, Any] | None:
    projects = ConfigStore(config_dir).load_config().get(_IMPORTS_KEY, {})
    if not isinstance(projects, dict):
        return None
    project = projects.get(str(project_root.resolve()), {})
    if not isinstance(project, dict):
        return None
    value = project.get(name)
    if not isinstance(value, dict):
        return None
    result = dict(value)
    result["modified"] = result.get("entry_digest") != _entry_digest(entry)
    return result


def _load_destination(
    path: Path,
) -> tuple[dict[str, Any], str, str | None, str | None]:
    if not path.exists():
        return {"mcpServers": {}}, "mcpServers", None, None
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except OSError as exc:
        return {}, "mcpServers", f"cannot read destination: {exc}", None
    except json.JSONDecodeError as exc:
        return (
            {},
            "mcpServers",
            f"destination contains invalid JSON at line {exc.lineno}, column {exc.colno}",
            None,
        )
    digest = hashlib.sha256(raw).hexdigest()
    if not isinstance(value, dict):
        return {}, "mcpServers", "destination root must be a JSON object", digest
    if "servers" in value:
        return (
            {},
            "mcpServers",
            "destination uses unsupported 'servers'; migrate it to 'mcpServers' before importing",
            digest,
        )
    if "mcpServers" not in value:
        value["mcpServers"] = {}
    if not isinstance(value["mcpServers"], dict):
        return {}, "mcpServers", "destination mcpServers must be a JSON object", digest
    return value, "mcpServers", None, digest


def _file_digest(path: Path) -> str | None:
    try:
        value = path.read_bytes()
    except FileNotFoundError:
        return None
    return hashlib.sha256(value).hexdigest()


def _entry_summary(value: dict[str, Any]) -> str:
    if isinstance(value.get("url"), str):
        target = redact_url(value["url"])
        mode = (
            "oauth"
            if value.get("auth") is not None or value.get("oauth") is not None
            else "headers"
        )
        return f"http {target} ({mode})"
    command = value.get("command")
    executable = command[0] if isinstance(command, list) and command else command
    return f"stdio {executable}"


def _entry_digest(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
