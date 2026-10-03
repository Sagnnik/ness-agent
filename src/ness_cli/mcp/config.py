"""Parse, normalize, redact, and fingerprint project MCP configuration."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlsplit

from dotenv import dotenv_values
from dotenv.parser import parse_stream

from ness_agent.mcp import (
    DEFAULT_STARTUP_TIMEOUT,
    MCPServerSpec,
    redact_text,
    redact_url,
    validate_mcp_http_url,
)
from ness_cli.mcp.models import (
    MCPConfigError,
    MCPOAuthSpec,
    MCPTrustPreview,
    ProjectMCPServer,
)
from ness_cli.mcp.schema import validate_server_structure
from ness_cli.terminal import terminal_safe_text

_PLACEHOLDER_RE = re.compile(r"\$\{([^{}]+)\}")


class ProjectMCPConfig:
    """Load project specs with a root resolved once when the instance is created."""

    def __init__(
        self,
        mcp_file: Path | None = None,
        *,
        project_root: Path | None = None,
    ) -> None:
        self.mcp_file = Path(mcp_file) if mcp_file is not None else None
        self.project_root = (
            Path.cwd() if project_root is None else Path(project_root)
        ).resolve()
        self._servers: dict[str, dict[str, Any]] = {}
        self._specs: dict[str, ProjectMCPServer] = {}
        self._fingerprint_material: dict[str, dict[str, Any]] = {}
        self._config_errors: list[str] = []
        self._loaded = False

    @property
    def servers(self) -> dict[str, dict[str, Any]]:
        return self._servers

    @property
    def specs(self) -> dict[str, ProjectMCPServer]:
        self.load()
        return dict(self._specs)

    @property
    def errors(self) -> tuple[str, ...]:
        self.load()
        return tuple(self._config_errors)

    def load(self) -> MCPTrustPreview:
        """Parse, normalize, and fingerprint project config without connecting."""
        if self._loaded:
            return self.trust_preview
        self._loaded = True
        self._specs.clear()
        self._fingerprint_material.clear()
        self._config_errors.clear()
        self._servers.clear()
        if self.mcp_file is None or not self.mcp_file.exists():
            return self.trust_preview
        try:
            config = json.loads(self.mcp_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            self._config_errors.append(
                f"invalid JSON at line {exc.lineno}, column {exc.colno}"
            )
            return self.trust_preview
        except OSError as exc:
            self._config_errors.append(f"cannot read config: {exc}")
            return self.trust_preview
        if not isinstance(config, dict):
            self._config_errors.append("config root must be a JSON object")
            return self.trust_preview
        if "servers" in config:
            self._config_errors.append("'servers' is not supported; use 'mcpServers'")
        raw_servers = config.get("mcpServers", {})
        if not isinstance(raw_servers, dict):
            self._config_errors.append("'mcpServers' must be a JSON object")
            return self.trust_preview

        for raw_name, raw_spec in raw_servers.items():
            name = str(raw_name)
            if not isinstance(raw_name, str) or not raw_name.strip():
                self._servers[name] = {
                    "status": "error",
                    "error": "server name must be a non-empty string",
                }
                continue
            if not isinstance(raw_spec, dict):
                self._servers[name] = {
                    "status": "error",
                    "error": "server definition must be an object",
                }
                continue
            try:
                spec, material = self._normalize_server(name, raw_spec)
            except MCPConfigError as exc:
                self._servers[name] = {
                    "status": "error",
                    "description": str(raw_spec.get("description") or ""),
                    "error": str(exc),
                }
                continue
            self._specs[name] = spec
            self._fingerprint_material[name] = material
            self._servers[name] = {
                "status": "configured",
                "description": spec.description,
                "transport": spec.transport,
                "tools": [],
            }
        return self.trust_preview

    @property
    def trust_preview(self) -> MCPTrustPreview:
        fingerprint: str | None = None
        if self._fingerprint_material:
            payload = json.dumps(
                self._fingerprint_material,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            fingerprint = hashlib.sha256(payload).hexdigest()
        summaries = tuple(
            self._server_preview(self._specs[name]) for name in sorted(self._specs)
        )
        return MCPTrustPreview(
            self.mcp_file.resolve() if self.mcp_file else None,
            fingerprint,
            summaries,
        )

    def reset(self) -> None:
        self._servers.clear()
        self._specs.clear()
        self._fingerprint_material.clear()
        self._config_errors.clear()
        self._loaded = False

    def server_spec(self, name: str) -> ProjectMCPServer | None:
        self.load()
        return self._specs.get(name)

    def _normalize_server(
        self, name: str, raw: dict[str, Any]
    ) -> tuple[ProjectMCPServer, dict[str, Any]]:
        redactions: list[str] = []
        transport, errors = validate_server_structure(raw)
        if errors:
            raise MCPConfigError(errors[0])
        description = raw.get("description", "")
        timeout = raw.get("startup_timeout", DEFAULT_STARTUP_TIMEOUT)

        if transport == "stdio":
            command_raw = raw.get("command")
            explicit_args = raw.get("args", [])
            command_parts = (
                [command_raw] if isinstance(command_raw, str) else list(command_raw)
            )
            command = self._expand(command_parts[0], "command", redactions)
            args = tuple(
                self._expand(value, "args", redactions)
                for value in [*command_parts[1:], *explicit_args]
            )
            raw_cwd = raw.get("cwd")
            cwd = (
                self._resolve_path(self._expand(raw_cwd, "cwd", redactions))
                if raw_cwd
                else self.project_root
            )
            raw_env_file = raw.get("envFile")
            env_file = (
                self._resolve_path(self._expand(raw_env_file, "envFile", redactions))
                if raw_env_file
                else None
            )
            raw_env = dict(raw.get("env") or {})
            explicit_env = {
                key: self._expand(value, f"env.{key}", redactions)
                for key, value in raw_env.items()
            }
            redactions.extend(explicit_env.values())
            env, env_file_values = self._stdio_environment(env_file, explicit_env)
            redactions.extend(env_file_values)
            connection = MCPServerSpec(
                name=name,
                transport="stdio",
                description=description,
                startup_timeout=float(timeout),
                command=command,
                args=args,
                cwd=cwd,
                env=tuple(sorted(env.items())),
                redactions=tuple(dict.fromkeys(value for value in redactions if value)),
            )
            material = {
                "transport": "stdio",
                "command": command,
                "args": list(args),
                "cwd": str(cwd),
                "envFile": str(env_file) if env_file else None,
                "env": raw_env,
            }
            return ProjectMCPServer(connection=connection, env_file=env_file), material

        raw_url = raw.get("url")
        url = self._expand(raw_url, "url", redactions)
        url_error = validate_mcp_http_url(url)
        if url_error:
            raise MCPConfigError(url_error)
        raw_headers = dict(raw.get("headers") or {})
        headers = {
            key: self._expand(value, f"headers.{key}", redactions)
            for key, value in raw_headers.items()
        }
        redactions.extend(headers.values())
        oauth, raw_oauth = self._normalize_oauth(raw, redactions)
        connection = MCPServerSpec(
            name=name,
            transport="http",
            description=description,
            startup_timeout=float(timeout),
            url=url,
            headers=tuple(sorted(headers.items())),
            redactions=tuple(dict.fromkeys(value for value in redactions if value)),
        )
        material = {
            "transport": "http",
            "url": url,
            "headers": raw_headers,
            "oauth": raw_oauth,
        }
        return ProjectMCPServer(connection=connection, oauth=oauth), material

    def _normalize_oauth(
        self, raw: dict[str, Any], redactions: list[str]
    ) -> tuple[MCPOAuthSpec | None, dict[str, Any] | None]:
        cursor_auth = raw.get("auth")
        claude_oauth = raw.get("oauth")
        if cursor_auth is None and claude_oauth is None:
            return None, None
        if cursor_auth is not None:
            client_id_raw = cursor_auth.get("CLIENT_ID")
            client_id = self._expand(client_id_raw, "auth.CLIENT_ID", redactions)
            secret_raw = cursor_auth.get("CLIENT_SECRET")
            client_secret = (
                self._expand(secret_raw, "auth.CLIENT_SECRET", redactions)
                if secret_raw
                else None
            )
            if client_secret:
                redactions.append(client_secret)
            scopes = self._normalize_scopes(
                cursor_auth.get("scopes", []), "auth.scopes", redactions
            )
            return (
                MCPOAuthSpec(
                    source="cursor",
                    client_id=client_id,
                    client_secret=client_secret,
                    scopes=scopes,
                    callback_port=8787,
                    token_endpoint_auth_method=(
                        "client_secret_post" if client_secret else "none"
                    ),
                ),
                dict(cursor_auth),
            )
        client_id_raw = claude_oauth.get("clientId")
        client_id = (
            self._expand(client_id_raw, "oauth.clientId", redactions)
            if client_id_raw
            else None
        )
        secret_raw = claude_oauth.get("clientSecret")
        client_secret = (
            self._expand(secret_raw, "oauth.clientSecret", redactions)
            if secret_raw
            else None
        )
        if client_secret:
            redactions.append(client_secret)
        callback_port = claude_oauth.get("callbackPort")
        scopes = self._normalize_scopes(
            claude_oauth.get("scopes", []), "oauth.scopes", redactions
        )
        default_method = "client_secret_post" if client_secret else "none"
        method = claude_oauth.get("tokenEndpointAuthMethod", default_method)
        return (
            MCPOAuthSpec(
                source="claude",
                client_id=client_id,
                client_secret=client_secret,
                scopes=scopes,
                callback_port=callback_port,
                token_endpoint_auth_method=method,
            ),
            dict(claude_oauth),
        )

    def _normalize_scopes(
        self, value: Any, field: str, redactions: list[str]
    ) -> tuple[str, ...]:
        if isinstance(value, str):
            expanded = self._expand(value, field, redactions)
            return tuple(part for part in expanded.split() if part)
        return tuple(
            scope
            for item in value
            for scope in self._expand(item, field, redactions).split()
            if scope
        )

    def _expand(
        self, value: str, field: str, redactions: list[str] | None = None
    ) -> str:
        reserved = {
            "userHome": str(Path.home()),
            "workspaceFolder": str(self.project_root),
            "workspaceFolderBasename": self.project_root.name,
            "pathSeparator": os.sep,
            "/": os.sep,
        }

        def replace_value(match: re.Match[str]) -> str:
            token = match.group(1)
            if token in reserved:
                return reserved[token]
            if token.startswith("env:"):
                key = token[4:]
                if key and key in os.environ:
                    resolved = os.environ[key]
                    if redactions is not None:
                        redactions.append(resolved)
                    return resolved
                raise MCPConfigError(
                    f"{field} references missing environment variable {key or '<empty>'}"
                )
            if ":-" in token:
                key, default = token.split(":-", 1)
                if not key:
                    raise MCPConfigError(
                        f"{field} contains an invalid environment placeholder"
                    )
                resolved = os.environ.get(key) or default
                if redactions is not None and key in os.environ:
                    redactions.append(resolved)
                return resolved
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", token):
                if token in os.environ:
                    resolved = os.environ[token]
                    if redactions is not None:
                        redactions.append(resolved)
                    return resolved
                raise MCPConfigError(
                    f"{field} references missing environment variable {token}"
                )
            return match.group(0)

        return _PLACEHOLDER_RE.sub(replace_value, value)

    def _resolve_path(self, value: str) -> Path:
        path = Path(value).expanduser()
        return (
            path.resolve()
            if path.is_absolute()
            else (self.project_root / path).resolve()
        )

    def _stdio_environment(
        self, env_file: Path | None, explicit_env: dict[str, str]
    ) -> tuple[dict[str, str], list[str]]:
        from mcp.client.stdio import get_default_environment

        result = get_default_environment()
        env_file_values: list[str] = []
        if env_file is not None:
            if not env_file.is_file():
                raise MCPConfigError("envFile does not exist or is not a file")
            try:
                with env_file.open(encoding="utf-8") as handle:
                    if any(binding.error for binding in parse_stream(handle)):
                        raise MCPConfigError("envFile contains invalid syntax")
                parsed = dotenv_values(env_file, interpolate=False)
            except OSError as exc:
                detail = exc.strerror or type(exc).__name__
                raise MCPConfigError(f"cannot read envFile: {detail}") from exc
            parsed_values = {
                key: value for key, value in parsed.items() if value is not None
            }
            result.update(parsed_values)
            env_file_values.extend(parsed_values.values())
        result.update(explicit_env)
        return result, env_file_values

    def _server_preview(self, spec: ProjectMCPServer) -> str:
        if spec.transport == "stdio":
            raw_env = self._fingerprint_material.get(spec.name, {}).get("env", {})
            env_names = sorted(raw_env) if isinstance(raw_env, dict) else []
            command = redact_text(spec.command or "", spec.redactions)
            args = redact_text(" ".join(spec.args), spec.redactions)
            cwd = redact_text(str(spec.cwd), spec.redactions)
            detail = f"{spec.name}: stdio — {command} {args}; cwd={cwd}"
            if spec.env_file:
                detail += (
                    f"; envFile={redact_text(str(spec.env_file), spec.redactions)}"
                )
            if env_names:
                detail += f"; env keys={', '.join(env_names)}"
            return terminal_safe_text(redact_text(detail, spec.redactions))
        header_names = ", ".join(sorted(dict(spec.headers))) or "none"
        safe_url = redact_url(redact_text(spec.url or "", spec.redactions))
        oauth = ""
        if spec.oauth is not None:
            mode = "static" if spec.oauth.client_id else "dynamic"
            oauth = f"; oauth={mode}"
            if spec.oauth.scopes:
                scopes = redact_text(", ".join(spec.oauth.scopes), spec.redactions)
                oauth += f" ({scopes})"
        return terminal_safe_text(
            redact_text(
                f"{spec.name}: http — {safe_url}; header keys={header_names}{oauth}",
                spec.redactions,
            )
        )


def validate_project_mcp_http_url(value: str) -> str | None:
    """Validate a possibly-placeholder-bearing imported project URL."""
    try:
        parts = urlsplit(value)
        _ = parts.port
    except ValueError:
        return "url is malformed"
    if parts.username is not None or parts.password is not None:
        return "url must not contain embedded credentials"
    if parts.fragment:
        return "url must not contain a fragment"
    unresolved = bool(_PLACEHOLDER_RE.search(value))
    if unresolved:
        if parts.scheme and parts.scheme not in {"http", "https"}:
            return "url must be an http(s) URL with a hostname"
        return None
    return validate_mcp_http_url(value)


def validate_import_entry(value: Any) -> tuple[list[str], list[str]]:
    """Validate a portable MCP entry without resolving its environment."""
    transport, errors = validate_server_structure(value)
    if not isinstance(value, dict):
        return errors, []
    warnings: list[str] = []
    url = value.get("url")
    if transport == "http" and isinstance(url, str) and url:
        url_error = validate_project_mcp_http_url(url)
        if url_error:
            errors.append(url_error)
    if _contains_placeholder(value):
        warnings.append(
            "contains unresolved placeholders; runtime environment must provide them"
        )
    if _contains_literal_secret(value):
        warnings.append(
            "contains literal credential values; environment placeholders are safer"
        )
    return errors, warnings


def _contains_placeholder(value: Any) -> bool:
    if isinstance(value, str):
        return bool(_PLACEHOLDER_RE.search(value))
    if isinstance(value, dict):
        return any(_contains_placeholder(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_placeholder(item) for item in value)
    return False


def _contains_literal_secret(value: dict[str, Any]) -> bool:
    candidates: list[str] = []
    for field_name in ("env", "headers"):
        mapping = value.get(field_name)
        if isinstance(mapping, dict):
            candidates.extend(
                item for item in mapping.values() if isinstance(item, str)
            )
    auth = value.get("auth")
    if isinstance(auth, dict) and isinstance(auth.get("CLIENT_SECRET"), str):
        candidates.append(auth["CLIENT_SECRET"])
    oauth = value.get("oauth")
    if isinstance(oauth, dict) and isinstance(oauth.get("clientSecret"), str):
        candidates.append(oauth["clientSecret"])
    url = value.get("url")
    if isinstance(url, str):
        try:
            parts = urlsplit(url)
            if parts.username:
                candidates.append(parts.username)
            if parts.password:
                candidates.append(parts.password)
            candidates.extend(query_value for _, query_value in parse_qsl(parts.query))
        except ValueError:
            pass
    return any(
        candidate and not _PLACEHOLDER_RE.search(candidate) for candidate in candidates
    )
