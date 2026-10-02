"""Typed project-level MCP configuration records."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from ness_agent.mcp import MCPServerSpec


@dataclass(frozen=True, slots=True)
class MCPOAuthSpec:
    source: Literal["cursor", "claude"]
    client_id: str | None = None
    client_secret: str | None = None
    scopes: tuple[str, ...] = ()
    callback_port: int | None = None
    token_endpoint_auth_method: Literal[
        "none", "client_secret_post", "client_secret_basic"
    ] = "none"


@dataclass(frozen=True, slots=True)
class ProjectMCPServer:
    connection: MCPServerSpec
    oauth: MCPOAuthSpec | None = None
    env_file: Path | None = None

    @property
    def name(self) -> str:
        return self.connection.name

    @property
    def transport(self) -> Literal["stdio", "http"]:
        return self.connection.transport

    @property
    def description(self) -> str:
        return self.connection.description

    @property
    def startup_timeout(self) -> float:
        return self.connection.startup_timeout

    @property
    def command(self) -> str | None:
        return self.connection.command

    @property
    def args(self) -> tuple[str, ...]:
        return self.connection.args

    @property
    def cwd(self) -> Path | None:
        return self.connection.cwd

    @property
    def env(self) -> tuple[tuple[str, str], ...]:
        return self.connection.env

    @property
    def url(self) -> str | None:
        return self.connection.url

    @property
    def headers(self) -> tuple[tuple[str, str], ...]:
        return self.connection.headers

    @property
    def redactions(self) -> tuple[str, ...]:
        return self.connection.redactions

    def with_startup_timeout(self, timeout: float) -> ProjectMCPServer:
        return replace(
            self,
            connection=replace(self.connection, startup_timeout=timeout),
        )


@dataclass(frozen=True, slots=True)
class MCPTrustPreview:
    config_path: Path | None
    fingerprint: str | None
    servers: tuple[str, ...]

    @property
    def has_runnable_servers(self) -> bool:
        return bool(self.servers and self.fingerprint)


class MCPConfigError(ValueError):
    """Project MCP configuration could not be normalized safely."""
