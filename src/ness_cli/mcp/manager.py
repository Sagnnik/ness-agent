"""Live project MCP state over the SDK MCP runtime."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from ness_agent.mcp import MCPRuntime, MCPServerSpec

from ness_cli.mcp.config import ProjectMCPConfig
from ness_cli.mcp.models import MCPTrustPreview, ProjectMCPServer
from ness_cli.terminal import terminal_safe_text

ProjectHTTPAuthFactory = Callable[[ProjectMCPServer], Awaitable[Any | None]]


class ProjectMCPManager:
    """Manage MCP servers with a project root resolved at instance creation."""

    def __init__(
        self,
        mcp_file: Path | None = None,
        *,
        project_root: Path | None = None,
        http_auth_factory: ProjectHTTPAuthFactory | None = None,
        runtime: MCPRuntime | None = None,
    ) -> None:
        self.mcp_file = Path(mcp_file) if mcp_file is not None else None
        self.project_root = (
            Path.cwd() if project_root is None else Path(project_root)
        ).resolve()
        self.http_auth_factory = http_auth_factory
        self.runtime = runtime or MCPRuntime(http_auth_factory=self._runtime_auth)
        self.runtime.http_auth_factory = self._runtime_auth
        self._config = ProjectMCPConfig(self.mcp_file, project_root=self.project_root)
        self._specs: dict[str, ProjectMCPServer] = {}
        self._states: dict[str, dict[str, Any]] = {}
        self._loaded = False

    @property
    def servers(self) -> dict[str, dict[str, Any]]:
        self.load()
        self._sync_runtime_states()
        return self._states

    @property
    def sessions(self) -> dict[str, Any]:
        return self.runtime.sessions

    @property
    def tools(self):  # type: ignore[no-untyped-def]
        return self.runtime.tools

    @property
    def config_errors(self) -> tuple[str, ...]:
        self.load()
        return self._config.errors

    def load(self) -> MCPTrustPreview:
        if self._loaded:
            return self._config.trust_preview
        preview = self._config.load()
        self._specs = self._config.specs
        self._states = {
            name: dict(state) for name, state in self._config.servers.items()
        }
        self._loaded = True
        return preview

    @property
    def trust_preview(self) -> MCPTrustPreview:
        return self.load()

    def mark_untrusted(self) -> None:
        self.load()
        for name in self._specs:
            self._states[name]["status"] = "pending_trust"
            self._states[name]["error"] = "configuration has not been trusted"

    async def start(self) -> None:
        self.load()
        await self.runtime.start(spec.connection for spec in self._specs.values())
        self._sync_runtime_states()

    async def start_server(self, name: str, spec: ProjectMCPServer) -> None:
        if name != spec.name:
            raise ValueError("MCP server name does not match its project spec")
        self.load()
        self._specs[name] = spec
        try:
            await self.runtime.start_server(spec.connection)
        finally:
            self._sync_runtime_states()

    async def stop(self) -> None:
        await self.runtime.stop()
        self._config.reset()
        self._specs.clear()
        self._states.clear()
        self._loaded = False

    def server_spec(self, name: str) -> ProjectMCPServer | None:
        self.load()
        return self._specs.get(name)

    def list_tools(self) -> list[str]:
        return self.runtime.list_tools()

    def catalog(self) -> dict[str, dict[str, Any]]:
        return self.runtime.catalog()

    async def call(
        self,
        server_name: str,
        tool_name: str,
        args: dict[str, Any],
        *,
        timeout: float = 60.0,
    ) -> str:
        return await self.runtime.call(server_name, tool_name, args, timeout=timeout)

    def startup_summary(self) -> tuple[str, str]:
        servers = self.servers
        errors = self.config_errors
        if not servers:
            if errors:
                return terminal_safe_text(
                    f"MCP: config error, {'; '.join(errors)}"
                ), "warn"
            return "MCP: none configured", "none"

        connected = [
            name for name, info in servers.items() if info.get("status") == "connected"
        ]
        failed = [
            name for name, info in servers.items() if info.get("status") != "connected"
        ]
        if not failed and not errors:
            return terminal_safe_text(
                f"MCP: {len(connected)} server(s), {len(self.tools)} tool(s) "
                f"({', '.join(connected)})"
            ), "ok"

        details = [
            f"{name}: {servers[name].get('error', servers[name].get('status', 'failed'))}"
            for name in failed
        ]
        details[:0] = errors
        if connected:
            return terminal_safe_text(
                f"MCP: {len(connected)}/{len(servers)} connected, "
                f"{len(self.tools)} tool(s) ({', '.join(connected)}), "
                + "; ".join(details)
            ), "warn"
        return terminal_safe_text(
            f"MCP: 0/{len(servers)} connected, " + "; ".join(details)
        ), "warn"

    def status(self) -> str:
        lines = [f"Config warning: {error}" for error in self.config_errors]
        servers = self.servers
        if not servers:
            lines.append("No MCP servers configured or started")
            return terminal_safe_text("\n".join(lines), multiline=True)
        for name, info in servers.items():
            status = info.get("status")
            if status == "connected":
                lines.append(
                    f"- {name}: connected ({len(info.get('tools', []))} tools)"
                )
                lines.extend(
                    f"  - mcp__{name}__{tool}" for tool in info.get("tools", [])
                )
            elif status == "pending_trust":
                lines.append(f"- {name}: pending trust")
            elif status == "configured":
                lines.append(f"- {name}: configured, not started")
            elif status == "auth_required":
                lines.append(
                    f"- {name}: authentication required (run `ness mcp login {name}`)"
                )
            else:
                lines.append(f"- {name}: error: {info.get('error', 'failed')}")
        return terminal_safe_text("\n".join(lines), multiline=True)

    async def _runtime_auth(self, spec: MCPServerSpec) -> Any | None:
        if self.http_auth_factory is None:
            return None
        project_spec = self._specs.get(spec.name)
        if project_spec is None:
            raise RuntimeError("MCP project server metadata is unavailable")
        return await self.http_auth_factory(project_spec)

    def _sync_runtime_states(self) -> None:
        for name, state in self.runtime.states.items():
            error = state.error
            if state.status == "auth_required":
                error = f"authentication required; run `ness mcp login {name}`"
            self._states[name] = {
                "status": state.status,
                "description": state.description,
                "transport": state.transport,
                "tools": list(state.tools),
            }
            if error:
                self._states[name]["error"] = terminal_safe_text(error)
