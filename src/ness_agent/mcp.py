"""Adapter-neutral MCP runtime for SDK consumers.

The runtime accepts fully resolved server specifications. Project config
formats, trust policy, OAuth persistence, and terminal presentation belong to
the embedding application (the Ness CLI provides one such adapter).
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Awaitable, Callable, Iterable
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit

from langchain_core.tools import StructuredTool
from pydantic import Field, create_model

DEFAULT_STARTUP_TIMEOUT = 20.0
DEFAULT_CALL_TIMEOUT = 60.0

MCPTransport = Literal["stdio", "http"]
MCPStatus = Literal["connecting", "connected", "auth_required", "error"]


@dataclass(frozen=True)
class MCPServerSpec:
    """A fully resolved MCP connection specification.

    ``env`` is the complete child environment when provided. Passing an empty
    tuple delegates safe default-environment construction to the MCP SDK.
    ``redactions`` contains resolved secret values that must not appear in
    runtime errors.
    """

    name: str
    transport: MCPTransport
    description: str = ""
    startup_timeout: float = DEFAULT_STARTUP_TIMEOUT
    command: str | None = None
    args: tuple[str, ...] = ()
    cwd: Path | None = None
    env: tuple[tuple[str, str], ...] = ()
    url: str | None = None
    headers: tuple[tuple[str, str], ...] = ()
    redactions: tuple[str, ...] = ()


@dataclass(frozen=True)
class MCPServerState:
    """Structured connection state for one runtime server."""

    name: str
    status: MCPStatus
    description: str
    transport: MCPTransport
    tools: tuple[str, ...] = ()
    error: str | None = None


class MCPAuthenticationRequired(RuntimeError):
    """An HTTP server needs authentication supplied by its embedding app."""


HTTPAuthFactory = Callable[[MCPServerSpec], Awaitable[Any | None]]


class MCPRuntime:
    """Connect resolved MCP servers and expose their tools as LangChain tools."""

    def __init__(self, *, http_auth_factory: HTTPAuthFactory | None = None) -> None:
        self.http_auth_factory = http_auth_factory
        self.states: dict[str, MCPServerState] = {}
        self.sessions: dict[str, Any] = {}
        self.tools: dict[str, StructuredTool] = {}
        self.tool_meta: dict[str, dict[str, Any]] = {}
        self._stacks: dict[str, AsyncExitStack] = {}
        self._server_tasks: dict[str, asyncio.Task[None]] = {}
        self._stop_events: dict[str, asyncio.Event] = {}
        self._specs: dict[str, MCPServerSpec] = {}

    async def start(self, specs: Iterable[MCPServerSpec]) -> None:
        """Start all provided servers concurrently, isolating failures."""
        resolved = list(specs)
        names: set[str] = set()
        for spec in resolved:
            _validate_server_spec(spec)
            if spec.name in names:
                raise ValueError(f"duplicate MCP server name: {spec.name}")
            names.add(spec.name)
        pending: list[MCPServerSpec] = []
        for spec in resolved:
            if spec.name in self._server_tasks:
                continue
            self._specs[spec.name] = spec
            self.states[spec.name] = MCPServerState(
                name=spec.name,
                status="connecting",
                description=spec.description,
                transport=spec.transport,
            )
            pending.append(spec)
        if pending:
            await asyncio.gather(
                *(self.start_server(spec) for spec in pending),
                return_exceptions=True,
            )

    async def start_server(self, spec: MCPServerSpec) -> None:
        """Start one resolved server and raise if initial connection fails."""
        _validate_server_spec(spec)
        if spec.name in self._server_tasks:
            return
        self._specs[spec.name] = spec
        self.states[spec.name] = MCPServerState(
            name=spec.name,
            status="connecting",
            description=spec.description,
            transport=spec.transport,
        )
        ready: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        stop_event = asyncio.Event()
        self._stop_events[spec.name] = stop_event
        self._server_tasks[spec.name] = asyncio.create_task(
            self._run_server(spec, ready, stop_event),
            name=f"mcp-server-{spec.name}",
        )
        await ready

    async def stop(self) -> None:
        """Stop every server and reset runtime state."""
        for event in self._stop_events.values():
            event.set()
        if self._server_tasks:
            await asyncio.gather(*self._server_tasks.values(), return_exceptions=True)
        self._stacks.clear()
        self._server_tasks.clear()
        self._stop_events.clear()
        self.sessions.clear()
        self.tools.clear()
        self.tool_meta.clear()
        self.states.clear()
        self._specs.clear()

    def list_tools(self) -> list[str]:
        return sorted(self.tools)

    def catalog(self) -> dict[str, dict[str, Any]]:
        """Return discovered tools grouped by server."""
        catalog: dict[str, dict[str, Any]] = {}
        for full_name, meta in self.tool_meta.items():
            server = str(meta.get("server") or "")
            state = self.states.get(server)
            entry = catalog.setdefault(
                server,
                {
                    "description": state.description if state else "",
                    "tools": [],
                },
            )
            entry["tools"].append(
                {
                    "name": full_name,
                    "tool": meta.get("tool", ""),
                    "description": meta.get("description", ""),
                    "arg_names": meta.get("arg_names", []),
                }
            )
        return catalog

    async def call(
        self,
        server_name: str,
        tool_name: str,
        args: dict[str, Any],
        *,
        timeout: float = DEFAULT_CALL_TIMEOUT,
    ) -> str:
        """Invoke one connected MCP tool and serialize its result."""
        session = self.sessions.get(server_name)
        if session is None:
            return f"Error: MCP server not connected: {server_name}"
        payload = _sanitize_tool_args(args)
        try:
            result = await asyncio.wait_for(
                session.call_tool(tool_name, payload), timeout=timeout
            )
        except asyncio.TimeoutError:
            return f"Error: MCP tool call timed out ({timeout}s): {server_name}/{tool_name}"
        except Exception as exc:
            spec = self._specs.get(server_name)
            detail = _redact_error(exc, spec) if spec else type(exc).__name__
            return f"Error: MCP call failed: {detail}"
        if result.isError:
            return "Error: " + _serialize_mcp_result(result)
        return _serialize_mcp_result(result)

    async def _run_server(
        self,
        spec: MCPServerSpec,
        ready: asyncio.Future[None],
        stop_event: asyncio.Event,
    ) -> None:
        name = spec.name
        stack = AsyncExitStack()
        await stack.__aenter__()
        try:
            connect = (
                self._connect_stdio if spec.transport == "stdio" else self._connect_http
            )
            await asyncio.wait_for(connect(spec, stack), timeout=spec.startup_timeout)
            self._stacks[name] = stack
            if not ready.done():
                ready.set_result(None)
            await stop_event.wait()
        except BaseException as exc:
            self._remove_server_tools(name)
            if not stop_event.is_set():
                self._set_failure(spec, exc)
            if not ready.done():
                ready.set_exception(exc)
        finally:
            self._stacks.pop(name, None)
            try:
                await stack.aclose()
            except Exception:
                pass
            current_task = asyncio.current_task()
            if self._server_tasks.get(name) is current_task:
                self._server_tasks.pop(name, None)
                self._stop_events.pop(name, None)

    def _set_failure(self, spec: MCPServerSpec, exc: BaseException) -> None:
        auth_required = _is_auth_required(exc)
        self.states[spec.name] = MCPServerState(
            name=spec.name,
            status="auth_required" if auth_required else "error",
            description=spec.description,
            transport=spec.transport,
            error="authentication required" if auth_required else _redact_error(exc, spec),
        )

    def _remove_server_tools(self, name: str) -> None:
        self.sessions.pop(name, None)
        for full_name in [
            key for key, meta in self.tool_meta.items() if meta.get("server") == name
        ]:
            self.tools.pop(full_name, None)
            self.tool_meta.pop(full_name, None)

    async def _connect_stdio(
        self, spec: MCPServerSpec, stack: AsyncExitStack
    ) -> None:
        from mcp.client.stdio import StdioServerParameters, stdio_client

        params = StdioServerParameters(
            command=spec.command or "",
            args=list(spec.args),
            env=dict(spec.env) if spec.env else None,
            cwd=str(spec.cwd) if spec.cwd else None,
        )
        errlog = open(os.devnull, "w", encoding="utf-8")
        stack.callback(errlog.close)
        read_stream, write_stream = await stack.enter_async_context(
            stdio_client(params, errlog=errlog)
        )
        await self._initialize_session(spec, read_stream, write_stream, stack)

    async def _connect_http(
        self, spec: MCPServerSpec, stack: AsyncExitStack
    ) -> None:
        import httpx
        from mcp.client.streamable_http import streamable_http_client

        auth = await self.http_auth_factory(spec) if self.http_auth_factory else None
        client = await stack.enter_async_context(
            httpx.AsyncClient(
                headers=dict(spec.headers),
                auth=auth,
                follow_redirects=True,
                timeout=httpx.Timeout(30.0, read=300.0),
            )
        )
        read_stream, write_stream, _ = await stack.enter_async_context(
            streamable_http_client(spec.url or "", http_client=client)
        )
        await self._initialize_session(spec, read_stream, write_stream, stack)

    async def _initialize_session(
        self,
        spec: MCPServerSpec,
        read_stream: Any,
        write_stream: Any,
        stack: AsyncExitStack,
    ) -> None:
        from mcp.client.session import ClientSession

        session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
        await session.initialize()
        result = await session.list_tools()
        self.sessions[spec.name] = session
        tool_names = tuple(tool.name for tool in result.tools)
        self.states[spec.name] = MCPServerState(
            name=spec.name,
            status="connected",
            description=spec.description,
            transport=spec.transport,
            tools=tool_names,
        )
        for mcp_tool in result.tools:
            full_name = f"mcp__{spec.name}__{mcp_tool.name}"
            self.tools[full_name] = self._wrap_tool(spec.name, mcp_tool)
            self.tool_meta[full_name] = {
                "server": spec.name,
                "tool": mcp_tool.name,
                "description": getattr(mcp_tool, "description", "") or "",
                "arg_names": _input_arg_names(getattr(mcp_tool, "inputSchema", None)),
            }

    def _wrap_tool(self, server_name: str, mcp_tool: Any) -> StructuredTool:
        full_name = f"mcp__{server_name}__{mcp_tool.name}"
        raw_schema = getattr(mcp_tool, "inputSchema", None)
        args_schema = _args_schema(full_name, raw_schema)
        description = (
            getattr(mcp_tool, "description", "")
            or f"MCP tool {mcp_tool.name} from {server_name}"
        )
        if isinstance(raw_schema, dict):
            description += f"\n\nInput schema: {json.dumps(raw_schema)}"

        async def _call(**kwargs: Any) -> str:
            return await self.call(server_name, mcp_tool.name, kwargs)

        return StructuredTool.from_function(
            name=full_name,
            description=description,
            coroutine=_call,
            args_schema=args_schema,
        )


def redact_url(value: str) -> str:
    """Strip URL credentials, query, and fragment; mask malformed URLs.

    This formats diagnostic text; it does not validate a connection URL.
    """
    try:
        parts = urlsplit(value)
        host = parts.hostname or ""
        if parts.port:
            host = f"{host}:{parts.port}"
        return urlunsplit((parts.scheme, host, parts.path, "", ""))
    except ValueError:
        return "[invalid URL]"


def redact_text(
    value: str, secrets: tuple[str, ...], *, fallback: str = "[redacted]"
) -> str:
    """Replace literal secrets, longest first, ignoring empty secrets.

    A matching secret shorter than four characters replaces the whole message
    with ``fallback`` to avoid exposing fragments or corrupting ordinary words.
    Callers still need to escape text for their output format.
    """
    result = value
    for secret in sorted(set(secrets), key=len, reverse=True):
        if not secret or secret not in result:
            continue
        if len(secret) < 4:
            return fallback
        result = result.replace(secret, "[redacted]")
    return result


def validate_mcp_http_url(value: str) -> str | None:
    """Return an error when a resolved MCP HTTP URL is unsafe or malformed."""
    try:
        parts = urlsplit(value)
        _ = parts.port
    except ValueError:
        return "url is malformed"
    if parts.username is not None or parts.password is not None:
        return "url must not contain embedded credentials"
    if parts.fragment:
        return "url must not contain a fragment"
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        return "url must be an http(s) URL with a hostname"
    return None


def _validate_server_spec(spec: MCPServerSpec) -> None:
    if not spec.name.strip():
        raise ValueError("MCP server name must not be empty")
    if spec.transport not in {"stdio", "http"}:
        raise ValueError("MCP transport must be stdio or http")
    if isinstance(spec.startup_timeout, bool) or spec.startup_timeout <= 0:
        raise ValueError("MCP startup timeout must be positive")
    if spec.transport == "stdio":
        if not spec.command:
            raise ValueError("stdio MCP server requires a command")
        if spec.url is not None:
            raise ValueError("stdio MCP server cannot contain a URL")
        return
    if spec.command is not None:
        raise ValueError("HTTP MCP server cannot contain a command")
    if not spec.url:
        raise ValueError("HTTP MCP server requires a URL")
    error = validate_mcp_http_url(spec.url)
    if error:
        raise ValueError(error)


def _redact_error(exc: BaseException, spec: MCPServerSpec) -> str:
    message = str(exc) or type(exc).__name__
    if spec.url:
        message = message.replace(spec.url, redact_url(spec.url))
    return redact_text(message, spec.redactions, fallback=type(exc).__name__)


def _is_auth_required(exc: BaseException) -> bool:
    if isinstance(exc, MCPAuthenticationRequired):
        return True
    if getattr(exc, "status_code", None) in {401, 403}:
        return True
    nested = getattr(exc, "exceptions", None)
    if nested and any(_is_auth_required(item) for item in nested):
        return True
    message = str(exc).lower()
    return "401 unauthorized" in message or "authentication required" in message


def _sanitize_tool_args(args: dict[str, Any]) -> dict[str, Any]:
    """Drop null optional padding before forwarding arguments to MCP."""
    if not isinstance(args, dict):
        return {}

    def clean(value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            if item is None:
                continue
            normalized = clean(item)
            if normalized == {}:
                continue
            cleaned[key] = normalized
        return cleaned

    cleaned = clean(args)
    return cleaned if isinstance(cleaned, dict) else {}


def _input_arg_names(schema: dict[str, Any] | None) -> list[str]:
    if not isinstance(schema, dict):
        return []
    properties = schema.get("properties")
    return list(properties.keys()) if isinstance(properties, dict) else []


def _args_schema(name: str, schema: dict[str, Any] | None) -> type:
    if not isinstance(schema, dict):
        return create_model(f"{name}_Args")
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    fields: dict[str, Any] = {}
    for prop, info in properties.items():
        if not isinstance(info, dict):
            fields[prop] = (Any, None)
            continue
        typ = _resolve_type(info)
        fields[prop] = (
            typ,
            Field(
                ... if prop in required else info.get("default", None),
                description=info.get("description", ""),
            ),
        )
    return create_model(f"{name}_Args", **fields)


def _resolve_type(info: dict[str, Any]) -> type:
    if "enum" in info:
        try:
            from typing import Literal as TypingLiteral

            return TypingLiteral[tuple(info["enum"])]
        except (TypeError, ValueError):
            return str
    return {
        "object": dict,
        "array": list,
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
    }.get(info.get("type"), Any)


def _serialize_mcp_result(result: Any) -> str:
    parts: list[str] = []
    for item in getattr(result, "content", []) or []:
        content_type = getattr(item, "type", None)
        if content_type == "image":
            parts.append("[image]")
        elif content_type == "resource":
            resource = getattr(item, "resource", None)
            parts.append(resource.text if resource and hasattr(resource, "text") else str(item))
        else:
            text = getattr(item, "text", None)
            if text is not None:
                parts.append(text)
            elif hasattr(item, "model_dump"):
                parts.append(json.dumps(item.model_dump(), ensure_ascii=False))
            else:
                parts.append(str(item))
    structured = getattr(result, "structuredContent", None)
    if structured:
        parts.append(json.dumps(structured, ensure_ascii=False))
    return "\n".join(parts) if parts else str(result)
