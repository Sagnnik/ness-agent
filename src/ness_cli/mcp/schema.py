"""Shared project MCP schema checks, before any environment resolution.

Import and config loading use the same structural rules. URL validation belongs
to the caller: imports retain placeholders, while connections require resolved
HTTP URLs. Credential warnings and expansion also belong to the caller.
"""

from typing import Any

from ness_agent.mcp import DEFAULT_STARTUP_TIMEOUT


def validate_server_structure(value: Any) -> tuple[str, list[str]]:
    """Return the normalized transport and all structural errors."""
    if not isinstance(value, dict):
        return "", ["server definition must be an object"]
    errors: list[str] = []
    raw_type = value.get("type")
    if raw_type is not None and not isinstance(raw_type, str):
        errors.append("type must be a string")
    has_command, has_url = "command" in value, "url" in value
    transport = raw_type.lower() if isinstance(raw_type, str) else ""
    if not transport:
        transport = "http" if has_url else "stdio" if has_command else ""
    if transport == "streamable-http":
        transport = "http"
    if has_command and has_url:
        errors.append("server cannot contain both command and url")
    if transport in {"sse", "ws", "websocket"}:
        errors.append(f"unsupported transport: {transport}")
    elif transport not in {"stdio", "http"}:
        errors.append("type must be stdio, http, or streamable-http")
    if not isinstance(value.get("description", ""), str):
        errors.append("description must be a string")
    timeout = value.get("startup_timeout", DEFAULT_STARTUP_TIMEOUT)
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or timeout <= 0
    ):
        errors.append("startup_timeout must be a positive number")
    if transport == "stdio":
        if not has_command or has_url:
            errors.append("stdio server requires command and cannot contain url")
        command = value.get("command")
        if isinstance(command, str):
            if not command.strip():
                errors.append("command must not be empty")
        elif not (
            isinstance(command, list)
            and command
            and all(isinstance(part, str) and part for part in command)
        ):
            errors.append("command must be a non-empty string or array of strings")
        args = value.get("args", [])
        if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
            errors.append("args must be an array of strings")
        for field in ("cwd", "envFile"):
            if value.get(field) is not None and not isinstance(value[field], str):
                errors.append(f"{field} must be a string")
    if transport == "http":
        if not has_url or has_command:
            errors.append("http server requires url and cannot contain command")
        if not isinstance(value.get("url"), str) or not value["url"]:
            errors.append("url must be a non-empty string")
        if value.get("envFile") is not None:
            errors.append("envFile is only supported for stdio servers")
    for field in ("env", "headers"):
        mapping = value.get(field)
        if mapping is not None and not (
            isinstance(mapping, dict)
            and all(
                isinstance(key, str) and isinstance(item, str)
                for key, item in mapping.items()
            )
        ):
            errors.append(f"{field} must be an object of string values")
    if value.get("headersHelper") is not None:
        errors.append("headersHelper is not supported")
    _validate_oauth(value, errors)
    has_oauth = value.get("auth") is not None or value.get("oauth") is not None
    headers = value.get("headers")
    if (
        has_oauth
        and isinstance(headers, dict)
        and any(
            isinstance(key, str) and key.lower() == "authorization" for key in headers
        )
    ):
        errors.append("OAuth cannot be combined with an explicit Authorization header")
    if has_oauth and transport != "http":
        errors.append("OAuth is supported only for HTTP servers")
    return transport, errors


def _validate_scopes(value: Any, field: str, errors: list[str]) -> None:
    if not (
        isinstance(value, str)
        or isinstance(value, list)
        and all(isinstance(scope, str) for scope in value)
    ):
        errors.append(f"{field} must be a string or array of strings")


def _validate_oauth(value: dict[str, Any], errors: list[str]) -> None:
    auth, oauth = value.get("auth"), value.get("oauth")
    if auth is not None and oauth is not None:
        errors.append("server cannot contain both auth and oauth")
    if auth is not None:
        if not isinstance(auth, dict):
            errors.append("auth must be an object")
        else:
            if not isinstance(auth.get("CLIENT_ID"), str) or not auth["CLIENT_ID"]:
                errors.append("auth.CLIENT_ID must be a non-empty string")
            secret = auth.get("CLIENT_SECRET")
            if secret is not None and not isinstance(secret, str):
                errors.append("auth.CLIENT_SECRET must be a string")
            _validate_scopes(auth.get("scopes", []), "auth.scopes", errors)
    if oauth is None:
        return
    if not isinstance(oauth, dict):
        errors.append("oauth must be an object")
        return
    if oauth.get("authServerMetadataUrl") is not None:
        errors.append("oauth.authServerMetadataUrl is not supported")
    for field in ("clientId", "clientSecret"):
        if oauth.get(field) is not None and not isinstance(oauth[field], str):
            errors.append(f"oauth.{field} must be a string")
    port = oauth.get("callbackPort")
    if port is not None and (
        isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535
    ):
        errors.append("oauth.callbackPort must be an integer from 1 to 65535")
    _validate_scopes(oauth.get("scopes", []), "oauth.scopes", errors)
    if "tokenEndpointAuthMethod" in oauth and oauth["tokenEndpointAuthMethod"] not in (
        "none",
        "client_secret_post",
        "client_secret_basic",
    ):
        errors.append(
            "oauth.tokenEndpointAuthMethod must be none, client_secret_post, or client_secret_basic"
        )
