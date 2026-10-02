"""OAuth authentication and secure token storage for MCP HTTP servers."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from mcp.client.auth import OAuthClientProvider
from mcp.shared.auth import (
    OAuthClientInformationFull,
    OAuthClientMetadata,
    OAuthToken,
)

from ness_agent.mcp import MCPAuthenticationRequired
from ness_cli.config.store import (
    atomic_write_json,
    locked_path,
)
from ness_cli.mcp.models import (
    MCPOAuthSpec,
    ProjectMCPServer,
)

_KEYRING_SERVICE = "ness-agent/mcp-oauth"
_FALLBACK_NAME = "mcp_oauth.json"
_RECORD_VERSION = 1


class OAuthCredentialClearError(RuntimeError):
    """Local OAuth credentials could not be fully removed or verified."""


@dataclass(frozen=True)
class _KeyringReadResult:
    available: bool
    value: str | None
    error: Exception | None = None

    def __iter__(self):  # type: ignore[no-untyped-def]
        yield self.available
        yield self.value


def credential_id(project_root: Path, spec: ProjectMCPServer) -> str:
    oauth = spec.oauth
    payload = "\0".join(
        (
            str(project_root.resolve()),
            spec.name,
            spec.url or "",
            oauth.client_id if oauth and oauth.client_id else "",
            " ".join(oauth.scopes) if oauth else "",
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ProjectOAuthTokenStorage:
    """Project-isolated SDK TokenStorage backed by keyring with file fallback."""

    def __init__(
        self,
        *,
        config_dir: Path,
        identity: str,
        static_client_info: OAuthClientInformationFull | None = None,
        warnings: list[str] | None = None,
    ) -> None:
        self.config_dir = config_dir
        self.identity = identity
        self.static_client_info = static_client_info
        self.warnings = warnings if warnings is not None else []
        self._lock = asyncio.Lock()
        self.token_expiry_time: float | None = None

    @property
    def fallback_path(self) -> Path:
        return self.config_dir / _FALLBACK_NAME

    async def get_tokens(self) -> OAuthToken | None:
        record = await self._read_record()
        value = record.get("tokens") if record else None
        if not isinstance(value, dict):
            return None
        try:
            token = OAuthToken.model_validate(value)
            expires_at = record.get("expires_at") if record else None
            self.token_expiry_time = (
                float(expires_at) if isinstance(expires_at, (int, float)) else None
            )
            return token
        except Exception:
            self._warn("Stored MCP OAuth tokens are corrupt; login is required.")
            return None

    async def set_tokens(self, tokens: OAuthToken) -> None:
        async with self._lock:
            record = await self._read_record_unlocked() or {}
            previous = record.get("tokens")
            if (
                tokens.refresh_token is None
                and isinstance(previous, dict)
                and isinstance(previous.get("refresh_token"), str)
            ):
                tokens.refresh_token = previous["refresh_token"]
            record["version"] = _RECORD_VERSION
            record["tokens"] = tokens.model_dump(mode="json", exclude_none=True)
            record["expires_at"] = (
                time.time() + tokens.expires_in
                if tokens.expires_in is not None
                else None
            )
            record["updated_at"] = datetime.now(timezone.utc).isoformat()
            await self._write_record_unlocked(record)

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        if self.static_client_info is not None:
            return self.static_client_info
        record = await self._read_record()
        value = record.get("client_info") if record else None
        if not isinstance(value, dict):
            return None
        try:
            return OAuthClientInformationFull.model_validate(value)
        except Exception:
            self._warn(
                "Stored MCP OAuth client registration is corrupt; login is required."
            )
            return None

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        if self.static_client_info is not None:
            return
        async with self._lock:
            record = await self._read_record_unlocked() or {}
            record["version"] = _RECORD_VERSION
            record["client_info"] = client_info.model_dump(
                mode="json", exclude_none=True
            )
            record["updated_at"] = datetime.now(timezone.utc).isoformat()
            await self._write_record_unlocked(record)

    async def has_credentials(self) -> bool:
        return await self.get_tokens() is not None

    async def backend_name(self) -> str:
        available, _ = await _keyring_get(self.identity)
        return "keyring" if available else f"file ({self.fallback_path})"

    async def clear(self) -> None:
        async with self._lock:
            failure: BaseException | None = None
            keyring_result = await _keyring_get(self.identity)
            keyring_available, encoded = keyring_result
            keyring_error = getattr(keyring_result, "error", None)
            if keyring_error is not None:
                failure = keyring_error
            elif keyring_available and encoded is not None:
                try:
                    await _keyring_delete(self.identity)
                    verification = await _keyring_get(self.identity)
                    verified_available, verified_value = verification
                    verification_error = getattr(verification, "error", None)
                    if verification_error is not None:
                        raise verification_error
                    if not verified_available or verified_value is not None:
                        raise RuntimeError(
                            "keyring credential deletion could not be verified"
                        )
                except Exception as exc:
                    failure = exc

            try:
                await asyncio.to_thread(self._remove_fallback_record, strict=True)
            except Exception as exc:
                failure = failure or exc

            if failure is not None:
                raise OAuthCredentialClearError(
                    "local OAuth credentials could not be fully removed or verified"
                ) from failure

    async def _read_record(self) -> dict[str, Any] | None:
        async with self._lock:
            return await self._read_record_unlocked()

    async def _read_record_unlocked(self) -> dict[str, Any] | None:
        keyring_available, encoded = await _keyring_get(self.identity)
        if encoded:
            record = _decode_record(encoded)
            if record is None:
                self._warn(
                    "Stored MCP OAuth keyring record is corrupt; login is required."
                )
            return record

        fallback = await asyncio.to_thread(self._fallback_record)
        if fallback is not None and keyring_available:
            migrated = await _keyring_set(
                self.identity,
                json.dumps(fallback, separators=(",", ":")),
            )
            if migrated:
                await asyncio.to_thread(self._remove_fallback_record)
            else:
                self._warn_fallback()
            return fallback
        if fallback is not None:
            self._warn_fallback()
        return fallback

    async def _write_record_unlocked(self, record: dict[str, Any]) -> None:
        available, _ = await _keyring_get(self.identity)
        encoded = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        if available and await _keyring_set(self.identity, encoded):
            await asyncio.to_thread(self._remove_fallback_record)
            return
        self._warn_fallback()
        await asyncio.to_thread(self._store_fallback_record, record)

    def _fallback_record(self) -> dict[str, Any] | None:
        with locked_path(self.fallback_path, secret=True):
            document = self._fallback_document()
            if document is None:
                return None
            records = document.get("records", {})
            value = records.get(self.identity) if isinstance(records, dict) else None
            return dict(value) if isinstance(value, dict) else None

    def _store_fallback_record(self, record: dict[str, Any]) -> None:
        with locked_path(self.fallback_path, secret=True):
            document = self._fallback_document(strict=True)
            assert document is not None
            records = document.get("records", {})
            records = dict(records) if isinstance(records, dict) else {}
            records[self.identity] = record
            atomic_write_json(
                self.fallback_path,
                {"version": _RECORD_VERSION, "records": records},
                secret=True,
            )

    def _remove_fallback_record(self, *, strict: bool = False) -> None:
        with locked_path(self.fallback_path, secret=True):
            document = self._fallback_document(strict=strict)
            if document is None:
                return
            records = document.get("records", {})
            if not isinstance(records, dict) or self.identity not in records:
                return
            updated = dict(records)
            del updated[self.identity]
            atomic_write_json(
                self.fallback_path,
                {"version": _RECORD_VERSION, "records": updated},
                secret=True,
            )

    def _fallback_document(self, *, strict: bool = False) -> dict[str, Any] | None:
        if not self.fallback_path.exists():
            return {"version": _RECORD_VERSION, "records": {}}
        try:
            value = json.loads(self.fallback_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            self._warn(
                "MCP OAuth fallback file is unreadable or corrupt; login is required."
            )
            if strict:
                raise RuntimeError(
                    "refusing to overwrite corrupt MCP OAuth fallback storage"
                ) from exc
            return None
        if not isinstance(value, dict) or not isinstance(
            value.get("records", {}), dict
        ):
            self._warn(
                "MCP OAuth fallback file has an invalid shape; login is required."
            )
            if strict:
                raise RuntimeError(
                    "refusing to overwrite invalid MCP OAuth fallback storage"
                )
            return None
        return value

    def _warn_fallback(self) -> None:
        self._warn(
            f"No usable system keyring; MCP OAuth credentials are stored in "
            f"{self.fallback_path} (mode 0600)."
        )

    def _warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)


class PinnedScopeOAuthClientProvider(OAuthClientProvider):
    """SDK v1 adapter that preserves explicitly configured OAuth scopes."""

    def __init__(
        self, *args: Any, pinned_scope: str | None = None, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self._ness_pinned_scope = pinned_scope

    async def _initialize(self) -> None:
        await super()._initialize()
        expiry = getattr(self.context.storage, "token_expiry_time", None)
        if expiry is not None:
            self.context.token_expiry_time = expiry

    async def _perform_authorization(self):  # type: ignore[no-untyped-def]
        if self._ness_pinned_scope:
            self.context.client_metadata.scope = self._ness_pinned_scope
        return await super()._perform_authorization()


class MCPOAuthService:
    """Build non-interactive or explicit-login OAuth providers for a project."""

    def __init__(self, *, project_root: Path, config_dir: Path) -> None:
        self.project_root = project_root.resolve()
        self.config_dir = config_dir
        self.warnings: list[str] = []

    def storage_for(
        self,
        spec: ProjectMCPServer,
        *,
        redirect_uri: str | None = None,
    ) -> ProjectOAuthTokenStorage:
        static = _static_client_info(spec.oauth, redirect_uri) if spec.oauth else None
        return ProjectOAuthTokenStorage(
            config_dir=self.config_dir,
            identity=credential_id(self.project_root, spec),
            static_client_info=static,
            warnings=self.warnings,
        )

    async def startup_auth(self, spec: ProjectMCPServer) -> Any | None:
        redirect_uri = _default_redirect_uri(spec.oauth)
        storage = self.storage_for(spec, redirect_uri=redirect_uri)
        if not await storage.has_credentials():
            if spec.oauth is not None:
                raise MCPAuthenticationRequired("OAuth login required")
            return None

        async def login_required(*args: Any, **kwargs: Any):
            raise MCPAuthenticationRequired("OAuth login required")

        return self._provider(
            spec,
            storage=storage,
            redirect_uri=redirect_uri,
            redirect_handler=login_required,
            callback_handler=login_required,
        )

    def interactive_auth(
        self,
        spec: ProjectMCPServer,
        *,
        redirect_uri: str,
        redirect_handler: Any,
        callback_handler: Any,
        timeout: float = 300.0,
    ) -> Any:
        storage = self.storage_for(spec, redirect_uri=redirect_uri)
        return self._provider(
            spec,
            storage=storage,
            redirect_uri=redirect_uri,
            redirect_handler=redirect_handler,
            callback_handler=callback_handler,
            timeout=timeout,
        )

    def _provider(
        self,
        spec: ProjectMCPServer,
        *,
        storage: ProjectOAuthTokenStorage,
        redirect_uri: str,
        redirect_handler: Any,
        callback_handler: Any,
        timeout: float = 300.0,
    ) -> PinnedScopeOAuthClientProvider:
        oauth = spec.oauth
        scope = " ".join(oauth.scopes) if oauth and oauth.scopes else None
        metadata = OAuthClientMetadata(
            redirect_uris=[redirect_uri],
            token_endpoint_auth_method=(
                oauth.token_endpoint_auth_method if oauth else "none"
            ),
            scope=scope,
            client_name="Ness Agent",
            software_id="ness-agent",
        )
        return PinnedScopeOAuthClientProvider(
            server_url=spec.url or "",
            client_metadata=metadata,
            storage=storage,
            redirect_handler=redirect_handler,
            callback_handler=callback_handler,
            timeout=timeout,
            pinned_scope=scope,
        )


@dataclass
class LoopbackOAuthCallback:
    port: int = 0
    timeout: float = 300.0

    def __post_init__(self) -> None:
        self._server: asyncio.AbstractServer | None = None
        self._result: asyncio.Future[tuple[str, str | None]] | None = None

    @property
    def redirect_uri(self) -> str:
        if self._server is None or not self._server.sockets:
            raise RuntimeError("callback server has not started")
        port = int(self._server.sockets[0].getsockname()[1])
        return f"http://localhost:{port}/callback"

    async def start(self) -> None:
        self._result = asyncio.get_running_loop().create_future()
        try:
            self._server = await asyncio.start_server(
                self._handle_client,
                host="127.0.0.1",
                port=self.port,
            )
        except OSError as exc:
            raise RuntimeError(
                f"cannot bind OAuth callback port {self.port}: {exc.strerror or exc}"
            ) from exc

    async def close(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    async def callback_handler(self) -> tuple[str, str | None]:
        if self._result is None:
            raise RuntimeError("callback server has not started")
        return await asyncio.wait_for(self._result, timeout=self.timeout)

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        status = "400 Bad Request"
        body = "OAuth callback was invalid."
        try:
            raw = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=5)
            if len(raw) > 16_384:
                raise ValueError("callback request is too large")
            request_line = raw.split(b"\r\n", 1)[0].decode("ascii", errors="strict")
            method, target, _ = request_line.split(" ", 2)
            parts = urlsplit(target)
            if method != "GET" or parts.path != "/callback":
                status = "404 Not Found"
                body = "OAuth callback path not found."
            else:
                code, state = parse_oauth_callback_url(target)
                if self._result is not None and not self._result.done():
                    self._result.set_result((code, state))
                status = "200 OK"
                body = "OAuth login complete. You can return to Ness."
        except Exception as exc:
            if self._result is not None and not self._result.done():
                self._result.set_exception(exc)
        response = (
            f"HTTP/1.1 {status}\r\nContent-Type: text/plain; charset=utf-8\r\n"
            f"Content-Length: {len(body.encode('utf-8'))}\r\nConnection: close\r\n\r\n{body}"
        )
        writer.write(response.encode("utf-8"))
        await writer.drain()
        writer.close()
        await writer.wait_closed()


def parse_oauth_callback_url(value: str) -> tuple[str, str | None]:
    parts = urlsplit(value)
    if parts.path != "/callback":
        raise ValueError("callback URL must use the /callback path")
    query = parse_qs(parts.query)
    if query.get("error"):
        raise ValueError("OAuth provider returned an authorization error")
    code = query.get("code", [""])[0]
    state = query.get("state", [None])[0]
    if not code:
        raise ValueError("callback URL does not contain an authorization code")
    return code, state


def _static_client_info(
    oauth: MCPOAuthSpec,
    redirect_uri: str | None,
) -> OAuthClientInformationFull | None:
    if not oauth.client_id:
        return None
    return OAuthClientInformationFull(
        client_id=oauth.client_id,
        client_secret=oauth.client_secret,
        redirect_uris=[redirect_uri or _default_redirect_uri(oauth)],
        token_endpoint_auth_method=oauth.token_endpoint_auth_method,
        scope=" ".join(oauth.scopes) if oauth.scopes else None,
        client_name="Ness Agent",
    )


def _default_redirect_uri(oauth: MCPOAuthSpec | None) -> str:
    port = oauth.callback_port if oauth and oauth.callback_port else 8765
    return f"http://localhost:{port}/callback"


def _decode_record(value: str) -> dict[str, Any] | None:
    try:
        record = json.loads(value)
    except (TypeError, ValueError):
        return None
    return record if isinstance(record, dict) else None


async def _keyring_get(identity: str) -> _KeyringReadResult:
    try:
        import keyring
    except ImportError:
        return _KeyringReadResult(False, None)
    try:
        backend = keyring.get_keyring()
        if float(getattr(backend, "priority", 0)) <= 0:
            return _KeyringReadResult(False, None)
        value = await asyncio.to_thread(
            keyring.get_password,
            _KEYRING_SERVICE,
            identity,
        )
        return _KeyringReadResult(True, value)
    except Exception as exc:
        return _KeyringReadResult(False, None, exc)


async def _keyring_set(identity: str, value: str) -> bool:
    try:
        import keyring

        await asyncio.to_thread(keyring.set_password, _KEYRING_SERVICE, identity, value)
        return True
    except Exception:
        return False


async def _keyring_delete(identity: str) -> None:
    import keyring

    await asyncio.to_thread(keyring.delete_password, _KEYRING_SERVICE, identity)
