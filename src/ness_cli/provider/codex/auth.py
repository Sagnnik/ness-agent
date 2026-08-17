from __future__ import annotations

import base64
import asyncio
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ness_cli.paths import config_dir_from_env
from ness_cli.provider.codex.app_server import CodexAppServer


def codex_home() -> Path:
    return config_dir_from_env() / "codex"


@dataclass(frozen=True)
class CodexCredentials:
    access_token: str
    account_id: str
    expires_at: int | None


class CodexAuth:
    def __init__(self, server: CodexAppServer | None = None) -> None:
        self.home = codex_home()
        self.server = server or CodexAppServer(self.home)

    @property
    def auth_path(self) -> Path:
        return self.home / "auth.json"

    def credentials(self) -> CodexCredentials | None:
        try:
            raw = json.loads(self.auth_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        tokens = raw.get("tokens") if isinstance(raw, dict) else None
        if not isinstance(tokens, dict):
            return None
        access = tokens.get("access_token")
        account = tokens.get("account_id")
        if not isinstance(access, str) or not access or not isinstance(account, str) or not account:
            return None
        return CodexCredentials(access, account, _jwt_expiry(access))

    def is_authenticated(self) -> bool:
        return self.credentials() is not None

    async def refresh(self) -> CodexCredentials:
        await self.server.start()
        result = await self.server.request("account/read", {"refreshToken": True})
        account = result.get("account")
        if not isinstance(account, dict) or account.get("type") != "chatgpt":
            raise RuntimeError("Codex ChatGPT authentication is required; run /login.")
        credentials = self.credentials() # edge case: codex says signed in but auth.json is not present
        if credentials is None:
            raise RuntimeError("Codex did not persist ChatGPT credentials.")
        try:
            os.chmod(self.auth_path, 0o600)
        except OSError:
            pass
        return credentials

    async def valid_credentials(self, *, force_refresh: bool = False) -> CodexCredentials:
        credentials = self.credentials()
        if force_refresh or credentials is None or (
            credentials.expires_at is not None and credentials.expires_at <= int(time.time()) + 60
        ):
            return await self.refresh()
        return credentials

    async def wait_until_ready(self, *, timeout: float = 8.0) -> CodexCredentials:
        """Wait for app-server's successful login to reach account + disk state."""
        deadline = asyncio.get_running_loop().time() + timeout # monotonic time
        while asyncio.get_running_loop().time() < deadline:
            credentials = self.credentials()
            # check if the credentials are ready
            if credentials is not None: 
                try:
                    # rediness of app-sever account check
                    result = await self.server.request(
                        "account/read", {"refreshToken": False}
                    )
                except RuntimeError:
                    result = {}
                account = result.get("account")
                if isinstance(account, dict) and account.get("type") == "chatgpt":
                    try:
                        os.chmod(self.auth_path, 0o600)
                    except OSError:
                        pass
                    return credentials
            await asyncio.sleep(0.1) # polling readiness every 100ms
        raise RuntimeError(
            "Codex reported a successful sign-in, but credentials were not ready. "
            "Please retry /login."
        )


def _jwt_expiry(token: str) -> int | None:
    try:
        # split: header.payload.signature
        payload = token.split(".")[1]  
        payload += "=" * (-len(payload) % 4) # padding with "=" to make len of payload multiple of 4
        data = json.loads(base64.urlsafe_b64decode(payload).decode())
        expiry = data.get("exp")
        return int(expiry) if expiry is not None else None
    except (IndexError, ValueError, TypeError, json.JSONDecodeError):
        return None
