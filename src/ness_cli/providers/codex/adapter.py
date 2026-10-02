"""Codex subscription provider adapter."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any

from langchain_core.language_models import BaseChatModel

from ness_cli.config import ProviderRuntimeConfig
from ness_cli.providers.base import (
    AccountDetails,
    AuthState,
    LoginMethod,
    LoginResult,
    ModelInfo,
    ProviderAdapter,
    ProviderStatus,
    RateLimitBucket,
)
from ness_cli.providers.codex.app_server import CodexAppServer
from ness_cli.providers.codex.auth import CodexAuth, codex_home
from ness_cli.providers.codex.browser import open_auth_url
from ness_cli.providers.codex.catalog import fallback_model_info, load_models
from ness_cli.providers.codex.responses import CodexSubscriptionChatModel


@dataclass(slots=True)
class _StatusCache:
    data: ProviderStatus
    fetched_at: float


class CodexProviderAdapter(ProviderAdapter):
    id = "codex"
    display_name = "Codex (ChatGPT)"
    login_description = "Sign in with ChatGPT"
    selection_priority = 10
    billing_label = "subscription"

    _device_login_guidance = (
        "Device-code login must first be enabled in ChatGPT Settings > Security: "
        'turn on "Device code authorization for Codex." If unavailable, choose '
        '"Open browser" instead.'
    )

    def __init__(self, runtime_config: ProviderRuntimeConfig) -> None:
        super().__init__(runtime_config)
        self.server = CodexAppServer(codex_home())
        self.auth = CodexAuth(self.server)
        self._models: tuple[ModelInfo, ...] = ()
        self._status_cache: _StatusCache | None = None

    def is_authenticated(self) -> bool:
        return self.auth.is_authenticated()

    def build_chat_model(
        self,
        thread_id: str,
        *,
        model_name: str,
        reasoning_effort: str | None,
        session_suffix: str = "",
    ) -> BaseChatModel:
        identity = f"ness-agent:{thread_id}:{session_suffix or 'main'}"
        cache_key = str(uuid.uuid5(uuid.NAMESPACE_URL, identity))
        return CodexSubscriptionChatModel(
            model=model_name,
            reasoning_effort=reasoning_effort,
            prompt_cache_key=cache_key,
            max_retries=self.runtime_config.max_retries,
            auth=self.auth,
        )

    async def models(self, *, refresh: bool = False) -> tuple[ModelInfo, ...]:
        if refresh or not self._models:
            await self.server.start()
            self._models = await load_models(self.server)
        return self._models

    def model_info(self, model_name: str) -> ModelInfo | None:
        found = next((item for item in self._models if item.id == model_name), None)
        if found is not None:
            return found
        # Runtime construction is synchronous, while the Codex catalog comes
        # from the app-server. Accept the configured model until the first
        # catalog load. Once loaded, unknown IDs have no catalog metadata.
        if not self._models and model_name.strip():
            return fallback_model_info(model_name)
        return None

    def login_methods(self) -> tuple[LoginMethod, ...]:
        return (
            LoginMethod(
                "browser",
                "Browser sign-in",
                description="Complete sign-in with ChatGPT",
                default=True,
            ),
            LoginMethod(
                "device",
                "Device code",
                description="Requires ChatGPT Settings > Security",
                guidance=self._device_login_guidance,
            ),
        )

    @classmethod
    def _login_error(cls, error: object) -> str:
        message = str(error or "Codex sign-in failed.")
        if "enable device code authorization" in message.casefold():
            return (
                "Device-code authorization is disabled for this ChatGPT account. "
                + cls._device_login_guidance
                + " Then run /login again."
            )
        return message

    async def login(
        self,
        *,
        method: str = "browser",
        secret: str | None = None,
    ) -> LoginResult:
        del secret
        if method not in {"browser", "device"}:
            return LoginResult("error", f"Unsupported Codex login method: {method}")
        await self.server.start()
        params = (
            {"type": "chatgptDeviceCode"}
            if method == "device"
            else {
                "type": "chatgpt",
                "appBrand": "codex",
                "codexStreamlinedLogin": False,
                "useHostedLoginSuccessPage": True,
            }
        )
        try:
            response = await self.server.request("account/login/start", params)
        except RuntimeError as exc:
            return LoginResult("error", self._login_error(exc))
        login_id = response.get("loginId")
        auth_url = response.get("authUrl")
        verification_url = response.get("verificationUrl")
        if not isinstance(login_id, str) or not login_id:
            return LoginResult("error", "Codex returned a malformed login response.")
        if method == "browser" and not isinstance(auth_url, str):
            return LoginResult("error", "Codex returned a malformed login response.")
        if method == "device" and not isinstance(verification_url, str):
            return LoginResult("error", "Codex returned a malformed login response.")
        return LoginResult(
            "pending",
            "Complete sign-in in your browser.",
            auth_url=auth_url,
            login_id=login_id,
            user_code=response.get("userCode"),
            verification_url=response.get("verificationUrl"),
        )

    async def wait_for_login(self, login_id: str) -> LoginResult:
        params = await self.server.wait_notification(
            "account/login/completed",
            predicate=lambda item: (
                not item.get("loginId") or item.get("loginId") == login_id
            ),
        )
        if params.get("success"):
            await self.auth.wait_until_ready()
            self._status_cache = None
            return LoginResult("complete", "Signed in to Codex with ChatGPT.")
        return LoginResult("error", self._login_error(params.get("error")))

    async def cancel_login(self, login_id: str) -> None:
        await self.server.request("account/login/cancel", {"loginId": login_id})

    async def open_login_url(self, url: str) -> bool:
        return open_auth_url(url)

    async def logout(self) -> str:
        await self.server.start()
        await self.server.request("account/logout", {})
        self._models = ()
        self._status_cache = None
        return "Signed out of Codex."

    async def status(self, *, refresh: bool = False) -> ProviderStatus:
        if not self.is_authenticated():
            return ProviderStatus(
                self.display_name,
                AuthState(False, "ChatGPT", "signed out"),
            )

        now = time.monotonic()
        if (
            not refresh
            and self._status_cache is not None
            and now - self._status_cache.fetched_at < 60
        ):
            return self._status_cache.data

        await self.server.start()
        account_response = await self.server.request(
            "account/read", {"refreshToken": refresh}
        )
        account = account_response.get("account") or {}
        warning: str | None = None
        try:
            limits_response = await self.server.request("account/rateLimits/read", {})
        except Exception as exc:
            limits_response = {}
            warning = f"Usage limits unavailable: {exc}"

        snapshots: list[tuple[str, dict[str, Any]]] = []
        by_id = limits_response.get("rateLimitsByLimitId")
        if isinstance(by_id, dict):
            snapshots.extend(
                (str(key), value)
                for key, value in by_id.items()
                if isinstance(value, dict)
            )
        legacy = limits_response.get("rateLimits")
        if isinstance(legacy, dict):
            snapshots.append((str(legacy.get("limitId") or "codex"), legacy))

        buckets: list[RateLimitBucket] = []
        seen: set[tuple[str, int | None, int | None]] = set()
        for limit_id, snapshot in snapshots:
            limit_name = str(snapshot.get("limitName") or limit_id)
            reached = bool(snapshot.get("rateLimitReachedType"))
            for slot, label in (("primary", "primary"), ("secondary", "secondary")):
                window = snapshot.get(slot)
                if not isinstance(window, dict):
                    continue
                duration = window.get("windowDurationMins")
                resets_at = window.get("resetsAt")
                key = (
                    limit_id,
                    int(duration) if duration is not None else None,
                    int(resets_at) if resets_at is not None else None,
                )
                if key in seen:
                    continue
                seen.add(key)
                used = float(window.get("usedPercent") or 0)
                buckets.append(
                    RateLimitBucket(
                        name=f"{limit_name} {label}",
                        window_minutes=key[1],
                        used_percent=used,
                        remaining_percent=max(0.0, 100.0 - used),
                        resets_at=key[2],
                        reached=reached or used >= 100,
                    )
                )

        credits = limits_response.get("rateLimitResetCredits")
        credits_text = (
            f"{int(credits.get('availableCount') or 0)} available"
            if isinstance(credits, dict)
            else None
        )
        status = ProviderStatus(
            provider=self.display_name,
            auth=AuthState(True, "ChatGPT", "managed by Codex CLI"),
            account=AccountDetails(
                email=account.get("email") if isinstance(account, dict) else None,
                tier=account.get("planType") if isinstance(account, dict) else None,
            ),
            limits=tuple(buckets),
            credits=credits_text,
            warning=warning,
        )
        self._status_cache = _StatusCache(status, time.monotonic())
        return status

    async def close(self) -> None:
        self._status_cache = None
        await self.server.close()
