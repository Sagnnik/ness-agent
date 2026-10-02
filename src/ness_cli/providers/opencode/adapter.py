"""OpenCode Go provider adapter."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from typing import Any

import httpx
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
from ness_cli.providers.opencode.catalog import (
    FALLBACK_MODEL_IDS,
    OPENCODE_GO_BASE_URL,
    OPENCODE_GO_USAGE_URL,
    fetch_model_ids,
    model_infos,
)
from ness_cli.providers.opencode.openai import OpenCodeChatOpenAI
from ness_cli.providers.openrouter.messages import OpenRouterAnthropicMessages

_MISSING_API_KEY = "sk-missing-opencode-key"
_DEFAULT_MODEL = "deepseek-v4-flash"
_RESPONSES_MODELS = frozenset({"gpt-5.6-luna", "grok-4.5"})
_MESSAGES_PREFIXES = ("minimax-", "qwen")
CredentialWriter = Callable[[str | None], ProviderRuntimeConfig]


@dataclass(slots=True)
class _StatusCache:
    data: ProviderStatus
    fetched_at: float
    key_fingerprint: str


def _timestamp(value: Any) -> int | None:
    if not isinstance(value, str):
        return None
    try:
        return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp())
    except ValueError:
        return None


def _usage_buckets(payload: Any) -> tuple[RateLimitBucket, ...]:
    if not isinstance(payload, dict) or not isinstance(payload.get("usage"), dict):
        raise ValueError("response is missing usage windows")
    usage = payload["usage"]
    windows = (
        ("rolling", "Go usage", 300),
        ("weekly", "Go usage", 10_080),
        ("monthly", "Go usage", 43_200),
    )
    buckets: list[RateLimitBucket] = []
    for key, name, duration in windows:
        value = usage.get(key)
        if not isinstance(value, dict):
            raise ValueError(f"response is missing the {key} usage window")
        percent = value.get("percent")
        resets_at = _timestamp(value.get("resetsAt"))
        if value.get("status") != "ok":
            raise ValueError(
                f"{key} usage status is {value.get('status') or 'unknown'}"
            )
        if not isinstance(percent, (int, float)) or not 0 <= float(percent) <= 100:
            raise ValueError(f"{key} usage percent is invalid")
        if resets_at is None:
            raise ValueError(f"{key} reset time is invalid")
        used = float(percent)
        buckets.append(
            RateLimitBucket(
                name=name,
                window_minutes=duration,
                used_percent=used,
                remaining_percent=100.0 - used,
                resets_at=resets_at,
                reached=used >= 100.0,
            )
        )
    return tuple(buckets)


class OpenCodeProviderAdapter(ProviderAdapter):
    id = "opencode"
    display_name = "OpenCode Go"
    login_description = "Go subscription API key"
    selection_priority = 30
    billing_label = "subscription"

    def __init__(
        self,
        runtime_config: ProviderRuntimeConfig,
        *,
        credential_writer: CredentialWriter | None = None,
    ) -> None:
        super().__init__(runtime_config)
        self._credential_writer = credential_writer
        self._models: tuple[ModelInfo, ...] = ()
        self._status_cache: _StatusCache | None = None

    def is_authenticated(self) -> bool:
        return bool(self.runtime_config.api_key)

    def build_chat_model(
        self,
        thread_id: str,
        *,
        model_name: str,
        reasoning_effort: str | None,
        session_suffix: str = "",
    ) -> BaseChatModel:
        del thread_id, session_suffix
        config = self.runtime_config
        api_key = config.api_key or _MISSING_API_KEY
        info = self.model_info(model_name)
        effort = reasoning_effort
        if not effort or effort == "none" or info is None or not info.reasoning_efforts:
            effort = None
        elif effort not in info.reasoning_efforts:
            effort = info.default_reasoning_effort or info.reasoning_efforts[0]

        if model_name.startswith(_MESSAGES_PREFIXES):
            return OpenRouterAnthropicMessages(
                model=model_name,
                api_key=api_key,
                base_url=OPENCODE_GO_BASE_URL,
                session_id="",
                cache_ttl=None,
                max_retries=config.max_retries,
                include_openrouter_extensions=False,
                billing_mode="subscription",
            )

        kwargs: dict[str, Any] = {
            "model": model_name,
            "api_key": api_key,
            "base_url": OPENCODE_GO_BASE_URL,
            "max_retries": config.max_retries,
            "stream_usage": True,
            "use_responses_api": model_name in _RESPONSES_MODELS,
        }
        if model_name in _RESPONSES_MODELS:
            kwargs["output_version"] = "responses/v1"
        if effort:
            kwargs["reasoning_effort"] = effort
        return OpenCodeChatOpenAI(**kwargs)

    async def models(self, *, refresh: bool = False) -> tuple[ModelInfo, ...]:
        if refresh or not self._models:
            try:
                ids = await fetch_model_ids(api_key=self.runtime_config.api_key)
            except Exception:
                ids = tuple(item.id for item in self._models) or FALLBACK_MODEL_IDS
            self._models = model_infos(
                ids,
                default_model=_DEFAULT_MODEL,
            )
        return self._models

    def model_info(self, model_name: str) -> ModelInfo | None:
        cached = next((item for item in self._models if item.id == model_name), None)
        if cached is not None:
            return cached
        if model_name not in FALLBACK_MODEL_IDS:
            return None
        return next(
            item
            for item in model_infos(
                (model_name,),
                default_model=_DEFAULT_MODEL,
            )
        )

    async def status(self, *, refresh: bool = False) -> ProviderStatus:
        key = self.runtime_config.api_key
        if not key:
            return ProviderStatus(
                self.display_name,
                AuthState(False, "API key", "missing"),
                account=AccountDetails(tier="Go"),
            )
        now = time.monotonic()
        fingerprint = sha256(key.encode("utf-8")).hexdigest()
        if (
            not refresh
            and self._status_cache is not None
            and self._status_cache.key_fingerprint == fingerprint
            and now - self._status_cache.fetched_at < 60
        ):
            return self._status_cache.data

        limits: tuple[RateLimitBucket, ...] = ()
        warning: str | None = None
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(
                    OPENCODE_GO_USAGE_URL,
                    headers={
                        "Authorization": f"Bearer {key}",
                        "Accept": "application/json",
                    },
                )
                response.raise_for_status()
                limits = _usage_buckets(response.json())
        except Exception as exc:
            warning = f"Usage limits unavailable: {exc}"

        status = ProviderStatus(
            provider=self.display_name,
            auth=AuthState(True, "API key", "configured"),
            account=AccountDetails(tier="Go"),
            limits=limits,
            warning=warning,
        )
        self._status_cache = _StatusCache(status, time.monotonic(), fingerprint)
        return status

    def login_methods(self) -> tuple[LoginMethod, ...]:
        return (
            LoginMethod(
                "api_key",
                "API key",
                description="OpenCode Go subscription key",
                default=True,
                input_kind="secret",
                input_label="OpenCode Go API key",
                input_example="sk-...",
            ),
        )

    async def login(
        self,
        *,
        method: str = "api_key",
        secret: str | None = None,
    ) -> LoginResult:
        if method != "api_key":
            return LoginResult(
                "error", f"Unsupported OpenCode Go login method: {method}"
            )
        key = (secret or "").strip()
        if not key:
            return LoginResult("cancelled", "OpenCode Go sign-in was cancelled.")
        if self._credential_writer is None:
            return LoginResult(
                "error", "OpenCode credential persistence is unavailable."
            )
        self.reconfigure(self._credential_writer(key))
        self._status_cache = None
        return LoginResult("complete", "Saved the OpenCode Go API key.")

    async def logout(self) -> str:
        if self._credential_writer is None:
            return "OpenCode credential persistence is unavailable."
        self.reconfigure(self._credential_writer(None))
        self._models = ()
        self._status_cache = None
        return "Removed the saved OpenCode Go API key."
