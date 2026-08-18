from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from typing import Any

import httpx
from langchain_core.language_models import BaseChatModel

from ness_cli.config import (
    VISION_MODELS,
    coerce_reasoning_effort,
    model_supports_reasoning,
    reasoning_efforts_for_model,
    settings,
)
from ness_cli.provider.base import (
    AccountDetails,
    AuthState,
    LoginMethod,
    LoginResult,
    ModelInfo,
    ProviderAdapter,
    ProviderStatus,
    RateLimitBucket,
)
from ness_cli.provider.opencode.catalog import (
    FALLBACK_MODEL_IDS,
    OPENCODE_GO_BASE_URL,
    OPENCODE_GO_USAGE_URL,
    fetch_model_ids,
    model_infos,
)
from ness_cli.provider.opencode.chat_model import OpenCodeChatOpenAI
from ness_cli.provider.openrouter.chat_model import OpenRouterAnthropicMessages

_MISSING_API_KEY = "sk-missing-opencode-key"
_DEFAULT_MODEL = "deepseek-v4-flash"
_RESPONSES_MODELS = frozenset({"gpt-5.6-luna", "grok-4.5"})
_MESSAGES_PREFIXES = ("minimax-", "qwen")
_EXTRA_VISION_MODELS = frozenset({"grok-4.5", "kimi-k3"})
_STATUS_TTL_SECONDS = 60.0


@dataclass
class _StatusCache:
    data: ProviderStatus
    fetched_at: float
    key_fingerprint: str


def _runtime_api_key() -> str | None:
    """Use the generic --api-key override, otherwise OpenCode's own secret."""
    facade = sys.modules.get("ness_cli.chat_model")
    overrides = getattr(facade, "_overrides", None)
    override = getattr(overrides, "openai_api_key", None)
    return str(override) if override else settings.opencode_api_key


def _key_fingerprint(api_key: str) -> str:
    return sha256(api_key.encode("utf-8")).hexdigest()


def _reasoning_effort(model_name: str, requested: str | None) -> str | None:
    if not model_supports_reasoning(model_name) or not requested or requested == "none":
        return None
    effort = requested
    if effort not in reasoning_efforts_for_model(model_name):
        effort = coerce_reasoning_effort(model_name, effort)
    return effort if effort and effort != "none" else None


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
        if value.get("status") != "ok":
            raise ValueError(
                f"{key} usage status is {value.get('status') or 'unknown'}"
            )
        percent = value.get("percent")
        if not isinstance(percent, (int, float)) or not 0 <= float(percent) <= 100:
            raise ValueError(f"{key} usage percent is invalid")
        used = float(percent)
        resets_at = _timestamp(value.get("resetsAt"))
        if resets_at is None:
            raise ValueError(f"{key} reset time is invalid")
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

    def __init__(self) -> None:
        self._models: tuple[ModelInfo, ...] = ()
        self._status_cache: _StatusCache | None = None

    def is_authenticated(self) -> bool:
        return bool(_runtime_api_key())

    def build_chat_model(
        self,
        thread_id: str,
        *,
        model_name: str,
        reasoning_effort: str | None,
        session_suffix: str = "",
    ) -> BaseChatModel:
        api_key = _runtime_api_key() or _MISSING_API_KEY
        effort = _reasoning_effort(model_name, reasoning_effort)
        max_retries = settings.api_max_retries

        if model_name.startswith(_MESSAGES_PREFIXES):
            return OpenRouterAnthropicMessages(
                model=model_name,
                api_key=api_key,
                base_url=OPENCODE_GO_BASE_URL,
                session_id="",
                cache_ttl=None,
                max_retries=max_retries,
                include_openrouter_extensions=False,
                billing_mode="subscription",
            )

        kwargs: dict[str, Any] = {
            "model": model_name,
            "api_key": api_key,
            "base_url": OPENCODE_GO_BASE_URL,
            "max_retries": max_retries,
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
                ids = await fetch_model_ids(api_key=_runtime_api_key())
            except Exception:
                if self._models:
                    return self._models
                ids = FALLBACK_MODEL_IDS
            efforts = {
                model_id: reasoning_efforts_for_model(model_id) for model_id in ids
            }
            vision_models = {
                model_id
                for model_id in ids
                if model_id in _EXTRA_VISION_MODELS
                or any(marker in model_id for marker in VISION_MODELS)
            }
            self._models = model_infos(
                ids,
                default_model=_DEFAULT_MODEL,
                reasoning_efforts=efforts,
                vision_models=vision_models,
            )
        return self._models

    async def status(self, *, refresh: bool = False) -> ProviderStatus:
        key = _runtime_api_key()
        if not key:
            return ProviderStatus(
                self.display_name,
                AuthState(False, "API key", "missing"),
                account=AccountDetails(tier="Go"),
            )
        now = time.monotonic()
        key_fingerprint = _key_fingerprint(key)
        if (
            not refresh
            and self._status_cache is not None
            and self._status_cache.key_fingerprint == key_fingerprint
            and now - self._status_cache.fetched_at < _STATUS_TTL_SECONDS
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
        self._status_cache = _StatusCache(status, time.monotonic(), key_fingerprint)
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
        self, *, method: str = "api_key", secret: str | None = None
    ) -> LoginResult:
        if method != "api_key":
            return LoginResult(
                "error", f"Unsupported OpenCode Go login method: {method}"
            )
        key = (secret or "").strip()
        if not key:
            return LoginResult("cancelled", "OpenCode Go sign-in was cancelled.")
        from ness_cli.config_store import write_secret

        write_secret("opencode_api_key", key)
        settings.opencode_api_key = key
        self._status_cache = None
        return LoginResult("complete", "Saved the OpenCode Go API key.")

    async def logout(self) -> str:
        from ness_cli.config_store import write_secret

        write_secret("opencode_api_key", None)
        settings.opencode_api_key = None
        self._models = ()
        self._status_cache = None
        return "Removed the saved OpenCode Go API key."
