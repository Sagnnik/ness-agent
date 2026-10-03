"""OpenRouter provider lifecycle and model construction."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_openrouter import ChatOpenRouter

from ness_cli.config import ProviderRuntimeConfig
from ness_cli.providers.base import (
    AuthState,
    LoginMethod,
    LoginResult,
    ModelInfo,
    ProviderAdapter,
    ProviderStatus,
)
from ness_cli.providers.openrouter.catalog import (
    ModelRecord,
    OpenRouterCatalog,
)
from ness_cli.providers.openrouter.messages import (
    OpenRouterAnthropicMessages,
)

_MISSING_API_KEY = "sk-missing-api-key"

CredentialWriter = Callable[[str | None], ProviderRuntimeConfig]


class OpenRouterProviderAdapter(ProviderAdapter):
    id = "openrouter"
    display_name = "OpenRouter"
    login_description = "API key"
    selection_priority = 20
    billing_label = "API billing"

    def __init__(
        self,
        runtime_config: ProviderRuntimeConfig,
        *,
        catalog: OpenRouterCatalog,
        credential_writer: CredentialWriter | None = None,
    ) -> None:
        super().__init__(runtime_config)
        self._catalog = catalog
        self._credential_writer = credential_writer

    def is_authenticated(self) -> bool:
        return bool(self.runtime_config.api_key)

    def _reasoning(
        self,
        model_name: str,
        requested_effort: str | None,
    ) -> dict[str, str] | None:
        if not requested_effort or requested_effort == "none":
            return None

        info = self.model_info(model_name)
        if info is None or not info.reasoning_efforts:
            return None

        effort = requested_effort
        if effort not in info.reasoning_efforts:
            effort = info.default_reasoning_effort or info.reasoning_efforts[0]

        if effort == "none":
            return None
        return {"effort": effort}

    def build_chat_model(
        self,
        thread_id: str,
        *,
        model_name: str,
        reasoning_effort: str | None,
        session_suffix: str = "",
    ) -> BaseChatModel:
        config = self.runtime_config
        session_base = config.session_id or thread_id
        session_id = (
            f"{session_base}:{session_suffix}" if session_suffix else session_base
        )
        base_url = config.base_url
        is_openrouter = not base_url or "openrouter.ai" in base_url.lower()
        reasoning = self._reasoning(model_name, reasoning_effort)
        api_key = config.api_key or _MISSING_API_KEY

        if (
            model_name.startswith("anthropic/")
            and is_openrouter
            and config.anthropic_messages
        ):
            return OpenRouterAnthropicMessages(
                model=model_name,
                api_key=api_key,
                base_url=(base_url or "https://openrouter.ai/api/v1").rstrip("/"),
                session_id=session_id,
                cache_ttl=config.cache_ttl,
                max_retries=config.max_retries,
                reasoning=reasoning,
            )

        kwargs: dict[str, Any] = {
            "model": model_name,
            "api_key": api_key,
            "session_id": session_id,
            "max_retries": config.max_retries,
        }

        if reasoning:
            kwargs["reasoning"] = reasoning
        if base_url:
            kwargs["base_url"] = base_url
        if model_name.startswith("anthropic/") and is_openrouter:
            kwargs["model_kwargs"] = {
                "cache_control": {
                    "type": "ephemeral",
                    "ttl": config.cache_ttl,
                }
            }

        return ChatOpenRouter(**kwargs)

    def _model_info(self, record: ModelRecord) -> ModelInfo:
        return ModelInfo(
            id=record.id,
            name=record.name,
            context_window=record.context_length,
            input_price=record.input_price,
            output_price=record.output_price,
            cache_read_ratio=record.cache_read_ratio,
            default_reasoning_effort=record.default_reasoning_effort,
            reasoning_efforts=record.reasoning_efforts,
            supports_vision=record.supports_vision,
            supports_anthropic_messages=record.supports_anthropic_messages,
        )

    async def models(self, *, refresh: bool = False) -> tuple[ModelInfo, ...]:
        """List cached/offline models; only an explicit refresh fetches data."""
        if refresh:
            await self._catalog.refresh(force=True)
        return tuple(self._model_info(record) for record in self._catalog.models())

    def model_info(self, model_name: str) -> ModelInfo | None:
        record = self._catalog.model_record(model_name)
        return self._model_info(record) if record is not None else None

    async def status(self, *, refresh: bool = False) -> ProviderStatus:
        del refresh
        authenticated = self.is_authenticated()
        return ProviderStatus(
            provider=self.display_name,
            auth=AuthState(
                authenticated,
                "API key",
                "configured" if authenticated else "missing",
            ),
        )

    def login_methods(self) -> tuple[LoginMethod, ...]:
        return (
            LoginMethod(
                "api_key",
                "API key",
                description="Stored in the Ness secrets file",
                default=True,
                input_kind="secret",
                input_label="OpenRouter API key",
                input_example="sk-or-v1-...",
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
                "error",
                f"Unsupported OpenRouter login method: {method}",
            )

        key = (secret or "").strip()
        if not key:
            return LoginResult(
                "cancelled",
                "OpenRouter sign-in was cancelled.",
            )
        if self._credential_writer is None:
            return LoginResult(
                "error",
                "OpenRouter credential persistence is unavailable.",
            )

        self.reconfigure(self._credential_writer(key))
        return LoginResult("complete", "Saved the OpenRouter API key.")

    async def logout(self) -> str:
        if self._credential_writer is None:
            return "OpenRouter credential persistence is unavailable."

        self.reconfigure(self._credential_writer(None))
        return "Removed the saved OpenRouter API key."
