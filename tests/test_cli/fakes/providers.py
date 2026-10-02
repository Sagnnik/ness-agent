from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ness_cli.config import ProviderRuntimeConfig
from ness_cli.providers import AuthState, ModelInfo, ProviderStatus


@dataclass
class FakeProvider:
    id: str
    runtime_config: ProviderRuntimeConfig
    model_ids: tuple[str, ...] = ("test/model",)
    display_name: str = "Fake provider"
    selection_priority: int = 50
    billing_label: str = "API billing"
    close_calls: int = 0
    build_calls: list[dict[str, Any]] = field(default_factory=list)
    model_info_calls: list[str] = field(default_factory=list)

    def reconfigure(self, runtime_config: ProviderRuntimeConfig) -> None:
        self.runtime_config = runtime_config

    def is_authenticated(self) -> bool:
        return bool(self.runtime_config.api_key)

    def build_chat_model(
        self,
        thread_id: str,
        *,
        model_name: str,
        reasoning_effort: str | None,
        session_suffix: str = "",
    ) -> object:
        call = {
            "thread_id": thread_id,
            "model_name": model_name,
            "reasoning_effort": reasoning_effort,
            "session_suffix": session_suffix,
        }
        self.build_calls.append(call)
        return call

    async def models(self, *, refresh: bool = False) -> tuple[ModelInfo, ...]:
        del refresh
        return tuple(self._info(model_id) for model_id in self.model_ids)

    def model_info(self, model_name: str) -> ModelInfo | None:
        self.model_info_calls.append(model_name)
        return self._info(model_name) if model_name in self.model_ids else None

    def _info(self, model_name: str) -> ModelInfo:
        return ModelInfo(
            id=model_name,
            name=model_name,
            context_window=128_000,
            reasoning_efforts=("low", "high"),
            supports_vision=True,
        )

    async def status(self, *, refresh: bool = False) -> ProviderStatus:
        del refresh
        return ProviderStatus(self.display_name, AuthState(self.is_authenticated()))

    async def close(self) -> None:
        self.close_calls += 1
