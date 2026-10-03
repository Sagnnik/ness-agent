"""Instance-owned provider registration and model construction."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from langchain_core.language_models import BaseChatModel
from ness_agent.tracing.config import PricingDict

from ness_cli.cleanup import cleanup_all
from ness_cli.config import ConfigManager, ProviderRuntimeConfig
from ness_cli.config.store import ConfigStore
from ness_cli.paths import NessPaths
from ness_cli.providers.base import ModelInfo, ProviderAdapter

ProviderPurpose = Literal["main", "reflection", "goal", "summary"]
ProviderFactory = Callable[[ProviderRuntimeConfig], ProviderAdapter]


@dataclass(frozen=True, slots=True)
class ModelRequest:
    thread_id: str
    purpose: ProviderPurpose
    model_name: str
    reasoning_effort: str | None = None
    provider_id: str | None = None

    @property
    def session_suffix(self) -> str:
        return "" if self.purpose == "main" else self.purpose


class ProviderRegistry:
    """Own provider factories and adapter lifetimes for one CLI runtime."""

    def __init__(
        self,
        *,
        paths: NessPaths,
        config: ConfigManager,
    ) -> None:
        self._paths = paths
        self._config = config
        self._store = ConfigStore(paths.config_dir)
        self._factories: dict[str, ProviderFactory] = {}
        self._instances: dict[str, ProviderAdapter] = {}
        self._pricing: PricingDict = {}
        self._pricing_models: dict[str, set[str]] = {}
        self._closed = False

    def bind_pricing(self, pricing: PricingDict) -> None:
        """Use the agent's shared rate table without replacing live trackers."""
        pricing.update(self._pricing)
        self._pricing = pricing

    def _update_pricing(
        self, adapter: ProviderAdapter, model_name: str, info: ModelInfo | None
    ) -> None:
        if adapter.billing_label == "subscription":
            return
        names = {model_name}
        if info is not None:
            names.add(info.id)
        self._pricing_models.setdefault(adapter.id, set()).update(names)
        for name in names:
            if (
                info is not None
                and info.input_price is not None
                and info.output_price is not None
            ):
                self._pricing[name] = (
                    info.input_price,
                    info.output_price,
                    info.cache_read_ratio,
                )
            else:
                self._pricing.pop(name, None)

    @classmethod
    def load(
        cls,
        *,
        paths: NessPaths,
        config: ConfigManager,
    ) -> ProviderRegistry:
        registry = cls(paths=paths, config=config)
        registry._register_builtins()
        return registry

    def _register_builtins(self) -> None:
        from ness_cli.providers.codex import CodexProviderAdapter
        from ness_cli.providers.opencode import OpenCodeProviderAdapter
        from ness_cli.providers.openrouter.adapter import (
            OpenRouterProviderAdapter,
        )
        from ness_cli.providers.openrouter.catalog import OpenRouterCatalog

        catalog = OpenRouterCatalog.from_cache_dir(self._paths.cache_dir.parent)

        def openrouter_factory(
            runtime_config: ProviderRuntimeConfig,
        ) -> ProviderAdapter:
            return OpenRouterProviderAdapter(
                runtime_config,
                catalog=catalog,
                credential_writer=self._write_openrouter_key,
            )

        self.register("openrouter", openrouter_factory)
        self.register("codex", CodexProviderAdapter)
        self.register(
            "opencode",
            lambda runtime_config: OpenCodeProviderAdapter(
                runtime_config,
                credential_writer=self._write_opencode_key,
            ),
        )

    def _write_openrouter_key(
        self,
        value: str | None,
    ) -> ProviderRuntimeConfig:
        self._store.write_secret("openai_api_key", value)
        self._config.reload()
        return self._config.provider_runtime("openrouter")

    def _write_opencode_key(
        self,
        value: str | None,
    ) -> ProviderRuntimeConfig:
        self._store.write_secret("opencode_api_key", value)
        self._config.reload()
        return self._config.provider_runtime("opencode")

    def register(self, provider_id: str, factory: ProviderFactory) -> None:
        if self._closed:
            raise RuntimeError("provider registry is closed")

        normalized = provider_id.strip()
        if not normalized:
            raise ValueError("provider id cannot be empty")
        if normalized in self._instances:
            raise RuntimeError(f"cannot replace initialized provider: {normalized}")
        self._factories[normalized] = factory

    def provider_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                self._factories,
                key=lambda provider_id: (
                    self.get(provider_id).selection_priority,
                    provider_id,
                ),
            )
        )

    @property
    def active_provider_id(self) -> str:
        return self._config.current.model.provider_id

    def get(self, provider_id: str) -> ProviderAdapter:
        if self._closed:
            raise RuntimeError("provider registry is closed")

        factory = self._factories.get(provider_id)
        if factory is None:
            raise ValueError(f"unknown model provider: {provider_id}")

        runtime_config = self._config.provider_runtime(provider_id)
        adapter = self._instances.get(provider_id)

        if adapter is None:
            adapter = factory(runtime_config)
            if adapter.id != provider_id:
                raise ValueError(
                    f"provider factory {provider_id!r} returned {adapter.id!r}"
                )
            self._instances[provider_id] = adapter
        else:
            adapter.reconfigure(runtime_config)

        return adapter

    def active(self) -> ProviderAdapter:
        return self.get(self.active_provider_id)

    def create_model(self, request: ModelRequest) -> BaseChatModel:
        adapter = self.get(request.provider_id or self.active_provider_id)
        # Catalog metadata is optional; the endpoint determines model support.
        model = adapter.build_chat_model(
            request.thread_id,
            model_name=request.model_name,
            reasoning_effort=request.reasoning_effort,
            session_suffix=request.session_suffix,
        )
        self._update_pricing(
            adapter, request.model_name, adapter.model_info(request.model_name)
        )
        return model

    async def models(
        self, *, provider_id: str | None = None, refresh: bool = False
    ) -> tuple[ModelInfo, ...]:
        """Load catalog metadata and refresh rates for catalog and live models."""
        adapter = self.get(provider_id or self.active_provider_id)
        models = await adapter.models(refresh=refresh)
        by_name = {info.id: info for info in models}
        names = self._pricing_models.get(adapter.id, set()) | by_name.keys()
        for name in sorted(names):
            self._update_pricing(
                adapter, name, by_name.get(name) or adapter.model_info(name)
            )
        return models

    def model_info(
        self,
        model_name: str,
        *,
        provider_id: str | None = None,
    ) -> ModelInfo | None:
        adapter = self.get(provider_id or self.active_provider_id)
        return adapter.model_info(model_name)

    def validate_reasoning_effort(
        self, model_name: str, effort: str, *, provider_id: str | None = None
    ) -> None:
        """Validate an explicit effort against the selected provider's metadata.

        Unavailable effort metadata leaves validation to the endpoint, as it
        does for uncatalogued models. Never infer another provider's options.
        """
        adapter = self.get(provider_id or self.active_provider_id)
        info = adapter.model_info(model_name)
        if info is None or not info.reasoning_efforts:
            return
        if effort not in info.reasoning_efforts:
            raise ValueError(
                f"reasoning effort must be one of: {', '.join(info.reasoning_efforts)} "
                f"for model {model_name!r} on provider {adapter.id!r}"
            )

    async def close(self) -> None:
        if self._closed:
            return

        self._closed = True
        instances = tuple(self._instances.items())
        self._instances.clear()

        await cleanup_all(
            (f"provider {provider_id}", adapter.close)
            for provider_id, adapter in instances
        )
