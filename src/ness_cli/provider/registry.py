from __future__ import annotations

from collections.abc import Callable

from ness_cli.provider.base import ProviderAdapter

_factories: dict[str, Callable[[], ProviderAdapter]] = {}
_instances: dict[str, ProviderAdapter] = {}
_builtins_registered = False


def register_provider(provider_id: str, factory: Callable[[], ProviderAdapter]) -> None:
    _factories[provider_id] = factory


def _register_builtins() -> None:
    global _builtins_registered
    if _builtins_registered:
        return
    from ness_cli.provider.codex.adapter import CodexProviderAdapter
    from ness_cli.provider.opencode.adapter import OpenCodeProviderAdapter
    from ness_cli.provider.openrouter.adapter import OpenRouterProviderAdapter

    register_provider("openrouter", OpenRouterProviderAdapter)
    register_provider("codex", CodexProviderAdapter)
    register_provider("opencode", OpenCodeProviderAdapter)
    _builtins_registered = True


def provider_ids() -> tuple[str, ...]:
    _register_builtins()
    return tuple(
        sorted(
            _factories,
            key=lambda provider_id: (
                get_provider(provider_id).selection_priority,
                provider_id,
            ),
        )
    )


def get_provider(provider_id: str) -> ProviderAdapter:
    _register_builtins()
    if provider_id not in _factories:
        raise ValueError(f"unknown model provider: {provider_id}")
    if provider_id not in _instances:
        _instances[provider_id] = _factories[provider_id]()
    return _instances[provider_id]


def active_provider() -> ProviderAdapter:
    from ness_cli.config import settings

    return get_provider(settings.model_provider)


async def close_providers() -> None:
    for adapter in tuple(_instances.values()):
        await adapter.close()
    _instances.clear()
