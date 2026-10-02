from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from ness_cli.config.settings import (
    Settings,
    app_settings,
    environment_value,
)
from ness_cli.config.types import (
    CliOverrides,
    ModelSelection,
    ProviderRuntimeConfig,
    RuntimeConfig,
)


_PRIMARY_REFLECTION_FALLBACK = frozenset({"codex", "opencode"})


def _profile(sett: Settings, provider_id: str) -> dict[str, Any]:
    value = sett.provider_profiles.get(provider_id)
    return dict(value) if isinstance(value, dict) else {}


def _text(value: Any) -> str | None:
    if value is None:
        return None

    result = str(value).strip()
    return result or None


def _select(
    *,
    override: Any,
    environment_field: str,
    profile: Mapping[str, Any],
    profile_field: str,
    fallback: Any,
    environment: Mapping[str, str],
) -> str | None:
    value = _text(override)

    if value is not None:
        return value

    value = _text(environment_value(environment_field, environment))

    if value is not None:
        return value

    value = _text(profile.get(profile_field))

    if value is not None:
        return value

    return _text(fallback)


def resolve_model_selection(
    sett: Settings,
    overrides: CliOverrides | None = None,
    environment: Mapping[str, str] | None = None,
) -> ModelSelection:
    env = os.environ if environment is None else environment
    cli = overrides or CliOverrides()

    provider_id = (
        _text(cli.provider_id)
        or _text(environment_value("model_provider", env))
        or _text(sett.model_provider)
    )

    if provider_id is None:
        raise ValueError("model provider cannot be empty")

    profile = _profile(sett, provider_id)

    model_name = _select(
        override=cli.model_name,
        environment_field="model_name",
        profile=profile,
        profile_field="model_name",
        fallback=sett.model_name,
        environment=env,
    )

    if model_name is None:
        raise ValueError("model name cannot be empty")

    reflection_model_name = _select(
        override=cli.reflection_model_name,
        environment_field="reflection_model_name",
        profile=profile,
        profile_field="reflection_model_name",
        fallback=None,
        environment=env,
    )

    if reflection_model_name is None:
        if provider_id in _PRIMARY_REFLECTION_FALLBACK:
            reflection_model_name = model_name
        else:
            reflection_model_name = _text(sett.reflection_model_name) or model_name

    reasoning_effort = _select(
        override=cli.reasoning_effort,
        environment_field="reasoning_effort",
        profile=profile,
        profile_field="reasoning_effort",
        fallback=sett.reasoning_effort,
        environment=env,
    )

    return ModelSelection(
        provider_id=provider_id,
        model_name=model_name,
        reflection_model_name=reflection_model_name,
        reasoning_effort=reasoning_effort,
    )


def resolve_provider_runtime(
    sett: Settings,
    *,
    provider_id: str,
    active_provider_id: str,
    overrides: CliOverrides | None = None,
) -> ProviderRuntimeConfig:
    cli = overrides or CliOverrides()
    profile = _profile(sett, provider_id)

    use_cli = provider_id == active_provider_id

    cli_api_key = cli.api_key if use_cli else None
    cli_base_url = cli.base_url if use_cli else None
    cli_session_id = cli.session_id if use_cli else None

    if provider_id == "opencode":
        stored_api_key = sett.opencode_api_key
    elif provider_id == "codex":
        stored_api_key = None
    else:
        stored_api_key = sett.openai_api_key

    api_key = _text(cli_api_key) or _text(stored_api_key)

    base_url = (
        _text(cli_base_url)
        or _text(profile.get("base_url"))
        or _text(sett.openai_base_url)
    )

    session_id = (
        _text(cli_session_id)
        or _text(profile.get("session_id"))
        or _text(sett.openrouter_session_id)
    )

    return ProviderRuntimeConfig(
        provider_id=provider_id,
        api_key=api_key,
        base_url=base_url,
        max_retries=sett.api_max_retries,
        cache_ttl=(sett.openrouter_cache_ttl if provider_id == "openrouter" else None),
        anthropic_messages=(
            sett.openrouter_anthropic_messages if provider_id == "openrouter" else False
        ),
        session_id=(session_id if provider_id == "openrouter" else None),
    )


def resolve_runtime_config(
    sett: Settings,
    overrides: CliOverrides | None = None,
    environment: Mapping[str, str] | None = None,
) -> tuple[RuntimeConfig, ProviderRuntimeConfig]:
    model = resolve_model_selection(
        sett,
        overrides,
        environment,
    )

    provider = resolve_provider_runtime(
        sett,
        provider_id=model.provider_id,
        active_provider_id=model.provider_id,
        overrides=overrides,
    )

    return (
        RuntimeConfig(
            settings=app_settings(sett),
            model=model,
        ),
        provider,
    )
