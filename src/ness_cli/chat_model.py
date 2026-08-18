from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, replace
from typing import cast

from langchain_core.language_models import BaseChatModel
from langchain_openrouter import ChatOpenRouter  # noqa: F401  # compatibility patch point

from ness_cli.config import (
    ReasoningEffort,
    coerce_reasoning_effort,
    reasoning_efforts_for_model,
    settings,
)
from ness_cli.provider.profile import provider_profile, update_provider_profile
from ness_cli.provider.registry import active_provider, get_provider


@dataclass(frozen=True)
class ModelOverrides:
    """Optional CLI/runtime overrides for model construction."""

    model_name: str | None = None
    reflection_model_name: str | None = None
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    openrouter_session_id: str | None = None
    reasoning_effort: ReasoningEffort | None = None


_overrides: ModelOverrides | None = None

def provider_key_missing() -> bool:
    """True when the active provider has no usable credentials."""
    return not active_provider().is_authenticated()


def configure_model(overrides: ModelOverrides | None = None) -> None:
    """Apply runtime overrides that take precedence over settings."""
    global _overrides
    _overrides = overrides


def set_active_model(model_name: str) -> str | None:
    """Switch the active chat model at runtime.

    Updates both the override (used when constructing the chat model) and
    ``settings.model_name`` so cost and context-window lookups follow the switch.
    Callers must rebuild the graph afterwards to bind the new model.

    Returns the coerced reasoning effort when it changed for the new model.
    """
    global _overrides
    base = _overrides or ModelOverrides()
    current_effort = cast(str | None, _resolved_setting("reasoning_effort"))
    coerced = coerce_reasoning_effort(model_name, current_effort)
    if coerced != current_effort:
        if coerced is not None:
            settings.reasoning_effort = cast(ReasoningEffort, coerced)
        _overrides = replace(
            base,
            model_name=model_name,
            reasoning_effort=cast(ReasoningEffort, coerced) if coerced is not None else base.reasoning_effort,
        )
    else:
        _overrides = replace(base, model_name=model_name)
    settings.model_name = model_name
    profile = dict(settings.provider_profiles.get(settings.model_provider, {}))
    profile["model_name"] = model_name
    settings.provider_profiles[settings.model_provider] = profile
    return coerced if coerced != current_effort else None


def set_active_reasoning_effort(reasoning_effort: ReasoningEffort) -> None:
    """Switch the active provider's reasoning effort at runtime."""
    allowed = reasoning_efforts_for_model(active_model_name())
    if reasoning_effort not in allowed:
        allowed_text = ", ".join(allowed) if allowed else "(none)"
        raise ValueError(f"invalid reasoning effort for model: {reasoning_effort} (allowed: {allowed_text})")
    global _overrides
    base = _overrides or ModelOverrides()
    _overrides = replace(base, reasoning_effort=reasoning_effort)
    settings.reasoning_effort = reasoning_effort
    profile = dict(settings.provider_profiles.get(settings.model_provider, {}))
    profile["reasoning_effort"] = reasoning_effort
    settings.provider_profiles[settings.model_provider] = profile


def _resolved_setting(field: str) -> str | int | bool | None:
    if _overrides is not None:
        value = getattr(_overrides, field, None)
        if value is not None:
            return value
    return getattr(settings, field)


def _environment_setting(field: str) -> str | None:
    """Return a process-environment value using Pydantic's field alias."""
    field_info = type(settings).model_fields[field]
    alias = field_info.validation_alias or field_info.alias or field
    if not isinstance(alias, str):
        return None
    expected = alias.casefold()
    return next(
        (value for key, value in os.environ.items() if key.casefold() == expected),
        None,
    )


def active_model_name() -> str:
    if _overrides is not None and _overrides.model_name is not None:
        return _overrides.model_name
    environment = _environment_setting("model_name")
    if environment is not None:
        return environment
    profile = settings.provider_profiles.get(settings.model_provider, {})
    return cast(str, profile.get("model_name") or _resolved_setting("model_name"))


def active_reasoning_effort() -> ReasoningEffort:
    if _overrides is not None and _overrides.reasoning_effort is not None:
        return _overrides.reasoning_effort
    environment = _environment_setting("reasoning_effort")
    if environment is not None:
        return cast(ReasoningEffort, environment)
    profile = settings.provider_profiles.get(settings.model_provider, {})
    return cast(ReasoningEffort, profile.get("reasoning_effort") or _resolved_setting("reasoning_effort"))


def active_provider_id() -> str:
    return settings.model_provider


def activate_provider(provider_id: str, *, model_name: str | None = None, reasoning_effort: str | None = None) -> None:
    """Activate a provider and its profile without storing provider secrets."""
    get_provider(provider_id)  # validate before mutating config
    previous_id = settings.model_provider
    if previous_id != provider_id:
        previous = provider_profile(previous_id)
        previous.setdefault("model_name", active_model_name())
        previous.setdefault("reasoning_effort", active_reasoning_effort())
        update_provider_profile(previous_id, previous)
        settings.provider_profiles[previous_id] = previous
    profile = provider_profile(provider_id)
    if model_name:
        profile["model_name"] = model_name
    if reasoning_effort:
        profile["reasoning_effort"] = reasoning_effort
    update_provider_profile(provider_id, profile)
    from ness_cli.config_store import write_config

    write_config("model_provider", provider_id)
    settings.model_provider = provider_id
    settings.provider_profiles[provider_id] = profile
    if profile.get("model_name"):
        settings.model_name = str(profile["model_name"])
    if profile.get("reasoning_effort"):
        settings.reasoning_effort = str(profile["reasoning_effort"])


def openrouter_session(thread_id: str, *, suffix: str = "") -> str:
    base = _resolved_setting("openrouter_session_id") or thread_id
    if suffix:
        return f"{base}:{suffix}"
    return cast(str, base)


def build_chat_model(
    thread_id: str,
    *,
    model_name: str | None = None,
    session_suffix: str = "",
) -> BaseChatModel:
    resolved_model = cast(str, model_name or active_model_name())
    return active_provider().build_chat_model(
        thread_id,
        model_name=resolved_model,
        reasoning_effort=active_reasoning_effort(),
        session_suffix=session_suffix,
    )


def create_model(thread_id: str) -> BaseChatModel:
    return build_chat_model(thread_id)


def create_reflection_model(thread_id: str) -> BaseChatModel:
    profile = settings.provider_profiles.get(settings.model_provider, {})
    environment = _environment_setting("reflection_model_name")
    if _overrides is not None and _overrides.reflection_model_name is not None:
        reflection_model = _overrides.reflection_model_name
    elif environment is not None:
        reflection_model = environment
    else:
        reflection_model = profile.get("reflection_model_name") or (
            active_model_name()
            if settings.model_provider in {"codex", "opencode"}
            else _resolved_setting("reflection_model_name")
        )
    return build_chat_model(
        thread_id,
        model_name=cast(str, reflection_model),
        session_suffix="reflection",
    )


def validate_effort(model_name: str, reasoning_effort: str) -> None:
    allowed = reasoning_efforts_for_model(model_name)
    if not allowed:
        return
    if reasoning_effort not in allowed:
        raise ValueError(
            f"reasoning effort must be one of: {', '.join(allowed)} for model {model_name!r}"
        )


def model_overrides_from_args(args: argparse.Namespace) -> ModelOverrides | None:
    fields = {
        "model_name": args.model,
        "reflection_model_name": args.reflection_model,
        "openai_api_key": args.api_key,
        "openai_base_url": args.base_url,
        "openrouter_session_id": args.openrouter_session_id,
        "reasoning_effort": getattr(args, "reasoning_effort", None),
    }
    active = {key: value for key, value in fields.items() if value is not None}
    if not active:
        return None
    return ModelOverrides(**active)


def add_model_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        help="Chat model name (overrides MODEL_NAME)",
    )
    parser.add_argument(
        "--reflection-model",
        help="Reflection model name (overrides REFLECTION_MODEL_NAME)",
    )
    parser.add_argument(
        "--api-key",
        help="OpenAI-compatible API key (overrides OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--base-url",
        help="OpenAI-compatible base URL (overrides OPENAI_BASE_URL)",
    )
    parser.add_argument(
        "--openrouter-session-id",
        help="Stable OpenRouter prompt-cache session id (overrides OPENROUTER_SESSION_ID)",
    )
    parser.add_argument(
        "--reasoning-effort",
        help="Provider-literal reasoning effort (overrides REASONING_EFFORT)",
    )
