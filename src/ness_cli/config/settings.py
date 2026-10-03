from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ness_cli.config.store import ConfigStore
from ness_cli.config.types import AppSettings


_ENV_ALIASES: dict[str, tuple[str, ...]] = {
    "model_provider": ("MODEL_PROVIDER",),
    "provider_profiles": ("PROVIDER_PROFILES",),
    "model_name": ("MODEL_NAME",),
    "reflection_model_name": ("REFLECTION_MODEL_NAME",),
    "reasoning_effort": ("REASONING_EFFORT",),
    "api_max_retries": ("API_MAX_RETRIES",),
    "enable_approval": ("ENABLE_APPROVAL",),
    "auto_save_threads": ("AUTO_SAVE_THREADS",),
    "session_end_reflection": ("SESSION_END_REFLECTION",),
    "reflection_token_ratio": ("REFLECTION_TOKEN_RATIO",),
    "compaction_token_budget": ("COMPACTION_TOKEN_BUDGET",),
    "compaction_buffer_tokens": ("COMPACTION_BUFFER_TOKENS",),
    "compaction_summary_max_tokens": ("COMPACTION_SUMMARY_MAX_TOKENS",),
    "openai_api_key": ("OPENAI_API_KEY",),
    "opencode_api_key": (
        "OPENCODE_GO_API_KEY",
        "OPENCODE_API_KEY",
    ),
    "openai_base_url": ("OPENAI_BASE_URL",),
    "openrouter_session_id": ("OPENROUTER_SESSION_ID",),
    "openrouter_cache_ttl": ("OPENROUTER_CACHE_TTL",),
    "openrouter_anthropic_messages": ("OPENROUTER_ANTHROPIC_MESSAGES",),
    "goal_judge_model": ("GOAL_JUDGE_MODEL",),
    "goal_max_attempts": ("GOAL_MAX_ATTEMPTS",),
    "ness_dir": ("NESS_DIR",),
    "format_on_write": ("FORMAT_ON_WRITE",),
    "exa_api_key": ("EXA_API_KEY",),
}


class Settings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model_provider: str = "openrouter"
    provider_profiles: dict[str, Any] = Field(default_factory=dict)

    model_name: str = "deepseek/deepseek-v4-flash"
    reflection_model_name: str = "deepseek/deepseek-v4-flash"
    reasoning_effort: str | None = "xhigh"

    api_max_retries: int = Field(default=3, ge=0)

    enable_approval: bool = True
    auto_save_threads: bool = True
    session_end_reflection: bool = False
    reflection_token_ratio: float = Field(default=0.4, ge=0.0, le=1.0)

    compaction_token_budget: int = Field(default=120_000, gt=0)
    compaction_buffer_tokens: int = Field(default=16_384, ge=0)
    compaction_summary_max_tokens: int = Field(default=4_096, gt=0)

    openai_api_key: str | None = None
    opencode_api_key: str | None = None
    openai_base_url: str | None = None

    openrouter_session_id: str | None = None
    openrouter_cache_ttl: str = "5m"
    openrouter_anthropic_messages: bool = True

    goal_judge_model: str | None = None
    goal_max_attempts: int = Field(default=3, gt=0)

    ness_dir: str = ".ness"
    format_on_write: bool = True
    exa_api_key: str | None = None


def environment_value(
    field_name: str,
    environment: Mapping[str, str],
) -> str | None:
    folded = {key.casefold(): value for key, value in environment.items()}

    for alias in _ENV_ALIASES.get(field_name, ()):
        value = folded.get(alias.casefold())
        if value is not None:
            return value

    return None


def _environment_field_value(field_name: str, value: str) -> Any:
    if field_name != "provider_profiles":
        return value

    try:
        return json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError("PROVIDER_PROFILES must contain valid JSON") from error


def load_documents(
    config: Mapping[str, Any],
    secrets: Mapping[str, Any],
    environment: Mapping[str, str] | None = None,
) -> Settings:
    env = os.environ if environment is None else environment
    values = {**dict(config), **dict(secrets)}

    for field_name in Settings.model_fields:
        value = environment_value(field_name, env)

        if value is not None:
            values[field_name] = _environment_field_value(
                field_name,
                value,
            )

    return Settings.model_validate(values)


def load_settings(
    store: ConfigStore,
    environment: Mapping[str, str] | None = None,
) -> Settings:
    return load_documents(
        store.load_config(),
        store.load_secrets(),
        environment,
    )


def app_settings(sett: Settings) -> AppSettings:
    return AppSettings(
        enable_approval=sett.enable_approval,
        auto_save_threads=sett.auto_save_threads,
        session_end_reflection=sett.session_end_reflection,
        reflection_token_ratio=sett.reflection_token_ratio,
        compaction_token_budget=sett.compaction_token_budget,
        compaction_buffer_tokens=sett.compaction_buffer_tokens,
        compaction_summary_max_tokens=(sett.compaction_summary_max_tokens),
        format_on_write=sett.format_on_write,
        goal_judge_model=sett.goal_judge_model,
        goal_max_attempts=sett.goal_max_attempts,
        ness_dir=sett.ness_dir,
        exa_api_key=sett.exa_api_key,
    )


def sdk_behavior_values(settings: AppSettings, *, yolo_mode: bool) -> dict[str, Any]:
    """Map mutable CLI settings for SDK construction or validated replacement."""
    return {
        "enable_approval": settings.enable_approval and not yolo_mode,
        "auto_save_threads": settings.auto_save_threads,
        "session_end_reflection": settings.session_end_reflection,
        "reflection_token_ratio": settings.reflection_token_ratio,
        "compaction_token_budget": settings.compaction_token_budget,
        "compaction_buffer_tokens": settings.compaction_buffer_tokens,
        "compaction_summary_max_tokens": settings.compaction_summary_max_tokens,
        "format_on_write": settings.format_on_write,
        "exa_api_key": settings.exa_api_key,
    }
