from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class CliOverrides:
    provider_id: str | None = None
    model_name: str | None = None
    reflection_model_name: str | None = None
    reasoning_effort: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    session_id: str | None = None


@dataclass(frozen=True, slots=True)
class AppSettings:
    enable_approval: bool
    auto_save_threads: bool
    session_end_reflection: bool
    reflection_token_ratio: float
    compaction_token_budget: int
    compaction_buffer_tokens: int
    compaction_summary_max_tokens: int
    format_on_write: bool
    goal_judge_model: str | None
    goal_max_attempts: int
    ness_dir: str
    exa_api_key: str | None


@dataclass(frozen=True, slots=True)
class ModelSelection:
    provider_id: str
    model_name: str
    reflection_model_name: str
    reasoning_effort: str | None


@dataclass(frozen=True, slots=True)
class ProviderRuntimeConfig:
    provider_id: str
    api_key: str | None
    base_url: str | None
    max_retries: int
    cache_ttl: str | None
    anthropic_messages: bool
    session_id: str | None


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    settings: AppSettings
    model: ModelSelection


@dataclass(frozen=True, slots=True)
class ConfigPatch:
    values: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))


@dataclass(frozen=True, slots=True)
class ConfigUpdate:
    previous: RuntimeConfig
    current: RuntimeConfig
    changed_keys: tuple[str, ...]
    model_changed: bool
    provider_runtime_changed: bool
    options_changed: bool
    reload_selected: bool
    future_sessions_changed: bool
    restart_required: bool
    message: str


class ConfigApplyError(RuntimeError):
    """A save failure with the verified effects, or unknown persistence state.

    Key names describe requested values that are present on disk, including
    already-saved values. No credential values appear in the error message.
    """

    def __init__(
        self,
        message: str,
        *,
        update: ConfigUpdate | None,
        saved_keys: tuple[str, ...] | None,
        unsaved_keys: tuple[str, ...] | None,
        failed_document: str,
    ) -> None:
        super().__init__(message)
        self.update = update
        self.saved_keys = saved_keys
        self.unsaved_keys = unsaved_keys
        self.failed_document = failed_document
