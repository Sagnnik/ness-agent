from __future__ import annotations

from typing import Any

from pydantic import AliasChoices, Field
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from ness_cli.config_store import load_configs, load_secrets
from ness_cli.provider.openrouter.catalog import cached_models, model_record


# USD per 1M tokens: (input, output, cache_read_ratio)
# Fallback when the provider does not return cost in response metadata.
# Order matters for substring matching — list more specific slugs first.
MODEL_PRICING: dict[str, tuple[float, float, float]] = {
    "gpt-5.5": (5.00, 30.00, 0.10),
    "gpt-5.4": (2.50, 15.00, 0.10),
    "gpt-5.2": (1.75, 14.00, 0.10),
    "gpt-5.1": (1.25, 10.00, 0.10),
    "gpt-5": (1.25, 10.00, 0.10),
    "gpt-4o-mini": (0.15, 0.60, 0.50),
    "gpt-4o": (2.50, 10.00, 0.50),
    "gpt-4.1": (2.00, 8.00, 0.25),
    "o4-mini": (1.10, 4.40, 0.25),
    "claude-opus-4.8": (5.00, 25.00, 0.10),
    "claude-opus-4.7": (5.00, 25.00, 0.10),
    "claude-opus-4.6": (5.00, 25.00, 0.10),
    "claude-sonnet-5": (2.00, 10.00, 0.10),
    "claude-sonnet-4.6": (3.00, 15.00, 0.10),
    "claude-sonnet-4.5": (3.00, 15.00, 0.10),
    "claude-sonnet-4": (3.00, 15.00, 0.10),
    "claude-haiku-4.5": (1.00, 5.00, 0.10),
    "claude-3.5-sonnet": (3.00, 15.00, 0.10),
    "claude-3-haiku": (0.25, 1.25, 0.12),
    "gemini-3.1-pro": (2.00, 12.00, 0.10),
    "gemini-2.5-pro": (1.25, 10.00, 0.10),
    "gemini-2.5-flash": (0.30, 2.50, 0.10),
    "gemini-2.0-flash": (0.10, 0.40, 0.10),
    "deepseek-v4-flash": (0.09, 0.18, 0.20),
    "deepseek-chat": (0.20, 0.80, 0.10),
    "kimi-k2.7-code": (0.74, 3.50, 0.20),
    "kimi-k2.6": (0.66, 3.41, 0.21),
    "glm-5.2": (0.69, 2.16, 0.19),
    "glm-5.1": (0.97, 3.04, 0.19),
}

MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "gpt-5.5": 1_050_000,
    "gpt-5.4": 1_050_000,
    "gpt-5.2": 400_000,
    "gpt-5.1": 400_000,
    "gpt-5": 400_000,
    "gpt-4o-mini": 128_000,
    "gpt-4o": 128_000,
    "gpt-4.1": 1_047_576,
    "o4-mini": 200_000,
    "o3": 200_000,
    "claude-opus-4.8": 1_000_000,
    "claude-opus-4.7": 1_000_000,
    "claude-opus-4.6": 1_000_000,
    "claude-sonnet-5": 1_000_000,
    "claude-sonnet-4.6": 1_000_000,
    "claude-sonnet-4.5": 1_000_000,
    "claude-sonnet-4": 1_000_000,
    "claude-haiku-4.5": 200_000,
    "claude-3.5-sonnet": 200_000,
    "claude-3-5-sonnet": 200_000,
    "claude-3-opus": 200_000,
    "claude-3-sonnet": 200_000,
    "claude-3-haiku": 200_000,
    "gemini-3.1-pro": 1_048_576,
    "gemini-2.5-pro": 1_048_576,
    "gemini-2.5-flash": 1_048_576,
    "gemini-2.0-flash": 1_000_000,
    "deepseek-chat": 131_072,
    "deepseek-v4-flash": 1_048_576,
    "glm-5.2": 1_048_576,
    "glm-5.1": 202_752,
    "kimi-k2.7-code": 262_144,
    "kimi-k2.6": 262_144,
}

VISION_MODELS = {
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-vision",
    "gpt-5",
    "claude-3.5-sonnet",
    "claude-sonnet-4",
    "claude-sonnet-5",
    "claude-opus-4",
    "claude-3-opus",
    "claude-3-haiku",
    "claude-haiku-4.5",
    "gemini-pro-vision",
    "gemini-2.0-flash",
    "gemini-2.5-flash",
    "gemini-3.1-pro",
    "gemini-2.5-pro",
    "glm-5.1",
    "kimi-k2.6",
    "kimi-k2.7-code",
}

# Curated OpenRouter slugs offered by the /config model switcher. Edit freely.
# Substring matching against MODEL_PRICING / MODEL_CONTEXT_WINDOWS keeps cost and
# context-window resolution working for the provider-prefixed slugs below.
AVAILABLE_MODELS: tuple[str, ...] = (
    "openai/gpt-4o-mini",
    "openai/gpt-4o",
    "openai/gpt-4.1",
    "openai/o4-mini",
    "openai/gpt-5",
    "openai/gpt-5.1",
    "openai/gpt-5.2",
    "openai/gpt-5.4",
    "openai/gpt-5.5",
    "anthropic/claude-3-haiku",
    "anthropic/claude-haiku-4.5",
    "anthropic/claude-sonnet-4",
    "anthropic/claude-sonnet-4.5",
    "anthropic/claude-sonnet-4.6",
    "anthropic/claude-sonnet-5",
    "anthropic/claude-opus-4.6",
    "anthropic/claude-opus-4.7",
    "anthropic/claude-opus-4.8",
    "google/gemini-2.5-flash",
    "google/gemini-2.5-pro",
    "google/gemini-3.1-pro-preview",
    "deepseek/deepseek-chat",
    "deepseek/deepseek-v4-flash",
    "moonshotai/kimi-k2.6",
    "moonshotai/kimi-k2.7-code",
    "z-ai/glm-5.1",
    "z-ai/glm-5.2",
)

# Per-model reasoning metadata (sync via scripts/fetch_openrouter_models.py).
# Keys are short slugs; matched with `key in model_name.lower()` like pricing.
# Order matters — more specific slugs must appear before shorter prefixes.
# `efforts: ()` = thinking model with no effort selector; `None` = all REASONING_EFFORTS.
MODEL_REASONING: dict[str, dict[str, Any]] = {
    "gpt-5.5": {"efforts": ("xhigh", "high", "medium", "low", "none"), "default": "medium", "mandatory": False},
    "gpt-5.4": {"efforts": ("xhigh", "high", "medium", "low", "none"), "default": "medium", "mandatory": False},
    "gpt-5.2": {"efforts": ("xhigh", "high", "medium", "low", "none"), "default": "medium", "mandatory": False},
    "gpt-5.1": {"efforts": ("high", "medium", "low", "none"), "default": "none", "mandatory": False},
    "gpt-5": {"efforts": ("high", "medium", "low", "minimal"), "default": "medium", "mandatory": True},
    "o4-mini": {"efforts": ("low", "medium", "high"), "default": None, "mandatory": False},
    "claude-opus-4.8": {"efforts": ("max", "xhigh", "high", "medium", "low"), "default": "medium", "mandatory": False},
    "claude-opus-4.7": {"efforts": ("max", "xhigh", "high", "medium", "low"), "default": "medium", "mandatory": False},
    "claude-opus-4.6": {"efforts": ("max", "high", "medium", "low"), "default": "medium", "mandatory": False},
    "claude-sonnet-5": {"efforts": ("max", "xhigh", "high", "medium", "low"), "default": "medium", "mandatory": False},
    "claude-sonnet-4.6": {"efforts": ("max", "high", "medium", "low"), "default": "medium", "mandatory": False},
    "claude-sonnet-4.5": {"efforts": (), "default": None, "mandatory": False},
    "claude-sonnet-4": {"efforts": (), "default": None, "mandatory": False},
    "claude-haiku-4.5": {"efforts": (), "default": None, "mandatory": False},
    "gemini-3.1-pro": {"efforts": (), "default": None, "mandatory": True},
    "gemini-2.5-pro": {"efforts": (), "default": None, "mandatory": True},
    "gemini-2.5-flash": {"efforts": (), "default": None, "mandatory": False},
    "deepseek-v4-flash": {"efforts": ("xhigh", "high"), "default": "high", "mandatory": False},
    "kimi-k2.7-code": {"efforts": (), "default": None, "mandatory": True},
    "kimi-k2.6": {"efforts": (), "default": None, "mandatory": False},
    "glm-5.2": {"efforts": ("high", "max"), "default": "high", "mandatory": False},
    "glm-5.1": {"efforts": (), "default": None, "mandatory": False},
}

ReasoningEffort = str
REASONING_EFFORTS: tuple[str, ...] = ("none", "minimal", "low", "medium", "high", "xhigh", "max")


class _JsonDirSource(PydanticBaseSettingsSource):
    """Settings source reading global ``configs.json`` + ``secrets.json``.

    The files are keyed by Settings field names (e.g. ``model_name``);
    values are re-keyed to validation aliases so pydantic matches them to
    the aliased fields. Missing/corrupt files yield an empty source.
    """

    def __call__(self) -> dict[str, Any]:
        raw = {**load_configs(), **load_secrets()}
        values: dict[str, Any] = {}
        for name, field in self.settings_cls.model_fields.items():
            if name not in raw:
                continue
            alias = field.validation_alias or field.alias or name
            if isinstance(alias, AliasChoices):
                alias = alias.choices[0]
            if isinstance(alias, str):
                values[alias] = raw[name]
        return values

    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        return None, field_name, False


class Settings(BaseSettings):
    """Adapter settings schema + defaults.

    Precedence: init kwargs / CLI overrides > process env vars >
    ``secrets.json`` / ``configs.json`` > the field defaults below (which
    are therefore what a first run starts with).
    """

    model_config = SettingsConfigDict(env_prefix="", populate_by_name=True)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (init_settings, env_settings, _JsonDirSource(settings_cls))

    model_provider: str = Field(default="openrouter", alias="MODEL_PROVIDER")
    provider_profiles: dict[str, Any] = Field(default_factory=dict, alias="PROVIDER_PROFILES")
    model_name: str = Field(default="deepseek/deepseek-v4-flash", alias="MODEL_NAME")
    reflection_model_name: str = Field(default="deepseek/deepseek-v4-flash", alias="REFLECTION_MODEL_NAME")
    reasoning_effort: ReasoningEffort = Field(default="xhigh", alias="REASONING_EFFORT")
    api_max_retries: int = Field(default=3, alias="API_MAX_RETRIES")
    enable_approval: bool = Field(default=True, alias="ENABLE_APPROVAL")
    auto_save_threads: bool = Field(default=True, alias="AUTO_SAVE_THREADS")
    session_end_reflection: bool = Field(default=False, alias="SESSION_END_REFLECTION")
    reflection_token_ratio: float = Field(default=0.4, alias="REFLECTION_TOKEN_RATIO")
    compaction_token_budget: int = Field(default=120_000, alias="COMPACTION_TOKEN_BUDGET")
    compaction_buffer_tokens: int = Field(default=16_384, alias="COMPACTION_BUFFER_TOKENS")
    compaction_summary_max_tokens: int = Field(
        default=4_096, alias="COMPACTION_SUMMARY_MAX_TOKENS"
    )
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    opencode_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("OPENCODE_GO_API_KEY", "OPENCODE_API_KEY"),
    )
    openai_base_url: str | None = Field(default=None, alias="OPENAI_BASE_URL")
    openrouter_session_id: str | None = Field(default=None, alias="OPENROUTER_SESSION_ID")
    openrouter_cache_ttl: str = Field(default="5m", alias="OPENROUTER_CACHE_TTL")
    openrouter_anthropic_messages: bool = Field(
        default=True,
        alias="OPENROUTER_ANTHROPIC_MESSAGES",
    )
    goal_judge_model: str | None = Field(default=None, alias="GOAL_JUDGE_MODEL")
    goal_max_attempts: int = Field(default=3, alias="GOAL_MAX_ATTEMPTS")
    ness_dir: str = Field(default=".ness", alias="NESS_DIR")
    format_on_write: bool = Field(default=True, alias="FORMAT_ON_WRITE")
    exa_api_key: str | None = Field(default=None, alias="EXA_API_KEY")

    @property
    def has_exa(self) -> bool:
        return bool(self.exa_api_key)

    @property
    def supports_vision(self) -> bool:
        from ness_cli.provider.registry import get_provider

        for item in getattr(get_provider(self.model_provider), "_models", ()):
            if item.id == self.model_name:
                return item.supports_vision
        record = model_record(self.model_name)
        if record is not None:
            return record.supports_vision
        model = self.model_name.lower()
        return any(marker in model for marker in VISION_MODELS)


settings = Settings()


def reload_settings() -> None:
    """Re-read env + global JSON config into the shared settings.

    Mutates the existing ``settings`` singleton in place so every module that did
    ``from config import settings`` observes the new values.
    """
    fresh = Settings()
    for field in type(fresh).model_fields:
        setattr(settings, field, getattr(fresh, field))
    from ness_agent.tools.web import reset_provider

    reset_provider()


def make_sdk_cost_tracker():
    """SDK CostTracker wired with CLI MODEL_PRICING estimates for non-provider costs."""
    from ness_agent.tracing.cost import CostTracker

    pricing = dict(MODEL_PRICING)
    for record in cached_models():
        if record.input_price is None or record.output_price is None:
            continue
        pricing[record.id] = (
            record.input_price,
            record.output_price,
            record.cache_read_ratio,
        )
    return CostTracker(pricing=pricing)


def resolve_model_key(model_name: str, catalog: dict[str, Any]) -> str | None:
    model = model_name.lower()
    return next((candidate for candidate in catalog if candidate in model), None)


def context_window_for(model_name: str) -> int | None:
    """Context window for *model_name* from the CLI catalog, if known.

    Fed into ``NessAgentOptions.context_window`` so the SDK's usable-budget
    resolution (window − reserves) applies instead of the flat
    ``compaction_token_budget`` fallback.
    """
    record = model_record(model_name)
    if record is not None and record.context_length:
        return record.context_length
    key = resolve_model_key(model_name, MODEL_CONTEXT_WINDOWS)
    return MODEL_CONTEXT_WINDOWS[key] if key is not None else None


def _reasoning_entry(model_name: str) -> dict[str, Any] | None:
    key = resolve_model_key(model_name, MODEL_REASONING)
    if key is None:
        return None
    return MODEL_REASONING[key]


def model_supports_reasoning(model_name: str) -> bool:
    from ness_cli.provider.registry import get_provider

    for item in getattr(get_provider(settings.model_provider), "_models", ()):
        if item.id == model_name:
            return bool(item.reasoning_efforts)
    record = model_record(model_name)
    if record is not None:
        return bool(record.reasoning_efforts) or "reasoning" in record.supported_parameters
    return _reasoning_entry(model_name) is not None


def reasoning_efforts_for_model(model_name: str) -> tuple[str, ...]:
    from ness_cli.provider.registry import get_provider

    for item in getattr(get_provider(settings.model_provider), "_models", ()):
        if item.id == model_name:
            return item.reasoning_efforts
    record = model_record(model_name)
    if record is not None:
        return record.reasoning_efforts
    entry = _reasoning_entry(model_name)
    if entry is None:
        return ()
    efforts = entry.get("efforts")
    supported = REASONING_EFFORTS if efforts is None else tuple(efforts)
    if entry.get("mandatory"):
        supported = tuple(level for level in supported if level != "none")
    return supported


def default_effort(model_name: str) -> str | None:
    from ness_cli.provider.registry import get_provider

    for item in getattr(get_provider(settings.model_provider), "_models", ()):
        if item.id == model_name and item.default_reasoning_effort:
            return item.default_reasoning_effort
    efforts = reasoning_efforts_for_model(model_name)
    if not efforts:
        return None
    entry = _reasoning_entry(model_name) or {}
    default = entry.get("default")
    if default and default in efforts:
        return str(default)
    return efforts[0]


def coerce_reasoning_effort(model_name: str, effort: str | None) -> str | None:
    efforts = reasoning_efforts_for_model(model_name)
    if not efforts:
        return None
    if effort and effort in efforts:
        return effort
    return default_effort(model_name)


def available_model_ids() -> tuple[str, ...]:
    """Return the cached dynamic catalog, or the packaged offline fallback."""
    if settings.model_provider != "openrouter":
        from ness_cli.provider.registry import get_provider

        models = getattr(get_provider(settings.model_provider), "_models", ())
        if models:
            return tuple(item.id for item in models)
        profile = settings.provider_profiles.get(settings.model_provider, {})
        return (str(profile.get("model_name") or settings.model_name),)
    records = cached_models()
    return tuple(record.id for record in records) if records else AVAILABLE_MODELS
