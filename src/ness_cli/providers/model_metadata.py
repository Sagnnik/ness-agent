"""Provider-independent fallback metadata for known model families.

Live provider catalogs take priority. Provider overrides describe differences in
supported options; pricing and protocol routing stay in provider modules.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelFallback:
    context_window: int | None = None
    reasoning_efforts: tuple[str, ...] = ()
    default_reasoning_effort: str | None = None
    supports_vision: bool = False


# API fallbacks. Codex subscription windows are maintained by that provider.
_MODEL_FALLBACKS: dict[str, ModelFallback] = {
    "gpt-5.6-luna": ModelFallback(1_050_000, (), None, True),
    "gpt-5.5": ModelFallback(
        1_050_000, ("xhigh", "high", "medium", "low", "none"), "medium", True
    ),
    "gpt-5.4": ModelFallback(
        1_050_000, ("xhigh", "high", "medium", "low", "none"), "medium", True
    ),
    "gpt-5.2": ModelFallback(
        400_000, ("xhigh", "high", "medium", "low", "none"), "medium", True
    ),
    "gpt-5.1": ModelFallback(400_000, ("high", "medium", "low", "none"), "none", True),
    "gpt-5": ModelFallback(
        400_000, ("high", "medium", "low", "minimal"), "medium", True
    ),
    "gpt-4o-mini": ModelFallback(128_000, (), None, True),
    "gpt-4o": ModelFallback(128_000, (), None, True),
    "gpt-4.1": ModelFallback(1_047_576, (), None, False),
    "o4-mini": ModelFallback(200_000, ("low", "medium", "high"), None, False),
    "o3": ModelFallback(200_000, (), None, False),
    "claude-opus-4.8": ModelFallback(
        1_000_000, ("max", "xhigh", "high", "medium", "low"), "medium", True
    ),
    "claude-opus-4.7": ModelFallback(
        1_000_000, ("max", "xhigh", "high", "medium", "low"), "medium", True
    ),
    "claude-opus-4.6": ModelFallback(
        1_000_000, ("max", "high", "medium", "low"), "medium", True
    ),
    "claude-sonnet-5": ModelFallback(
        1_000_000, ("max", "xhigh", "high", "medium", "low"), "medium", True
    ),
    "claude-sonnet-4.6": ModelFallback(
        1_000_000, ("max", "high", "medium", "low"), "medium", True
    ),
    "claude-sonnet-4.5": ModelFallback(1_000_000, (), None, True),
    "claude-sonnet-4": ModelFallback(1_000_000, (), None, True),
    "claude-haiku-4.5": ModelFallback(200_000, (), None, True),
    "claude-3.5-sonnet": ModelFallback(200_000, (), None, True),
    "claude-3-5-sonnet": ModelFallback(200_000, (), None, False),
    "claude-3-opus": ModelFallback(200_000, (), None, True),
    "claude-3-sonnet": ModelFallback(200_000, (), None, False),
    "claude-3-haiku": ModelFallback(200_000, (), None, True),
    "gemini-3.1-pro": ModelFallback(1_048_576, (), None, True),
    "gemini-2.5-pro": ModelFallback(1_048_576, (), None, True),
    "gemini-2.5-flash": ModelFallback(1_048_576, (), None, True),
    "gemini-2.0-flash": ModelFallback(1_000_000, (), None, True),
    "deepseek-chat": ModelFallback(131_072, (), None, False),
    "deepseek-v4-flash": ModelFallback(1_048_576, ("xhigh", "high"), "high", False),
    "glm-5.3-flash": ModelFallback(1_310_720, ("max", "high", "low"), "max", True),
    "glm-5.3": ModelFallback(1_310_720, ("max", "high", "low"), "max", False),
    "glm-5.2": ModelFallback(1_048_576, ("high", "max"), "high", False),
    "glm-5.1": ModelFallback(202_752, (), None, True),
    "kimi-k2.7-code": ModelFallback(262_144, (), None, True),
    "kimi-k2.6": ModelFallback(262_144, (), None, True),
}


def matching_model_family(model_id: str, families: Iterable[str]) -> str | None:
    """Match the most specific slug, allowing snapshots but not new versions.

    A namespace is optional. Hyphenated variants and colon suffixes share their
    family's fallback; a different numeric version needs its own metadata.
    """
    slug = model_id.rsplit("/", 1)[-1].casefold()
    for family in sorted(families, key=len, reverse=True):
        marker = family.casefold()
        if slug == marker or slug.startswith((marker + "-", marker + ":")):
            return family
    return None


def fallback_metadata_for(
    model_id: str, *, overrides: Mapping[str, ModelFallback] | None = None
) -> ModelFallback | None:
    """Match one complete fallback record, preferring provider overrides.

    Exact versions and the most specific family win. Snapshots and colon
    suffixes share the same record. Unknown numeric versions remain unknown.
    """
    records = (
        _MODEL_FALLBACKS if overrides is None else {**_MODEL_FALLBACKS, **overrides}
    )
    family = matching_model_family(model_id, records)
    return records.get(family) if family is not None else None


def context_window_for(model_id: str) -> int | None:
    """Return an API fallback; provider catalog metadata takes priority.

    GPT-5.6 Luna's API fallback follows the official model documentation:
    https://developers.openai.com/api/docs/models/gpt-5.6-luna
    """
    metadata = fallback_metadata_for(model_id)
    return metadata.context_window if metadata is not None else None
