from __future__ import annotations

from dataclasses import replace
from typing import Any

import httpx

from ness_cli.providers.base import ModelInfo
from ness_cli.providers.model_metadata import ModelFallback, fallback_metadata_for

OPENCODE_GO_BASE_URL = "https://opencode.ai/zen/go/v1"
OPENCODE_GO_MODELS_URL = f"{OPENCODE_GO_BASE_URL}/models"
OPENCODE_GO_USAGE_URL = f"{OPENCODE_GO_BASE_URL}/usage"

# The live endpoint is authoritative. This seed only keeps login and model
# selection usable during a temporary catalog outage.
FALLBACK_MODEL_IDS: tuple[str, ...] = (
    "deepseek-v4-flash",
    "glm-5.3",
    "glm-5.2",
    "glm-5.1",
    "gpt-5.6-luna",
    "grok-4.5",
    "kimi-k3",
    "kimi-k2.7-code",
    "kimi-k2.6",
    "mimo-v2.5",
    "mimo-v2.5-pro",
    "minimax-m3",
    "minimax-m2.7",
    "qwen3.8-max",
    "qwen3.7-max",
    "qwen3.7-plus",
    "qwen3.6-plus",
    "deepseek-v4-pro",
    "hy3",
)


# Go exposes its own Luna effort set and leaves these defaults unspecified.
# GLM 5.3 has no selectable effort metadata in this provider's seed catalog.
# Other known families reuse API capabilities; protocol/billing stay separate.
_METADATA_OVERRIDES: dict[str, ModelFallback] = {
    "gpt-5.6-luna": replace(
        fallback_metadata_for("gpt-5.6-luna"),
        reasoning_efforts=("high", "medium", "low", "none"),
        default_reasoning_effort=None,
    ),
    "deepseek-v4-flash": replace(
        fallback_metadata_for("deepseek-v4-flash"),
        default_reasoning_effort=None,
    ),
    "glm-5.2": replace(
        fallback_metadata_for("glm-5.2"),
        default_reasoning_effort=None,
    ),
    "glm-5.3": replace(
        fallback_metadata_for("glm-5.3"),
        reasoning_efforts=(),
        default_reasoning_effort=None,
    ),
    "grok-4.5": ModelFallback(supports_vision=True),
    "kimi-k3": ModelFallback(supports_vision=True),
}


def display_name(model_id: str) -> str:
    words: list[str] = []
    for word in model_id.replace("-", " ").split():
        lowered = word.casefold()
        if lowered in {"gpt", "glm", "qwen", "mimo", "hy"}:
            words.append(word.upper())
        elif lowered == "minimax":
            words.append("MiniMax")
        elif lowered == "deepseek":
            words.append("DeepSeek")
        elif lowered == "kimi":
            words.append("Kimi")
        elif lowered == "grok":
            words.append("Grok")
        else:
            words.append(word.title())
    return " ".join(words)


def parse_models(payload: Any) -> tuple[str, ...]:
    if not isinstance(payload, dict):
        raise ValueError("OpenCode Go models response must be an object")
    data = payload.get("data")
    if not isinstance(data, list):
        raise ValueError("OpenCode Go models response is missing data")
    model_ids = {
        str(item.get("id")).strip()
        for item in data
        if isinstance(item, dict) and item.get("id")
    }
    if not model_ids:
        raise ValueError("OpenCode Go returned no models")
    return tuple(sorted(model_ids))


async def fetch_model_ids(
    *, api_key: str | None = None, timeout: float = 15.0
) -> tuple[str, ...]:
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(OPENCODE_GO_MODELS_URL, headers=headers)
        response.raise_for_status()
        return parse_models(response.json())


def model_infos(
    model_ids: tuple[str, ...],
    *,
    default_model: str,
) -> tuple[ModelInfo, ...]:
    records: list[ModelInfo] = []
    for model_id in model_ids:
        metadata = (
            fallback_metadata_for(model_id, overrides=_METADATA_OVERRIDES)
            or ModelFallback()
        )
        records.append(
            ModelInfo(
                id=model_id,
                name=display_name(model_id),
                context_window=metadata.context_window,
                reasoning_efforts=metadata.reasoning_efforts,
                default_reasoning_effort=metadata.default_reasoning_effort,
                supports_vision=metadata.supports_vision,
                is_default=model_id == default_model,
            )
        )
    return tuple(records)
