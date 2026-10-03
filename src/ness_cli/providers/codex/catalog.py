from __future__ import annotations

from ness_cli.providers.base import ModelInfo
from ness_cli.providers.codex.app_server import CodexAppServer
from ness_cli.providers.model_metadata import ModelFallback, matching_model_family


# Subscription defaults, not API limits. Captured from the upstream Codex
# bundled catalog on 2026-10-01; app-server metadata always takes precedence.
# https://github.com/openai/codex/blob/main/codex-rs/models-manager/models.json
_FALLBACK_CAPABILITIES: dict[str, ModelFallback] = {
    "gpt-5.6-luna": ModelFallback(
        context_window=272_000,
        reasoning_efforts=("low", "medium", "high", "xhigh", "max"),
        default_reasoning_effort="medium",
    ),
    "gpt-5.6-sol": ModelFallback(
        context_window=272_000,
        reasoning_efforts=("low", "medium", "high", "xhigh", "max", "ultra"),
        default_reasoning_effort="low",
    ),
    "gpt-5.6-terra": ModelFallback(
        context_window=272_000,
        reasoning_efforts=("low", "medium", "high", "xhigh", "max", "ultra"),
        default_reasoning_effort="medium",
    ),
    "gpt-5.5": ModelFallback(
        context_window=272_000,
        reasoning_efforts=("low", "medium", "high", "xhigh"),
        default_reasoning_effort="medium",
    ),
}


def fallback_model_info(model_id: str) -> ModelInfo:
    """Keep cold startup usable without guessing API capabilities for Codex."""
    lookup = "gpt-5.6-sol" if model_id == "gpt-5.6" else model_id
    family = matching_model_family(lookup, _FALLBACK_CAPABILITIES)
    metadata = _FALLBACK_CAPABILITIES.get(family, ModelFallback())
    return ModelInfo(
        id=model_id,
        name=model_id,
        context_window=metadata.context_window,
        reasoning_efforts=metadata.reasoning_efforts,
        default_reasoning_effort=metadata.default_reasoning_effort,
        supports_vision=metadata.supports_vision,
    )


def _context_window(item: dict, model_id: str) -> int | None:
    for key in (
        "contextWindow",
        "contextWindowTokens",
        "contextLength",
        "context_length",
    ):
        value = item.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return fallback_model_info(model_id).context_window


async def load_models(server: CodexAppServer) -> tuple[ModelInfo, ...]:
    records: list[ModelInfo] = []
    cursor: str | None = None
    while True:
        params = {"limit": 100}
        if cursor:
            params["cursor"] = cursor
        response = await server.request("model/list", params)
        for item in response.get("data") or []:
            if not isinstance(item, dict) or item.get("hidden"):
                continue
            efforts = tuple(
                str(option.get("reasoningEffort"))
                for option in item.get("supportedReasoningEfforts") or []
                if isinstance(option, dict) and option.get("reasoningEffort")
            )
            model_id = str(item.get("model") or item.get("id") or "")
            records.append(
                ModelInfo(
                    id=model_id,
                    name=str(item.get("displayName") or item.get("model") or ""),
                    context_window=_context_window(item, model_id),
                    default_reasoning_effort=item.get("defaultReasoningEffort"),
                    reasoning_efforts=efforts,
                    supports_vision="image" in (item.get("inputModalities") or []),
                    is_default=bool(item.get("isDefault")),
                )
            )
        cursor = response.get("nextCursor")
        if not cursor:
            return tuple(records)
