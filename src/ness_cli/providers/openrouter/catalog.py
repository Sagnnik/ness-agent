"""OpenRouter model catalog with an offline fallback."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import requests

from ness_cli.providers.model_metadata import (
    ModelFallback,
    fallback_metadata_for,
    matching_model_family,
)

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
MODELS_DEV_URL = "https://models.dev/api.json"
OPENROUTER_CATALOG_CACHE_NAME = "openrouter-models-v2.json"
LEGACY_OPENROUTER_CATALOG_CACHE_NAMES: tuple[str, ...] = ("openrouter-models-v1.json",)
CATALOG_TTL_SECONDS = 24 * 60 * 60
CATALOG_VERSION = 2

OFFLINE_MODEL_IDS: tuple[str, ...] = (
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
    "z-ai/glm-5.3",
    "z-ai/glm-5.3-flash",
    "z-ai/glm-5.1",
    "z-ai/glm-5.2",
)

_PRICING: dict[str, tuple[float, float, float]] = {
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
    "glm-5.3-flash": (0.075, 0.25, 0.20),
    "glm-5.3": (1.40, 4.40, 0.19),
    "glm-5.2": (0.69, 2.16, 0.19),
    "glm-5.1": (0.97, 3.04, 0.19),
}

_PROVIDER_ALIASES = {
    "z-ai": "zhipuai",
    "moonshotai": "moonshotai",
    "anthropic": "anthropic",
    "openai": "openai",
    "google": "google",
    "deepseek": "deepseek",
    "mistralai": "mistral",
}


@dataclass(frozen=True, slots=True)
class ModelRecord:
    id: str
    name: str
    input_modalities: tuple[str, ...] = ("text",)
    output_modalities: tuple[str, ...] = ("text",)
    context_length: int | None = None
    input_price: float | None = None
    output_price: float | None = None
    cache_read_price: float | None = None
    supported_parameters: tuple[str, ...] = ()
    reasoning_efforts: tuple[str, ...] = ()
    default_reasoning_effort: str | None = None
    supports_anthropic_messages: bool = False

    @property
    def supports_vision(self) -> bool:
        return "image" in self.input_modalities

    @property
    def cache_read_ratio(self) -> float:
        if not self.input_price or self.cache_read_price is None:
            return 0.1
        return self.cache_read_price / self.input_price


@dataclass(frozen=True, slots=True)
class RefreshResult:
    refreshed: bool
    models: int
    error: str | None = None


def _number(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _per_million(value: Any) -> float | None:
    parsed = _number(value)
    return parsed * 1_000_000 if parsed is not None else None


def _models_dev_reasoning(
    model_id: str,
    models_dev: dict[str, Any],
) -> tuple[str, ...]:
    author, _, slug = model_id.partition("/")
    provider = models_dev.get(_PROVIDER_ALIASES.get(author, author), {})
    candidates = [
        (provider.get("models") or {}).get(slug),
        (provider.get("models") or {}).get(model_id),
        ((models_dev.get("openrouter") or {}).get("models") or {}).get(model_id),
    ]

    for entry in candidates:
        if not isinstance(entry, dict):
            continue
        for option in entry.get("reasoning_options") or []:
            if option.get("type") == "effort":
                return tuple(
                    str(value)
                    for value in option.get("values") or []
                    if value is not None
                )

    return ()


def parse_catalog(
    openrouter_payload: dict[str, Any],
    models_dev_payload: dict[str, Any] | None = None,
) -> list[ModelRecord]:
    """Normalize text, tool-capable models from provider payloads."""
    models_dev = models_dev_payload or {}
    records: list[ModelRecord] = []

    for raw in openrouter_payload.get("data") or []:
        if not isinstance(raw, dict):
            continue

        architecture = raw.get("architecture") or {}
        inputs = tuple(str(item) for item in architecture.get("input_modalities") or [])
        outputs = tuple(
            str(item) for item in architecture.get("output_modalities") or []
        )
        parameters = tuple(str(item) for item in raw.get("supported_parameters") or [])

        if "text" not in inputs or "text" not in outputs or "tools" not in parameters:
            continue

        model_id = str(raw.get("id") or "").strip()
        if not model_id:
            continue

        pricing = raw.get("pricing") or {}
        fallback = fallback_metadata_for(model_id) or ModelFallback()
        fallback_efforts = fallback.reasoning_efforts
        fallback_default = fallback.default_reasoning_effort
        native_reasoning = raw.get("reasoning") or {}
        if isinstance(native_reasoning, dict):
            native_efforts = tuple(
                str(value)
                for value in native_reasoning.get("supported_efforts") or ()
                if value is not None
            )
            native_default_value = native_reasoning.get("default_effort")
            native_default = (
                str(native_default_value) if native_default_value is not None else None
            )
        else:
            native_efforts = ()
            native_default = None
        reasoning_efforts = (
            native_efforts
            or _models_dev_reasoning(model_id, models_dev)
            or fallback_efforts
        )
        default_effort = next(
            (
                effort
                for effort in (native_default, fallback_default)
                if effort in reasoning_efforts
            ),
            reasoning_efforts[0] if reasoning_efforts else None,
        )

        records.append(
            ModelRecord(
                id=model_id,
                name=str(raw.get("name") or model_id),
                input_modalities=inputs,
                output_modalities=outputs,
                context_length=(
                    int(raw["context_length"]) if raw.get("context_length") else None
                ),
                input_price=_per_million(pricing.get("prompt")),
                output_price=_per_million(pricing.get("completion")),
                cache_read_price=_per_million(
                    pricing.get("input_cache_read") or pricing.get("cache_read")
                ),
                supported_parameters=parameters,
                reasoning_efforts=reasoning_efforts,
                default_reasoning_effort=default_effort,
                supports_anthropic_messages=model_id.startswith("anthropic/"),
            )
        )

    return sorted(records, key=lambda item: item.id.lower())


def _offline_record(model_id: str) -> ModelRecord:
    pricing_key = matching_model_family(model_id, _PRICING)
    fallback = fallback_metadata_for(model_id) or ModelFallback()
    pricing = _PRICING[pricing_key] if pricing_key is not None else (None, None, 0.1)
    input_price, output_price, cache_ratio = pricing
    input_modalities = ("text", "image") if fallback.supports_vision else ("text",)
    cache_read_price = input_price * cache_ratio if input_price is not None else None

    return ModelRecord(
        id=model_id,
        name=model_id,
        input_modalities=input_modalities,
        context_length=fallback.context_window,
        input_price=input_price,
        output_price=output_price,
        cache_read_price=cache_read_price,
        supported_parameters=("tools",),
        reasoning_efforts=fallback.reasoning_efforts,
        default_reasoning_effort=fallback.default_reasoning_effort,
        supports_anthropic_messages=model_id.startswith("anthropic/"),
    )


_OFFLINE_MODELS = tuple(_offline_record(model_id) for model_id in OFFLINE_MODEL_IDS)
_OFFLINE_BY_ID = {record.id: record for record in _OFFLINE_MODELS}


def offline_models() -> tuple[ModelRecord, ...]:
    return _OFFLINE_MODELS


def _fallback_model(model_id: str) -> ModelRecord | None:
    exact = _OFFLINE_BY_ID.get(model_id)
    if exact is not None:
        return exact

    known_variant = (
        fallback_metadata_for(model_id) is not None
        or matching_model_family(model_id, _PRICING) is not None
    )
    return _offline_record(model_id) if known_variant else None


def fetch_catalog(timeout: float = 20.0) -> list[ModelRecord]:
    """Fetch OpenRouter data and optional reasoning metadata."""
    headers = {"User-Agent": "ness_agent/0.2"}
    openrouter = requests.get(
        OPENROUTER_MODELS_URL,
        headers=headers,
        timeout=timeout,
    )
    openrouter.raise_for_status()

    models_dev_payload: dict[str, Any] = {}
    try:
        models_dev = requests.get(
            MODELS_DEV_URL,
            headers=headers,
            timeout=timeout,
        )
        models_dev.raise_for_status()
        models_dev_payload = models_dev.json()
    except (requests.RequestException, TypeError, ValueError):
        pass

    records = parse_catalog(openrouter.json(), models_dev_payload)
    if not records:
        raise ValueError("OpenRouter returned no text tool-capable models")
    return records


class OpenRouterCatalog:
    """Own cached/offline model data; reads never fetch automatically.

    The TTL gates an explicit ``refresh()`` call. ``refresh(force=True)``
    bypasses it, including refreshes requested by the provider adapter/script.
    """

    def __init__(
        self,
        cache_path: Path,
        *,
        ttl_seconds: int = CATALOG_TTL_SECONDS,
        legacy_cache_paths: Iterable[Path] = (),
    ) -> None:
        self.cache_path = Path(cache_path).expanduser().resolve()
        self.legacy_cache_paths = tuple(
            Path(path).expanduser().resolve() for path in legacy_cache_paths
        )
        self.ttl_seconds = ttl_seconds
        self._records: dict[str, ModelRecord] | None = None
        self._fetched_at = 0.0
        self._refresh_task: asyncio.Task[RefreshResult] | None = None

    @classmethod
    def from_cache_dir(cls, cache_dir: Path) -> OpenRouterCatalog:
        """Use the shared runtime cache and its supported legacy files."""
        root = Path(cache_dir)
        return cls(
            root / OPENROUTER_CATALOG_CACHE_NAME,
            legacy_cache_paths=tuple(
                root / name for name in LEGACY_OPENROUTER_CATALOG_CACHE_NAMES
            ),
        )

    def _load_cache(self) -> None:
        if self._records is not None:
            return

        self._records = {}
        candidates = (
            (self.cache_path, CATALOG_VERSION),
            *((path, 1) for path in self.legacy_cache_paths),
        )
        for path, expected_version in candidates:
            if not path.is_file():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("version") != expected_version:
                    continue

                records: dict[str, ModelRecord] = {}
                for value in payload.get("models") or []:
                    raw = dict(value)
                    for key in (
                        "input_modalities",
                        "output_modalities",
                        "supported_parameters",
                        "reasoning_efforts",
                    ):
                        raw[key] = tuple(raw.get(key) or ())
                    record = ModelRecord(**raw)
                    records[record.id] = record
                if records:
                    self._records = records
                    self._fetched_at = float(payload.get("fetched_at") or 0)
                    return
            except (OSError, ValueError, TypeError):
                continue

    def cached_models(self) -> tuple[ModelRecord, ...]:
        self._load_cache()
        return tuple((self._records or {}).values())

    def models(self) -> tuple[ModelRecord, ...]:
        return self.cached_models() or offline_models()

    def model_record(self, model_id: str) -> ModelRecord | None:
        self._load_cache()
        return (self._records or {}).get(model_id) or _fallback_model(model_id)

    def is_stale(self, *, now: float | None = None) -> bool:
        self._load_cache()
        current = time.time() if now is None else now
        return not self._records or current - self._fetched_at >= self.ttl_seconds

    def _write_cache(self, records: Iterable[ModelRecord]) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": CATALOG_VERSION,
            "fetched_at": time.time(),
            "models": [asdict(record) for record in records],
        }
        fd, temporary_name = tempfile.mkstemp(
            dir=self.cache_path.parent,
            prefix=f"{self.cache_path.name}.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)

        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.cache_path)
        except BaseException:
            try:
                temporary.unlink()
            except OSError:
                pass
            raise

    async def refresh(self, *, force: bool = False) -> RefreshResult:
        if not force and not self.is_stale():
            return RefreshResult(False, len(self.cached_models()))

        if self._refresh_task is not None and not self._refresh_task.done():
            return await self._refresh_task

        async def run() -> RefreshResult:
            try:
                records = await asyncio.to_thread(fetch_catalog)
                self._write_cache(records)
                self._records = {record.id: record for record in records}
                self._fetched_at = time.time()
                return RefreshResult(True, len(records))
            except Exception as error:
                return RefreshResult(
                    False,
                    len(self.cached_models()),
                    str(error),
                )

        self._refresh_task = asyncio.create_task(run())
        try:
            return await self._refresh_task
        finally:
            self._refresh_task = None

    def reset(self) -> None:
        self._records = None
        self._fetched_at = 0.0
        self._refresh_task = None
