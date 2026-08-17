from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ness_agent.tracing.config import PricingDict


# Sentinel for empty-model aggregation so we can start with zero-valued dicts.
_EMPTY = {
    "input_tokens": 0,
    "uncached_input_tokens": 0,
    "cached_input_tokens": 0,
    "cache_write_input_tokens": 0,
    "output_tokens": 0,
    "calls": 0,
    "cost_usd": 0.0,
}

# --- helpers ------------------------------------------------------------
def _value(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _usage_value(usage: Any, *names: str) -> int:
    for name in names:
        value = _value(usage, name)
        if value is not None:
            return int(value or 0)
    return 0


def _detail_value(usage: Any, details_key: str, *names: str) -> int:
    details = _value(usage, details_key) or {}
    for name in names:
        value = _value(details, name)
        if value is not None:
            return int(value or 0)
    return 0


def _provider_cost(metadata: dict[str, Any]) -> float | None:
    for key in ("cost", "total_cost"):
        value = metadata.get(key)
        if value is not None:
            return float(value)
    cost_details = metadata.get("cost_details") or {}
    for key in ("total", "total_cost", "cost"):
        value = cost_details.get(key)
        if value is not None:
            return float(value)
    return None


def _resolve_model_key(model_name: str, catalog: PricingDict) -> str | None:
    name = model_name.lower()
    return next((candidate for candidate in catalog if candidate in name), None)

def _model_cost_to_token_usage(model: str, mc: dict[str, int | float]) -> TokenUsage:
    input_tokens = int(mc["input_tokens"])
    output_tokens = int(mc["output_tokens"])
    cached = int(mc["cached_input_tokens"])
    calls = int(mc["calls"])
    cost_usd = float(mc["cost_usd"])
    cache_hit_rate = cached / input_tokens if input_tokens else 0.0
    return TokenUsage(
        model=model,
        input_tokens=input_tokens,
        uncached_input_tokens=int(mc["uncached_input_tokens"]),
        cached_input_tokens=cached,
        cache_write_input_tokens=int(mc["cache_write_input_tokens"]),
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        cost_usd=cost_usd or None,
        cache_hit_rate=cache_hit_rate,
        calls=calls,
    )


# --- data classes -------------------------------------------------------
@dataclass(kw_only=True, slots=True)
class TokenUsage:
    """Snapshot of one LLM call's token + cost consumption.

    Returned by :meth:`CostTracker.add` and aggregated by
    :meth:`CostTracker.for_model` / :meth:`CostTracker.total`. ``cost_usd``
    is ``None`` when neither the provider nor the pricing catalog could
    produce a figure; ``cost_source`` reports which path was used
    (``"provider"`` | ``"estimated"`` | ``None``).
    """

    model: str
    input_tokens: int = 0
    uncached_input_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float | None = None
    cost_source: str | None = None
    cache_hit_rate: float = 0.0
    calls: int = 0

    def as_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (used for persistence + events)."""
        return dataclasses.asdict(self)

    def __repr__(self) -> str:
        return (
            f"TokenUsage(model={self.model!r}, in={self.input_tokens}, "
            f"out={self.output_tokens}, cost={self.cost_usd})"
        )



# --- CostTracker --------------------------------------------------------
class CostTracker:
    """Aggregates token usage + cost across one or more models.

    Cost source priority:

    1. Provider-reported cost (``response_metadata.cost`` / ``cost_details``).
    2. ``estimate_cost`` callback when supplied.
    3. ``pricing`` dict (per-1M-token USD rates + cache-read ratio) when
       the model name matches a key as a case-insensitive substring.

    When neither path produces a value, ``cost_usd`` is ``None``.
    Per-model snapshots are available via :meth:`for_model`; :meth:`total`
    returns the aggregate across every model seen.
    """

    def __init__(
        self,
        pricing: PricingDict | None = None,
        estimate_cost: Callable[[str, int, int, int], float | None] | None = None,
        *,
        _aggregate: "CostTracker | None" = None,
        _share_pricing: bool = False,
    ) -> None:
        self.pricing: PricingDict = (
            pricing if _share_pricing and pricing is not None else dict(pricing or {})
        )
        self.estimate_cost = estimate_cost
        self._models: dict[str, dict[str, int | float]] = {}
        self._aggregate = _aggregate

    def fork_for_session(self) -> "CostTracker":
        """Return an empty tracker that also rolls new usage into this tracker.

        Pricing is intentionally shared so catalog updates are immediately
        visible to every live session. Durable replay uses :meth:`restore`,
        which bypasses the live agent aggregate.
        """
        return CostTracker(
            pricing=self.pricing,
            estimate_cost=self.estimate_cost,
            _aggregate=self,
            _share_pricing=True,
        )

    # --- ingestion ------------------------------------------------------
    def add(
        self,
        usage: Any,
        model_name: str | None = None,
        response_metadata: dict[str, Any] | None = None,
    ) -> TokenUsage | None:
        """Ingest live usage and propagate it to the agent aggregate, if any."""
        return self._add(usage, model_name, response_metadata, propagate=True)

    def restore(
        self,
        usage: Any,
        model_name: str | None = None,
        response_metadata: dict[str, Any] | None = None,
    ) -> TokenUsage | None:
        """Restore durable usage into this tracker without billing it again."""
        return self._add(usage, model_name, response_metadata, propagate=False)

    def _add(
        self,
        usage: Any,
        model_name: str | None,
        response_metadata: dict[str, Any] | None,
        *,
        propagate: bool,
    ) -> TokenUsage | None:
        if not usage:
            return None
        model = model_name or ""
        input_tokens = _usage_value(usage, "input_tokens", "prompt_tokens")
        output_tokens = _usage_value(usage, "output_tokens", "completion_tokens")
        total_tokens = _usage_value(usage, "total_tokens") or input_tokens + output_tokens
        cache_read = _detail_value(usage, "input_token_details", "cache_read", "cached_tokens")
        cache_write = _detail_value(
            usage,
            "input_token_details",
            "cache_creation",
            "cache_write_tokens",
        )
        uncached_input = max(input_tokens - cache_read, 0)

        metadata = response_metadata or {}
        provider_cost = _provider_cost(metadata)
        estimated_cost = (
            None
            if metadata.get("billing_mode") == "subscription"
            else self._estimate(model, uncached_input, cache_read, output_tokens)
        )
        cost_usd = provider_cost if provider_cost is not None else estimated_cost
        if provider_cost is not None:
            cost_source: str | None = "provider"
        elif estimated_cost is not None:
            cost_source = "estimated"
        else:
            cost_source = None

        cache_hit_rate = cache_read / input_tokens if input_tokens else 0.0
        result = TokenUsage(
            model=model,
            input_tokens=input_tokens,
            uncached_input_tokens=uncached_input,
            cached_input_tokens=cache_read,
            cache_write_input_tokens=cache_write,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
            cost_source=cost_source,
            cache_hit_rate=cache_hit_rate,
            calls=1,
        )
        self._accumulate(result)
        if propagate and self._aggregate is not None:
            self._aggregate._accumulate(result)
        return result

    def _accumulate(self, usage: TokenUsage) -> None:
        """Accumulate an already-priced usage record without re-estimation."""
        mc = self._models.setdefault(usage.model, dict(_EMPTY))
        mc["input_tokens"] += usage.input_tokens
        mc["uncached_input_tokens"] += usage.uncached_input_tokens
        mc["cached_input_tokens"] += usage.cached_input_tokens
        mc["cache_write_input_tokens"] += usage.cache_write_input_tokens
        mc["output_tokens"] += usage.output_tokens
        mc["calls"] += usage.calls
        if usage.cost_usd is not None:
            mc["cost_usd"] += usage.cost_usd

    def _estimate(self, model_name: str, uncached: int, cached: int, output: int) -> float | None:
        if self.estimate_cost is not None:
            return self.estimate_cost(model_name, uncached, cached, output)
        key = _resolve_model_key(model_name, self.pricing)
        if key is None:
            return None
        input_per_m, output_per_m, read_ratio = self.pricing[key]
        return (
            uncached * input_per_m
            + cached * input_per_m * read_ratio
            + output * output_per_m
        ) / 1_000_000

    # --- reads ----------------------------------------------------------
    def for_model(self, model: str) -> TokenUsage:
        mc = self._models.get(model)
        if mc is None:
            return TokenUsage(model=model)
        return _model_cost_to_token_usage(model, mc)

    def total(self) -> TokenUsage:
        total = dict(_EMPTY)
        for mc in self._models.values():
            total["input_tokens"] += mc["input_tokens"]
            total["uncached_input_tokens"] += mc["uncached_input_tokens"]
            total["cached_input_tokens"] += mc["cached_input_tokens"]
            total["cache_write_input_tokens"] += mc["cache_write_input_tokens"]
            total["output_tokens"] += mc["output_tokens"]
            total["calls"] += mc["calls"]
            total["cost_usd"] += mc["cost_usd"]
        return _model_cost_to_token_usage("*", total)

    def models(self) -> list[str]:
        return list(self._models.keys())

    # --- scalar conveniences (back-compat with aggregate-only callers) --
    @property
    def input_tokens(self) -> int:
        return int(sum(m["input_tokens"] for m in self._models.values()))

    @property
    def uncached_input_tokens(self) -> int:
        return int(sum(m["uncached_input_tokens"] for m in self._models.values()))

    @property
    def cached_input_tokens(self) -> int:
        return int(sum(m["cached_input_tokens"] for m in self._models.values()))

    @property
    def cache_write_input_tokens(self) -> int:
        return int(sum(m["cache_write_input_tokens"] for m in self._models.values()))

    @property
    def output_tokens(self) -> int:
        return int(sum(m["output_tokens"] for m in self._models.values()))

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def calls(self) -> int:
        return int(sum(m["calls"] for m in self._models.values()))

    @property
    def cost_usd(self) -> float:
        return sum(m["cost_usd"] for m in self._models.values())

    @property
    def cache_hit_rate(self) -> float:
        return self.cached_input_tokens / self.input_tokens if self.input_tokens else 0.0

    @property
    def total_cost_usd(self) -> float | None:
        return self.cost_usd if self.cost_usd > 0 else None

    def report(self) -> str:
        """One-shot multi-line summary across every model seen."""
        lines = [
            f"Calls: {self.calls}",
            f"Input tokens: {self.input_tokens:,}",
            f"Uncached input: {self.uncached_input_tokens:,}",
            f"Cached read: {self.cached_input_tokens:,}",
            f"Cache write: {self.cache_write_input_tokens:,}",
            f"Output tokens: {self.output_tokens:,}",
            f"Total tokens: {self.total_tokens:,}",
            f"Cache hit rate: {self.cache_hit_rate:.1%}",
            f"Cost: ${self.cost_usd:.4f}" if self.cost_usd > 0 else "Cost: unknown",
        ]
        for model in self._models:
            mu = self.for_model(model)
            lines.append(
                f"  - {model or '(unknown)'}: in={mu.input_tokens:,} "
                f"out={mu.output_tokens:,} cost=${mu.cost_usd or 0:.4f} "
                f"calls={mu.calls}"
            )
        return "\n".join(lines)
