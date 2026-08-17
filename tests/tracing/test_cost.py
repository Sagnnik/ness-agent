"""Tests for ness_agent.tracing.cost (CostTracker + TokenUsage)."""

from __future__ import annotations

import pytest

from ness_agent.tracing.cost import CostTracker, TokenUsage


def _fake_usage(input_tokens=100, output_tokens=20, cache_read=0, cache_write=0):
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "input_token_details": {
            "cache_read": cache_read,
            "cache_creation": cache_write,
        },
    }


def test_cost_tracker_pricing_dict_calculates_cost():
    tracker = CostTracker(pricing={"gpt-4o": (2.50, 10.00, 0.50)})
    usage = tracker.add(_fake_usage(input_tokens=1_000, output_tokens=50), model_name="gpt-4o")
    assert usage is not None
    assert usage.cost_usd is not None
    # 1000 input * 2.50 + 50 output * 10.00 = 2500 + 500 = 3000 / 1M = 0.003
    assert usage.cost_usd == pytest.approx(0.003)


def test_cost_tracker_pricing_dict_applies_cache_read_ratio():
    tracker = CostTracker(pricing={"gpt-4o": (2.50, 10.00, 0.50)})
    usage = tracker.add(
        _fake_usage(input_tokens=1_000, output_tokens=0, cache_read=200),
        model_name="gpt-4o",
    )
    assert usage is not None
    # 800 uncached * 2.50 + 200 cached * 2.50 * 0.50 + 0 = 2000 + 250 = 2250 / 1M
    assert usage.cost_usd == pytest.approx(0.00225)
    assert usage.cache_hit_rate == 0.2


def test_cost_tracker_records_cache_writes_without_changing_input_accounting():
    tracker = CostTracker()
    usage = tracker.add(
        _fake_usage(input_tokens=1_000, cache_read=200, cache_write=750),
        model_name="gpt-test",
    )

    assert usage is not None
    assert usage.cache_write_input_tokens == 750
    assert usage.uncached_input_tokens == 800
    assert tracker.cache_write_input_tokens == 750


def test_cost_tracker_per_model_breakdown():
    tracker = CostTracker(pricing={"gpt-4o": (2.50, 10.00, 0.50), "gpt-4o-mini": (0.15, 0.60, 0.50)})
    tracker.add(_fake_usage(input_tokens=1_000, output_tokens=10), model_name="gpt-4o")
    tracker.add(_fake_usage(input_tokens=2_000, output_tokens=5), model_name="gpt-4o-mini")
    tracker.add(_fake_usage(input_tokens=500, output_tokens=3), model_name="gpt-4o")
    assert set(tracker.models()) == {"gpt-4o", "gpt-4o-mini"}
    gpt4o = tracker.for_model("gpt-4o")
    assert gpt4o.input_tokens == 1_500
    assert gpt4o.output_tokens == 13
    assert gpt4o.calls == 2
    mini = tracker.for_model("gpt-4o-mini")
    assert mini.input_tokens == 2_000
    assert mini.calls == 1
    total = tracker.total()
    assert total.input_tokens == 3_500
    assert total.output_tokens == 18
    assert total.calls == 3


def test_cost_tracker_unknown_model_returns_none_cost():
    tracker = CostTracker(pricing={"gpt-4o": (2.50, 10.00, 0.50)})
    usage = tracker.add(_fake_usage(), model_name="totally-unknown-model")
    assert usage is not None
    assert usage.cost_usd is None
    assert usage.cost_source is None


def test_cost_tracker_estimate_cost_callback_takes_priority_over_pricing():
    """When BOTH pricing and estimate_cost are provided, callback wins."""
    tracker = CostTracker(
        pricing={"gpt-4o": (2.50, 10.00, 0.50)},
        estimate_cost=lambda model, u, c, o: 9.99,
    )
    usage = tracker.add(_fake_usage(), model_name="gpt-4o")
    assert usage is not None
    assert usage.cost_usd == 9.99
    assert usage.cost_source == "estimated"


def test_cost_tracker_provider_cost_beats_estimate_and_pricing():
    tracker = CostTracker(
        pricing={"gpt-4o": (2.50, 10.00, 0.50)},
        estimate_cost=lambda model, u, c, o: 9.99,
    )
    usage = tracker.add(
        _fake_usage(),
        model_name="gpt-4o",
        response_metadata={"cost": 0.01},
    )
    assert usage is not None
    assert usage.cost_usd == 0.01
    assert usage.cost_source == "provider"


def test_cost_tracker_handles_none_usage():
    tracker = CostTracker(pricing={"gpt-4o": (2.50, 10.00, 0.50)})
    assert tracker.add(None, model_name="gpt-4o") is None
    assert tracker.add({}, model_name="gpt-4o") is None  # falsy usage object
