from __future__ import annotations

from types import SimpleNamespace


def _text(fragments: list[tuple[str, str]]) -> str:
    return "".join(text for _, text in fragments)


def test_stats_line_labels_subscription_instead_of_zero_cost(make_app, monkeypatch):
    app = make_app()
    app.coding.cost_tracker.input_tokens = 5_202_125
    app.coding.cost_tracker.output_tokens = 25_499
    monkeypatch.setattr(
        "ness_cli.tui.chrome.get_provider",
        lambda _provider_id: SimpleNamespace(billing_label="subscription"),
    )

    text = _text(app._stats_line())

    assert "subscription" in text
    assert "$0.0000" not in text


def test_stats_line_keeps_dollar_cost_for_api_provider(make_app, monkeypatch):
    app = make_app()
    app.coding.cost_tracker.cost_usd = 1.23456
    monkeypatch.setattr(
        "ness_cli.tui.chrome.get_provider",
        lambda _provider_id: SimpleNamespace(billing_label="usage-based"),
    )

    assert "$1.2346" in _text(app._stats_line())


def test_stats_line_cache_tracks_billing_mode(make_app, monkeypatch):
    app = make_app()
    provider = SimpleNamespace(billing_label="usage-based")
    monkeypatch.setattr("ness_cli.tui.chrome.get_provider", lambda _provider_id: provider)

    assert "$0.0000" in _text(app._stats_line())

    provider.billing_label = "subscription"

    assert "subscription" in _text(app._stats_line())
