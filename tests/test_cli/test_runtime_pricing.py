from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from ness_agent import CostTracker

from ness_cli import runtime
from ness_cli.config import ConfigManager, ConfigPatch
from ness_cli.providers import ModelInfo
from ness_cli.providers.registry import ModelRequest, ProviderRegistry
from tests.test_cli.fakes.providers import FakeProvider


class PricedChatModel(FakeMessagesListChatModel):
    model_name: str

    def bind_tools(self, tools, **kwargs):
        del tools, kwargs
        return self


class PricedProvider(FakeProvider):
    def __init__(self, config):
        super().__init__("openrouter", config)
        self.infos = {
            name: ModelInfo(
                id=name,
                name=name,
                input_price=rate,
                output_price=rate * 2,
                cache_read_ratio=0.25,
            )
            for name, rate in (
                ("model-a", 1),
                ("model-b", 4),
                ("reflect-b", 2),
                ("judge", 3),
            )
        }

    def build_chat_model(self, *args, **kwargs):
        super().build_chat_model(*args, **kwargs)
        return PricedChatModel(
            model_name=kwargs["model_name"],
            responses=[
                AIMessage(
                    content="ok",
                    usage_metadata={
                        "input_tokens": 1000,
                        "output_tokens": 100,
                        "total_tokens": 1100,
                        "input_token_details": {"cache_read": 200},
                    },
                ),
            ],
        )

    def model_info(self, name):
        return self.infos.get(name)

    async def models(self, *, refresh=False):
        del refresh
        return tuple(self.infos.values())


@pytest.fixture
def priced_runtime(isolated_cli_env):
    manager = ConfigManager.load(isolated_cli_env.paths, environment={})
    manager.apply(
        ConfigPatch({"model_name": "model-a", "reflection_model_name": "model-a"})
    )
    providers = ProviderRegistry(paths=isolated_cli_env.paths, config=manager)
    provider = PricedProvider(manager.provider_runtime())
    providers.register("openrouter", lambda _config: provider)
    agent = runtime._build_agent(
        paths=isolated_cli_env.paths,
        config=manager,
        providers=providers,
        options=runtime.InteractiveOptions(),
    )
    sessions = runtime._SessionFactory(
        paths=isolated_cli_env.paths,
        config=manager,
        providers=providers,
        agent=agent,
    )
    facade = runtime.InteractiveRuntime(
        SimpleNamespace(
            config=manager, providers=providers, agent=agent, sessions=sessions
        ),
        resume_thread_id=None,
    )
    return facade, sessions, agent.config.cost_tracker, provider


def usage():
    return {
        "input_tokens": 1000,
        "output_tokens": 100,
        "input_token_details": {"cache_read": 200},
    }


def test_model_switch_refreshes_main_and_reflection_rates_without_resetting_usage(
    priced_runtime,
):
    facade, sessions, aggregate, _provider = priced_runtime

    async def exercise():
        session = await sessions.new(thread_id="first")
        tracker = session._session.cost_tracker
        first = tracker.add(usage(), "model-a")
        await facade.apply_config(
            ConfigPatch(
                {
                    "model_name": "model-b",
                    "reflection_model_name": "reflect-b",
                }
            ),
            selected=session,
        )
        assert tracker.pricing is aggregate.pricing
        assert tracker.calls == 1
        assert tracker.cost_usd == first.cost_usd
        main = tracker.add(usage(), "model-b")
        reflection = tracker.add(usage(), "reflect-b")
        assert main.cost_usd == pytest.approx(0.0042)
        assert reflection.cost_usd == pytest.approx(0.0021)
        assert main.cost_source == reflection.cost_source == "estimated"
        assert tracker.calls == aggregate.calls == 3
        assert tracker.cost_usd == aggregate.cost_usd == pytest.approx(0.00735)

    asyncio.run(exercise())


def test_new_session_populates_new_model_rates_without_reloading_existing_session(
    priced_runtime,
):
    facade, sessions, aggregate, _provider = priced_runtime

    async def exercise():
        first = await sessions.new(thread_id="first")
        await facade.apply_config(ConfigPatch({"model_name": "model-b"}))
        second = await sessions.new(thread_id="second")
        result = second._session.cost_tracker.add(usage(), "model-b")
        assert result.cost_usd == pytest.approx(0.0042)
        assert first.runtime_config.model.model_name == "model-a"
        assert first._session.cost_tracker.pricing is aggregate.pricing

    asyncio.run(exercise())


def test_goal_judge_creation_populates_its_fallback_price(priced_runtime):
    facade, _sessions, aggregate, _provider = priced_runtime
    asyncio.run(facade.apply_config(ConfigPatch({"goal_judge_model": "judge"})))

    facade.goal_judge_model("thread")
    result = aggregate.add(usage(), "judge")

    assert result.cost_usd == pytest.approx(0.00315)
    assert result.cost_source == "estimated"


def test_catalog_refresh_updates_existing_session_rates_without_repricing_history(
    priced_runtime,
):
    facade, sessions, aggregate, provider = priced_runtime

    async def exercise():
        first = await sessions.new(thread_id="first")
        sibling = await sessions.new(thread_id="sibling")
        tracker = first._session.cost_tracker
        before = tracker.add(usage(), "model-a")
        provider.infos["model-a"] = replace(
            provider.infos["model-a"],
            input_price=10,
            output_price=20,
            cache_read_ratio=0.5,
        )
        await facade.providers.models(refresh=True)
        assert (
            tracker.pricing
            is sibling._session.cost_tracker.pricing
            is aggregate.pricing
        )
        assert tracker.cost_usd == before.cost_usd
        assert tracker.calls == aggregate.calls == 1
        after = sibling._session.cost_tracker.add(usage(), "model-a")
        assert after.cost_usd == pytest.approx(0.011)
        assert aggregate.cost_usd == pytest.approx(0.01205)
        assert tracker.cost_usd == before.cost_usd

    asyncio.run(exercise())


@pytest.mark.parametrize("missing", ["input", "output", "model"])
def test_catalog_refresh_removes_unavailable_rates(priced_runtime, missing):
    facade, _sessions, aggregate, provider = priced_runtime
    if missing == "model":
        provider.infos.pop("model-a")
    else:
        provider.infos["model-a"] = replace(
            provider.infos["model-a"], **{f"{missing}_price": None}
        )

    asyncio.run(facade.providers.models(refresh=True))
    result = aggregate.add(usage(), "model-a")

    assert "model-a" not in aggregate.pricing
    assert result.cost_usd is None
    assert result.cost_source is None


def test_catalog_refresh_updates_alias_rates_for_live_models(
    priced_runtime, monkeypatch
):
    facade, _sessions, aggregate, provider = priced_runtime
    original_info = provider.model_info
    monkeypatch.setattr(
        provider,
        "model_info",
        lambda name: original_info("model-a" if name == "alias" else name),
    )
    facade.providers.create_model(ModelRequest("thread", "main", "alias"))
    provider.infos["model-a"] = replace(
        provider.infos["model-a"], input_price=10, output_price=20
    )

    asyncio.run(facade.providers.models(refresh=True))
    result = aggregate.add(usage(), "alias")

    assert aggregate.pricing["alias"] == aggregate.pricing["model-a"]
    assert result.cost_usd == pytest.approx(0.0105)


def test_failed_catalog_refresh_preserves_previous_rates_and_usage(
    priced_runtime, monkeypatch
):
    facade, _sessions, aggregate, provider = priced_runtime
    aggregate.add(usage(), "model-a")
    previous_pricing = dict(aggregate.pricing)
    previous_usage = aggregate.total()

    async def fail(*, refresh):
        assert refresh
        raise RuntimeError("catalog unavailable")

    monkeypatch.setattr(provider, "models", fail)
    with pytest.raises(RuntimeError, match="catalog unavailable"):
        asyncio.run(facade.providers.models(refresh=True))
    assert aggregate.pricing == previous_pricing
    assert aggregate.total() == previous_usage


@pytest.mark.parametrize(
    "metadata,expected,source",
    [
        ({"cost": 0.25}, 0.25, "provider"),
        ({"cost": 0}, 0, "provider"),
        ({"billing_mode": "subscription"}, None, None),
        ({"billing_mode": "subscription", "cost": 0.125}, 0.125, "provider"),
    ],
)
def test_refreshed_fallback_rates_preserve_reported_cost_and_subscription_rules(
    priced_runtime, metadata, expected, source
):
    facade, _sessions, aggregate, _provider = priced_runtime
    asyncio.run(facade.providers.models(refresh=True))

    result = aggregate.add(usage(), "model-a", metadata)

    assert result.cost_usd == expected
    assert result.cost_source == source


def test_subscription_catalog_does_not_overwrite_api_rates(priced_runtime):
    facade, _sessions, aggregate, provider = priced_runtime
    subscription = PricedProvider(provider.runtime_config)
    subscription.id = "codex"
    subscription.billing_label = "subscription"
    subscription.infos["model-a"] = replace(
        subscription.infos["model-a"], input_price=100, output_price=200
    )
    facade.providers.register("codex", lambda _config: subscription)
    original = dict(aggregate.pricing)

    facade.providers.create_model(
        ModelRequest("thread", "main", "model-a", provider_id="codex")
    )
    asyncio.run(facade.providers.models(provider_id="codex", refresh=True))

    assert aggregate.pricing == original
    assert aggregate.add(usage(), "model-a").cost_usd == pytest.approx(0.00105)
    assert (
        aggregate.add(usage(), "model-a", {"billing_mode": "subscription"}).cost_usd
        is None
    )


def test_model_picker_refresh_updates_live_prices(priced_runtime, monkeypatch):
    from tests.test_cli.test_config_flow import FakeContext, FakeRenderer, FakeUI, _spec
    from ness_cli.tui.input.config import _edit

    facade, _sessions, aggregate, provider = priced_runtime
    original_models = provider.models

    async def models(*, refresh=False):
        if refresh:
            provider.infos["model-a"] = replace(
                provider.infos["model-a"], input_price=10, output_price=20
            )
        return await original_models(refresh=refresh)

    class RefreshUI(FakeUI):
        async def choose(self, _title, _items, **kwargs):
            await kwargs["refresh"]()
            return None

    monkeypatch.setattr(provider, "models", models)
    context = FakeContext(facade, RefreshUI(), FakeRenderer())
    asyncio.run(_edit(context, _spec("model_name")))

    assert aggregate.add(usage(), "model-a").cost_usd == pytest.approx(0.0105)
    assert context.renderer.errors == []


def test_streamed_turns_use_new_model_rates_after_switch(priced_runtime):
    facade, sessions, aggregate, _provider = priced_runtime

    async def exercise():
        session = await sessions.new(thread_id="streamed")
        first = [event async for event in session.stream("first")]
        await facade.apply_config(
            ConfigPatch({"model_name": "model-b"}), selected=session
        )
        second = [event async for event in session.stream("second")]
        assert not [event for event in first + second if event.kind == "error"]
        assert session._session.cost_tracker.for_model(
            "model-a"
        ).cost_usd == pytest.approx(0.00105)
        assert session._session.cost_tracker.for_model(
            "model-b"
        ).cost_usd == pytest.approx(0.0042)
        assert aggregate.calls == 2
        assert aggregate.cost_usd == pytest.approx(0.00525)

    asyncio.run(exercise())


def test_runtime_pricing_tables_are_isolated(priced_runtime, isolated_cli_env):
    facade, _sessions, aggregate, _provider = priced_runtime
    manager = ConfigManager.load(isolated_cli_env.paths, environment={})
    separate = ProviderRegistry(paths=isolated_cli_env.paths, config=manager)
    provider = PricedProvider(manager.provider_runtime())
    provider.infos["model-a"] = replace(
        provider.infos["model-a"], input_price=10, output_price=20
    )
    separate.register("openrouter", lambda _config: provider)
    separate_tracker = CostTracker()
    separate.bind_pricing(separate_tracker.pricing)

    asyncio.run(separate.models(refresh=True))

    assert aggregate.add(usage(), "model-a").cost_usd == pytest.approx(0.00105)
    assert separate_tracker.add(usage(), "model-a").cost_usd == pytest.approx(0.0105)
    assert facade.providers is not separate


def test_free_catalog_prices_are_recorded_as_zero_estimates(priced_runtime):
    facade, _sessions, aggregate, provider = priced_runtime
    provider.infos["model-a"] = replace(
        provider.infos["model-a"], input_price=0, output_price=0
    )

    asyncio.run(facade.providers.models(refresh=True))
    result = aggregate.add(usage(), "model-a")

    assert result.cost_usd == 0
    assert result.cost_source == "estimated"
