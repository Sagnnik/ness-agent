import asyncio
import builtins
import sys
from importlib.metadata import PackageNotFoundError
from types import ModuleType, SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from evals.codex import codex_chat_model as snapshot
from ness_cli.providers.codex.responses import CodexSubscriptionChatModel


class FakeAuth:
    pass


class FakeTransport:
    def __init__(self, auth, *, max_retries):
        self.auth = auth
        self.max_retries = max_retries
        self.response = {}
        self.payloads = []

    async def create(self, payload):
        self.payloads.append(payload)
        return dict(self.response)


@pytest.fixture
def model(monkeypatch):
    monkeypatch.setattr(
        snapshot, "eval_provider_types", lambda: (FakeAuth, FakeTransport)
    )
    return snapshot.CodexChatModel(
        model="gpt-5.6-luna",
        reasoning_effort="high",
        prompt_cache_key="eval-thread",
        max_retries=2,
    )


@pytest.mark.parametrize("installed", ["0.2.3", "0.2.5", "0.2.4.dev1"])
def test_snapshot_rejects_other_package_versions_before_construction(
    monkeypatch, installed
):
    monkeypatch.setattr(snapshot, "version", lambda name: installed)
    with pytest.raises(
        RuntimeError, match=f"requires ness-agent==0.2.4; found {installed}"
    ):
        snapshot.CodexChatModel(model="gpt-5.6-luna")


def test_snapshot_reports_missing_package(monkeypatch):
    def missing(name):
        raise PackageNotFoundError(name)

    monkeypatch.setattr(snapshot, "version", missing)
    with pytest.raises(RuntimeError, match="package is not installed"):
        snapshot.eval_provider_types()


def test_snapshot_rejects_same_version_with_ported_cli_layout(monkeypatch):
    monkeypatch.setattr(snapshot, "version", lambda name: "0.2.4")
    with pytest.raises(RuntimeError, match="incompatible CLI layout"):
        snapshot.eval_provider_types()


def test_snapshot_loads_only_the_pinned_legacy_provider(monkeypatch):
    monkeypatch.setattr(snapshot, "version", lambda name: "0.2.4")
    for name in ("ness_cli.provider", "ness_cli.provider.codex"):
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    auth = ModuleType("ness_cli.provider.codex.auth")
    auth.CodexAuth = FakeAuth
    transport = ModuleType("ness_cli.provider.codex.transport")
    transport.CodexResponsesTransport = FakeTransport
    monkeypatch.setitem(sys.modules, auth.__name__, auth)
    monkeypatch.setitem(sys.modules, transport.__name__, transport)
    assert snapshot.eval_provider_types() == (FakeAuth, FakeTransport)
    supplied_auth = FakeAuth()
    model = snapshot.CodexChatModel(
        model="gpt-5.6-luna", auth=supplied_auth, max_retries=2
    )
    assert model._auth is supplied_auth and model._transport.auth is supplied_auth
    assert model._transport.max_retries == 2


def test_missing_provider_dependency_keeps_its_original_error(monkeypatch):
    original = builtins.__import__
    monkeypatch.setattr(snapshot, "version", lambda name: "0.2.4")

    def import_module(name, *args, **kwargs):
        if name == "ness_cli.provider.codex.auth":
            raise ModuleNotFoundError(
                "dependency unavailable", name="provider_dependency"
            )
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_module)
    with pytest.raises(ModuleNotFoundError) as error:
        snapshot.eval_provider_types()
    assert error.value.name == "provider_dependency"


def test_frozen_conversation_conversion_preserves_tools_images_and_options(model):
    messages = [
        SystemMessage(content="first"),
        SystemMessage(content="second"),
        HumanMessage(
            content=[
                {"type": "text", "text": "inspect"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AA==", "detail": "low"},
                },
            ]
        ),
        AIMessage(
            content="calling",
            tool_calls=[{"id": "call-1", "name": "echo", "args": {"text": "hello"}}],
        ),
        ToolMessage(content="tool result", tool_call_id="call-1"),
    ]
    tool = {
        "type": "function",
        "function": {
            "name": "echo",
            "description": "Echo text",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
            },
        },
    }
    bound = model.bind_tools([tool], tool_choice="any")
    payload = model._payload(messages, **bound.kwargs, max_tokens=100)
    assert payload == {
        "model": "gpt-5.6-luna",
        "instructions": "first\n\nsecond",
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "inspect"},
                    {
                        "type": "input_image",
                        "image_url": "data:image/png;base64,AA==",
                        "detail": "low",
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "output_text", "text": "calling"}],
            },
            {
                "type": "function_call",
                "call_id": "call-1",
                "name": "echo",
                "arguments": '{"text":"hello"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call-1",
                "output": "tool result",
            },
        ],
        "store": False,
        "parallel_tool_calls": True,
        "prompt_cache_key": "eval-thread",
        "tools": [
            {
                "type": "function",
                "name": "echo",
                "description": "Echo text",
                "parameters": tool["function"]["parameters"],
                "strict": False,
            }
        ],
        "tool_choice": "required",
        "reasoning": {"effort": "high", "summary": "auto"},
        "max_output_tokens": 100,
    }


def test_frozen_registry_binding_captures_tools_without_mutating_source(model):
    tool = {
        "type": "function",
        "function": {
            "name": "echo",
            "description": "Echo",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    registry = SimpleNamespace(active_tools=[tool])
    bound = model.bind_tool_registry(registry)
    registry.active_tools.clear()
    assert bound is not model and bound._transport is model._transport
    assert bound._payload([])["tools"][0]["name"] == "echo"
    assert "tools" not in model._payload([])


def test_frozen_output_replay_retains_reasoning_and_function_call_items(model):
    output = [
        {"type": "reasoning", "id": "reasoning-1", "summary": []},
        {
            "type": "function_call",
            "id": "function-1",
            "call_id": "call-1",
            "name": "echo",
            "arguments": '{"text":"hello"}',
        },
    ]
    message = snapshot.CodexChatModel._message(
        {"model": "gpt-5.6-luna", "output": output}
    )
    assert message.tool_calls[0]["args"] == {"text": "hello"}
    assert model._payload([message])["input"] == output


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("gpt-5.6-luna", 0.0000289),
        ("gpt-5.6-terra", 0.000289),
        ("gpt-5.6-sol", 0.000538),
        ("gpt-5.6", 0.000538),
        ("gpt-5.6-sol-snapshot", 0.000538),
    ],
)
def test_frozen_api_cost_rates_include_reads_and_writes(model, expected):
    usage = {
        "input_tokens": 100,
        "output_tokens": 10,
        "input_tokens_details": {"cached_tokens": 20, "cache_write_tokens": 10},
    }
    assert snapshot._estimate_api_cost(model, usage) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("input_tokens", "expected"),
    [(272_000, 1.088), (272_001, 2.176008)],
)
def test_frozen_long_context_cost_boundary(input_tokens, expected):
    assert snapshot._estimate_api_cost(
        "gpt-5.6-sol", {"input_tokens": input_tokens}
    ) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("details", "expected"),
    [
        ({"cached_tokens": 1_000}, 0.000002),
        ({"cache_write_tokens": 1_000}, 0.000025),
        ({"cached_tokens": -1, "cache_write_tokens": -1}, 0.00002),
        ({"cache_creation": 40}, 0.000022),
    ],
)
def test_frozen_api_cost_clamps_cache_counts_and_supports_creation_alias(
    details, expected
):
    assert snapshot._estimate_api_cost(
        "gpt-5.6-luna", {"input_tokens": 100, "input_tokens_details": details}
    ) == pytest.approx(expected)


def test_eval_estimates_cost_while_cli_keeps_subscription_billing():
    response = {
        "model": "gpt-5.6-luna",
        "output_text": "answer",
        "id": "response-1",
        "usage": {
            "input_tokens": 100,
            "output_tokens": 10,
            "input_tokens_details": {"cached_tokens": 20, "cache_write_tokens": 10},
        },
        "_cache_diagnostics": {"cache": "hit"},
    }
    evaluated = snapshot.CodexChatModel._message(response)
    cli = CodexSubscriptionChatModel._message(response)
    assert evaluated.content == "answer"
    assert evaluated.response_metadata["cost"] == pytest.approx(0.0000289)
    assert evaluated.response_metadata["cost_source"] == "estimated"
    assert evaluated.response_metadata["cost_basis"] == "openai-api-standard"
    assert (
        evaluated.response_metadata["billing_mode"]
        == cli.response_metadata["billing_mode"]
        == "subscription"
    )
    assert "cost" not in cli.response_metadata
    assert evaluated.usage_metadata["input_token_details"] == {
        "cache_read": 20,
        "cache_creation": 10,
    }
    assert evaluated.response_metadata["cache_diagnostics"] == {"cache": "hit"}


def test_generation_fills_missing_model_for_cost_estimation(model):
    model._transport.response = {
        "output_text": "answer",
        "usage": {"input_tokens": 100},
    }
    result = asyncio.run(model._agenerate([HumanMessage(content="hello")]))
    assert result.generations[0].message.response_metadata["cost"] == pytest.approx(
        0.00002
    )
    assert model._transport.payloads[0]["input"][0]["content"][0]["text"] == "hello"


@pytest.mark.parametrize("name", ["gpt-5.60", "gpt-5.6-unknown", "unknown"])
def test_unknown_models_keep_unknown_eval_cost_and_context(name):
    assert snapshot.context_window_for_model(name) is None
    assert snapshot._estimate_api_cost(name, {"input_tokens": 100}) is None
    assert (
        "cost"
        not in snapshot.CodexChatModel._message({"model": name}).response_metadata
    )


@pytest.mark.parametrize(
    "name", ["gpt-5.6", "gpt-5.6-sol", "gpt-5.6-luna", "gpt-5.6-terra-snapshot"]
)
def test_eval_context_windows_remain_pinned(name):
    assert snapshot.context_window_for_model(name) == 272_000
