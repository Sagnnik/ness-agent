from __future__ import annotations

import asyncio
import base64
import json
import time
import uuid
from types import SimpleNamespace

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from ness_agent.compaction import summarize
from ness_cli.config import ProviderRuntimeConfig
from ness_cli.providers.codex.adapter import CodexProviderAdapter
from ness_cli.providers.codex.app_server import CodexAppServer, CodexUnavailable
from ness_cli.providers.codex.auth import CodexAuth, CodexCredentials, _jwt_expiry
from ness_cli.providers.codex.catalog import load_models
from ness_cli.providers.codex.responses import CodexSubscriptionChatModel
from ness_cli.providers.codex.transport import (
    CodexResponsesTransport,
    _stream_error,
    merge_streamed_response,
)


def _runtime() -> ProviderRuntimeConfig:
    return ProviderRuntimeConfig(
        provider_id="codex",
        api_key=None,
        base_url=None,
        max_retries=2,
        cache_ttl=None,
        anthropic_messages=False,
        session_id=None,
    )


def _jwt(expiry: int) -> str:
    payload = base64.urlsafe_b64encode(json.dumps({"exp": expiry}).encode()).decode().rstrip("=")
    return f"header.{payload}.signature"


@tool
def read_file(path: str) -> str:
    """Read a file."""
    return path


def test_auth_reads_only_the_isolated_codex_home(isolated_cli_env) -> None:
    home = isolated_cli_env.config / "codex"
    home.mkdir(parents=True)
    expiry = int(time.time()) + 3_600
    (home / "auth.json").write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "tokens": {
                    "access_token": _jwt(expiry),
                    "account_id": "acct-test",
                    "refresh_token": "secret",
                },
            }
        ),
        encoding="utf-8",
    )

    auth = CodexAuth()
    credentials = auth.credentials()

    assert auth.home == home
    assert credentials == CodexCredentials(_jwt(expiry), "acct-test", expiry)
    assert _jwt_expiry("invalid") is None


def test_expired_credentials_refresh_through_the_owned_server(
    isolated_cli_env,
) -> None:
    calls: list[tuple[str, dict]] = []

    class Server:
        async def start(self):
            calls.append(("start", {}))

        async def request(self, method, params):
            calls.append((method, params))
            return {"account": {"type": "chatgpt"}}

    auth = CodexAuth(Server())
    auth.home.mkdir(parents=True)
    auth.auth_path.write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": _jwt(int(time.time()) - 10),
                    "account_id": "acct-test",
                }
            }
        ),
        encoding="utf-8",
    )

    credentials = asyncio.run(auth.valid_credentials())

    assert credentials.account_id == "acct-test"
    assert calls == [("start", {}), ("account/read", {"refreshToken": True})]
    assert auth.auth_path.stat().st_mode & 0o777 == 0o600


def test_responses_payload_preserves_tools_reasoning_history_and_images() -> None:
    model = CodexSubscriptionChatModel(
        model="gpt-test",
        reasoning_effort="high",
        prompt_cache_key="thread-key",
    )
    prior = AIMessage(
        content="",
        tool_calls=[{"name": "read_file", "args": {"path": "a.py"}, "id": "call-1"}],
    )
    payload = model._payload(
        [
            SystemMessage("stable"),
            HumanMessage(
                content=[
                    {"type": "text", "text": "inspect"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                ]
            ),
            prior,
            ToolMessage("contents", tool_call_id="call-1"),
        ],
        tools=[read_file],
        tool_choice="any",
        max_tokens=99,
    )

    assert payload["instructions"] == "stable"
    assert payload["input"][0]["content"][1] == {
        "type": "input_image",
        "image_url": "data:image/png;base64,AAAA",
    }
    assert payload["input"][1]["type"] == "function_call"
    assert payload["input"][2]["type"] == "function_call_output"
    assert payload["input"][2]["output"] == "contents"
    assert payload["tools"][0]["name"] == "read_file"
    assert payload["tool_choice"] == "required"
    assert payload["reasoning"] == {"effort": "high", "summary": "auto"}
    assert payload["prompt_cache_key"] == "thread-key"
    assert "max_output_tokens" not in payload
    assert "max_tokens" not in payload


def test_compaction_omits_unsupported_token_limit_and_preserves_parent_request(
    monkeypatch,
) -> None:
    model = CodexSubscriptionChatModel(
        model="gpt-test", reasoning_effort="high", prompt_cache_key="thread-key"
    )
    bound = model.bind_tool_registry(SimpleNamespace(active_tools=[read_file]))
    messages = [SystemMessage("stable"), HumanMessage("Fix the build")]
    parent_payload = bound._payload(messages)
    calls = []

    async def create(payload):
        # Model the Codex backend's rejection of output-token limits.
        if "max_output_tokens" in payload or "max_tokens" in payload:
            raise RuntimeError("Unsupported parameter: max_output_tokens")
        calls.append(payload)
        return {"model": "gpt-test", "output_text": "Build fixed; tests remain."}

    monkeypatch.setattr(model._transport, "create", create)
    result = asyncio.run(
        summarize(
            messages, bound, instruction="Summarize progress", max_output_tokens=4096
        )
    )

    assert result == "Build fixed; tests remain."
    assert len(calls) == 1
    payload = calls[0]
    assert payload["input"][:-1] == parent_payload["input"]
    assert payload["input"][-1] == {
        "role": "user",
        "content": [{"type": "input_text", "text": "Summarize progress"}],
    }
    assert {key: value for key, value in payload.items() if key != "input"} == {
        key: value for key, value in parent_payload.items() if key != "input"
    }


@pytest.mark.parametrize("asynchronous", [False, True])
def test_registry_bindings_share_transport_and_keep_tool_snapshots(
    monkeypatch, asynchronous
) -> None:
    model = CodexSubscriptionChatModel(model="gpt-test")
    registry = SimpleNamespace(active_tools=[read_file])
    calls = []

    async def create(payload):
        calls.append(payload)
        return {"model": "gpt-test", "output_text": "done"}

    monkeypatch.setattr(model._transport, "create", create)
    before = model.model_dump()
    bound = model.bind_tool_registry(registry)
    registry.active_tools.clear()
    without_tools = model.bind_tool_registry(registry)

    assert bound is not model
    assert bound._auth is without_tools._auth is model._auth
    assert bound._transport is without_tools._transport is model._transport
    assert model.model_dump() == before
    assert bound._llm_type == "codex-subscription-responses"
    assert bound._identifying_params["billing_mode"] == "subscription"

    for candidate in (bound, model, without_tools):
        response = (
            asyncio.run(candidate.ainvoke([HumanMessage("hello")]))
            if asynchronous
            else candidate.invoke([HumanMessage("hello")])
        )
        assert response.content == "done"
        assert response.response_metadata["billing_mode"] == "subscription"

    assert [tool["name"] for tool in calls[0]["tools"]] == ["read_file"]
    assert "tools" not in calls[1]
    assert "tools" not in calls[2]


def test_response_maps_previous_id_tools_usage_and_subscription() -> None:
    message = CodexSubscriptionChatModel._message(
        {
            "id": "response-1",
            "model": "gpt-test",
            "output": [
                {"type": "reasoning", "summary": []},
                {"type": "message", "content": [{"type": "output_text", "text": "done"}]},
                {"type": "function_call", "call_id": "call-1", "name": "read_file", "arguments": '{"path":"a.py"}'},
            ],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
                "input_tokens_details": {"cached_tokens": 3},
            },
        }
    )

    assert message.content == "done"
    assert message.tool_calls[0]["args"] == {"path": "a.py"}
    assert message.response_metadata["response_id"] == "response-1"
    assert message.response_metadata["billing_mode"] == "subscription"
    assert message.usage_metadata["input_token_details"]["cache_read"] == 3
    assert message.additional_kwargs["codex_output_items"][0]["type"] == "reasoning"


def test_prompt_cache_key_is_stable_and_thread_scoped() -> None:
    adapter = CodexProviderAdapter(_runtime())

    first = adapter.build_chat_model("thread-1", model_name="gpt-test", reasoning_effort="high")
    again = adapter.build_chat_model("thread-1", model_name="gpt-test", reasoning_effort="high")
    reflection = adapter.build_chat_model(
        "thread-1", model_name="gpt-test", reasoning_effort="high", session_suffix="reflection"
    )

    assert str(uuid.UUID(str(first.prompt_cache_key))) == first.prompt_cache_key
    assert first.prompt_cache_key == again.prompt_cache_key
    assert first.prompt_cache_key != reflection.prompt_cache_key


def test_sparse_completed_response_keeps_streamed_output() -> None:
    merged = merge_streamed_response(
        {"id": "response-1", "usage": {"output_tokens": 2}},
        [{"type": "message", "content": [{"type": "output_text", "text": "done"}]}],
        ["do", "ne"],
    )

    assert merged["output_text"] == "done"
    assert merged["output"][0]["type"] == "message"
    assert CodexSubscriptionChatModel._message(merged).content == "done"


def test_stream_errors_distinguish_retryable_and_terminal() -> None:
    retryable = _stream_error(
        {"type": "error", "error": {"type": "server_error", "message": "busy"}},
        httpx.Headers({"retry-after": "2"}),
    )
    terminal = _stream_error(
        {"type": "error", "error": {"type": "invalid_request_error", "message": "bad"}},
        httpx.Headers(),
    )

    assert retryable.retryable is True
    assert retryable.retry_after == 2
    assert terminal.retryable is False


def test_transport_refreshes_once_after_unauthorized(monkeypatch) -> None:
    refresh_calls: list[bool] = []

    class Auth:
        async def valid_credentials(self, *, force_refresh=False):
            refresh_calls.append(force_refresh)
            return CodexCredentials("token", "account", None)

    class Response:
        def __init__(self, status, lines=()):
            self.status_code = status
            self.headers = httpx.Headers()
            self.request = httpx.Request("POST", "https://example.test")
            self._lines = lines
            self.is_success = 200 <= status < 300
            self.text = ""

        async def aread(self):
            return b""

        async def aiter_lines(self):
            for line in self._lines:
                yield line

        def raise_for_status(self):
            if not self.is_success:
                raise httpx.HTTPStatusError("bad", request=self.request, response=self)

    responses = [
        Response(401),
        Response(200, ['data: {"type":"response.completed","response":{"id":"ok"}}']),
    ]

    class Stream:
        def __init__(self, response):
            self.response = response

        async def __aenter__(self):
            return self.response

        async def __aexit__(self, *_args):
            return None

    class Client:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def stream(self, *_args, **_kwargs):
            return Stream(responses.pop(0))

    monkeypatch.setattr("ness_cli.providers.codex.transport.httpx.AsyncClient", Client)

    result = asyncio.run(CodexResponsesTransport(Auth(), max_retries=0).create({"model": "gpt"}))

    assert result["id"] == "ok"
    assert refresh_calls == [False, True, False]


def test_transport_retries_retryable_sse_failure(monkeypatch) -> None:
    class Auth:
        async def valid_credentials(self, *, force_refresh=False):
            del force_refresh
            return CodexCredentials("token", "account", None)

    class Response:
        status_code = 200
        is_success = True
        headers = httpx.Headers()
        request = httpx.Request("POST", "https://example.test")
        text = ""

        def __init__(self, lines):
            self.lines = lines

        async def aread(self):
            return b""

        async def aiter_lines(self):
            for line in self.lines:
                yield line

        def raise_for_status(self):
            return None

    responses = [
        Response(
            [
                'data: {"type":"error","error":{"type":"server_error","message":"busy"}}'
            ]
        ),
        Response(
            [
                'data: {"type":"response.output_text.delta","delta":"done"}',
                'data: {"type":"response.completed","response":{"id":"ok"}}',
            ]
        ),
    ]

    class Stream:
        def __init__(self, response):
            self.response = response

        async def __aenter__(self):
            return self.response

        async def __aexit__(self, *_args):
            return None

    class Client:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def stream(self, *_args, **_kwargs):
            return Stream(responses.pop(0))

    sleeps: list[float] = []

    async def no_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr("ness_cli.providers.codex.transport.httpx.AsyncClient", Client)
    monkeypatch.setattr("ness_cli.providers.codex.transport._sleep", no_sleep)
    monkeypatch.setattr("ness_cli.providers.codex.transport.uniform", lambda *_args: 0)

    result = asyncio.run(CodexResponsesTransport(Auth(), max_retries=1).create({"model": "gpt"}))

    assert result["id"] == "ok"
    assert result["output_text"] == "done"
    assert sleeps == [1]


def test_login_success_cancel_timeout_and_malformed_paths() -> None:
    class Server:
        def __init__(self):
            self.requests = []
            self.notification = {"success": True, "loginId": "login-1"}

        async def start(self):
            return None

        async def request(self, method, params):
            self.requests.append((method, params))
            if method == "account/login/start":
                return {"authUrl": "https://auth.test", "loginId": "login-1"}
            return {}

        async def wait_notification(self, *_args, **_kwargs):
            return self.notification

        async def close(self):
            return None

    adapter = CodexProviderAdapter(_runtime())
    server = Server()
    adapter.server = server
    adapter.auth = SimpleNamespace(wait_until_ready=lambda: None)

    pending = asyncio.run(adapter.login(method="browser"))

    class ReadyAuth:
        async def wait_until_ready(self):
            return CodexCredentials("token", "account", None)

    adapter.auth = ReadyAuth()
    complete = asyncio.run(adapter.wait_for_login("login-1"))
    asyncio.run(adapter.cancel_login("login-1"))

    assert pending.status == "pending"
    assert pending.login_id == "login-1"
    assert complete.status == "complete"
    assert server.requests[-1] == ("account/login/cancel", {"loginId": "login-1"})

    class TimeoutServer(Server):
        async def wait_notification(self, *_args, **_kwargs):
            raise TimeoutError("login timed out")

    adapter.server = TimeoutServer()
    with pytest.raises(TimeoutError, match="timed out"):
        asyncio.run(adapter.wait_for_login("login-1"))

    class MalformedServer(Server):
        async def request(self, method, params):
            return {"authUrl": "https://auth.test"}

    adapter.server = MalformedServer()
    malformed = asyncio.run(adapter.login(method="browser"))
    assert malformed.status == "error"
    assert "malformed" in malformed.message.lower()


def test_device_login_uses_device_code_contract() -> None:
    class Server:
        def __init__(self):
            self.request_call = None

        async def start(self):
            return None

        async def request(self, method, params):
            self.request_call = (method, params)
            return {
                "loginId": "device-1",
                "userCode": "ABCD-EFGH",
                "verificationUrl": "https://auth.test/device",
            }

    adapter = CodexProviderAdapter(_runtime())
    server = Server()
    adapter.server = server

    result = asyncio.run(adapter.login(method="device"))

    assert result.status == "pending"
    assert result.user_code == "ABCD-EFGH"
    assert result.verification_url == "https://auth.test/device"
    assert server.request_call == (
        "account/login/start",
        {"type": "chatgptDeviceCode"},
    )
    unsupported = asyncio.run(adapter.login(method="password"))
    assert unsupported.status == "error"


def test_codex_context_windows_use_server_value_then_family_fallback() -> None:
    class Server:
        async def request(self, method, params):
            assert method == "model/list"
            assert params == {"limit": 100}
            return {
                "data": [
                    {
                        "model": "gpt-explicit",
                        "displayName": "Explicit",
                        "contextWindow": 777_000,
                    },
                    {
                        "model": "gpt-5.6-luna",
                        "displayName": "Codex",
                    },
                ]
            }

    records = asyncio.run(load_models(Server()))
    assert records[0].context_window == 777_000
    assert records[1].context_window == 272_000

    cold = CodexProviderAdapter(_runtime())
    assert cold.model_info("gpt-5.6-luna").context_window == 272_000


def test_app_server_correlates_ids_and_keeps_notifications_separate() -> None:
    class Stdout:
        def __init__(self):
            self.lines = [
                b'{"jsonrpc":"2.0","method":"turn/started","params":{"id":"turn"}}\n',
                b'{"jsonrpc":"2.0","id":2,"result":{"value":"two"}}\n',
                b'{"jsonrpc":"2.0","id":1,"result":{"value":"one"}}\n',
                b"",
            ]

        async def readline(self):
            return self.lines.pop(0)

    async def run():
        server = CodexAppServer(SimpleNamespace())
        server._process = SimpleNamespace(stdout=Stdout())
        loop = asyncio.get_running_loop()
        first = loop.create_future()
        second = loop.create_future()
        server._pending = {1: first, 2: second}
        await server._read_loop()
        return first.result(), second.result(), server._notification_history

    first, second, history = asyncio.run(run())

    assert first["result"]["value"] == "one"
    assert second["result"]["value"] == "two"
    assert history["turn/started"] == [{"id": "turn"}]


def test_app_server_close_terminates_process_tasks_and_pending_requests() -> None:
    class Process:
        returncode = None

        def __init__(self):
            self.terminated = False
            self.killed = False

        def terminate(self):
            self.terminated = True
            self.returncode = 0

        def kill(self):
            self.killed = True
            self.returncode = -9

        async def wait(self):
            return self.returncode

    async def run():
        server = CodexAppServer(SimpleNamespace())
        process = Process()
        server._process = process
        server._reader_task = asyncio.create_task(asyncio.Event().wait())
        future = asyncio.get_running_loop().create_future()
        server._pending[1] = future
        await server.close()
        return process, server._reader_task, future

    process, task, future = asyncio.run(run())

    assert process.terminated is True
    assert task is None
    assert future.done()
    with pytest.raises(CodexUnavailable, match="closed"):
        future.result()


def test_codex_chat_model_projects_image_tool_results():
    model = CodexSubscriptionChatModel(model="gpt-test")
    data_url = "data:image/png;base64,cG5n"

    _instructions, items = model._input(
        [
            ToolMessage(
                content=[
                    {"type": "text", "text": "Read image: code.png, 2x1"},
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url, "detail": "high"},
                    },
                ],
                tool_call_id="call-image",
            )
        ]
    )

    assert items == [
        {
            "type": "function_call_output",
            "call_id": "call-image",
            "output": [
                {"type": "input_text", "text": "Read image: code.png, 2x1"},
                {"type": "input_image", "image_url": data_url, "detail": "high"},
            ],
        }
    ]
