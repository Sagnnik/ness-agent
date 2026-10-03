from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from ness_cli import runtime
from ness_cli.config import ConfigManager
from ness_cli.providers.registry import ProviderRegistry


class Closer:
    def __init__(self, name, calls, error=None):
        self.name = name
        self.calls = calls
        self.error = error

    async def close(self):
        self.calls.append(self.name)
        if self.error is not None:
            raise self.error

    stop = close


@pytest.mark.parametrize("owner", ["sessions", "providers"])
@pytest.mark.parametrize("failed", [(0,), (0, 1)])
def test_all_owned_resources_are_attempted_and_repeat_close_is_safe(
    isolated_cli_env, monkeypatch, owner, failed
):
    calls = []
    errors = {index: RuntimeError(f"failure {index}") for index in failed}
    closers = {
        f"resource-{index}": Closer(f"resource-{index}", calls, errors.get(index))
        for index in range(3)
    }
    if owner == "sessions":
        monkeypatch.setattr(runtime, "repo_root", lambda _: None)
        manager = runtime._SessionFactory(
            paths=isolated_cli_env.paths, config=Mock(), providers=Mock(), agent=Mock()
        )
        manager._sessions.update(closers)
    else:
        manager = ProviderRegistry(
            paths=isolated_cli_env.paths,
            config=ConfigManager.load(isolated_cli_env.paths, environment={}),
        )
        manager._instances.update(closers)

    with pytest.raises(Exception) as raised:
        asyncio.run(manager.close())

    assert calls == ["resource-0", "resource-1", "resource-2"]
    if len(failed) == 1:
        assert raised.value is errors[failed[0]]
    else:
        assert isinstance(raised.value, ExceptionGroup)
        assert raised.value.exceptions == tuple(errors.values())
        assert all(str(error) in str(raised.value) for error in errors.values())
    asyncio.run(manager.close())
    assert calls == ["resource-0", "resource-1", "resource-2"]


def make_core(calls, errors):
    return runtime._RuntimeCore(
        paths=SimpleNamespace(),
        config=SimpleNamespace(),
        oauth=SimpleNamespace(),
        warnings=(),
        agent=SimpleNamespace(),
        sessions=Closer("sessions", calls, errors.get("sessions")),
        mcp=Closer("mcp", calls, errors.get("mcp")),
        providers=Closer("providers", calls, errors.get("providers")),
    )


def test_runtime_reports_all_shutdown_failures_and_remains_idempotent():
    calls = []
    errors = {
        name: RuntimeError(f"{name} failed")
        for name in ("sessions", "mcp", "providers")
    }
    core = make_core(calls, errors)

    with pytest.raises(ExceptionGroup) as raised:
        asyncio.run(core.close())

    assert calls == ["sessions", "mcp", "providers"]
    assert raised.value.exceptions == tuple(errors.values())
    assert all(str(error) in str(raised.value) for error in errors.values())
    asyncio.run(core.close())
    assert calls == ["sessions", "mcp", "providers"]


def test_cancelled_close_still_attempts_later_owners(
    isolated_cli_env, monkeypatch, caplog
):
    calls = []
    cancellation = asyncio.CancelledError("stop closing")
    monkeypatch.setattr(runtime, "repo_root", lambda _: None)
    factory = runtime._SessionFactory(
        paths=isolated_cli_env.paths, config=Mock(), providers=Mock(), agent=Mock()
    )
    factory._sessions.update(
        {
            "first": Closer("first", calls, cancellation),
            "middle": Closer("middle", calls, RuntimeError("middle close failed")),
            "last": Closer("last", calls),
        }
    )

    with (
        caplog.at_level(logging.WARNING),
        pytest.raises(asyncio.CancelledError) as raised,
    ):
        asyncio.run(factory.close())

    assert raised.value is cancellation
    assert calls == ["first", "middle", "last"]
    assert "middle close failed" in caplog.text
    assert any("middle close failed" in note for note in cancellation.__notes__)


@pytest.fixture
def startup_env(isolated_cli_env, monkeypatch):
    calls = []
    providers = SimpleNamespace(
        active=Mock(return_value=SimpleNamespace(is_authenticated=lambda: True)),
        close=AsyncMock(side_effect=lambda: calls.append("providers")),
    )
    mcp = SimpleNamespace(
        start=AsyncMock(),
        stop=AsyncMock(side_effect=lambda: calls.append("mcp")),
        tools={},
        catalog=lambda: {},
        startup_summary=lambda: ("ready", "ok"),
    )
    oauth = SimpleNamespace(startup_auth=Mock(), warnings=())
    monkeypatch.setattr(runtime.ProviderRegistry, "load", Mock(return_value=providers))
    monkeypatch.setattr(runtime, "ProjectMCPManager", Mock(return_value=mcp))
    monkeypatch.setattr(runtime, "MCPOAuthService", Mock(return_value=oauth))
    monkeypatch.setattr(runtime, "is_mcp_trusted", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        runtime, "authorize_mcp_interactively", lambda *args, **kwargs: True
    )
    agent = SimpleNamespace(
        config=SimpleNamespace(
            tool_registry=SimpleNamespace(
                register_dynamic=Mock(), set_mcp_catalog=Mock()
            )
        )
    )
    monkeypatch.setattr(runtime, "_build_agent", Mock(return_value=agent))
    return SimpleNamespace(
        calls=calls,
        providers=providers,
        mcp=mcp,
        agent=agent,
        options=runtime.HeadlessOptions(
            project_root=isolated_cli_env.paths.project_root
        ),
    )


@pytest.mark.parametrize(
    "stage",
    ["mcp_start", "build_agent", "register_tools", "session_factory", "runtime_core"],
)
def test_startup_error_survives_all_cleanup_failures(
    startup_env, monkeypatch, caplog, stage
):
    env = startup_env
    original = ValueError("original startup failure")
    env.mcp.stop.side_effect = RuntimeError("MCP stop failed")
    env.providers.close.side_effect = RuntimeError("providers close failed")
    if stage == "mcp_start":
        env.mcp.start.side_effect = original
    elif stage == "build_agent":
        monkeypatch.setattr(runtime, "_build_agent", Mock(side_effect=original))
    elif stage == "register_tools":
        env.agent.config.tool_registry.register_dynamic.side_effect = original
    elif stage == "session_factory":
        monkeypatch.setattr(runtime, "_SessionFactory", Mock(side_effect=original))
    else:
        monkeypatch.setattr(runtime, "_RuntimeCore", Mock(side_effect=original))

    with caplog.at_level(logging.WARNING), pytest.raises(ValueError) as raised:
        asyncio.run(runtime._open_core(env.options))

    assert raised.value is original
    env.mcp.stop.assert_awaited_once()
    env.providers.close.assert_awaited_once()
    assert "MCP stop failed" in caplog.text and "providers close failed" in caplog.text
    assert len(original.__notes__) == 2


@pytest.mark.parametrize("constructor", ["MCPOAuthService", "ProjectMCPManager"])
def test_partial_startup_closes_providers_when_later_construction_fails(
    startup_env, monkeypatch, constructor
):
    env = startup_env
    original = ValueError("constructor failed")
    monkeypatch.setattr(runtime, constructor, Mock(side_effect=original))

    with pytest.raises(ValueError) as raised:
        asyncio.run(runtime._open_core(env.options))

    assert raised.value is original
    env.providers.close.assert_awaited_once()
    env.mcp.stop.assert_not_awaited()


@pytest.mark.parametrize("kind", ["headless", "interactive"])
def test_runtime_context_preserves_body_error_during_failed_cleanup(
    monkeypatch, kind, caplog
):
    calls = []
    core = make_core(calls, {"mcp": RuntimeError("MCP close failed")})
    monkeypatch.setattr(runtime, "_open_core", AsyncMock(return_value=core))
    monkeypatch.setattr(runtime, "_require_provider_authentication", lambda _: None)
    original = ValueError("original runtime failure")
    opener = (
        runtime.open_headless_runtime
        if kind == "headless"
        else runtime.open_interactive_runtime
    )
    options = (
        runtime.HeadlessOptions()
        if kind == "headless"
        else runtime.InteractiveOptions()
    )

    async def scenario():
        async with opener(options):
            raise original

    with caplog.at_level(logging.WARNING), pytest.raises(ValueError) as raised:
        asyncio.run(scenario())

    assert raised.value is original
    assert calls == ["sessions", "mcp", "providers"]
    assert "MCP close failed" in caplog.text
    assert any("MCP close failed" in note for note in original.__notes__)


def test_headless_authentication_error_survives_failed_cleanup(monkeypatch, caplog):
    calls = []
    core = make_core(calls, {"providers": RuntimeError("provider close failed")})
    core.providers.active = lambda: SimpleNamespace(
        is_authenticated=lambda: False, display_name="Test"
    )
    monkeypatch.setattr(runtime, "_open_core", AsyncMock(return_value=core))

    async def scenario():
        async with runtime.open_headless_runtime(runtime.HeadlessOptions()):
            pytest.fail("unauthenticated runtime must not be yielded")

    with (
        caplog.at_level(logging.WARNING),
        pytest.raises(runtime.ProviderAuthenticationError),
    ):
        asyncio.run(scenario())

    assert calls == ["sessions", "mcp", "providers"]
    assert "provider close failed" in caplog.text


def test_cancelled_startup_preserves_cancellation_and_attempts_all_cleanup(
    startup_env, caplog
):
    env = startup_env
    env.mcp.stop.side_effect = RuntimeError("MCP stop failed")
    env.providers.close.side_effect = RuntimeError("providers close failed")

    async def scenario():
        entered = asyncio.Event()

        async def start():
            entered.set()
            await asyncio.Event().wait()

        env.mcp.start.side_effect = start
        task = asyncio.create_task(runtime._open_core(env.options))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as raised:
            await task
        assert len(raised.value.__notes__) == 2

    with caplog.at_level(logging.WARNING):
        asyncio.run(scenario())

    env.mcp.stop.assert_awaited_once()
    env.providers.close.assert_awaited_once()
    assert "MCP stop failed" in caplog.text and "providers close failed" in caplog.text


@pytest.mark.parametrize("kind", ["headless", "interactive"])
def test_successful_startup_and_context_exit_close_resources_once(startup_env, kind):
    env = startup_env
    opener = (
        runtime.open_headless_runtime
        if kind == "headless"
        else runtime.open_interactive_runtime
    )
    options = (
        env.options
        if kind == "headless"
        else runtime.InteractiveOptions(project_root=env.options.project_root)
    )

    async def scenario():
        async with opener(options) as opened:
            assert opened.mcp is env.mcp
        await opened.close()

    asyncio.run(scenario())

    assert env.calls == ["mcp", "providers"]
    env.mcp.start.assert_awaited_once()
    env.mcp.stop.assert_awaited_once()
    env.providers.close.assert_awaited_once()


@pytest.mark.parametrize("kind", ["headless", "interactive"])
def test_context_reports_cleanup_failures_when_body_succeeds(monkeypatch, kind):
    calls = []
    errors = {
        "mcp": RuntimeError("MCP stop failed"),
        "providers": RuntimeError("provider close failed"),
    }
    core = make_core(calls, errors)
    monkeypatch.setattr(runtime, "_open_core", AsyncMock(return_value=core))
    monkeypatch.setattr(runtime, "_require_provider_authentication", lambda _: None)
    opener = (
        runtime.open_headless_runtime
        if kind == "headless"
        else runtime.open_interactive_runtime
    )
    options = (
        runtime.HeadlessOptions()
        if kind == "headless"
        else runtime.InteractiveOptions()
    )

    async def scenario():
        async with opener(options):
            pass

    with pytest.raises(ExceptionGroup) as raised:
        asyncio.run(scenario())

    assert raised.value.exceptions == tuple(errors.values())
    assert calls == ["sessions", "mcp", "providers"]
