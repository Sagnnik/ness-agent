from __future__ import annotations

import asyncio
from collections import deque
from types import SimpleNamespace

from ness_cli.providers.base import LoginMethod, LoginResult, ModelInfo
from ness_cli.tui.commands.provider import (
    _authenticate_provider,
    _wait_for_provider_login,
    login_command,
)
from ness_cli.tui.context import CommandContext


class Renderer:
    def __init__(self):
        self.notices = []
        self.warnings = []
        self.errors = []

    def notice(self, title, *lines):
        self.notices.append((title, lines))

    def warning(self, message):
        self.warnings.append(message)

    def error(self, message):
        self.errors.append(message)


class UI:
    def __init__(self, *choices, inputs=()):
        self.choices = deque(choices)
        self.inputs = deque(inputs)
        self.choose_calls = []

    async def choose(self, title, items, **kwargs):
        self.choose_calls.append((title, items, kwargs))
        return self.choices.popleft() if self.choices else None

    async def request_input(self, label, **kwargs):
        del label, kwargs
        return self.inputs.popleft() if self.inputs else None


class Session:
    thread_id = "thread-1"

    def __init__(self):
        self.reloads = 0

    async def reload_model(self):
        self.reloads += 1


class Provider:
    login_description = "test login"
    billing_label = "subscription"

    def __init__(
        self,
        provider_id,
        *,
        authenticated,
        models=(),
        methods=(),
        login_result=None,
    ):
        self.id = provider_id
        self.display_name = provider_id.title()
        self.authenticated = authenticated
        self._models = tuple(models)
        self._methods = tuple(methods)
        self.login_result = login_result or LoginResult("complete", "signed in")
        self.login_calls = []
        self.cancelled = []
        self.opened = []
        self.wait_cancelled = False
        self.logout_calls = 0

    def is_authenticated(self):
        return self.authenticated

    def login_methods(self):
        return self._methods

    async def login(self, **kwargs):
        self.login_calls.append(kwargs)
        return self.login_result

    async def wait_for_login(self, login_id):
        del login_id
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.wait_cancelled = True
            raise

    async def cancel_login(self, login_id):
        self.cancelled.append(login_id)

    async def open_login_url(self, url):
        self.opened.append(url)
        return False

    async def models(self, *, refresh=False):
        del refresh
        return self._models

    async def logout(self):
        self.logout_calls += 1
        self.authenticated = False
        return f"Signed out of {self.display_name}."


class Registry:
    def __init__(self, providers):
        self.providers = providers

    def provider_ids(self):
        return tuple(self.providers)

    def get(self, provider_id):
        return self.providers[provider_id]

    async def models(self, *, provider_id, refresh=False):
        return await self.get(provider_id).models(refresh=refresh)


class Runtime:
    def __init__(self, providers, active, *, profiles=None):
        self.providers = Registry(providers)
        self.config = SimpleNamespace(
            model=SimpleNamespace(
                provider_id=active,
                model_name=f"{active}-model",
                reflection_model_name=f"{active}-model",
                reasoning_effort=None,
            )
        )
        self.profiles = profiles or {}
        self.patches = []

    def provider_profile(self, provider_id):
        return dict(self.profiles.get(provider_id, {}))

    async def apply_config(self, patch, *, selected):
        values = dict(patch.values)
        previous = vars(self.config.model).copy()
        for key, value in values.items():
            setattr(self.config.model, key, value)
        changed = any(previous.get(key) != value for key, value in values.items())
        self.patches.append(values)
        if changed:
            await selected.reload_model()
        return SimpleNamespace(reload_selected=changed)


def context(runtime, ui):
    renderer = Renderer()
    session = Session()
    threads = SimpleNamespace(session=session, sink=renderer)
    return CommandContext(runtime=runtime, threads=threads, ui=ui), renderer, session


def model(model_id, *, default=False):
    return ModelInfo(
        model_id,
        model_id,
        default_reasoning_effort="medium",
        reasoning_efforts=("low", "medium", "high"),
        is_default=default,
    )


def test_device_login_displays_verification_details_and_can_cancel():
    provider = Provider(
        "codex",
        authenticated=False,
        methods=(LoginMethod("device", "Device code"),),
        login_result=LoginResult(
            "pending",
            "Complete sign-in.",
            login_id="device-1",
            verification_url="https://auth.test/device",
            user_code="ABCD-EFGH",
        ),
    )
    ctx, renderer, _session = context(
        Runtime({"codex": provider}, "codex"),
        UI("cancel"),
    )

    result = asyncio.run(_authenticate_provider(ctx, provider))

    assert result == "stop"
    rendered = repr(renderer.notices)
    assert "https://auth.test/device" in rendered
    assert "ABCD-EFGH" in rendered
    assert provider.opened == ["https://auth.test/device"]
    assert provider.cancelled == ["device-1"]
    assert provider.wait_cancelled is True


def test_completed_login_dismisses_cancel_picker_without_cancelling_provider():
    picker_cancelled = False

    class CompletedProvider(Provider):
        async def wait_for_login(self, login_id):
            del login_id
            await asyncio.sleep(0)
            return LoginResult("complete", "signed in")

    class WaitingUI(UI):
        async def choose(self, title, items, **kwargs):
            nonlocal picker_cancelled
            del title, items, kwargs
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                picker_cancelled = True
                raise

    provider = CompletedProvider("codex", authenticated=False)
    ctx, _renderer, _session = context(
        Runtime({"codex": provider}, "codex"),
        WaitingUI(),
    )

    result = asyncio.run(_wait_for_provider_login(ctx, provider, "login-1"))

    assert result == LoginResult("complete", "signed in")
    assert picker_cancelled is True
    assert provider.cancelled == []


def test_connected_provider_activates_saved_profile_without_reauthentication():
    openrouter = Provider(
        "openrouter",
        authenticated=True,
        models=(model("openrouter-model", default=True),),
    )
    codex = Provider(
        "codex",
        authenticated=True,
        models=(model("codex-default", default=True), model("codex-preferred")),
    )
    runtime = Runtime(
        {"codex": codex, "openrouter": openrouter},
        "openrouter",
        profiles={
            "codex": {
                "model_name": "codex-preferred",
                "reflection_model_name": "codex-default",
                "reasoning_effort": "high",
            }
        },
    )
    ui = UI(None)
    ctx, _renderer, _session = context(runtime, ui)

    asyncio.run(login_command(ctx, "codex"))

    assert runtime.config.model.provider_id == "codex"
    assert runtime.config.model.model_name == "codex-preferred"
    assert runtime.config.model.reflection_model_name == "codex-default"
    assert runtime.config.model.reasoning_effort == "high"
    assert codex.login_calls == []
    assert [item.key for item in ui.choose_calls[0][1]] == ["reconnect", "logout"]


def test_logout_prefers_previous_connected_provider():
    openrouter = Provider(
        "openrouter",
        authenticated=True,
        models=(model("openrouter-saved", default=True),),
    )
    codex = Provider(
        "codex",
        authenticated=True,
        models=(model("codex-model", default=True),),
    )
    runtime = Runtime(
        {"codex": codex, "openrouter": openrouter},
        "codex",
        profiles={"openrouter": {"model_name": "openrouter-saved"}},
    )
    ctx, _renderer, session = context(runtime, UI("logout"))

    asyncio.run(login_command(ctx, "codex"))

    assert codex.logout_calls == 1
    assert runtime.config.model.provider_id == "openrouter"
    assert runtime.config.model.model_name == "openrouter-saved"
    assert session.reloads == 1


def test_logout_without_fallback_preserves_thread_and_reloads_model():
    codex = Provider(
        "codex",
        authenticated=True,
        models=(model("codex-model", default=True),),
    )
    runtime = Runtime({"codex": codex}, "codex")
    ctx, renderer, session = context(runtime, UI("logout"))

    asyncio.run(login_command(ctx, "codex"))

    assert codex.logout_calls == 1
    assert session.reloads == 1
    assert any("No connected model provider remains" in item for item in renderer.warnings)
