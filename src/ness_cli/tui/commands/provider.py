"""Provider status, login, and configuration commands."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import Literal

from ness_cli.config import ConfigPatch
from ness_cli.providers.base import LoginResult, ModelInfo, ProviderAdapter
from ness_cli.tui.context import CommandContext
from ness_cli.tui.input import MenuItem


async def status_command(context: CommandContext, args: str) -> None:
    if args:
        context.renderer.error("Usage: /status")
        return
    provider = context.runtime.providers.active()
    usage = context.session.usage_summary()
    try:
        status = await provider.status(refresh=False)
    except Exception as error:
        status = None
        context.renderer.warning(f"Provider status unavailable: {error}")

    input_tokens = usage["input_tokens"]
    cached_tokens = usage["cached_input_tokens"]
    cache_hit = cached_tokens / input_tokens if input_tokens else None
    lines = [
        f"provider       {provider.display_name}",
        f"thread         {context.session.thread_id}",
        f"model          {context.runtime.config.model.model_name}",
        f"reasoning      {context.runtime.config.model.reasoning_effort or 'none'}",
        f"turns          {usage['turns']}",
        f"input tokens   {input_tokens:,}",
        f"output tokens  {usage['output_tokens']:,}",
        f"cache read     {cached_tokens:,}",
        (
            f"cache hit      {cache_hit:.0%}"
            if cache_hit is not None
            else "cache hit      n/a"
        ),
        (
            "cost           subscription"
            if provider.billing_label == "subscription"
            else f"cost           ${usage['cost_usd']:.4f}"
        ),
    ]
    if status is not None:
        lines.append(
            "auth           "
            + ("connected" if status.auth.authenticated else "not connected")
            + (f" via {status.auth.method}" if status.auth.method else "")
        )
        if status.account.email:
            lines.append(f"account        {status.account.email}")
        if status.account.tier:
            lines.append(f"tier           {status.account.tier}")
        if status.credits:
            lines.append(f"credits        {status.credits}")
        if status.warning:
            lines.append(f"provider note  {status.warning}")
    context.renderer.notice("session status", *lines)


LoginFlowResult = Literal["complete", "back", "stop"]


def _provider_items(context: CommandContext) -> list[MenuItem]:
    active_id = context.runtime.config.model.provider_id
    items: list[MenuItem] = []
    for provider_id in context.runtime.providers.provider_ids():
        provider = context.runtime.providers.get(provider_id)
        connected = provider.is_authenticated()
        status = "connected" if connected else "not connected"
        if provider_id == active_id:
            status = f"active · {status}"
        items.append(
            MenuItem(
                provider_id,
                provider.display_name,
                description=provider.login_description,
                suffix=status,
            )
        )
    return items


def _preferred_model(
    models: tuple[ModelInfo, ...],
    preferred_id: str,
) -> ModelInfo | None:
    return next((model for model in models if model.id == preferred_id), None) or next(
        (model for model in models if model.is_default),
        models[0] if models else None,
    )


async def _provider_selection(
    context: CommandContext,
    provider_id: str,
) -> tuple[str, str, str | None]:
    provider = context.runtime.providers.get(provider_id)
    models = await context.runtime.providers.models(provider_id=provider_id, refresh=False)
    current = context.runtime.config.model
    profile = context.runtime.provider_profile(provider_id)
    if current.provider_id == provider_id:
        preferred_id = current.model_name
        preferred_reflection = current.reflection_model_name
        preferred_effort = current.reasoning_effort
    else:
        preferred_id = str(profile.get("model_name") or "")
        preferred_reflection = str(profile.get("reflection_model_name") or "")
        preferred_effort = str(profile.get("reasoning_effort") or "") or None

    selected = _preferred_model(models, preferred_id)
    if selected is None:
        if not preferred_id:
            raise RuntimeError(f"{provider.display_name} did not return any models.")
        return preferred_id, preferred_reflection or preferred_id, preferred_effort

    reasoning_effort = preferred_effort
    if (
        selected.reasoning_efforts
        and reasoning_effort not in selected.reasoning_efforts
    ):
        reasoning_effort = (
            selected.default_reasoning_effort
            or (
                "medium"
                if "medium" in selected.reasoning_efforts
                else selected.reasoning_efforts[0]
            )
        )
    reasoning_effort = reasoning_effort or selected.default_reasoning_effort
    model_ids = {model.id for model in models}
    reflection_model = (
        preferred_reflection
        if preferred_reflection in model_ids
        else selected.id
    )
    return selected.id, reflection_model, reasoning_effort


async def _activate_provider(
    context: CommandContext,
    provider_id: str,
    *,
    force_reload: bool = False,
) -> str:
    model_name, reflection_model, reasoning_effort = await _provider_selection(
        context,
        provider_id,
    )
    update = await context.runtime.apply_config(
        ConfigPatch(
            {
                "provider_id": provider_id,
                "model_name": model_name,
                "reflection_model_name": reflection_model,
                "reasoning_effort": reasoning_effort,
            }
        ),
        selected=context.session,
    )
    if force_reload and not update.reload_selected:
        await context.session.reload_model()
    return model_name


async def _wait_for_provider_login(
    context: CommandContext,
    provider: ProviderAdapter,
    login_id: str,
) -> LoginResult | None:
    login_task = asyncio.create_task(provider.wait_for_login(login_id))
    cancel_task = asyncio.create_task(
        context.ui.choose(
            f"{provider.display_name} sign-in",
            [
                MenuItem(
                    "cancel",
                    "Cancel pending login",
                    description="Return to the current session",
                )
            ],
            initial_key="cancel",
            hint="Enter cancel · Esc cancel",
        )
    )
    try:
        done, _ = await asyncio.wait(
            {login_task, cancel_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if login_task in done:
            cancel_task.cancel()
            await asyncio.gather(cancel_task, return_exceptions=True)
            return login_task.result()

        login_task.cancel()
        await asyncio.gather(login_task, return_exceptions=True)
        with suppress(Exception):
            await asyncio.wait_for(provider.cancel_login(login_id), timeout=5)
        return None
    finally:
        for task in (login_task, cancel_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(login_task, cancel_task, return_exceptions=True)


async def _authenticate_provider(
    context: CommandContext,
    provider: ProviderAdapter,
) -> LoginFlowResult:
    methods = provider.login_methods()
    if not methods:
        context.renderer.error(
            f"{provider.display_name} does not support interactive login."
        )
        return "stop"
    method = next((item for item in methods if item.default), methods[0])
    if len(methods) > 1:
        selected = await context.ui.choose(
            f"{provider.display_name} login",
            [
                MenuItem(item.id, item.label, description=item.description)
                for item in methods
            ],
            initial_key=method.id,
        )
        if selected is None:
            return "back"
        method = next(item for item in methods if item.id == selected)

    secret = None
    if method.input_kind == "secret":
        secret = await context.ui.request_input(
            method.input_label or "API key",
            secret=True,
        )
        if not secret:
            return "back"
    if method.guidance:
        context.renderer.warning(method.guidance)
    result = await provider.login(method=method.id, secret=secret)
    if result.status == "cancelled":
        context.renderer.warning(result.message)
        return "stop"
    if result.status == "error":
        context.renderer.error(result.message)
        return "stop"
    if result.status == "complete":
        return "complete"

    url = result.auth_url or result.verification_url
    lines = [result.message]
    if url:
        lines.append(f"Open this URL: {url}")
    if result.user_code:
        lines.append(f"Device code: {result.user_code}")
    if url:
        opened = await provider.open_login_url(url)
        if opened:
            lines.append("A browser window was opened.")
    context.renderer.notice("login", *lines)
    if not result.login_id:
        context.renderer.error("The provider did not return a login ID.")
        return "stop"
    try:
        completed = await _wait_for_provider_login(
            context,
            provider,
            result.login_id,
        )
    except asyncio.CancelledError:
        with suppress(Exception):
            await provider.cancel_login(result.login_id)
        raise
    except TimeoutError:
        with suppress(Exception):
            await provider.cancel_login(result.login_id)
        context.renderer.error(
            f"{provider.display_name} sign-in timed out and was cancelled."
        )
        return "stop"

    if completed is None:
        context.renderer.warning(f"{provider.display_name} sign-in was cancelled.")
        return "stop"
    result = completed
    if result.status != "complete":
        if result.status == "cancelled":
            context.renderer.notice("login", result.message)
        else:
            context.renderer.error(result.message)
        return "stop"
    return "complete"


async def _logout_with_fallback(
    context: CommandContext,
    provider_id: str,
    previous_active_id: str,
) -> None:
    provider = context.runtime.providers.get(provider_id)
    context.renderer.notice("login", await provider.logout())

    candidates: list[str] = []
    if previous_active_id != provider_id:
        candidates.append(previous_active_id)
    candidates.extend(
        candidate
        for candidate in context.runtime.providers.provider_ids()
        if candidate != provider_id and candidate not in candidates
    )
    for candidate in candidates:
        fallback = context.runtime.providers.get(candidate)
        if not fallback.is_authenticated():
            continue
        try:
            model_name = await _activate_provider(
                context,
                candidate,
                force_reload=True,
            )
        except Exception as error:
            context.renderer.warning(
                f"Could not switch to {fallback.display_name}: {error}"
            )
            continue
        context.renderer.notice(
            "login",
            f"Switched to {fallback.display_name} with {model_name}.",
        )
        return

    await context.session.reload_model()
    context.renderer.warning(
        "No connected model provider remains. The current thread was preserved. "
        "Run /login before sending another message."
    )


async def _manage_connected_provider(
    context: CommandContext,
    provider_id: str,
    previous_active_id: str,
) -> LoginFlowResult:
    provider = context.runtime.providers.get(provider_id)
    while True:
        action = await context.ui.choose(
            f"{provider.display_name} connected",
            [
                MenuItem(
                    "reconnect",
                    "Reconnect",
                    description="Replace the current credentials",
                ),
                MenuItem(
                    "logout",
                    "Log out",
                    description="Remove the saved credentials",
                ),
            ],
            initial_key="reconnect",
        )
        if action is None:
            return "back"
        if action == "logout":
            await _logout_with_fallback(
                context,
                provider_id,
                previous_active_id,
            )
            return "complete"

        result = await _authenticate_provider(context, provider)
        if result == "back":
            continue
        if result == "complete":
            model_name = await _activate_provider(
                context,
                provider_id,
                force_reload=True,
            )
            context.renderer.notice(
                "login",
                f"{provider.display_name} reconnected with {model_name}. "
                "The current thread was preserved.",
            )
        return result


async def login_command(context: CommandContext, args: str) -> None:
    requested = args.strip()
    provider_ids = context.runtime.providers.provider_ids()
    if requested and requested not in provider_ids:
        context.renderer.error(f"Unknown provider: {requested}")
        return

    while True:
        if requested:
            provider_id = requested
        elif len(provider_ids) == 1:
            provider_id = provider_ids[0]
        else:
            selected = await context.ui.choose(
                "model provider",
                _provider_items(context),
                initial_key=context.runtime.config.model.provider_id,
            )
            if selected is None:
                return
            provider_id = selected

        provider = context.runtime.providers.get(provider_id)
        previous_active_id = context.runtime.config.model.provider_id

        if provider.is_authenticated():
            if provider_id != previous_active_id:
                try:
                    await _activate_provider(context, provider_id)
                except Exception as error:
                    context.renderer.error(
                        f"Could not activate {provider.display_name}: {error}"
                    )
                    if requested:
                        return
                    continue
            result = await _manage_connected_provider(
                context,
                provider_id,
                previous_active_id,
            )
            if result == "back" and not requested:
                continue
            return

        result = await _authenticate_provider(context, provider)
        if result == "back" and not requested:
            continue
        if result != "complete":
            return

        try:
            model_name = await _activate_provider(
                context,
                provider_id,
                force_reload=True,
            )
        except Exception as error:
            context.renderer.error(
                f"Signed in to {provider.display_name}, but could not activate it: "
                f"{error}"
            )
            return
        context.renderer.notice(
            "login",
            f"{provider.display_name} is active with {model_name}. "
            "The current thread was preserved.",
        )
        return


async def config_command(context: CommandContext, args: str) -> None:
    if args:
        context.renderer.error("Usage: /config")
        return
    from ness_cli.tui.input.config import run_config_flow

    await run_config_flow(context)
