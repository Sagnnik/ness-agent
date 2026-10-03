"""Interactive configuration editor built on the generic picker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from ness_cli.config import ConfigApplyError, ConfigPatch
from ness_cli.providers.base import ModelInfo
from ness_cli.tui.context import CommandContext
from ness_cli.tui.input.menus import MenuItem

ConfigKind = Literal[
    "bool",
    "int",
    "float",
    "text",
    "secret",
    "select",
    "provider",
    "model",
    "reasoning",
]


@dataclass(frozen=True, slots=True)
class ConfigSpec:
    key: str
    label: str
    section: str
    kind: ConfigKind
    optional: bool = False
    choices: tuple[str, ...] = ()
    example: str = ""


SECTIONS: tuple[tuple[str, str], ...] = (
    ("provider", "Provider"),
    ("model", "Model"),
    ("behavior", "Behavior"),
    ("compaction", "Compaction"),
    ("advanced", "Advanced"),
)

SPECS: tuple[ConfigSpec, ...] = (
    ConfigSpec("provider_id", "Active provider", "provider", "provider"),
    ConfigSpec("api_key", "Provider API key", "provider", "secret", optional=True),
    ConfigSpec(
        "openai_base_url",
        "OpenAI-compatible base URL",
        "provider",
        "text",
        optional=True,
    ),
    ConfigSpec("exa_api_key", "Exa API key", "provider", "secret", optional=True),
    ConfigSpec(
        "openrouter_session_id",
        "OpenRouter session ID",
        "provider",
        "text",
        optional=True,
    ),
    ConfigSpec("model_name", "Chat model", "model", "model"),
    ConfigSpec(
        "reflection_model_name",
        "Reflection model",
        "model",
        "model",
        optional=True,
    ),
    ConfigSpec(
        "goal_judge_model",
        "Goal judge model",
        "model",
        "model",
        optional=True,
    ),
    ConfigSpec(
        "reasoning_effort",
        "Reasoning effort",
        "model",
        "reasoning",
        optional=True,
    ),
    ConfigSpec("enable_approval", "Tool approval", "behavior", "bool"),
    ConfigSpec("auto_save_threads", "Thread autosave", "behavior", "bool"),
    ConfigSpec(
        "session_end_reflection",
        "Session-end reflection",
        "behavior",
        "bool",
    ),
    ConfigSpec("format_on_write", "Format after writes", "behavior", "bool"),
    ConfigSpec(
        "reflection_token_ratio",
        "Reflection token ratio",
        "compaction",
        "float",
        example="0.0 to 1.0",
    ),
    ConfigSpec(
        "compaction_token_budget",
        "Compaction token budget",
        "compaction",
        "int",
    ),
    ConfigSpec(
        "compaction_buffer_tokens",
        "Compaction buffer tokens",
        "compaction",
        "int",
    ),
    ConfigSpec(
        "compaction_summary_max_tokens",
        "Summary max tokens",
        "compaction",
        "int",
    ),
    ConfigSpec("api_max_retries", "API retries", "advanced", "int"),
    ConfigSpec("goal_max_attempts", "Goal attempts", "advanced", "int"),
    ConfigSpec(
        "openrouter_cache_ttl",
        "OpenRouter cache TTL",
        "advanced",
        "select",
        choices=("5m", "1h"),
    ),
    ConfigSpec(
        "openrouter_anthropic_messages",
        "Anthropic Messages endpoint",
        "advanced",
        "bool",
    ),
)


def _specs(section: str) -> list[ConfigSpec]:
    return [spec for spec in SPECS if spec.section == section]


def _visible_specs(context: CommandContext, section: str) -> list[ConfigSpec]:
    provider_id = context.runtime.config.model.provider_id
    provider_scopes = {
        "api_key": {"openrouter", "opencode"},
        "openai_base_url": {"openrouter"},
        "openrouter_session_id": {"openrouter"},
        "openrouter_cache_ttl": {"openrouter"},
        "openrouter_anthropic_messages": {"openrouter"},
    }
    return [
        spec
        for spec in _specs(section)
        if spec.key not in provider_scopes
        or provider_id in provider_scopes[spec.key]
    ]


def _current_value(context: CommandContext, spec: ConfigSpec) -> Any:
    if spec.key == "auto_save_threads" and context.session is not None:
        return context.session.runtime_config.settings.auto_save_threads
    if spec.key == "provider_id":
        return context.runtime.config.model.provider_id
    if spec.key == "api_key":
        return context.runtime.providers.active().is_authenticated()
    if spec.key in {
        "model_name",
        "reflection_model_name",
        "reasoning_effort",
    }:
        return getattr(context.runtime.config.model, spec.key)
    settings = context.runtime.settings
    return getattr(settings, spec.key)


def _display_value(context: CommandContext, spec: ConfigSpec) -> str:
    value = _current_value(context, spec)
    if spec.kind == "bool":
        return "on" if value else "off"
    if spec.kind == "secret":
        return "set" if value else "missing"
    if value is None or value == "":
        return "unset"
    return str(value)


async def run_config_flow(context: CommandContext) -> None:
    while True:
        section = await context.ui.choose(
            "configuration",
            [MenuItem(section_id, title) for section_id, title in SECTIONS]
            + [MenuItem("view", "View current values"), MenuItem("done", "Done")],
        )
        if section is None or section == "done":
            return
        if section == "view":
            context.renderer.notice("configuration", *_config_lines(context))
            continue
        await _run_section(context, section)


async def _run_section(context: CommandContext, section: str) -> None:
    while True:
        specs = _visible_specs(context, section)
        selected = await context.ui.choose(
            dict(SECTIONS).get(section, section),
            [
                MenuItem(
                    spec.key,
                    spec.label,
                    suffix=_display_value(context, spec),
                )
                for spec in specs
            ]
            + [MenuItem("back", "Back")],
            hint="Up/Down select · Left/Right toggle · Enter edit · Esc back",
            horizontal_keys=frozenset(
                spec.key for spec in specs if spec.kind == "bool"
            ),
        )
        if selected is None or selected == "back":
            return
        spec = next(spec for spec in specs if spec.key == selected)
        await _edit(context, spec)


async def _edit(context: CommandContext, spec: ConfigSpec) -> None:
    current = _current_value(context, spec)
    value: Any
    if spec.kind == "bool":
        value = not bool(current)
    elif spec.kind == "provider":
        value = await context.ui.choose(
            "model provider",
            [
                MenuItem(
                    provider_id,
                    context.runtime.providers.get(provider_id).display_name,
                )
                for provider_id in context.runtime.providers.provider_ids()
            ],
            initial_key=str(current),
        )
        if value is None:
            return
    elif spec.kind == "model":
        try:
            models = await context.runtime.providers.models(refresh=False)
        except Exception as error:
            context.renderer.error(f"Model catalog unavailable: {error}")
            return
        if not models:
            context.renderer.error("The active provider returned no models.")
            return
        items = _model_items(context, spec, models)

        async def refresh_items() -> list[MenuItem]:
            refreshed = await context.runtime.providers.models(refresh=True)
            return _model_items(context, spec, refreshed)

        value = await context.ui.choose(
            spec.label,
            items,
            initial_key=str(current),
            hint="Type to filter · Up/Down select · Enter confirm · Esc cancel",
            filterable=True,
            refresh=refresh_items,
        )
        if value is None:
            return
        if spec.optional and value == "":
            value = None
    elif spec.kind == "reasoning":
        info = context.runtime.providers.active().model_info(
            context.runtime.config.model.model_name
        )
        choices = info.reasoning_efforts if info else ()
        choices = choices or ("none", "low", "medium", "high", "xhigh", "max")
        value = await context.ui.choose(
            spec.label,
            [MenuItem(choice, choice) for choice in choices],
            initial_key=str(current or "none"),
        )
        if value is None:
            return
        if value == "none":
            value = None
    elif spec.kind == "select":
        value = await context.ui.choose(
            spec.label,
            [MenuItem(choice, choice) for choice in spec.choices],
            initial_key=str(current),
        )
        if value is None:
            return
    else:
        default = "" if spec.kind == "secret" or spec.optional else str(current or "")
        label = spec.label
        if spec.optional and spec.kind != "secret":
            label += f" (current: {current or 'unset'}; blank clears)"
        value = await context.ui.request_input(
            label,
            default=default,
            secret=spec.kind == "secret",
        )
        if value is None:
            return
        value = value.strip()
        if not value and spec.optional:
            value = None
        elif spec.kind == "int":
            try:
                value = int(value)
            except ValueError:
                context.renderer.error(f"{spec.label} must be an integer.")
                return
        elif spec.kind == "float":
            try:
                value = float(value)
            except ValueError:
                context.renderer.error(f"{spec.label} must be a number.")
                return

    if value is None and not spec.optional and spec.kind not in {"reasoning"}:
        return
    try:
        update = await context.runtime.apply_config(
            ConfigPatch({spec.key: value}),
            selected=context.session,
        )
    except (ConfigApplyError, TypeError, ValueError) as error:
        context.renderer.error(str(error))
        return
    if spec.key == "auto_save_threads":
        state = _display_value(context, spec)
        context.renderer.notice(
            "configuration",
            f"Thread autosave is {state}. New threads use the same setting.",
        )
    else:
        context.renderer.notice("configuration", update.message)


def _model_items(
    context: CommandContext,
    spec: ConfigSpec,
    models: tuple[ModelInfo, ...],
) -> list[MenuItem]:
    current = str(_current_value(context, spec) or "")
    provider = context.runtime.providers.active()
    provider_id = context.runtime.config.model.provider_id
    provider_name = str(getattr(provider, "display_name", provider_id))
    items: list[MenuItem] = []
    for model in models:
        metadata = ["vision" if model.supports_vision else "text"]
        if model.context_window:
            metadata.append(f"{model.context_window:,} context")
        if model.input_price is not None and model.output_price is not None:
            metadata.append(
                f"${model.input_price:g}/${model.output_price:g} per 1M in/out"
            )
        display_name = model.name if model.name != model.id else ""
        if display_name:
            metadata.insert(0, display_name)
        items.append(
            MenuItem(
                model.id,
                model.id,
                description=" · ".join(metadata),
                suffix="current" if model.id == current else "",
                search_terms=(model.name, provider_name, provider_id),
            )
        )
    if spec.optional:
        items.insert(
            0,
            MenuItem(
                "",
                "Use provider default",
                suffix="current" if not current else "",
            ),
        )
    return items


def _config_lines(context: CommandContext) -> list[str]:
    lines: list[str] = []
    for section_id, title in SECTIONS:
        lines.append(title)
        for spec in _visible_specs(context, section_id):
            lines.append(f"  {spec.label:<30} {_display_value(context, spec)}")
    return lines
