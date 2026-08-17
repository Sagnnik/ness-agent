from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from ness_cli.tui import render
from ness_cli.tui.config_registry import (
    CONFIG_SECTIONS,
    SECTION_TITLES,
    SPEC_BY_KEY,
    ConfigSpec,
    format_current,
    parse_number,
    specs_for_section,
)
from ness_cli.config_store import write_config, write_secret
from ness_cli.chat_model import (
    active_reasoning_effort,
    set_active_model,
    set_active_reasoning_effort,
)
from ness_cli.config import reasoning_efforts_for_model, reload_settings, settings
from ness_cli.provider.openrouter.catalog import refresh_catalog
from ness_cli.provider.profile import update_provider_profile
from ness_cli.provider.registry import active_provider


# --- shared /config data + delegator ---------------------------------------
@dataclass
class ConfigResult:
    rebuild: bool = False
    session_update: bool = False
    messages: list[str] = field(default_factory=list)

    def note(self, message: str) -> None:
        self.messages.append(message)

    def mark_session_update(self) -> None:
        self.session_update = True


def note_model_reasoning_changes(
    result: ConfigResult,
    *,
    model_changed: bool,
    model_name: str,
    reasoning_changed: bool,
    reasoning: str,
) -> None:
    parts: list[str] = []
    if model_changed:
        parts.append(f"Model switched to {model_name}")
    if reasoning_changed:
        parts.append(f"Reasoning effort set to {reasoning}")
    if parts:
        result.note("; ".join(parts) + ".")


def current_config_lines() -> list[str]:
    lines: list[str] = []
    for section_id, title in CONFIG_SECTIONS:
        lines.append(f"{title}:")
        for spec in specs_for_section(section_id):
            lines.append(f"  {spec.key}  {format_current(spec)}")
    return lines


async def run_config() -> ConfigResult:
    """Run the active TUI config picker; fall back to a notice when headless."""
    sink = render.get_sink()
    if sink is None:
        result = ConfigResult()
        result.note("/config is available only in the interactive TUI.")
        return result
    return await sink.run_config()


# --- interactive /config picker + forms -------------------------------------
class ConfigFlowMixin:
    """Interactive /config sections, pickers, toggles, and entry forms."""

    async def run_config(self) -> ConfigResult:
        if self._config_future is not None and not self._config_future.done():
            return await self._config_future
        self._config_future = asyncio.get_running_loop().create_future()
        self._config_result = ConfigResult()
        self._model_pick_changed = False
        self._model_pick_name = ""
        self._config_section = ""
        self._config_select_key = ""
        self._config_model_target = "model_name"
        self._open_picker("config_action", "/config", index=0)
        result = await self._config_future
        self._config_future = None
        self._config_result = None
        return result

    def _finish_config(self) -> None:
        if self._config_future is not None and not self._config_future.done():
            self._config_future.set_result(self._config_result or ConfigResult())
        self._config_model_target = "model_name"
        self._config_select_key = ""
        self._close_menu()
        self._close_form(reset_buffer=True)
        self._reset_buffer()

    def _config_note(self, message: str) -> None:
        if self._config_result is not None:
            self._config_result.note(message)

    def _config_rebuild(self) -> None:
        if self._config_result is not None:
            self._config_result.rebuild = True

    def _config_session_update(self) -> None:
        if self._config_result is not None:
            self._config_result.mark_session_update()

    def _apply_spec_flags(self, spec: ConfigSpec) -> None:
        if spec.rebuild:
            self._config_rebuild()
        if spec.session_update:
            self._config_session_update()

    def _persist_spec(self, spec: ConfigSpec, value: Any) -> None:
        if spec.secret:
            write_secret(spec.key, value)
        else:
            write_config(spec.key, value)

    # --- top-level sections ------------------------------------------------
    def _apply_config_action(self, key: str) -> None:
        if key == "view":
            self._config_note("\n".join(current_config_lines()))
            self._finish_config()
            return
        if key in SECTION_TITLES:
            self._config_section = key
            self._open_picker("config_section", "/config", index=0)

    def _apply_config_section_item(self, key: str) -> None:
        spec = SPEC_BY_KEY.get(key)
        if spec is None:
            self._finish_config()
            return
        if spec.kind == "bool":
            self._toggle_config_bool(spec)
            return
        if spec.kind == "reasoning":
            self._config_model_target = "model_name"
            self._open_config_reasoning_picker()
            return
        if spec.kind == "model":
            self._config_model_target = spec.key
            current = self._active_config_model_slug()
            index = next(
                (i for i, item in enumerate(self._config_model_items()) if item.key == current),
                0,
            )
            self._open_picker("config_models", "", index=index)
            if self._catalog_refresh_task is None or self._catalog_refresh_task.done():
                self._catalog_refresh_task = asyncio.create_task(
                    self._refresh_model_catalog_menu()
                )
            return
        if spec.kind == "select":
            self._config_select_key = spec.key
            items = self._config_select_items()
            current = str(getattr(settings, spec.key))
            index = next((i for i, item in enumerate(items) if item.key == current), 0)
            self._open_picker("config_select", "/config", index=index)
            return
        # text / number / secret entry form
        self._open_form(spec)

    # --- bool toggle (Enter or left/right arrows) ---------------------------
    def _toggle_config_bool(self, spec: ConfigSpec) -> None:
        new_value = not bool(getattr(settings, spec.key))
        setattr(settings, spec.key, new_value)
        self._persist_spec(spec, new_value)
        self._apply_spec_flags(spec)
        self.invalidate()

    def _toggle_menu_value(self) -> None:
        """Left/right arrow toggle for bool rows in the open section menu."""
        if self._menu_kind != "config_section":
            return
        items = self._visible_menu_items()
        if not (0 <= self._menu_index < len(items)):
            return
        spec = SPEC_BY_KEY.get(items[self._menu_index].key)
        if spec is not None and spec.kind == "bool":
            self._toggle_config_bool(spec)

    async def _refresh_model_catalog_menu(self) -> None:
        selected_model: str | None = None
        if self._menu_kind == "config_models":
            items = self._visible_menu_items()
            if 0 <= self._menu_index < len(items):
                selected_model = items[self._menu_index].key
        error: str | None = None
        if settings.model_provider == "openrouter":
            result = await refresh_catalog()
            error = result.error
        else:
            try:
                await active_provider().models(refresh=True)
            except Exception as exc:
                error = str(exc)
        if error and not self._catalog_refresh_warned:
            self._catalog_refresh_warned = True
            self.append_warning(
                f"Model catalog refresh failed; using cached fallback: {error}"
            )
        if self._menu_kind == "config_models":
            if selected_model is not None:
                refreshed = self._visible_menu_items()
                self._menu_index = next(
                    (
                        index
                        for index, item in enumerate(refreshed)
                        if item.key == selected_model
                    ),
                    self._menu_index,
                )
            self._clamp_menu_index()
            self.invalidate()

    def _open_config_reasoning_picker(self) -> None:
        model_name = self.coding.model_name
        efforts = reasoning_efforts_for_model(model_name)
        if not efforts:
            self._config_note("Current model does not support reasoning effort.")
            self._finish_config()
            return
        current_effort = self.coding.reasoning_effort
        index = next((i for i, item in enumerate(self._config_reasoning_items()) if item.key == current_effort), 0)
        self._open_picker("config_reasoning", "/config", index=index)

    def _apply_config_model(self, model_name: str) -> None:
        target = self._config_model_target or "model_name"
        if target != "model_name":
            # reflection / goal-judge model: persist only, no reasoning chain.
            spec = SPEC_BY_KEY[target]
            if model_name == "":
                self._persist_spec(spec, None)
                update_provider_profile(settings.model_provider, {target: None})
                reload_settings()
                self._config_note(f"{spec.label} cleared.")
            else:
                self._persist_spec(spec, model_name)
                update_provider_profile(settings.model_provider, {target: model_name})
                reload_settings()
                self._config_note(f"{spec.label} set to {model_name}.")
            self._apply_spec_flags(spec)
            self._config_model_target = "model_name"
            self._finish_config()
            return
        current = self._current_model_slug()
        self._model_pick_changed = model_name != current
        self._model_pick_name = model_name
        coerced: str | None = None
        if self._model_pick_changed:
            coerced = set_active_model(model_name)
            write_config("model_name", model_name)
            update_provider_profile(
                settings.model_provider,
                {
                    "model_name": model_name,
                    "reasoning_effort": coerced or active_reasoning_effort(),
                },
            )
            if coerced:
                write_config("reasoning_effort", coerced)
            self._config_rebuild()
            self._config_session_update()
        efforts = reasoning_efforts_for_model(model_name)
        if not efforts:
            if self._config_result is not None:
                note_model_reasoning_changes(
                    self._config_result,
                    model_changed=self._model_pick_changed,
                    model_name=self._model_pick_name,
                    reasoning_changed=bool(coerced),
                    reasoning=coerced or active_reasoning_effort(),
                )
            self._finish_config()
            return
        self._open_config_reasoning_picker()

    def _apply_config_reasoning(self, effort: str) -> None:
        current = self.coding.reasoning_effort
        reasoning_changed = effort != current
        if reasoning_changed:
            set_active_reasoning_effort(effort)  # type: ignore[arg-type]
            write_config("reasoning_effort", effort)
            update_provider_profile(settings.model_provider, {"reasoning_effort": effort})
            self._config_rebuild()
        if self._config_result is not None:
            note_model_reasoning_changes(
                self._config_result,
                model_changed=self._model_pick_changed,
                model_name=self._model_pick_name,
                reasoning_changed=reasoning_changed,
                reasoning=effort,
            )
        self._finish_config()

    def _apply_config_select(self, value: str) -> None:
        key = self._config_select_key
        self._config_select_key = ""
        spec = SPEC_BY_KEY.get(key)
        if spec is None:
            self._finish_config()
            return
        setattr(settings, spec.key, value)
        self._persist_spec(spec, value)
        self._apply_spec_flags(spec)
        self._config_note(f"{spec.label} set to {value}.")
        self._finish_config()

    # --- text / number / secret entry forms ---------------------------------
    def _open_form(self, spec: ConfigSpec) -> None:
        self._close_menu()
        self._form_kind = spec.key
        self._form_label = spec.label
        self._form_example = spec.example
        self._form_buffer.text = ""
        self._form_buffer.cursor_position = 0
        self._set_buffer_text("/config")
        self._focus_form_field()
        self.invalidate()

    def _close_form(self, *, reset_buffer: bool = True) -> None:
        had_form = self._form_kind is not None
        self._form_kind = None
        self._form_label = ""
        self._form_example = ""
        self._form_buffer.text = ""
        if had_form and reset_buffer:
            self._reset_buffer()
        self._focus_command_input()

    def _submit_form(self) -> None:
        kind = self._form_kind
        if kind is None:
            return
        if getattr(self, "_prompt_kind", None) == "secret":
            value = self._form_buffer.text.strip()
            if self._prompt_future is not None and not self._prompt_future.done():
                self._prompt_future.set_result(value)
            self._close_form(reset_buffer=False)
            return
        spec = SPEC_BY_KEY.get(kind)
        if spec is None:
            return
        value = self._form_buffer.text.strip()
        if not value:
            if spec.optional:
                self._persist_spec(spec, None)
                reload_settings()
                self._apply_spec_flags(spec)
                self._config_note(f"{spec.label} cleared.")
                self._finish_config()
            else:
                self.append_error(f"{spec.label} cannot be empty.")
            return
        parsed: Any = value
        if spec.kind == "number":
            try:
                parsed = parse_number(spec, value)
            except ValueError:
                self.append_error(f"{spec.label} must be a number ({spec.example}).")
                return
        self._persist_spec(spec, parsed)
        reload_settings()
        self._apply_spec_flags(spec)
        note = f"{spec.label} saved to {'secrets' if spec.secret else 'configs'}.json"
        if spec.deferred:
            note += " (applies to new sessions)"
        self._config_note(note + ".")
        self._finish_config()
