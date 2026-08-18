"""Tests for the /config UI flows: sections, toggles, forms, selects, model targets."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from ness_cli.config import settings
from ness_cli.config_store import load_configs, load_secrets
from ness_cli.provider.openrouter.catalog import RefreshResult
from ness_cli.tui.config_flow import current_config_lines
from ness_cli.tui.config_registry import (
    CONFIG_SECTIONS,
    specs_for_section,
)


@pytest.fixture
def config_dir(tmp_path: Path, monkeypatch) -> Path:
    cfg = tmp_path / "cfg"
    monkeypatch.setenv("NESS_AGENT_CONFIG_DIR", str(cfg))
    return cfg


def _open_section(app, section: str, key: str) -> None:
    app._config_section = section
    app._open_picker("config_section", "/config", index=0)
    items = app._visible_menu_items()
    app._menu_index = next(i for i, item in enumerate(items) if item.key == key)


def test_action_menu_lists_sections(make_app):
    app = make_app()
    items = app._config_action_items()
    keys = [item.key for item in items]
    assert keys == ["provider", "model", "behavior", "compaction", "advanced", "view"]


def test_section_menu_shows_current_values(make_app):
    app = make_app()
    app._config_section = "behavior"
    items = app._config_section_items()
    approval = next(item for item in items if item.key == "enable_approval")
    assert approval.suffix == ("on" if settings.enable_approval else "off")


def test_bool_toggle_via_arrow_keys(make_app, config_dir):
    app = make_app()
    previous = settings.auto_save_threads
    settings.auto_save_threads = True
    try:
        _open_section(app, "behavior", "auto_save_threads")
        app._toggle_menu_value()
        assert settings.auto_save_threads is False
        assert load_configs()["auto_save_threads"] is False
        # Menu stays open so several toggles can be flipped in one visit.
        assert app._menu_kind == "config_section"
        # Toggling a non-bool row is a no-op.
        _open_section(app, "provider", "openai_api_key")
        app._toggle_menu_value()
        assert load_secrets() == {}
    finally:
        settings.auto_save_threads = previous


def test_number_form_persists_parsed_value(make_app, config_dir):
    app = make_app()
    previous = settings.api_max_retries
    try:
        _open_section(app, "advanced", "api_max_retries")
        app._apply_picker_selection()
        assert app._form_kind == "api_max_retries"
        assert app._form_example == "e.g. 3"
        app._form_buffer.text = "5"
        app._submit_form()
        assert settings.api_max_retries == 5
        assert load_configs()["api_max_retries"] == 5
        assert app._form_kind is None
    finally:
        settings.api_max_retries = previous


def test_number_form_rejects_invalid_input(make_app, config_dir):
    app = make_app()
    previous = settings.api_max_retries
    try:
        _open_section(app, "advanced", "api_max_retries")
        app._apply_picker_selection()
        app._form_buffer.text = "abc"
        app._submit_form()
        # Form stays open with an error; nothing persisted.
        assert app._form_kind == "api_max_retries"
        assert load_configs() == {}
        text = "\n".join(line.text for line in app._lines)
        assert "must be a number" in text
    finally:
        settings.api_max_retries = previous


def test_secret_form_masks_and_persists(make_app, config_dir, monkeypatch):
    # Env wins over JSON; clear so reload_settings picks up the written secret.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    app = make_app()
    previous = settings.openai_api_key
    try:
        _open_section(app, "provider", "openai_api_key")
        app._apply_picker_selection()
        assert app._form_kind == "openai_api_key"
        assert app._form_buffer.password() is True
        app._form_buffer.text = "sk-new-key"
        app._submit_form()
        assert settings.openai_api_key == "sk-new-key"
        assert load_secrets()["openai_api_key"] == "sk-new-key"
    finally:
        settings.openai_api_key = previous


def test_login_secret_prompt_is_masked(make_app):
    async def run():
        app = make_app()
        task = asyncio.create_task(app.ask_secret("OpenRouter API key"))
        await asyncio.sleep(0)
        assert app._form_buffer.password() is True
        app._form_buffer.text = "sk-secret"
        app._submit_form()
        assert await task == "sk-secret"

    asyncio.run(run())


def test_optional_form_empty_input_clears(make_app, config_dir):
    app = make_app()
    previous = settings.openai_base_url
    settings.openai_base_url = "https://example.com/v1"
    try:
        _open_section(app, "provider", "openai_base_url")
        app._apply_picker_selection()
        assert app._form_buffer.password() is False
        app._form_buffer.text = ""
        app._submit_form()
        assert settings.openai_base_url is None
        assert "openai_base_url" not in load_configs()
    finally:
        settings.openai_base_url = previous


def test_select_picker_applies_choice(make_app, config_dir):
    app = make_app()
    previous = settings.openrouter_cache_ttl
    settings.openrouter_cache_ttl = "5m"
    try:
        _open_section(app, "provider", "openrouter_cache_ttl")
        app._apply_picker_selection()
        assert app._menu_kind == "config_select"
        items = app._visible_menu_items()
        assert [item.key for item in items] == ["5m", "1h"]
        current = next(item for item in items if item.key == "5m")
        assert current.suffix == "(current)"
        app._menu_index = next(i for i, item in enumerate(items) if item.key == "1h")
        app._apply_picker_selection()
        assert settings.openrouter_cache_ttl == "1h"
        assert load_configs()["openrouter_cache_ttl"] == "1h"
        assert app._menu_kind is None
    finally:
        settings.openrouter_cache_ttl = previous


def test_model_picker_targets_reflection_model(make_app, config_dir, monkeypatch):
    import ness_cli.tui.config_flow as config_flow

    monkeypatch.setattr(
        config_flow,
        "refresh_catalog",
        AsyncMock(return_value=RefreshResult(refreshed=False, models=0)),
    )
    app = make_app()
    previous = settings.reflection_model_name
    try:
        app._config_section = "model"
        asyncio.run(_async_open(app, "reflection_model_name"))
        assert app._menu_kind == "config_models"
        assert app._config_model_target == "reflection_model_name"
        app._apply_config_model("openai/gpt-4o")
        assert settings.reflection_model_name == "openai/gpt-4o"
        assert load_configs()["reflection_model_name"] == "openai/gpt-4o"
        assert app._config_model_target == "model_name"
        assert app._menu_kind is None
    finally:
        settings.reflection_model_name = previous


async def _async_open(app, key: str) -> None:
    app._apply_config_section_item(key)


def test_model_picker_unset_clears_optional_target(make_app, config_dir):
    app = make_app()
    previous = settings.goal_judge_model
    settings.goal_judge_model = "openai/gpt-4o"
    try:
        app._config_model_target = "goal_judge_model"
        app._open_picker("config_models", "", index=0)
        items = app._visible_menu_items()
        assert items[0].key == ""  # "(unset) - use default"
        app._menu_index = 0
        app._apply_config_model("")
        assert settings.goal_judge_model is None
        assert "goal_judge_model" not in load_configs()
    finally:
        settings.goal_judge_model = previous


def test_form_example_rendered_in_gray_hint(make_app):
    app = make_app()
    _open_section(app, "provider", "openai_base_url")
    app._apply_picker_selection()
    fragments = app._form_example_fragments()
    assert fragments
    style, text = fragments[0]
    assert style == "class:chrome.form.hint"
    assert "https://openrouter.ai/api/v1" in text
    app._close_form()


def test_current_config_lines_cover_all_sections():
    text = "\n".join(current_config_lines())
    for title in ("Provider:", "Model:", "Behavior:", "Compaction:", "Advanced:"):
        assert title in text
    visible_keys = {
        spec.key
        for section, _title in CONFIG_SECTIONS
        for spec in specs_for_section(section)
    }
    for key in visible_keys:
        assert key in text
