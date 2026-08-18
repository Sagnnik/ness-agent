"""Declarative registry of /config-editable adapter settings.

Each :class:`ConfigSpec` describes one ``Settings`` field: which /config
section it belongs to, which input widget edits it (bool toggle, text or
number form, literal select, model picker), where it persists
(``configs.json`` vs ``secrets.json``), and what side-effects a change has
(model rebuild, live session-options sync, or deferred to new sessions).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ness_cli.config import Settings, settings

# (section id, menu title) in display order.
CONFIG_SECTIONS: tuple[tuple[str, str], ...] = (
    ("provider", "Provider"),
    ("model", "Model"),
    ("behavior", "Behavior"),
    ("compaction", "Compaction"),
    ("advanced", "Advanced"),
)

SECTION_DESCRIPTIONS: dict[str, str] = {
    "provider": "provider credentials, exa, endpoint and cache options",
    "model": "chat model, reasoning, reflection, judge",
    "behavior": "approval, autosave, reflection, formatting",
    "compaction": "budget, reserves, ratio",
    "advanced": "retries, goal attempts",
}


@dataclass(frozen=True)
class ConfigSpec:
    """One /config-editable setting."""

    key: str  # Settings field name; also the JSON key
    label: str  # menu row label
    section: str  # CONFIG_SECTIONS id
    kind: str  # "bool" | "text" | "number" | "select" | "model" | "secret" | "reasoning"
    example: str = ""  # gray hint rendered above the form field
    choices: tuple[str, ...] = ()  # literal options for "select"
    secret: bool = False  # persist to secrets.json (0600)
    optional: bool = False  # empty form input clears the value
    rebuild: bool = False  # rebuild the model/graph after a change
    session_update: bool = False  # sync live session options after a change
    deferred: bool = False  # change only applies to new sessions


CONFIG_SPECS: tuple[ConfigSpec, ...] = (
    # --- Provider ---------------------------------------------------------
    ConfigSpec(
        "openai_api_key",
        "Provider API key",
        "provider",
        "secret",
        example="e.g. sk-or-v1-...",
        secret=True,
        rebuild=True,
    ),
    ConfigSpec(
        "opencode_api_key",
        "OpenCode Go API key",
        "provider",
        "secret",
        example="e.g. sk-...",
        secret=True,
        rebuild=True,
    ),
    ConfigSpec(
        "openai_base_url",
        "Base URL",
        "provider",
        "text",
        example="e.g. https://openrouter.ai/api/v1 (empty = provider default)",
        optional=True,
        rebuild=True,
    ),
    ConfigSpec(
        "exa_api_key",
        "Exa API key (web search)",
        "provider",
        "secret",
        example="e.g. exa-...",
        secret=True,
    ),
    ConfigSpec(
        "openrouter_session_id",
        "OpenRouter session id",
        "provider",
        "text",
        example="e.g. my-project (empty = per-thread id)",
        optional=True,
        rebuild=True,
    ),
    ConfigSpec(
        "openrouter_cache_ttl",
        "Prompt cache TTL",
        "provider",
        "select",
        choices=("5m", "1h"),
        rebuild=True,
    ),
    # --- Model ------------------------------------------------------------
    ConfigSpec("model_name", "Chat model", "model", "model", rebuild=True),
    ConfigSpec("reasoning_effort", "Reasoning effort", "model", "reasoning", rebuild=True),
    ConfigSpec("reflection_model_name", "Reflection model", "model", "model", rebuild=True),
    ConfigSpec(
        "goal_judge_model",
        "Goal judge model",
        "model",
        "model",
        optional=True,
        deferred=True,
    ),
    # --- Behavior ---------------------------------------------------------
    ConfigSpec("enable_approval", "Tool approval", "behavior", "bool", session_update=True),
    ConfigSpec("auto_save_threads", "Thread autosave", "behavior", "bool", session_update=True),
    ConfigSpec(
        "session_end_reflection", "Session end reflection", "behavior", "bool", session_update=True
    ),
    ConfigSpec("format_on_write", "Format on write", "behavior", "bool", deferred=True),
    ConfigSpec(
        "openrouter_anthropic_messages",
        "Anthropic messages endpoint",
        "behavior",
        "bool",
        rebuild=True,
    ),
    # --- Compaction -------------------------------------------------------
    ConfigSpec(
        "reflection_token_ratio",
        "Reflection token ratio",
        "compaction",
        "number",
        example="e.g. 0.4",
        deferred=True,
    ),
    ConfigSpec(
        "compaction_token_budget",
        "Token budget fallback",
        "compaction",
        "number",
        example="e.g. 120000",
        deferred=True,
    ),
    ConfigSpec(
        "compaction_buffer_tokens",
        "Compaction buffer",
        "compaction",
        "number",
        example="e.g. 16384",
        deferred=True,
    ),
    ConfigSpec(
        "compaction_summary_max_tokens",
        "Summary max tokens",
        "compaction",
        "number",
        example="e.g. 4096",
        deferred=True,
    ),
    # --- Advanced -----------------------------------------------------------
    ConfigSpec(
        "api_max_retries",
        "API max retries",
        "advanced",
        "number",
        example="e.g. 3",
        rebuild=True,
    ),
    ConfigSpec(
        "goal_max_attempts",
        "Goal max attempts",
        "advanced",
        "number",
        example="e.g. 3",
        deferred=True,
    ),
)

SPEC_BY_KEY: dict[str, ConfigSpec] = {spec.key: spec for spec in CONFIG_SPECS}

SECTION_TITLES: dict[str, str] = dict(CONFIG_SECTIONS)

# Form kinds (Settings field names) whose input is password-masked.
SECRET_FORM_KINDS: frozenset[str] = frozenset(
    spec.key for spec in CONFIG_SPECS if spec.kind == "secret"
)


def specs_for_section(section: str) -> list[ConfigSpec]:
    specs = [spec for spec in CONFIG_SPECS if spec.section == section]
    openrouter_only = {
        "openai_api_key",
        "openai_base_url",
        "openrouter_session_id",
        "openrouter_cache_ttl",
        "openrouter_anthropic_messages",
    }
    opencode_only = {"opencode_api_key"}
    if settings.model_provider != "openrouter":
        specs = [spec for spec in specs if spec.key not in openrouter_only]
    if settings.model_provider != "opencode":
        specs = [spec for spec in specs if spec.key not in opencode_only]
    return specs


def format_current(spec: ConfigSpec) -> str:
    """Short current-value text shown in the section menu row."""
    value = getattr(settings, spec.key, None)
    if spec.kind == "bool":
        return "on" if value else "off"
    if spec.secret:
        return "set" if value else "missing"
    if value is None or value == "":
        return "(unset)"
    return str(value)


def parse_number(spec: ConfigSpec, text: str) -> Any:
    """Parse form input for a ``number`` spec using the Settings field type.

    Raises ``ValueError`` on invalid input.
    """
    annotation = Settings.model_fields[spec.key].annotation
    if annotation is float:
        return float(text)
    return int(text)
