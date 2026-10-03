from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class NessAgentOptions:
    """Runtime knobs including context limit and cache-safe compaction buffer."""

    context_window: int | None = None
    compaction_token_budget: int = 120_000
    compaction_buffer_tokens: int = 16_384
    compaction_summary_max_tokens: int = 4_096
    enable_approval: bool = True
    yolo_mode: bool = False
    auto_save_threads: bool = True
    reflection_token_ratio: float = 0.0
    session_end_reflection: bool = False
    format_on_write: bool = True
    exa_api_key: str | None = None
    project_root: Path | None = None
    ness_dir: Path | None = None
    shell_default_timeout: float = 30.0
    shell_max_timeout: float | None = None
    # AIMessage cap text injected on a pure-LLM cancel with no partial text or
    # pending tool calls, so the model does not silently resume the abandoned
    # request next turn
    interruption_marker: str = ("… [turn interrupted by user] ")
    # LangGraph recursion_limit for Session.run / Session.stream turns.
    recursion_limit: int = 500

    def __post_init__(self) -> None:
        for name in ("shell_default_timeout", "shell_max_timeout"):
            value = getattr(self, name)
            if value is None and name == "shell_max_timeout":
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive finite number")
        if self.context_window is not None and self.context_window <= 0:
            raise ValueError("context_window must be positive when set")
        if self.compaction_token_budget <= 0:
            raise ValueError("compaction_token_budget must be positive")
        if self.compaction_buffer_tokens <= 0:
            raise ValueError("compaction_buffer_tokens must be positive")
        if self.compaction_summary_max_tokens <= 0:
            raise ValueError("compaction_summary_max_tokens must be positive")
        if self.compaction_summary_max_tokens >= self.compaction_buffer_tokens:
            raise ValueError(
                "compaction_summary_max_tokens must be smaller than compaction_buffer_tokens"
            )
        if not 0.0 <= self.reflection_token_ratio <= 1.0:
            raise ValueError("reflection_token_ratio must be between 0 and 1")
        if self.recursion_limit < 1:
            raise ValueError("recursion_limit must be at least 1")
        limit = self.context_window or self.compaction_token_budget
        if limit <= self.compaction_buffer_tokens:
            raise ValueError("context limit must be larger than compaction_buffer_tokens")


@dataclass
class MemoryConfig:
    disabled: bool = False
    project_memory: Path | None = None
    user_memory: Path | None = None
    session_memory_dir: Path | None = None


@dataclass
class ModeConfig:
    """Optional plan/act."""
    default: str = "act"
    plans_dir: Path | None = None
    plan_mode_template: str | None = None
    act_mode_template: str | None = None
    plan_mode_readonly: bool = True


@dataclass
class SubagentConfig:
    """Optional parallel sub-agent tasks (spawn_subagent tool uses this)."""
    prompt_template: str | None = None   # default subagent prompt body
    max_parallel: int = 3
    default_tools: tuple[str, ...] = ("read", "grep", "glob", "web_search", "fetch_url")
    default_timeout_seconds: int = 300


@dataclass
class PermissionRules:
    allow: list[str] = field(default_factory=list)
    deny: list[str] = field(default_factory=list)
    ask: list[str] = field(default_factory=lambda: ["*"])
