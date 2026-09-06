from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict

from langchain_core.messages import BaseMessage

CompactionTrigger = Literal["automatic", "manual", "safety"]
CompactionSkipReason = Literal[
    "retry_suppressed",
    "no_completed_history",
    "disabled",
    "failed",
]
CompactionNoticeReason = Literal[
    "pre_act_hard_threshold",
    "pre_act_checkpoint",
]
CompactionBridgeReason = CompactionTrigger | CompactionSkipReason | CompactionNoticeReason


class CompactionStatus(TypedDict, total=False):
    """Graph-state snapshot written by ``context_gate`` each boundary."""

    compacted: bool
    token_count: int
    ratio: float
    context_limit: int
    overlay_note: str
    trigger: CompactionTrigger
    skip_reason: CompactionSkipReason
    forced: bool
    error: str
    after_tokens: int
    active_suffix_messages: int


class CompactionBridgeEvent(TypedDict, total=False):
    """Payload for ``config._compaction_bridge`` and ``SessionEvent("compaction")``."""

    status: Literal["success", "failed"]
    trigger: CompactionTrigger
    skip_reason: CompactionSkipReason
    notice_reason: CompactionNoticeReason
    forced: bool
    info: str
    before_tokens: int
    after_tokens: int
    active_suffix_messages: int
    advisory: bool

COMPACTION_WARN_RATIO = 0.70
COMPACTION_SUMMARY_RATIO = 0.80
COMPACTION_HARD_RATIO = 0.92
COMPACTION_ACTIVE_TURN_RATIO = 0.40
COMPACTION_ACTIVE_TURN_MIN_TOKENS = 8_000
COMPACTION_ACTIVE_TURN_MAX_TOKENS = 65_000

# Fallback allowance per image when provider usage is unavailable for a slice.
# Actual cost varies by model, resolution, and detail; encoded byte size is not
# a token count. Keep this separate from the safe display placeholder.
IMAGE_TOKEN_ALLOWANCE = 4_096
_IMAGE_BLOCK_TYPES = frozenset({"image_url", "image", "input_image"})


def content_text(content) -> str:
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") in _IMAGE_BLOCK_TYPES:
                    parts.append("[image]")
                elif "text" in item:
                    parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return " ".join(parts)
    return str(content)


def estimate_tokens(messages: list[BaseMessage]) -> int:
    text = "\n\n".join(f"{m.type}: {content_text(m.content)}" for m in messages)
    symbols = sum(1 for char in text if not char.isalnum() and not char.isspace())
    image_count = sum(
        1
        for message in messages
        if isinstance(message.content, list)
        for block in message.content
        if isinstance(block, dict) and block.get("type") in _IMAGE_BLOCK_TYPES
    )
    return max(
        1,
        len(text) // 3 + symbols // 2 + 6 * len(messages)
        + image_count * IMAGE_TOKEN_ALLOWANCE,
    )


def resolve_token_count(
    messages: list[BaseMessage], *, known_input_tokens: int | None = None
) -> int:
    if known_input_tokens and known_input_tokens > 0:
        return int(known_input_tokens)
    return estimate_tokens(messages)


def context_limit(options=None) -> int:
    if options is None:
        return 120_000
    return int(options.context_window or options.compaction_token_budget)


def resolve_usable_context_budget(model_name: str = "", options=None) -> int:
    """Compatibility name for the model's configured context limit."""
    _ = model_name
    return context_limit(options)


@dataclass(frozen=True)
class ContextPressure:
    token_count: int
    context_limit: int
    ratio: float
    warning: bool
    should_compact: bool
    safety_threshold_reached: bool
    hard_threshold_reached: bool

    @property
    def usable_budget(self) -> int:
        return self.context_limit


def calculate_context_pressure(
    messages: list[BaseMessage],
    *,
    known_input_tokens: int | None = None,
    options=None,
    max_tokens: int | None = None,
    **_ignored,
) -> ContextPressure:
    limit = int(max_tokens or context_limit(options))
    used = resolve_token_count(messages, known_input_tokens=known_input_tokens)
    ratio = used / limit if limit > 0 else 1.0
    buffer_tokens = int(getattr(options, "compaction_buffer_tokens", 16_384))
    safety_at = max(1, limit - buffer_tokens)
    return ContextPressure(
        token_count=used,
        context_limit=limit,
        ratio=ratio,
        warning=ratio >= COMPACTION_WARN_RATIO,
        should_compact=ratio >= COMPACTION_SUMMARY_RATIO or used >= safety_at,
        safety_threshold_reached=used >= safety_at,
        hard_threshold_reached=ratio >= COMPACTION_HARD_RATIO,
    )

def pressure_note(
    pressure: ContextPressure,
    *,
    compacted: bool = False,
    had_stored_compaction: bool = False,
) -> str:
    if compacted:
        return (
            "Conversation was summarized at this model boundary. "
            "A coherent recent continuation was retained verbatim; re-read files if needed."
        )
    parts: list[str] = []
    if had_stored_compaction:
        parts.append("Conversation includes a compacted-history summary.")
    if pressure.warning:
        parts.append(
            f"Context ~{pressure.token_count:,} tokens "
            f"({pressure.ratio:.0%} of the configured context limit). "
            + ("Summary compaction is due." if pressure.should_compact else "Summary compaction may begin soon.")
        )
    return " ".join(parts)
