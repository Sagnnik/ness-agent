from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# USD per 1M tokens: (input_per_1m, output_per_1m, cache_read_ratio).
# Keys match case-insensitive substrings; the longest matching key wins.
PricingDict = dict[str, tuple[float, float, float]]


@dataclass
class TracingConfig:
    """Configuration for ness_agent tracing and cost tracking.

    ``enabled`` gates whether ``build_tracer`` constructs a real tracer.
    When ``False`` (or ``exporter == "none"``) a :class:`NoopTracer` is
    returned and no exporter is initialised.

    ``pricing`` is an optional :data:`PricingDict` consumed by the
    :class:`~ness_agent.tracing.cost.CostTracker` to estimate USD cost
    when the provider does not return one in ``response_metadata``.
    ``capture_tool_args`` records tool call arguments (truncated) on the
    tool-execution span — disabled by default to avoid leaking secrets.

    ``capture_messages`` records the full conversation content (system prompt,
    user messages, AI responses, tool results) on LLM-boundary spans via the
    OTel GenAI ``gen_ai.prompt`` / ``gen_ai.completion`` / ``gen_ai.tool.call.*``
    attributes. Messages may contain PII and bloat OTLP payloads, so this is
    opt-in only. When enabled, tool-call arguments and results are emitted as
    JSON strings, with results truncated to ``max_message_length`` characters
    to prevent span rejection by OTel backends.
    """

    enabled: bool = False
    exporter: Literal["otlp", "console", "none"] = "none"
    endpoint: str | None = None
    headers: dict[str, str] | None = None
    service_name: str = "ness-agent"
    resource_attrs: dict[str, str] = field(default_factory=dict)
    capture_tool_args: bool = False
    capture_messages: bool = False
    max_message_length: int = 10000
    pricing: PricingDict | None = None
