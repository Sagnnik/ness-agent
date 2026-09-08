"""Serialisation helpers for capturing conversation content on spans.

OTel span attributes only accept primitive types or homogeneous sequences of
primitives; nested ``list[dict]`` values are *silently dropped* by the SDK.
Conversation content therefore must be JSON-serialised to strings before
being attached to a span.

Backends that parse GenAI spans (Langfuse, Arize Phoenix, Datadog) expect
OpenAI-style ``{"role": ..., "content": ...}`` objects rather than langchain's
``{"type": ...}`` shape. These helpers perform that translation so the chat UI
in those backends renders the conversation correctly.
"""

from __future__ import annotations

import json
from typing import Any, Iterable

from langchain_core.messages import AIMessage, BaseMessage

# Map langchain ``message.type`` to the OpenAI ``role`` vocabulary understood
# by GenAI span parsers (Langfuse, Arize, Datadog).
_ROLE_BY_TYPE: dict[str, str] = {
    "system": "system",
    "human": "user",
    "ai": "assistant",
    "tool": "tool",
    "function": "function",
}


# Content-block types that carry embedded media (images, audio, etc.).
# These are replaced with a short text marker to avoid base64-bloating the
# span attribute payload. Backend chat UIs (Langfuse, Arize) cannot render
# inline images from span attributes anyway.
_MEDIA_BLOCK_TYPES: frozenset[str] = frozenset(
    {"image_url", "input_image", "input_audio", "image"}
)


def _sanitize_content_blocks(blocks: list) -> list:
    """Iterate over content blocks, replacing media blocks with a text marker."""
    out: list = []
    for block in blocks:
        if not isinstance(block, dict):
            out.append(block)
            continue
        btype = block.get("type", "")
        if btype in _MEDIA_BLOCK_TYPES:
            # Replace the media block with a short textual placeholder so
            # the structural integrity of the message (and ordering of the
            # surrounding text blocks) is preserved in the backend UI.
            detail = ""
            if btype == "image_url":
                src = block.get("image_url", {})
                url = src.get("url", "") if isinstance(src, dict) else str(src)
                if url.startswith("data:"):
                    detail = " [base64]"
                else:
                    detail = " [url]"
            out.append({"type": "text", "text": f"[{btype}{detail}]"})
        else:
            out.append(block)
    return out


def _coerce_content(content: Any) -> Any:
    """Return content suitable for JSON span attributes.

    langchain message ``content`` is either:
    * a plain ``str`` — returned as-is.
    * a list of content blocks (e.g.
      ``[{"type": "text", "text": "..."}, {"type": "image_url", ...}]``) —
      media blocks are replaced with text placeholders to avoid embedding
      base64 blobs in OTLP payloads; the rest remains structured.
    * anything else — stringified as fallback.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return _sanitize_content_blocks(content)
    return str(content)


def _message_to_openai_dict(message: BaseMessage) -> dict[str, Any]:
    """Convert a langchain message to an OpenAI-style ``{role, content, ...}`` dict."""
    role = _ROLE_BY_TYPE.get(message.type, message.type)
    obj: dict[str, Any] = {"role": role, "content": _coerce_content(message.content)}
    # Preserve tool-call structure on assistant messages — backends use this to
    # render the "model called a tool" step in the chat UI.
    if isinstance(message, AIMessage):
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            obj["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": tc["args"]},
                }
                for tc in tool_calls
            ]
    # ToolMessage carries the tool-call id that produced it; backends group
    # tool results to their originating assistant tool_call via this id.
    tool_call_id = getattr(message, "tool_call_id", None)
    if tool_call_id:
        obj["tool_call_id"] = tool_call_id
    name = getattr(message, "name", None)
    if name:
        obj["name"] = name
    return obj


def serialize_messages(messages: Iterable[BaseMessage]) -> str:
    """Serialise a list of langchain messages to an OpenAI-style JSON string.

    Returns a JSON array string of ``[{"role": ..., "content": ...}, ...]``.
    Suitable as the value of the ``gen_ai.prompt`` span attribute.
    """
    payload = [_message_to_openai_dict(m) for m in messages]
    return json.dumps(payload, default=str)


def serialize_completion(message: BaseMessage) -> str:
    """Serialise an LLM completion (``AIMessage``) to an OpenAI-style JSON string.

    Returns a JSON array string containing a single assistant message — the
    shape Langfuse/Arize expect for ``gen_ai.completion``.
    """
    return json.dumps([_message_to_openai_dict(message)], default=str)


def serialize_completion_dict(obj: Any) -> str:
    """Serialise a non-AIMessage completion object (e.g. a pydantic structured
    output) to a JSON string. Used by reflection where ``with_structured_output``
    returns a parsed pydantic model rather than an ``AIMessage``.
    """
    if hasattr(obj, "model_dump"):
        obj = obj.model_dump()
    return json.dumps(obj, default=str)


def truncate_for_span(value: str, max_length: int) -> str:
    """Truncate a string for use as a span attribute value.

    Append ``...[truncated]`` when truncated so consumers can tell the value
    was shortened. ``max_length`` bounds the *returned* string including the
    sentinel so the value never exceeds the configured limit.
    """
    if len(value) <= max_length:
        return value
    sentinel = "...[truncated]"
    if max_length <= len(sentinel):
        return value[:max_length]
    return value[: max_length - len(sentinel)] + sentinel