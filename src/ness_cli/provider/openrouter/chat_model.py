from __future__ import annotations

import asyncio
import json
import time
from typing import Any, AsyncIterator, Sequence

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    ChatResult,
)
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ConfigDict, Field, PrivateAttr


def _image_source(url: str) -> dict[str, str]:
    """Convert image data URLs to Anthropic's base64 source shape."""
    header, separator, data = url.partition(",")
    if separator and header.startswith("data:") and ";base64" in header:
        media_type = header[5:].split(";", 1)[0]
        if media_type.startswith("image/"):
            return {
                "type": "base64",
                "media_type": media_type,
                "data": data,
            }
    return {"type": "url", "url": url}


class OpenRouterAnthropicMessages(BaseChatModel):
    """LangChain chat model backed by OpenRouter's Anthropic Messages API."""

    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True)

    model_name: str = Field(alias="model")
    api_key: str
    base_url: str = "https://openrouter.ai/api/v1"
    session_id: str = ""
    cache_ttl: str | None = "5m"
    request_timeout: float = 120.0
    max_retries: int = 3
    reasoning: dict[str, Any] | None = None
    include_openrouter_extensions: bool = True
    billing_mode: str | None = None
    _tool_registry: Any = PrivateAttr(default=None)
    _tool_snapshot: list[dict[str, Any]] | None = PrivateAttr(default=None)
    _tool_additions: tuple[str, ...] = PrivateAttr(default=())

    @property
    def model(self) -> str:
        return self.model_name

    @property
    def _llm_type(self) -> str:
        return (
            "openrouter-anthropic-messages"
            if self.include_openrouter_extensions
            else "anthropic-messages-compatible"
        )

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "session_id": self.session_id,
            **({"billing_mode": self.billing_mode} if self.billing_mode else {}),
        }

    def bind_tool_registry(self, registry: Any) -> BaseChatModel:
        # Return an immutable request binding. A live mutable registry makes
        # historical tool definitions change underneath cached parent calls.
        clone = self.model_copy()
        clone._tool_registry = registry
        if self.include_openrouter_extensions:
            clone._tool_snapshot = clone._tools()
            clone._tool_additions = tuple(sorted(registry.active_mcp_tools))
        else:
            clone._tool_snapshot = [
                clone._format_tool(tool) for tool in registry.active_tools
            ]
            clone._tool_additions = ()
        clone._tool_registry = None
        return clone

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ):
        formatted = [self._format_tool(tool) for tool in tools]
        return self.bind(tools=formatted, tool_choice=tool_choice, **kwargs)

    @staticmethod
    def _format_tool(tool: dict[str, Any] | type | BaseTool) -> dict[str, Any]:
        converted = convert_to_openai_tool(tool)
        function = converted.get("function") or {}
        return {
            "name": function.get("name", ""),
            "description": function.get("description", ""),
            "input_schema": function.get("parameters") or {"type": "object"},
        }

    def _tools(self, supplied: Sequence[Any] | None = None) -> list[dict[str, Any]]:
        if self._tool_snapshot is not None:
            return [dict(tool) for tool in self._tool_snapshot]
        if self._tool_registry is None:
            return [
                dict(tool) if isinstance(tool, dict) and "input_schema" in tool else self._format_tool(tool)
                for tool in (supplied or [])
            ]
        deferred = self._tool_registry.deferred_tool_names()
        tools: list[dict[str, Any]] = []
        for tool in self._tool_registry.all_tools():
            item = self._format_tool(tool)
            if tool.name in deferred or tool.name in self._tool_registry.active_mcp_tools:
                item["defer_loading"] = True
            tools.append(item)
        return tools

    @staticmethod
    def _content_blocks(content: Any) -> list[dict[str, Any]]:
        if isinstance(content, str):
            return [{"type": "text", "text": content}] if content else []
        blocks: list[dict[str, Any]] = []
        for block in content if isinstance(content, list) else []:
            if not isinstance(block, dict):
                blocks.append({"type": "text", "text": str(block)})
                continue
            if block.get("type") == "image_url":
                value = block.get("image_url") or {}
                url = value.get("url") if isinstance(value, dict) else value
                blocks.append({"type": "image", "source": _image_source(str(url))})
            else:
                blocks.append(dict(block))
        return blocks

    def _convert_messages(
        self,
        messages: Sequence[BaseMessage],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        system: list[dict[str, Any]] = []
        converted: list[dict[str, Any]] = []
        for index, message in enumerate(messages):
            if isinstance(message, SystemMessage) and not converted:
                system.extend(self._content_blocks(message.content))
            elif isinstance(message, HumanMessage):
                converted.append(
                    {"role": "user", "content": self._content_blocks(message.content)}
                )
            elif isinstance(message, AIMessage):
                raw_blocks = message.additional_kwargs.get(
                    "anthropic_content_blocks"
                )
                blocks = (
                    [dict(block) for block in raw_blocks]
                    if isinstance(raw_blocks, list)
                    else self._content_blocks(message.content)
                )
                if not raw_blocks:
                    for call in message.tool_calls:
                        blocks.append(
                            {
                                "type": "tool_use",
                                "id": call.get("id") or f"tool-{index}",
                                "name": call.get("name"),
                                "input": call.get("args") or {},
                            }
                        )
                converted.append({"role": "assistant", "content": blocks})
            elif isinstance(message, ToolMessage):
                content = (
                    message.content
                    if isinstance(message.content, str)
                    else self._content_blocks(message.content)
                )
                converted.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": message.tool_call_id,
                                "content": content,
                            }
                        ],
                    }
                )
            elif isinstance(message, SystemMessage):
                converted.append(
                    {"role": "system", "content": self._content_blocks(message.content)}
                )
        if self._tool_registry is not None or self._tool_additions:
            additions = (
                sorted(self._tool_registry.active_mcp_tools)
                if self._tool_registry is not None
                else list(self._tool_additions)
            )
            if additions:
                converted.append(
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "tool_addition",
                                "tool": {"type": "tool_reference", "name": name},
                            }
                            for name in additions
                        ],
                    }
                )
        return system, converted

    def _payload(
        self,
        messages: Sequence[BaseMessage],
        *,
        stop: list[str] | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        system, converted = self._convert_messages(messages)
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": converted,
            "max_tokens": int(kwargs.pop("max_tokens", 8192)),
            "stream": stream,
        }
        if self.include_openrouter_extensions:
            payload["session_id"] = self.session_id
            if self.cache_ttl:
                payload["cache_control"] = {
                    "type": "ephemeral",
                    "ttl": self.cache_ttl,
                }
        if system:
            payload["system"] = system
        tools = self._tools(kwargs.pop("tools", None))
        if tools:
            payload["tools"] = tools
        if stop:
            payload["stop_sequences"] = stop
        if self.reasoning:
            payload["output_config"] = self.reasoning
        payload.update({key: value for key, value in kwargs.items() if value is not None})
        return payload

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _retryable(exc: Exception) -> bool:
        if isinstance(exc, httpx.TransportError):
            return True
        if isinstance(exc, httpx.HTTPStatusError):
            return exc.response.status_code == 429 or exc.response.status_code >= 500
        return False

    def _message_from_response(self, data: dict[str, Any]) -> AIMessage:
        text: list[str] = []
        reasoning: list[str] = []
        calls: list[dict[str, Any]] = []
        raw_blocks = [
            dict(block) for block in data.get("content") or [] if isinstance(block, dict)
        ]
        for block in raw_blocks:
            if block.get("type") == "text":
                text.append(str(block.get("text") or ""))
            elif block.get("type") == "thinking":
                reasoning.append(str(block.get("thinking") or ""))
            elif block.get("type") == "tool_use":
                calls.append(
                    {
                        "name": block.get("name"),
                        "args": block.get("input") or {},
                        "id": block.get("id"),
                        "type": "tool_call",
                    }
                )
        usage = data.get("usage") or {}
        input_tokens = int(usage.get("input_tokens") or 0)
        output_tokens = int(usage.get("output_tokens") or 0)
        cache_read = int(usage.get("cache_read_input_tokens") or 0)
        response_metadata = {"model_name": data.get("model")}
        if self.billing_mode:
            response_metadata["billing_mode"] = self.billing_mode
        if usage.get("cost") is not None:
            response_metadata["cost"] = usage.get("cost")
        return AIMessage(
            content="".join(text),
            tool_calls=calls,
            additional_kwargs={
                "anthropic_content_blocks": raw_blocks,
                **(
                    {"reasoning_content": "".join(reasoning)}
                    if reasoning
                    else {}
                ),
            },
            usage_metadata={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "input_token_details": {"cache_read": cache_read},
            },
            response_metadata=response_metadata,
        )

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        del run_manager
        for attempt in range(self.max_retries + 1):
            try:
                response = httpx.post(
                    f"{self.base_url.rstrip('/')}/messages",
                    headers=self._headers,
                    json=self._payload(messages, stop=stop, **kwargs),
                    timeout=self.request_timeout,
                )
                response.raise_for_status()
                break
            except Exception as exc:
                if attempt >= self.max_retries or not self._retryable(exc):
                    raise
                time.sleep(min(2**attempt, 8))
        return ChatResult(
            generations=[ChatGeneration(message=self._message_from_response(response.json()))]
        )

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        del run_manager
        async with httpx.AsyncClient(timeout=self.request_timeout) as client:
            for attempt in range(self.max_retries + 1):
                try:
                    response = await client.post(
                        f"{self.base_url.rstrip('/')}/messages",
                        headers=self._headers,
                        json=self._payload(messages, stop=stop, **kwargs),
                    )
                    response.raise_for_status()
                    break
                except Exception as exc:
                    if attempt >= self.max_retries or not self._retryable(exc):
                        raise
                    await asyncio.sleep(min(2**attempt, 8))
        return ChatResult(
            generations=[ChatGeneration(message=self._message_from_response(response.json()))]
        )

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        for attempt in range(self.max_retries + 1):
            emitted = False
            try:
                async for chunk in self._astream_once(
                    messages,
                    stop=stop,
                    run_manager=run_manager,
                    **kwargs,
                ):
                    emitted = True
                    yield chunk
                return
            except Exception as exc:
                if (
                    emitted
                    or attempt >= self.max_retries
                    or not self._retryable(exc)
                ):
                    raise
                await asyncio.sleep(min(2**attempt, 8))

    async def _astream_once(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        del run_manager
        payload = self._payload(messages, stop=stop, stream=True, **kwargs)
        async with httpx.AsyncClient(timeout=self.request_timeout) as client:
            async with client.stream(
                "POST",
                f"{self.base_url.rstrip('/')}/messages",
                headers=self._headers,
                json=payload,
            ) as response:
                response.raise_for_status()
                raw_blocks: dict[int, dict[str, Any]] = {}
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if not raw or raw == "[DONE]":
                        continue
                    event = json.loads(raw)
                    kind = event.get("type")
                    if kind == "message_start":
                        usage = (event.get("message") or {}).get("usage") or {}
                        input_tokens = int(usage.get("input_tokens") or 0)
                        cache_read = int(usage.get("cache_read_input_tokens") or 0)
                        if input_tokens:
                            yield ChatGenerationChunk(
                                message=AIMessageChunk(
                                    content="",
                                    usage_metadata={
                                        "input_tokens": input_tokens,
                                        "output_tokens": 0,
                                        "total_tokens": input_tokens,
                                        "input_token_details": {
                                            "cache_read": cache_read
                                        },
                                    },
                                )
                            )
                    elif kind == "content_block_start":
                        block = dict(event.get("content_block") or {})
                        index = int(event.get("index") or 0)
                        raw_blocks[index] = block
                        if block.get("type") == "tool_use":
                            block["_partial_json"] = ""
                            chunk = AIMessageChunk(
                                content="",
                                tool_call_chunks=[
                                    {
                                        "name": block.get("name"),
                                        "args": "",
                                        "id": block.get("id"),
                                        "index": index,
                                    }
                                ],
                            )
                            yield ChatGenerationChunk(message=chunk)
                    elif kind == "content_block_delta":
                        delta = event.get("delta") or {}
                        delta_type = delta.get("type")
                        index = int(event.get("index") or 0)
                        block = raw_blocks.setdefault(index, {})
                        if delta_type == "text_delta":
                            text = str(delta.get("text") or "")
                            block["text"] = str(block.get("text") or "") + text
                            chunk = AIMessageChunk(content=text)
                        elif delta_type == "thinking_delta":
                            thinking = str(delta.get("thinking") or "")
                            block["thinking"] = (
                                str(block.get("thinking") or "") + thinking
                            )
                            chunk = AIMessageChunk(
                                content="",
                                additional_kwargs={
                                    "reasoning_content": thinking
                                },
                            )
                        elif delta_type == "signature_delta":
                            block["signature"] = (
                                str(block.get("signature") or "")
                                + str(delta.get("signature") or "")
                            )
                            continue
                        elif delta_type == "input_json_delta":
                            partial = str(delta.get("partial_json") or "")
                            block["_partial_json"] = (
                                str(block.get("_partial_json") or "") + partial
                            )
                            chunk = AIMessageChunk(
                                content="",
                                tool_call_chunks=[
                                    {
                                        "name": None,
                                        "args": partial,
                                        "id": block.get("id"),
                                        "index": index,
                                    }
                                ],
                            )
                        else:
                            continue
                        yield ChatGenerationChunk(message=chunk)
                    elif kind == "content_block_stop":
                        index = int(event.get("index") or 0)
                        block = raw_blocks.get(index)
                        if block is not None:
                            partial = block.pop("_partial_json", "")
                            if block.get("type") == "tool_use" and partial:
                                try:
                                    block["input"] = json.loads(partial)
                                except ValueError:
                                    block["input"] = {}
                            yield ChatGenerationChunk(
                                message=AIMessageChunk(
                                    content="",
                                    additional_kwargs={
                                        "anthropic_content_blocks": [dict(block)]
                                    },
                                )
                            )
                    elif kind == "message_delta":
                        usage = event.get("usage") or {}
                        output_tokens = int(usage.get("output_tokens") or 0)
                        if output_tokens:
                            yield ChatGenerationChunk(
                                message=AIMessageChunk(
                                    content="",
                                    usage_metadata={
                                        "input_tokens": 0,
                                        "output_tokens": output_tokens,
                                        "total_tokens": output_tokens,
                                    },
                                )
                            )
