from __future__ import annotations

import asyncio
import json
from typing import Any, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ConfigDict, Field, PrivateAttr

from ness_cli.provider.codex.auth import CodexAuth
from ness_cli.provider.codex.transport import CodexResponsesTransport


class CodexSubscriptionChatModel(BaseChatModel):
    """LangChain model using ChatGPT credentials managed by Codex app-server."""

    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True)

    model_name: str = Field(alias="model")
    reasoning_effort: str | None = None
    prompt_cache_key: str | None = None
    max_retries: int = 3
    _auth: CodexAuth = PrivateAttr()
    _transport: CodexResponsesTransport = PrivateAttr()
    _tool_registry: Any = PrivateAttr(default=None)
    _tool_snapshot: list[dict[str, Any]] | None = PrivateAttr(default=None)

    def __init__(self, *, auth: CodexAuth | None = None, **data: Any) -> None:
        super().__init__(**data)
        self._auth = auth or CodexAuth()
        self._transport = CodexResponsesTransport(self._auth, max_retries=self.max_retries)

    @property
    def model(self) -> str:
        return self.model_name

    @property
    def _llm_type(self) -> str:
        return "codex-subscription-responses"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "billing_mode": "subscription",
            "prompt_cache_key": self.prompt_cache_key,
        }

    def bind_tool_registry(self, registry: Any) -> BaseChatModel:
        # make a deep copy of codex subscription chat model; avoid modifying the original instance
        clone = self.model_copy() # this is shallow copy
        clone._auth = self._auth
        clone._transport = self._transport
        clone._tool_snapshot = [self._format_tool(tool) for tool in registry.active_tools]
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
        # convert the tool from langchain tool to OpenAI tool format
        converted = convert_to_openai_tool(tool)
        function = converted.get("function") or {}
        return {
            "type": "function",
            "name": function.get("name", ""),
            "description": function.get("description", ""),
            "parameters": function.get("parameters") or {"type": "object", "properties": {}},
            "strict": False,
        }

    @staticmethod
    def _content(content: Any, *, output: bool = False) -> list[dict[str, Any]]:
        # converts message content into Responses-style content blocks
        text_type = "output_text" if output else "input_text"
        if isinstance(content, str):
            return [{"type": text_type, "text": content}] if content else []
        blocks: list[dict[str, Any]] = []
        for item in content if isinstance(content, list) else []:
            if not isinstance(item, dict):
                blocks.append({"type": text_type, "text": str(item)})
            elif item.get("type") == "image_url":
                value = item.get("image_url")
                url = value.get("url") if isinstance(value, dict) else value
                block = {"type": "input_image", "image_url": str(url)}
                if isinstance(value, dict) and value.get("detail"):
                    block["detail"] = str(value["detail"])
                blocks.append(block)
            elif item.get("type") in {"text", "input_text", "output_text"}:
                blocks.append({"type": text_type, "text": str(item.get("text") or "")})
        return blocks

    def _input(self, messages: Sequence[BaseMessage]) -> tuple[str, list[dict[str, Any]]]:
        # converts an entire LangChain conversation into the input format expected by Codex Responses request
        instructions: list[str] = []
        items: list[dict[str, Any]] = []
        for message in messages:
            if isinstance(message, SystemMessage):
                instructions.append(str(message.content))
            elif isinstance(message, HumanMessage):
                items.append({"role": "user", "content": self._content(message.content)})
            elif isinstance(message, ToolMessage):
                output = (
                    message.content
                    if isinstance(message.content, str)
                    else self._content(message.content)
                )
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": message.tool_call_id,
                        "output": output,
                    }
                )
            elif isinstance(message, AIMessage):
                raw_items = message.additional_kwargs.get("codex_output_items")
                if isinstance(raw_items, list):
                    items.extend(dict(item) for item in raw_items if isinstance(item, dict))
                    continue
                if message.content:
                    items.append({"role": "assistant", "content": self._content(message.content, output=True)})
                for call in message.tool_calls:
                    items.append(
                        {
                            "type": "function_call",
                            "call_id": call.get("id"),
                            "name": call.get("name"),
                            "arguments": json.dumps(call.get("args") or {}, separators=(",", ":")),
                        }
                    )
        return "\n\n".join(instructions), items

    def _payload(self, messages: Sequence[BaseMessage], **kwargs: Any) -> dict[str, Any]:
        # Takes those converted messages and builds the final request body. 
        # It adds model name, instructions, conversation input, tool definitions, tool choice, reasoning effort, and max output tokens. 
        # This is basically the last formatting step before the HTTP/transport layer.
        instructions, items = self._input(messages)
        supplied = kwargs.pop("tools", None)
        tools = self._tool_snapshot or [
            dict(tool) if isinstance(tool, dict) and tool.get("type") == "function" else self._format_tool(tool)
            for tool in (supplied or [])
        ]
        payload: dict[str, Any] = {
            "model": self.model_name,
            "instructions": instructions,
            "input": items,
            "store": False,
            "parallel_tool_calls": True,
        }
        if self.prompt_cache_key:
            # The key must remain stable for all requests that share a
            # conversation prefix. The adapter scopes it to one thread (and
            # to auxiliary model roles such as reflection).
            payload["prompt_cache_key"] = self.prompt_cache_key
        if tools:
            payload["tools"] = tools
            tool_choice = kwargs.pop("tool_choice", "auto")
            # LangChain uses ``any`` for "at least one tool" while the
            # Responses API names the same mode ``required``.
            payload["tool_choice"] = "required" if tool_choice == "any" else tool_choice
        if self.reasoning_effort and self.reasoning_effort != "none":
            payload["reasoning"] = {"effort": self.reasoning_effort, "summary": "auto"}
        max_tokens = kwargs.pop("max_tokens", None)
        if max_tokens is not None:
            payload["max_output_tokens"] = int(max_tokens)
        return payload

    @staticmethod
    def _message(response: dict[str, Any]) -> AIMessage:
        # _message() takes the raw response returned by Codex and turns it back into a LangChain AIMessage. 
        # It extracts generated text, parses function calls, stores raw output items for future history replay, 
        # and attaches token usage plus response metadata.
        text: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        raw_items = [dict(item) for item in response.get("output") or [] if isinstance(item, dict)]
        for item in raw_items:
            if item.get("type") == "message":
                for block in item.get("content") or []:
                    if isinstance(block, dict) and block.get("type") in {"output_text", "text"}:
                        text.append(str(block.get("text") or ""))
            elif item.get("type") == "function_call":
                arguments = item.get("arguments") or "{}"
                try:
                    parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
                except ValueError:
                    parsed = {"_raw": arguments}
                tool_calls.append(
                    {
                        "name": item.get("name"),
                        "args": parsed,
                        "id": item.get("call_id") or item.get("id"),
                        "type": "tool_call",
                    }
                )
        if not text and response.get("output_text"):
            fallback_text = str(response["output_text"])
            text.append(fallback_text)
            # Keep manual-history replay correct when the completed envelope
            # omitted its message item but streaming still delivered text.
            raw_items.append(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": fallback_text}],
                }
            )
        usage = response.get("usage") or {}
        details = usage.get("input_tokens_details") or {}
        input_tokens = int(usage.get("input_tokens") or 0)
        output_tokens = int(usage.get("output_tokens") or 0)
        cache_read = int(details.get("cached_tokens") or 0)
        cache_write = int(details.get("cache_write_tokens") or 0)
        cache_diagnostics = response.get("_cache_diagnostics")
        response_metadata: dict[str, Any] = {
            "model_name": response.get("model") or "",
            "response_id": response.get("id"),
            "billing_mode": "subscription",
            "cache_write_tokens": cache_write,
        }
        if isinstance(cache_diagnostics, dict):
            response_metadata["cache_diagnostics"] = dict(cache_diagnostics)
        return AIMessage(
            content="".join(text),
            tool_calls=tool_calls,
            additional_kwargs={"codex_output_items": raw_items},
            usage_metadata={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": int(usage.get("total_tokens") or input_tokens + output_tokens),
                "input_token_details": {
                    "cache_read": cache_read,
                    "cache_creation": cache_write,
                },
            },
            response_metadata=response_metadata,
        )

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, run_manager
        response = await self._transport.create(self._payload(messages, **kwargs))
        return ChatResult(generations=[ChatGeneration(message=self._message(response))])

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return asyncio.run(self._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs))
