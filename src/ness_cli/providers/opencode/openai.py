from __future__ import annotations

from typing import Any

from langchain_core.outputs import ChatResult
from langchain_openai import ChatOpenAI


class OpenCodeChatOpenAI(ChatOpenAI):
    """OpenAI-compatible model that marks OpenCode Go calls as subscription usage."""

    @staticmethod
    def _mark_subscription(result: ChatResult) -> ChatResult:
        for generation in result.generations:
            metadata = dict(generation.message.response_metadata or {})
            metadata["billing_mode"] = "subscription"
            generation.message.response_metadata = metadata
        return result

    def _generate(self, *args: Any, **kwargs: Any) -> ChatResult:
        return self._mark_subscription(super()._generate(*args, **kwargs))

    async def _agenerate(self, *args: Any, **kwargs: Any) -> ChatResult:
        return self._mark_subscription(await super()._agenerate(*args, **kwargs))
