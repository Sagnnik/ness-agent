from __future__ import annotations

import asyncio
import os

import pytest
from langchain_core.messages import HumanMessage

from ness_cli.config import ProviderRuntimeConfig
from ness_cli.providers.opencode.adapter import OpenCodeProviderAdapter


@pytest.mark.live
@pytest.mark.skipif(
    os.environ.get("OPENCODE_GO_LIVE_TEST") != "1",
    reason="set OPENCODE_GO_LIVE_TEST=1 to spend a minimal subscription request",
)
def test_opencode_go_live_chat_smoke() -> None:
    api_key = os.environ.get("OPENCODE_GO_API_KEY") or os.environ.get(
        "OPENCODE_API_KEY"
    )
    if not api_key:
        pytest.skip("OPENCODE_GO_API_KEY or OPENCODE_API_KEY is required")
    model_name = os.environ.get("OPENCODE_GO_LIVE_MODEL", "deepseek-v4-flash")
    adapter = OpenCodeProviderAdapter(
        ProviderRuntimeConfig(
            provider_id="opencode",
            api_key=api_key,
            base_url=None,
            max_retries=2,
            cache_ttl=None,
            anthropic_messages=False,
            session_id=None,
        )
    )

    async def run() -> None:
        try:
            response = await adapter.build_chat_model(
                "live-smoke",
                model_name=model_name,
                reasoning_effort="high",
            ).ainvoke([HumanMessage(content="Reply with exactly: OK")])
            assert "OK" in str(response.content)
        finally:
            await adapter.close()

    asyncio.run(run())
