from __future__ import annotations

import asyncio
import os

import pytest
from langchain_core.messages import HumanMessage

from ness_cli.config import ProviderRuntimeConfig
from ness_cli.providers.openrouter.adapter import OpenRouterProviderAdapter
from ness_cli.providers.openrouter.catalog import OpenRouterCatalog


@pytest.mark.live
@pytest.mark.skipif(
    os.environ.get("OPENROUTER_LIVE_TEST") != "1",
    reason="set OPENROUTER_LIVE_TEST=1 to spend a minimal live request",
)
def test_openrouter_live_chat_smoke(tmp_path) -> None:
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get(
        "OPENROUTER_API_KEY"
    )
    if not api_key:
        pytest.skip("OPENAI_API_KEY or OPENROUTER_API_KEY is required")
    model_name = os.environ.get(
        "OPENROUTER_LIVE_MODEL", "anthropic/claude-haiku-4.5"
    )
    adapter = OpenRouterProviderAdapter(
        ProviderRuntimeConfig(
            provider_id="openrouter",
            api_key=api_key,
            base_url=None,
            max_retries=2,
            cache_ttl="5m",
            anthropic_messages=True,
            session_id="live-smoke",
        ),
        catalog=OpenRouterCatalog(tmp_path / "catalog.json"),
    )

    async def run() -> None:
        try:
            response = await adapter.build_chat_model(
                "live-smoke",
                model_name=model_name,
                reasoning_effort=None,
            ).ainvoke([HumanMessage(content="Reply with exactly: OK")])
            assert "OK" in str(response.content)
        finally:
            await adapter.close()

    asyncio.run(run())
