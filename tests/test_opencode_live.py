from __future__ import annotations

import asyncio
import os

import pytest
from langchain_core.messages import HumanMessage

from ness_cli.provider.opencode.adapter import OpenCodeProviderAdapter


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
    adapter = OpenCodeProviderAdapter()
    response = asyncio.run(
        adapter.build_chat_model(
            "live-smoke", model_name=model_name, reasoning_effort="high"
        ).ainvoke([HumanMessage(content="Reply with exactly: OK")])
    )
    assert "OK" in str(response.content)
