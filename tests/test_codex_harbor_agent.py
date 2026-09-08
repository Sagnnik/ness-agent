from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

pytest.importorskip("harbor")

from evals.codex.ness_harbor_agent import NessAgent as CodexNessAgent
from evals.ness_harbor_agent import NessAgent as OpenRouterNessAgent


@pytest.mark.parametrize("failed_command", ["chown ", "chmod "])
def test_remote_auth_cleanup_covers_permission_setup_failures(
    tmp_path: Path, failed_command: str
) -> None:
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(
        json.dumps({
            "tokens": {
                "access_token": "live-access-token",
                "account_id": "account-id",
            }
        }),
        encoding="utf-8",
    )
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    agent = CodexNessAgent(
        logs_dir=logs_dir,
        model_name="codex/test-model",
    )
    environment = SimpleNamespace(
        default_user="sandbox-user",
        task_env_config=SimpleNamespace(workdir="/workspace"),
        upload_file=AsyncMock(),
    )
    root_commands: list[str] = []

    async def exec_as_root(_environment, *, command: str, **_kwargs) -> None:
        root_commands.append(command)
        if command.startswith(failed_command):
            raise RuntimeError("permission setup failed")

    run_agent = AsyncMock()
    with (
        patch.object(agent, "_resolve_codex_auth_path", return_value=auth_path),
        patch.object(agent, "_upload_agent_owned_file", new=AsyncMock()),
        patch.object(agent, "exec_as_root", side_effect=exec_as_root),
        patch.object(agent, "exec_as_agent", new=run_agent),
        pytest.raises(RuntimeError, match="permission setup failed"),
    ):
        asyncio.run(agent.run("task", environment, SimpleNamespace()))

    environment.upload_file.assert_awaited_once()
    staged_path = environment.upload_file.await_args.args[0]
    assert not staged_path.exists()
    assert root_commands[-1] == (
        "rm -rf /tmp/ness-codex-secrets /tmp/ness-codex-config"
    )
    run_agent.assert_not_awaited()


@pytest.mark.parametrize("agent_class", [CodexNessAgent, OpenRouterNessAgent])
def test_compaction_usage_is_not_charged_to_next_agent_step(
    tmp_path: Path, agent_class: type
) -> None:
    logs_dir = tmp_path / agent_class.__module__.replace(".", "-")
    logs_dir.mkdir()
    (logs_dir / "instruction.md").write_text("task", encoding="utf-8")
    agent = agent_class(
        logs_dir=logs_dir,
        model_name="test-model",
        version="0.2.3",
    )
    events = [
        {"kind": "user", "content": "task"},
        {
            "kind": "usage",
            "operation": "compaction",
            "model": "test-model",
            "input_tokens": 100,
            "output_tokens": 10,
            "cached_input_tokens": 40,
            "cost_usd": 0.10,
        },
        {"kind": "compaction_llm", "response": "summary"},
        {
            "kind": "usage",
            "model": "test-model",
            "input_tokens": 20,
            "output_tokens": 2,
            "cached_input_tokens": 5,
            "cost_usd": 0.02,
        },
        {"kind": "assistant", "content": "answer"},
    ]

    trajectory = agent._convert_ness_events_to_aitf(events, "session-usage")

    assert trajectory is not None
    agent_step = trajectory.steps[-1]
    assert agent_step.llm_call_count == 1
    assert agent_step.metrics is not None
    assert agent_step.metrics.prompt_tokens == 20
    assert agent_step.metrics.completion_tokens == 2
    assert agent_step.metrics.cached_tokens == 5
    assert agent_step.metrics.cost_usd == pytest.approx(0.02)
    assert agent_step.metrics.extra == {"llm_calls": 1}
    assert trajectory.final_metrics is not None
    assert trajectory.final_metrics.total_prompt_tokens == 120
    assert trajectory.final_metrics.total_completion_tokens == 12
    assert trajectory.final_metrics.total_cached_tokens == 45
    assert trajectory.final_metrics.total_cost_usd == pytest.approx(0.12)
