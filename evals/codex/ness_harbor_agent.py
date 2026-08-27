"""
Harbor adapter for Ness Agent SDK

Needs to inherit from BaseInstalledAgent and implement the following methods:
- install
- run
- populate_context_post_run

can also add:
- name
- get_version_command

Implemented from Harbor Docs, Codex, OpenCode and Pi examples on github
"""

from __future__ import annotations

import json
import os
import re
import shlex
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, override

from harbor.agents.installed.base import BaseInstalledAgent, with_prompt_template
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.agents.model_connection import ModelConnectionSpec
from harbor.models.trial.paths import EnvironmentPaths
from harbor.utils.env import parse_bool_env_value
from harbor.utils.trajectory_utils import format_trajectory_json
from harbor.models.trajectories import (
    Agent,
    FinalMetrics,
    Metrics,
    Observation,
    ObservationResult,
    Step,
    ToolCall,
    Trajectory,
)


class NessAgent(BaseInstalledAgent):
    SUPPORTS_ATIF: bool = True
    MODEL_CONNECTION = ModelConnectionSpec(passthrough=True)
    NESS_VERSION: str = "0.2.3"
    INSTRUCTION_PATH = EnvironmentPaths.agent_dir / "instruction.md"
    OUTPUT_PATH = EnvironmentPaths.agent_dir / "ness.txt"
    NESS_DIR = EnvironmentPaths.agent_dir / "ness"
    NESS_SCRIPT_PATH = EnvironmentPaths.agent_dir / "ness_session.py"
    CODEX_MODEL_SCRIPT_PATH = EnvironmentPaths.agent_dir / "codex_chat_model.py"
    CODEX_SECRET_DIR = "/tmp/ness-codex-secrets"
    CODEX_CONFIG_DIR = "/tmp/ness-codex-config"

    @staticmethod
    @override
    def name() -> str:
        return "ness"

    @override
    def get_version_command(self) -> str:
        return 'export PATH="$HOME/.local/bin:$PATH"; ness --version'

    @override
    def parse_version(self, stdout: str) -> str:
        match = re.search(r"\b\d+\.\d+\.\d+\b", stdout)
        return match.group(0) if match else stdout.strip()

    def _resolve_codex_auth_path(self) -> Path | None:
        if parse_bool_env_value(
            self._get_env("CODEX_AUTH_JSON"),
            name="CODEX_AUTH_JSON",
            default=False,
        ):
            path = Path.home() / ".codex" / "auth.json"
            if not path.is_file():
                raise ValueError(f"Codex auth file was not found: {path}")
            return path

        return None

    @staticmethod
    def _minimal_codex_auth(source: Path) -> str:
        """Return only the bearer credentials needed by the eval transport."""
        try:
            document = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"Could not read Codex auth file: {source}") from exc

        tokens = document.get("tokens") if isinstance(document, dict) else None
        access_token = tokens.get("access_token") if isinstance(tokens, dict) else None
        account_id = tokens.get("account_id") if isinstance(tokens, dict) else None
        if not isinstance(access_token, str) or not access_token:
            raise ValueError(f"Codex auth file has no access token: {source}")
        if not isinstance(account_id, str) or not account_id:
            raise ValueError(f"Codex auth file has no account ID: {source}")

        return json.dumps(
            {
                "auth_mode": "chatgpt",
                "tokens": {
                    "access_token": access_token,
                    "account_id": account_id,
                },
            },
            separators=(",", ":"),
        )

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        await self.ensure_system_dependencies(environment, ("curl", "coreutils"))
        await self.exec_as_agent(
            environment,
            command=(
                "set -euo pipefail; "
                "if ! command -v uv >/dev/null 2>&1; then "
                "curl -LsSf https://astral.sh/uv/install.sh | sh; "
                "fi; "
                'if [ -f "$HOME/.local/bin/env" ]; then '
                '. "$HOME/.local/bin/env"; '
                "fi; "
                'export PATH="$HOME/.local/bin:$PATH"; '
                f"uv tool install --force --python 3.12 ness-agent=={self.NESS_VERSION}; "
                "ness --version"
            ),
            timeout_sec=1200,
        )

    @override
    @with_prompt_template
    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        instruction_file = self.logs_dir / "instruction.md"
        instruction_file.write_text(instruction, encoding="utf-8")

        await self._upload_agent_owned_file(
            environment,
            instruction_file,
            self.INSTRUCTION_PATH.as_posix(),
        )

        await self._upload_agent_owned_file(
            environment,
            Path(__file__).with_name("ness_session.py"),
            self.NESS_SCRIPT_PATH.as_posix(),
        )

        await self._upload_agent_owned_file(
            environment,
            Path(__file__).with_name("codex_chat_model.py"),
            self.CODEX_MODEL_SCRIPT_PATH.as_posix(),
        )

        if not self.model_name:
            raise ValueError("A model is required. Use --model provider/model.")

        model = self.model_name
        model_provider = "codex" if model.startswith("codex/") else ""
        if model.startswith("codex/"):
            model = model.split("/", 1)[1]

        auth_path = self._resolve_codex_auth_path()
        if auth_path is None:
            raise ValueError("Codex evals require user to be signed in with Codex")

        env: dict[str, str] = {
            "NESS_DIR": self.NESS_DIR.as_posix(),
            "AUTO_SAVE_THREADS": "true",
            "SESSION_END_REFLECTION": "false",
            "NESS_MODEL": model,
            "NESS_MODEL_PROVIDER": model_provider,
        }

        staged_auth_path: Path | None = None
        # CodexAuth reads this path inside the sandbox. The source auth file
        # itself is uploaded below and removed in the finally block.
        env["NESS_AGENT_CONFIG_DIR"] = self.CODEX_CONFIG_DIR
        env["NESS_MODEL_REASONING_EFFORT"] = (self._get_env("NESS_MODEL_REASONING_EFFORT") or "xhigh")

        remote_secret_dir = self.CODEX_SECRET_DIR
        remote_auth_path = f"{remote_secret_dir}/auth.json"
        await self.exec_as_root(
            environment,
            command=(
                f"mkdir -p {shlex.quote(remote_secret_dir)} "
                f"{shlex.quote(self.CODEX_CONFIG_DIR + '/codex')}"
            ),
        )
        # Do not upload the refresh token
        fd, staged_name = tempfile.mkstemp(
            prefix="ness-codex-auth-", suffix=".json"
        )
        os.close(fd)
        staged_auth_path = Path(staged_name)
        staged_auth_path.write_text(self._minimal_codex_auth(auth_path), encoding="utf-8")
        os.chmod(staged_auth_path, 0o600)
        try:
            await environment.upload_file(staged_auth_path, remote_auth_path)
        finally:
            staged_auth_path.unlink(missing_ok=True)
            staged_auth_path = None
        if environment.default_user is not None:
            await self.exec_as_root(
                environment,
                command=(
                    f"chown {shlex.quote(str(environment.default_user))} "
                    f"{shlex.quote(remote_auth_path)}"
                ),
            )
        await self.exec_as_root(
            environment,
            command=(
                f"chmod 600 {shlex.quote(remote_auth_path)} && "
                f"ln -sf {shlex.quote(remote_auth_path)} "
                f"{shlex.quote(self.CODEX_CONFIG_DIR + '/codex/auth.json')}"
            ),
        )

        cwd = environment.task_env_config.workdir
        command = (
            "set -euo pipefail; "
            'export PATH="$HOME/.local/bin:$PATH"; '
            'tool_root="$(uv tool dir)"; '
            'ness_python="$tool_root/ness-agent/bin/python"; '
            'test -x "$ness_python"; '
            f'"$ness_python" {shlex.quote(self.NESS_SCRIPT_PATH.as_posix())} '
            f'< {shlex.quote(self.INSTRUCTION_PATH.as_posix())} '
            f'2>&1 | stdbuf -oL tee {shlex.quote(self.OUTPUT_PATH.as_posix())}'
        )

        try:
            await self.exec_as_agent(environment, command, env, cwd)
        finally:
            if staged_auth_path is not None:
                try:
                    staged_auth_path.unlink(missing_ok=True)
                except OSError:
                    pass
            try:
                await self.exec_as_root(
                    environment,
                    command=(
                        f"rm -rf {shlex.quote(self.CODEX_SECRET_DIR)} "
                        f"{shlex.quote(self.CODEX_CONFIG_DIR)}"
                    ),
                )
            except Exception:
                self.logger.warning("Failed to clean up temporary Codex credentials")

    @override
    def populate_context_post_run(self, context: AgentContext) -> None:
        """Convert Ness's SQLite events to Harbor AITF."""
        db_path = self.logs_dir / "ness" / "threads" / "threads.db"

        if not db_path.exists():
            self.logger.debug(f"Ness thread database was not found: {db_path}")
            return

        try:
            with sqlite3.connect(db_path) as conn:
                row = conn.execute(
                    "SELECT thread_id from threads "
                    "WHERE thread_id LIKE 'session-%' "
                    "ORDER BY updated_at DESC LIMIT 1"
                ).fetchone()

                if row is None:
                    return

                session_id = str(row[0])
                event_rows = conn.execute(
                    "SELECT payload FROM events WHERE thread_id = ? ORDER BY seq",
                    (session_id,),
                ).fetchall()

                events = [json.loads(row[0]) for row in event_rows]

        except (OSError, sqlite3.Error, json.JSONDecodeError) as exc:
            self.logger.exception(f"Failed to read Ness event log: {exc}")
            return

        try:
            traj = self._convert_ness_events_to_aitf(events, session_id)
        except Exception as exc:
            self.logger.exception(f"Failed to convert Ness events to ATIF: {exc}")
            return

        if traj is None:
            return

        traj_path = self.logs_dir / "trajectory.json"
        try:
            traj_path.write_text(format_trajectory_json(traj.to_json_dict()), encoding='utf-8')
        except OSError as exc:
            self.logger.exception(f"Failed to write ATIF trajectory: {exc}")
            return

        if traj.final_metrics:
            metrics = traj.final_metrics
            context.cost_usd = metrics.total_cost_usd
            context.n_input_tokens = metrics.total_prompt_tokens or 0
            context.n_output_tokens = metrics.total_completion_tokens or 0
            context.n_cache_tokens = metrics.total_cached_tokens or 0

    def _convert_ness_events_to_aitf(self, events: list[dict[str, Any]], session_id: str) -> Trajectory | None:
        instruction_path = self.logs_dir / "instruction.md"
        instruction = instruction_path.read_text(encoding="utf-8", errors="replace")

        records: list[dict[str, Any]] = []
        pending_usage: list[dict[str, Any]] = []
        current_agent: dict[str, Any] | None = None
        all_usage: list[dict[str, Any]] = []
        resolved_model = self.model_name or "unknown"

        for event in events:
            kind = event.get("kind")

            if kind == "usage":
                pending_usage.append(event)
                all_usage.append(event)
                event_model = event.get("model")

                if isinstance(event_model, str) and event_model and event_model != "*":
                    resolved_model = event_model
                continue

            if kind == "user":
                record = {
                    "kind": "user",
                    "message": str(event.get("content") or ""),
                    "timestamp": event.get("t"),
                }
                records.append(record)
                current_agent = None
                continue

            if kind == "assistant":
                raw_tool_calls = event.get("tool_calls") or []
                if not isinstance(raw_tool_calls, list):
                    raw_tool_calls = []
                current_agent = {
                    "kind": "assistant",
                    "message": str(event.get("content") or ""),
                    "timestamp": event.get("t"),
                    "reasoning": (event.get("additional_kwargs") or {}).get(
                        "reasoning_content"
                    ),
                    "tool_calls": raw_tool_calls,
                    "tool_events": [],
                    "usage": pending_usage,
                }
                records.append(current_agent)
                pending_usage = []
                continue

            if kind == "tool" and current_agent is not None:
                current_agent["tool_events"].append(event)

            if instruction and not any(record["kind"] == "user" for record in records):
                records.insert(
                    0,
                    {
                        "kind": "user",
                        "message": instruction,
                        "timestamp": None,
                    },
                )

        steps: list[Step] = []
        for step_id, record in enumerate(records, start=1):
            if record["kind"] == "user":
                steps.append(
                    Step(
                        step_id=step_id,
                        timestamp=record.get("timestamp"),
                        source="user",
                        message=record["message"],
                    )
                )
                continue

            tool_calls: list[ToolCall] = []
            tool_call_ids: set[str] = set()
            for index, raw_call in enumerate(record["tool_calls"]):
                if not isinstance(raw_call, dict):
                    continue
                call_id = str(
                    raw_call.get("id")
                    or raw_call.get("call_id")
                    or f"ness-call-{step_id}-{index}"
                )
                tool_call_ids.add(call_id)
                tool_calls.append(
                    ToolCall(
                        tool_call_id=call_id,
                        function_name=str(raw_call.get("name") or "unknown"),
                        arguments=_tool_arguments(
                            raw_call.get("args", raw_call.get("arguments"))
                        ),
                    )
                )

            observations: list[ObservationResult] = []
            for index, tool_event in enumerate(record["tool_events"]):
                call_id = str(
                    tool_event.get("call_id") or f"ness-call-{step_id}-{index}"
                )
                if call_id not in tool_call_ids:
                    tool_call_ids.add(call_id)
                    tool_calls.append(
                        ToolCall(
                            tool_call_id=call_id,
                            function_name=str(tool_event.get("tool") or "unknown"),
                            arguments=_tool_arguments(tool_event.get("args")),
                        )
                    )
                observations.append(
                    ObservationResult(
                        source_call_id=call_id,
                        content=str(tool_event.get("result") or ""),
                        extra={
                            "exit": tool_event.get("exit"),
                            "duration_ms": _number(tool_event.get("duration_ms")),
                        },
                    )
                )

            step_metrics = self._usage_metrics(record["usage"])
            reasoning = record.get("reasoning")
            step_kwargs: dict[str, Any] = {
                "step_id": step_id,
                "timestamp": record.get("timestamp"),
                "source": "agent",
                "message": record["message"],
                "model_name": resolved_model,
                "llm_call_count": 1,
            }
            if isinstance(reasoning, str) and reasoning:
                step_kwargs["reasoning_content"] = reasoning
            if tool_calls:
                step_kwargs["tool_calls"] = tool_calls
            if observations:
                step_kwargs["observation"] = Observation(results=observations)
            if step_metrics:
                step_kwargs["metrics"] = step_metrics
            steps.append(Step(**step_kwargs))

        if not steps:
            return None

        total_prompt = sum(_number(event.get("input_tokens")) for event in all_usage)
        total_completion = sum(
            _number(event.get("output_tokens")) for event in all_usage
        )
        total_cached = sum(
            _number(event.get("cached_input_tokens")) for event in all_usage
        )
        total_costs = [
            value
            for value in (_cost(event.get("cost_usd")) for event in all_usage)
            if value is not None
        ]

        return Trajectory(
            schema_version="ATIF-v1.7",
            session_id=session_id,
            agent=Agent(
                name="ness",
                version=self.version(),
                model_name=resolved_model,
            ),
            steps=steps,
            final_metrics=FinalMetrics(
                total_prompt_tokens=total_prompt or None,
                total_completion_tokens=total_completion or None,
                total_cached_tokens=total_cached or None,
                total_cost_usd=sum(total_costs) if total_costs else None,
                total_steps=len(steps),
            ),
        )

    def _usage_metrics(self, events: list[dict[str, Any]]) -> Metrics | None:
        if not events:
            return None

        prompt_tokens = sum(_number(event.get("input_tokens")) for event in events)
        completion_tokens = sum(
            _number(event.get("output_tokens")) for event in events
        )
        cached_tokens = sum(
            _number(event.get("cached_input_tokens")) for event in events
        )
        costs = [
            value
            for value in (_cost(event.get("cost_usd")) for event in events)
            if value is not None
        ]
        calls = sum(_number(event.get("calls"), 1) for event in events)

        if not any((prompt_tokens, completion_tokens, cached_tokens, costs)):
            return None

        return Metrics(
            prompt_tokens=prompt_tokens or None,
            completion_tokens=completion_tokens or None,
            cached_tokens=cached_tokens or None,
            cost_usd=sum(costs) if costs else None,
            extra={"llm_calls": calls} if calls else None,
        )

def _number(value: Any, default: int = 0) -> int:
    try:
        return int(value or default)
    except (TypeError, ValueError):
        return default


def _cost(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _tool_arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {"value": value} if value is not None else {}
