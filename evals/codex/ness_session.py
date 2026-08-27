from __future__ import annotations

import asyncio
import os
import sys
import time
import uuid
from pathlib import Path

from langchain_openrouter import ChatOpenRouter
from ness_agent import (
    MemoryConfig,
    ModeConfig,
    NessAgent,
    NessAgentOptions,
    PromptLayersConfig,
)
from ness_cli.provider.codex.auth import CodexAuth

from codex_chat_model import CodexChatModel


def auto_answer_question(questions: list[dict]) -> list[dict]:
    answers: list[dict] = []

    for index, question in enumerate(questions, start=1):
        options = list(question.get("options") or [])
        selected = next((o for o in options if o.get("recommended")), None)
        if selected is None and options:
            selected = options[0]
        if selected is None:
            selected = {"id": "0", "label": "proceed"}

        answers.append({
                "id": question.get("id", str(index)),
                "selected": {
                    "id": selected.get("id"),
                    "label": selected.get("label"),
                },
                "note": "auto-answered (headless agent)",
        })

    return answers


async def run() -> int:
    instructions = sys.stdin.read().strip()
    if not instructions:
        print("Ness SDK runner did not receive any instructions")
        return 2

    thread_id = f"session-{uuid.uuid4().hex[:8]}"
    model_name = os.environ.get("NESS_MODEL")
    provider = os.environ.get("NESS_MODEL_PROVIDER")
    session_id = os.environ.get("NESS_SESSION_ID") or thread_id
    reasoning_effort = os.environ.get("NESS_MODEL_REASONING_EFFORT") or "xhigh"

    auth = CodexAuth()
    credentials = auth.credentials()
    if credentials is None:
        print("Ness Codex runner did not find a usable auth.json")
        return 1
    if credentials.expires_at is not None and credentials.expires_at <= int(time.time()) + 60:
        print("Ness Codex runner received an expired auth.json")
        return 1

    project_root = Path.cwd().resolve()
    ness_dir = Path(os.environ.get("NESS_DIR", "/logs/agent/ness")).resolve() # default harbor environment path /logs/agent
    ness_dir.mkdir(parents=True, exist_ok=True)
    project_memory = ness_dir / "NESS.md"
    skills_dir = ness_dir / "skills"


    model = CodexChatModel(
        model=model_name,
        auth=auth,
        reasoning_effort=reasoning_effort,
        prompt_cache_key=str(uuid.uuid5(uuid.NAMESPACE_URL, f"ness-agent:{thread_id}")),
    )

    ness_options = NessAgentOptions(
        yolo_mode=True,
        enable_approval=False,
        auto_save_threads=True,
        session_end_reflection=False,
        project_root=project_root,
        ness_dir=ness_dir,
    )

    agent = NessAgent(
        model=model,
        prompt=PromptLayersConfig(),
        options=ness_options,
        memory=MemoryConfig(
            project_memory=project_memory,
            user_memory=ness_dir / "USER.md",
            session_memory_dir=ness_dir / "runtime" / "sessions",
        ),
        modes=ModeConfig(default="act", plans_dir=ness_dir / "plans"),
        skills_dir=skills_dir,
        question_handler=auto_answer_question,
    )

    agent.config.thread_store.append_event(thread_id, {"kind": "user", "content": instructions})

    session = agent.session(
        thread_id=thread_id,
        mode="act",
        git_available=(project_root / ".git").exists(),
        vision=True
    )

    result = await session.run(instructions, mode="act")
    if result.assistant_message:
        sys.stdout.write(result.assistant_message.rstrip() + "\n")

    errors = []
    for event in result.events:
        if event.kind == "error":
            errors.append(str(event.data.get("message") or "unknown Ness session error"))
    
    if errors:
        for message in errors:
            print(f"Ness session error: {message}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run()))
