"""Offline headless lifecycle probe, reusable against an installed wheel."""

from __future__ import annotations

import asyncio
import io
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from ness_agent import ThreadStore, message_to_text
from ness_cli.config import ConfigManager, ConfigPatch
from ness_cli.headless import run_headless, run_headless_turn
from ness_cli.paths import resolve_paths
from ness_cli.providers import AuthState, ModelInfo, ProviderAdapter, ProviderStatus
from ness_cli.providers.registry import ProviderRegistry
from ness_cli.runtime import HeadlessOptions, open_headless_runtime
from ness_cli.session import CodingSession, TurnRequest


async def exercise_headless(project_root: Path) -> None:
    requests = []
    providers = []
    streamed = []
    sessions = []

    class Model(BaseChatModel):
        @property
        def _llm_type(self):
            return "offline-release-probe"

        def bind_tools(self, tools, **kwargs):
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            requests.append(list(messages))
            return ChatResult(generations=[ChatGeneration(message=AIMessage("offline answer"))])

        async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
            requests.append(list(messages))
            for text in ("offline ", "answer"):
                yield ChatGenerationChunk(message=AIMessageChunk(content=text))

    class Provider(ProviderAdapter):
        id = "openrouter"
        display_name = "Offline provider"
        billing_label = "API billing"

        def __init__(self, config):
            super().__init__(config)
            self.close_calls = 0
            providers.append(self)

        def is_authenticated(self):
            return True

        def build_chat_model(self, thread_id, **kwargs):
            return Model()

        def model_info(self, model_name):
            return ModelInfo(id=model_name, name=model_name, context_window=128_000)

        async def models(self, *, refresh=False):
            return (self.model_info("offline/model"),)

        async def status(self, *, refresh=False):
            return ProviderStatus(self.display_name, AuthState(True))

        async def close(self):
            self.close_calls += 1

    def register(registry):
        registry.register("openrouter", Provider)

    original_stream = CodingSession.stream

    async def observe(session, request):
        if session not in sessions:
            sessions.append(session)
        async for event in original_stream(session, request):
            streamed.append(event)
            yield event

    paths = resolve_paths(project_root=project_root)
    ConfigManager.load(paths, environment={}).apply(ConfigPatch({
        "model_provider": "openrouter",
        "model_name": "offline/model",
        "reflection_model_name": "offline/model",
        "reasoning_effort": None,
        "auto_save_threads": True,
        "reflection_token_ratio": 0,
        "session_end_reflection": False,
    }))
    with (
        patch.object(ProviderRegistry, "_register_builtins", register),
        patch.object(CodingSession, "stream", observe),
    ):
        output, errors = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            code = await run_headless("first prompt", options=HeadlessOptions(
                project_root=project_root, yolo=True,
            ))
        assert code == 0, errors.getvalue()
        assert output.getvalue().strip() == "offline answer"
        thread_id = sessions[0].thread_id
        assert f"Resume: ness --resume {thread_id}" in errors.getvalue()
        assert providers[0].close_calls == 1
        assert any(event.kind == "assistant_delta" for event in streamed)
        assert any(event.kind == "assistant_final" for event in streamed)

        store = ThreadStore(threads_dir=paths.threads_dir)
        saved = store.load_thread_events(thread_id)
        assert [(event["kind"], event.get("content")) for event in saved
                if event["kind"] in {"user", "assistant"}] == [
            ("user", "first prompt"), ("assistant", "offline answer"),
        ]
        async with open_headless_runtime(HeadlessOptions(
            project_root=project_root, resume_thread_id=thread_id, yolo=True,
        )) as runtime:
            resumed = await runtime.session()
            assert resumed.thread_id == thread_id
            assert await run_headless_turn(resumed, "second prompt") == ("offline answer", 0)
            restored = await resumed.get_messages()
            assert any(isinstance(message, HumanMessage) and message_to_text(message) == "first prompt"
                       for message in restored)
            assert any(isinstance(message, AIMessage) and message_to_text(message) == "offline answer"
                       for message in restored)
            result = await resumed.finalize_and_save()
            assert result.resume_thread_id == thread_id

        assert [provider.close_calls for provider in providers] == [1, 1]
        await runtime.close()
        assert [provider.close_calls for provider in providers] == [1, 1]
        for session in sessions:
            try:
                async for _event in session.stream(TurnRequest("after close")):
                    raise AssertionError("closed session accepted a turn")
            except RuntimeError as error:
                assert "closed" in str(error)
            else:
                raise AssertionError("closed session accepted a turn")

    final = store.load_thread_events(thread_id)
    assert final[:len(saved)] == saved
    assert [event["content"] for event in final if event["kind"] == "user"] == [
        "first prompt", "second prompt",
    ]
    assert len(requests) == 2
    assert any(isinstance(message, HumanMessage) and message_to_text(message) == "first prompt"
               for message in requests[-1])
    assert any(isinstance(message, AIMessage) and message_to_text(message) == "offline answer"
               for message in requests[-1])


if __name__ == "__main__":
    asyncio.run(exercise_headless(Path(sys.argv[1])))
    print("headless save/resume/close ok")
