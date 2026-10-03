from __future__ import annotations

import asyncio
import runpy
import sys
from pathlib import Path

import pytest

from ness_agent.tools.ask import question, set_question_runtime


@pytest.fixture(params=["ness_session.py", "codex/ness_session.py"])
def eval_question_handler(request, monkeypatch):
    if request.param.startswith("codex/"):
        from evals.codex import codex_chat_model

        # The runner checks its pinned sandbox provider at import time. The
        # callback needs no provider, credentials, or model initialization.
        monkeypatch.setitem(sys.modules, "codex_chat_model", codex_chat_model)
        monkeypatch.setattr(codex_chat_model, "eval_provider_types", lambda: (object, object))

    path = Path(__file__).resolve().parents[1] / "evals" / request.param
    return runpy.run_path(str(path))["auto_answer_question"]


def test_eval_question_handler_works_through_sdk_tool(eval_question_handler):
    async def invoke():
        set_question_runtime(eval_question_handler)
        try:
            return await question.ainvoke(
                {
                    "questions": [
                        {
                            "id": "backend",
                            "prompt": "Which backend?",
                            "options": [
                                {"id": "redis", "label": "Redis"},
                                {"id": "memory", "label": "In-memory", "recommended": True},
                            ],
                        },
                        {
                            "prompt": "Which format?",
                            "options": [{"label": "JSON"}, {"label": "YAML"}],
                        },
                    ]
                }
            )
        finally:
            set_question_runtime(None)

    assert asyncio.run(invoke()) == (
        "User clarifications:\n"
        "- Q: Which backend?\n"
        "  A: In-memory\n"
        "  Note: auto-answered (headless agent)\n"
        "- Q: Which format?\n"
        "  A: JSON\n"
        "  Note: auto-answered (headless agent)"
    )
