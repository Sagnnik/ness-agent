from __future__ import annotations

import sys
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from ness_agent import SessionEvent

from ness_cli.session.turn import TurnOutcome
if TYPE_CHECKING:
    from ness_cli.runtime import HeadlessOptions
    from ness_cli.session import CodingSession


async def auto_answer_questions(
    questions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    answers: list[dict[str, Any]] = []

    for index, question in enumerate(questions, start=1):
        options = list(question.get("options") or [])

        selected = next(
            (
                option
                for option in options
                if isinstance(option, Mapping) and option.get("recommended")
            ),
            None,
        )

        if selected is None and options:
            selected = options[0]

        if not isinstance(selected, Mapping):
            selected = {"id": "0", "label": "proceed"}

        answers.append(
            {
                "id": question.get("id", str(index)),
                "selected": {
                    "id": selected.get("id"),
                    "label": selected.get("label"),
                },
                "note": "auto-answered (headless mode)",
            }
        )

    return answers


def merge_prompt_parts(
    prompt_parts: list[str] | None,
    stdin_text: str | None,
) -> str | None:
    query = " ".join(part for part in (prompt_parts or []) if part).strip()
    piped = (stdin_text or "").strip()

    if piped and query:
        return f"{piped}\n\n{query}"

    return piped or query or None


async def run_headless_turn(
    coding: CodingSession,
    prompt: str,
) -> tuple[str, int]:
    final_text = ""
    outcome = TurnOutcome()

    try:
        from ness_cli.session import TurnRequest

        async for event in coding.stream(
            TurnRequest(message=prompt),
        ):
            if not isinstance(event, SessionEvent):
                continue
            outcome = outcome.observe(event)

            if event.kind == "assistant_final":
                content = str(event.data.get("content") or "").strip()
                if content:
                    final_text = content

            elif event.kind == "error":
                message = event.data.get("message") or event.data
                print(f"error: {message}", file=sys.stderr)

            elif event.kind == "warning":
                message = event.data.get("message") or event.data
                print(f"warning: {message}", file=sys.stderr)

            elif event.kind == "interrupted":
                partial = str(
                    event.data.get("partial_text") or "",
                ).strip()

                if partial:
                    final_text = partial

    except KeyboardInterrupt:
        return final_text, outcome.with_interruption().exit_code
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return final_text, outcome.with_error(str(exc)).exit_code

    return final_text, outcome.exit_code


async def run_headless(
    prompt: str,
    *,
    options: HeadlessOptions,
) -> int:
    from dataclasses import replace

    from ness_cli.runtime import open_headless_runtime

    options = replace(
        options,
        question_handler=auto_answer_questions,
        approval_handler=None,
    )

    try:
        async with open_headless_runtime(options) as runtime:
            session = await runtime.session()

            text, code = await run_headless_turn(session, prompt)

            if text:
                sys.stdout.write(text + "\n")

            save_result = await session.finalize_and_save()
            resume_id = getattr(save_result, "resume_thread_id", None)

            if resume_id:
                print(
                    f"Resume: ness --resume {resume_id}",
                    file=sys.stderr,
                )

            for warning in runtime.warnings:
                print(f"warning: {warning}", file=sys.stderr)

            return code

    except LookupError:
        thread_id = options.resume_thread_id
        print(
            f"error: no saved thread '{thread_id}'",
            file=sys.stderr,
        )
        return 1
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
