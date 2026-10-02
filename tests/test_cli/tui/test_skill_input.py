from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from prompt_toolkit.document import Document
from prompt_toolkit.keys import Keys

from ness_agent import SessionEvent
from ness_agent.skills import SkillLoader, default_skill_search_dirs
from ness_cli.session import TurnRequest
from ness_cli.tui.app import TuiApp
from ness_cli.tui.commands.context import skill_command
from ness_cli.tui.controller import ThreadRuntime
from ness_cli.tui.layout import build_key_bindings
from ness_cli.tui.sink import NullPromptDriver, TuiRenderSink
from ness_cli.tui.transcript.store import TranscriptStore


class SkillSession:
    thread_id = "skills-thread"
    mode = "act"

    def __init__(self):
        self.catalog = (
            {"name": "review", "description": "Review changes", "available": True},
            {"name": "write", "description": "Edit prose", "available": True},
            {"name": "review", "description": "Lower priority duplicate"},
            {"name": "disabled", "description": "Hidden", "available": False},
        )
        self.requests = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.release.set()

    def skills(self):
        return self.catalog, ()

    async def stream(self, request):
        self.requests.append(request)
        self.started.set()
        await self.release.wait()
        yield SessionEvent("assistant_final", {"content": "done"})


@pytest.fixture
def skill_app(tmp_path):
    app = TuiApp(
        SimpleNamespace(
            paths=SimpleNamespace(
                cli_history=tmp_path / "cache" / "cli_history",
                project_root=tmp_path,
                ness_dir=tmp_path / ".ness",
            )
        )
    )
    session = SkillSession()
    sink = TuiRenderSink(
        TranscriptStore(), invalidate=lambda: None, prompts=NullPromptDriver()
    )
    app.controller._selected = ThreadRuntime(session, sink=sink)
    return app, session


def type_prompt(app, text, cursor=None):
    app.input_buffer.set_document(
        Document(text, cursor_position=len(text) if cursor is None else cursor)
    )


def press_tab(app):
    binding = next(
        binding
        for binding in build_key_bindings(app).bindings
        if binding.keys == (Keys.ControlI,) and binding.filter()
    )
    binding.handler(SimpleNamespace(app=SimpleNamespace(invalidate=lambda: None)))


async def send_prompt(app):
    app.submit_input()
    if app._turn_tasks:
        await asyncio.gather(*app._turn_tasks)


def test_dollar_opens_editable_skill_list_in_chrome(skill_app):
    app, _ = skill_app
    app.input_buffer.insert_text("Please use $")

    assert app._menu.kind == "skill"
    assert [item.key for item in app._menu.items] == ["review", "write"]
    assert app._menu.items[0].description == "Review changes"
    assert app.menu_height().preferred > 0
    chrome = "".join(text for _, text in app.menu_fragments())
    assert "skills" in chrome and "review" in chrome and "⇥ insert" in chrome
    assert "".join(text for _, text in app.prompt_fragments()) == "act > "
    assert not app.input_buffer.read_only()
    assert app.input_buffer.text == "Please use $"


def test_dollar_picker_uses_shared_skill_discovery(skill_app, monkeypatch, tmp_path):
    app, session = skill_app
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    for root, name in (
        (tmp_path / ".agents/skills", "project-skill"),
        (home / ".agents/skills", "personal-skill"),
        (tmp_path / ".ness/skills", "legacy-skill"),
    ):
        path = root / name / "SKILL.md"
        path.parent.mkdir(parents=True)
        path.write_text(f"---\nname: {name}\ndescription: Test skill\n---\nBody\n")
    session.catalog = SkillLoader(
        skills_dirs=default_skill_search_dirs(tmp_path)
    ).discover()

    app.input_buffer.insert_text("$")

    assert [item.key for item in app._menu.items] == ["project-skill", "personal-skill"]
    press_tab(app)
    assert app.input_buffer.text == "$project-skill "
    assert app._skills_for_text(app.input_buffer.text) == ("project-skill",)


@pytest.mark.parametrize("query", ["wri", "PROSE"])
def test_dollar_filters_skill_names_and_descriptions(skill_app, query):
    app, _ = skill_app
    app.input_buffer.insert_text("$" + query)
    assert [item.key for item in app._menu.items] == ["write"]
    press_tab(app)
    assert app.input_buffer.text == "$write "
    assert app._skills_for_text(app.input_buffer.text) == ("write",)
    assert not app.menu_open


def test_selection_only_requests_skills_for_that_prompt(skill_app):
    app, session = skill_app

    async def exercise():
        app.input_buffer.insert_text("$rev")
        press_tab(app)
        app.input_buffer.insert_text("check this and $")
        app.move_menu(1)
        press_tab(app)
        app.input_buffer.insert_text("improve wording")
        await send_prompt(app)
        assert session.requests == [
            TurnRequest(
                "$review check this and $write improve wording",
                requested_skills=("review", "write"),
            )
        ]
        assert not app._selected_skills
        type_prompt(app, "$review")
        await send_prompt(app)
        assert session.requests[-1] == TurnRequest("$review")

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "text", ["$", "$review", "$5", "$$", "cost$review", r"\$review"]
)
@pytest.mark.parametrize("dismiss", [False, True])
def test_unselected_dollar_is_literal_when_sent(skill_app, text, dismiss):
    app, session = skill_app

    async def exercise():
        type_prompt(app, text)
        if dismiss and app.menu_open:
            app.cancel_prompt()
            assert app.input_buffer.text == text
        await send_prompt(app)
        assert session.requests == [TurnRequest(text)]
        assert not app.menu_open

    asyncio.run(exercise())


def test_deleting_marker_removes_request_and_retyping_stays_literal(skill_app):
    app, session = skill_app

    async def exercise():
        app.input_buffer.insert_text("$")
        press_tab(app)
        app.input_buffer.delete_before_cursor(len(app.input_buffer.text))
        app.input_buffer.insert_text("$review")
        await send_prompt(app)
        assert session.requests == [TurnRequest("$review")]

    asyncio.run(exercise())


def test_completion_in_middle_preserves_surrounding_prompt(skill_app):
    app, _ = skill_app
    type_prompt(app, "Use $rev for this", cursor=len("Use $rev"))
    press_tab(app)
    assert app.input_buffer.text == "Use $review  for this"
    assert app.input_buffer.cursor_position == len("Use $review ")
    assert app._skills_for_text(app.input_buffer.text) == ("review",)


def test_queued_skill_selections_stay_with_each_prompt(skill_app):
    app, session = skill_app

    async def exercise():
        session.release.clear()
        type_prompt(app, "$rev")
        press_tab(app)
        app.submit_input()
        await session.started.wait()
        first_task = next(iter(app._turn_tasks))

        type_prompt(app, "$wri")
        press_tab(app)
        app.submit_input()
        type_prompt(app, "ordinary $review")
        app.submit_input()
        assert [
            request.requested_skills for request in app.controller._selected.prompts
        ] == [("write",), ()]

        session.release.set()
        await first_task
        assert session.requests == [
            TurnRequest("$review", requested_skills=("review",)),
            TurnRequest("$write", requested_skills=("write",)),
            TurnRequest("ordinary $review"),
        ]

    asyncio.run(exercise())


def test_empty_skill_list_keeps_dollar_literal(skill_app):
    app, session = skill_app
    session.catalog = ()

    async def exercise():
        type_prompt(app, "$")
        assert app._menu.items == []
        assert "no matches" in "".join(text for _, text in app.menu_fragments()).lower()
        press_tab(app)
        assert app.input_buffer.text == "$"
        await send_prompt(app)
        assert session.requests == [TurnRequest("$")]

    asyncio.run(exercise())


def test_multiline_paste_after_dollar_preserves_payload(skill_app):
    app, session = skill_app

    async def exercise():
        app.input_buffer.insert_text("$ ")
        app.input_buffer.insert_text("$")
        assert app.menu_open
        app.insert_paste("first line\nsecond line")
        await send_prompt(app)
        assert session.requests == [TurnRequest("$ $first line\nsecond line")]

    asyncio.run(exercise())


def test_file_completion_still_works_and_shell_dollars_do_not_open_skills(
    skill_app, tmp_path
):
    app, _ = skill_app
    (tmp_path / "notes.md").write_text("notes")
    type_prompt(app, "@notes")
    assert app._menu.kind == "mention"
    press_tab(app)
    assert app.input_buffer.text == "@notes.md "
    assert not app._selected_skills
    type_prompt(app, "!echo $review")
    assert not app.menu_open


def test_named_skill_command_points_to_dollar_picker():
    errors = []
    context = SimpleNamespace(renderer=SimpleNamespace(error=errors.append))
    asyncio.run(skill_command(context, "review"))
    assert len(errors) == 1 and "Type $" in errors[0]
