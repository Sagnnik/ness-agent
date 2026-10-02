from __future__ import annotations

import asyncio
from collections import deque
from types import SimpleNamespace

import pytest
from ness_agent import SessionEvent
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.data_structures import Point
from prompt_toolkit.history import FileHistory
from prompt_toolkit.keys import Keys
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType
from prompt_toolkit.utils import get_cwidth

from ness_cli.session import TurnRequest
from ness_cli.tui.app import TuiApp
from ness_cli.tui.commands import command_registry
from ness_cli.tui.commands.context import skill_command
from ness_cli.tui.commands.registry import CommandRegistry, CommandSpec
from ness_cli.tui.input.config import SPECS, _model_items
from ness_cli.tui.controller import ThreadRuntime, TuiController
from ness_cli.tui.input.files import FileIndex
from ness_cli.tui.input.menus import ChecklistResult, MenuItem, MenuState
from ness_cli.tui.layout import build_key_bindings
from ness_cli.tui.sink import NullPromptDriver, TuiRenderSink
from ness_cli.tui.transcript import render
from ness_cli.tui.transcript.store import (
    TranscriptLine,
    TranscriptStore,
    VisualPosition,
)
from ness_cli.tui.transcript.view import TranscriptView


def line(text: str, style: str = "class:test") -> TranscriptLine:
    return TranscriptLine(style, text)


def test_transcript_wraps_rows_and_invalidates_cached_text():
    store = TranscriptStore([line("abcdef")], width=3)
    assert store.total_rows == 2
    assert [store.row_text(index) for index in range(2)] == ["abc", "def"]
    assert store.plain_text() == "abcdef"
    store.append([line("next")])
    assert store.plain_text() == "abcdef\nnext"
    assert store.set_width(4) and store.total_rows == 3
    assert not store.set_width(4)


def test_transcript_tracked_blocks_remain_stable_across_mutations():
    store = TranscriptStore([line("before")])
    first = store.append_tracked([line("one"), line("two")])
    second = store.append_tracked([line("three")])
    store.replace_tracked(first, [line("replacement")])
    assert (first.start, first.count, second.start) == (1, 1, 2)
    store.move_tracked_to_end(first)
    assert [item.text for item in store.lines] == ["before", "three", "replacement"]
    store.delete_tracked(second)
    assert not second.attached
    with pytest.raises(ValueError, match="detached"):
        store.delete_tracked(second)


def test_transcript_raw_mutations_cannot_split_or_overlap_tracked_blocks():
    store = TranscriptStore([line("a")])
    block = store.append_tracked([line("b"), line("c")])
    with pytest.raises(ValueError, match="splits"):
        store.insert(block.start + 1, [line("x")])
    with pytest.raises(ValueError, match="overlaps"):
        store.delete(block.start, 1)


def test_transcript_fragments_are_sliced_with_wrapped_rows():
    store = TranscriptStore(
        [TranscriptLine("", "abcdef", [("class:a", "abc"), ("class:b", "def")])],
        width=4,
    )
    assert store.row_fragments(0) == [("class:a", "abc"), ("class:b", "d")]
    assert store.row_fragments(1) == [("class:b", "ef")]
    assert store.row_fragments(99) == []


def test_transcript_selection_spans_wrapped_rows_and_copies_plain_text():
    store = TranscriptStore(
        [TranscriptLine("", "abcdef", [("class:a", "abc"), ("class:b", "def")])],
        width=3,
    )
    selection = (VisualPosition(0, 1), VisualPosition(1, 2))

    first = store.row_fragments(0, selection=selection)
    second = store.row_fragments(1, selection=selection)

    assert any("class:transcript.selection" in style for style, _ in first)
    assert any("class:transcript.selection" in style for style, _ in second)
    assert store.copy_range(*selection) == "bc\nde"


def test_transcript_view_pauses_follow_mode_and_preserves_thread_scroll_state():
    first = TranscriptStore([line(f"row {index}") for index in range(12)], width=20)
    second = TranscriptStore([line(f"other {index}") for index in range(8)], width=20)
    view = TranscriptView(first)

    view.create_content(20, 4)
    assert view.vertical_scroll == 8 and view.is_following

    view.scroll_by(-3)
    assert view.vertical_scroll == 5 and not view.is_following
    first.append([line("new output")])
    view.create_content(20, 4)
    assert view.vertical_scroll == 5

    view.store = second
    view.create_content(20, 4)
    assert view.vertical_scroll == 4 and view.is_following
    view.store = first
    assert view.vertical_scroll == 5 and not view.is_following

    view.follow_end()
    assert view.vertical_scroll == 9 and view.is_following


def test_transcript_view_mouse_selection_uses_visible_content_rows():
    focused = []
    store = TranscriptStore([line("alpha"), line("bravo")], width=20)
    view = TranscriptView(store, focus=lambda: focused.append(True))
    view.create_content(20, 2)
    modifiers = frozenset()

    view.mouse_handler(
        MouseEvent(Point(1, 0), MouseEventType.MOUSE_DOWN, MouseButton.LEFT, modifiers)
    )
    view.mouse_handler(
        MouseEvent(Point(3, 1), MouseEventType.MOUSE_UP, MouseButton.LEFT, modifiers)
    )

    assert focused == [True]
    assert view.selected_text() == "lpha\nbra"


def test_final_assistant_markdown_is_styled_while_streaming_stays_literal():
    source = (
        "# Heading\n\n- first\n\n> quoted\n\n"
        "[docs](https://example.test)\n\n---\n\n"
        "```python\nprint('ok')\n```"
    )
    streaming = render.assistant_message(source, streaming=True, width=48)
    final = render.assistant_message(source, width=48)

    assert any(line.text == "# Heading" for line in streaming)
    assert all("```" not in line.text for line in final)
    assert "Heading" in "\n".join(line.text for line in final)
    assert "quoted" in "\n".join(line.text for line in final)
    assert "docs" in "\n".join(line.text for line in final)
    assert "https://example.test" not in "\n".join(line.text for line in final)
    assert "print('ok')" in "\n".join(line.text for line in final)
    assert any(
        line.fragments and any(style for style, _ in line.fragments)
        for line in final
    )


def test_assistant_markdown_reflows_from_retained_source():
    store = TranscriptStore(width=24)
    sink = TuiRenderSink(
        store,
        invalidate=lambda: None,
        prompts=NullPromptDriver(),
    )
    source = "## Build status\n\nA paragraph that needs to wrap at narrow widths."
    sink.assistant_message(source)
    narrow_rows = store.total_rows

    sink.reflow_assistants(60)

    assert store.width == 60
    assert store.total_rows < narrow_rows
    assert "Build status" in store.plain_text()


def test_startup_header_is_informative_responsive_and_reflows():
    store = TranscriptStore(width=120)
    sink = TuiRenderSink(
        store,
        invalidate=lambda: None,
        prompts=NullPromptDriver(),
    )
    sink.append_header(
        model="z-ai/glm-5.3-flash",
        provider="openrouter",
        project="/home/sagnnik/projects/liteharness",
        thread_id="thread-12345678901234567890",
        version="0.2.3",
        mode="act",
        approval="on",
        integrations="2 servers, 14 tools",
    )

    wide = store.plain_text()
    assert "NessAgent v0.2.3" in wide
    assert "Mode    : Act (auto-approval)" in wide
    assert "Session : openrouter/z-ai/glm-5.3-flash" in wide
    assert "Project : /home/sagnnik/projects/liteharne…" in wide
    assert "Add-ons : 2 servers, 14 tools" in wide
    assert "Shift+Tab toggle Act/Plan" in wide and "/help" in wide
    assert any(
        "\u2800" <= character <= "\u28ff"
        for row in store.lines
        for character in row.text
    )
    assert any(
        style.startswith("fg:#")
        for row in store.lines
        for style, _ in (row.fragments or ())
    )

    sink.set_header_mode("plan")
    sink.reflow_assistants(80)
    compact = store.plain_text()
    assert "NessAgent v0.2.3" in compact
    assert "Mode    : Plan" in compact
    assert not any("\u2800" <= character <= "\u28ff" for character in compact)

    sink.reflow_assistants(39)
    narrow = store.plain_text()
    assert narrow.startswith("[session]    model openrouter/")
    assert all(get_cwidth(row.text) <= 39 for row in store.lines)


def test_user_message_band_reflows_with_transcript_width():
    store = TranscriptStore(width=32)
    sink = TuiRenderSink(
        store,
        invalidate=lambda: None,
        prompts=NullPromptDriver(),
    )
    sink.user_message("A user message that needs wrapping.")

    band = [line for line in store.lines if line.style == "class:transcript.user"]
    assert len(band) == 4
    assert all(len(line.text) == 32 for line in band)

    sink.reflow_assistants(18)
    band = [line for line in store.lines if line.style == "class:transcript.user"]
    assert len(band) == 5
    assert all(len(line.text) == 18 for line in band)


def test_usage_footer_uses_compact_symbols_and_cache_rate():
    lines = render.usage(
        {
            "input_tokens": 4_716,
            "output_tokens": 204,
            "cached_input_tokens": 0,
            "cost_usd": 0.0004,
        }
    )

    assert lines[0].text == "↑ 4,716  ↓ 204  ⟳ 0 (0%)  $0.0004"
    assert lines[1].text == ""


def test_tool_rendering_batches_reads_suppresses_success_and_colors_diffs():
    store = TranscriptStore()
    sink = TuiRenderSink(
        store,
        invalidate=lambda: None,
        prompts=NullPromptDriver(),
    )
    sink.tool_call("read", {"path": "a.py"})
    sink.tool_call("read", {"path": "b.py"})
    sink.tool_result("read", "file contents")
    sink.tool_result("read", "more file contents")

    sink.tool_call("edit", {"path": "a.py", "old_string": "a", "new_string": "b"})
    sink.tool_result(
        "edit",
        "Applied 1 edit to a.py\ndiff:\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-a\n+b",
    )

    text = store.plain_text()
    assert "read ×2" in text and "a.py" in text and "b.py" in text
    assert "file contents" not in text
    assert "Applied 1 edit" in text
    styles = {item.style for item in store.lines}
    assert "class:transcript.diff.add" in styles
    assert "class:transcript.diff.del" in styles
    assert "class:transcript.diff.hunk" in styles


def test_shell_panel_is_bounded_and_todos_replace_in_place():
    store = TranscriptStore()
    sink = TuiRenderSink(
        store,
        invalidate=lambda: None,
        prompts=NullPromptDriver(),
    )
    sink.tool_result("shell", "x" * 20_100, exit_status="ok")
    sink.todos([{"id": "1", "content": "old task", "status": "pending"}])
    sink.todos([{"id": "2", "content": "current task", "status": "in_progress"}])

    text = store.plain_text()
    assert "showing last 20,000 characters" in text
    assert "old task" not in text
    assert "current task" in text
    assert text.count("todos") == 1


def test_menu_selection_wraps_and_visible_window_tracks_selection():
    items = [MenuItem(str(index), f"item {index}") for index in range(10)]
    menu = MenuState("test", "Test", items)
    menu.move(-1)
    assert menu.selected == items[-1]
    start, visible = menu.visible(4)
    assert start == 6 and visible[-1] == items[-1]
    empty = MenuState("empty", "Empty")
    empty.move(2)
    assert empty.selected is None and empty.index == 0


def test_skill_checklist_toggles_and_keeps_every_row_on_one_line(tmp_path):
    async def exercise():
        app = _input_app(tmp_path)
        app._terminal_size = lambda: (80, 24)
        items = [
            MenuItem(
                "codebase-design",
                "codebase-design",
                "Compare several architectural approaches before implementation.",
            ),
            MenuItem(
                "unslop",
                "unslop",
                "A description that is deliberately long enough to be truncated " * 3,
            ),
        ]
        app._menu = MenuState(
            "checklist",
            "skills (2)",
            items,
            hint="Up/Down select · Left/Right toggle · Enter view · Esc close",
            horizontal_keys=frozenset(item.key for item in items),
            disabled_keys=set(),
        )
        app._menu_future = asyncio.get_running_loop().create_future()

        rendered = "".join(text for _, text in app.menu_fragments())
        rows = rendered.splitlines()
        assert "[ ] codebase-design" in rows[1]
        assert rows[2].endswith("...")
        assert all(get_cwidth(row) <= 80 for row in rows)

        app.change_menu_horizontal()
        assert "codebase-design" in app._menu.disabled_keys
        assert "[X] codebase-design" in "".join(
            text for _, text in app.menu_fragments()
        )

        app.submit_input()
        assert await app._menu_future == ChecklistResult(
            selected_key="codebase-design",
            disabled_keys=frozenset({"codebase-design"}),
        )

    asyncio.run(exercise())


def test_skill_command_opens_checklist_and_renders_selected_detail(tmp_path):
    source = str(tmp_path / ".agents/skills/unslop/SKILL.md")
    alternate = str(tmp_path / ".cursor/skills/unslop/SKILL.md")
    skill_id = "sha256:unslop"

    class SkillUI:
        def __init__(self):
            self.call = None

        async def choose_checklist(self, title, items, **kwargs):
            self.call = (title, items, kwargs)
            return ChecklistResult(skill_id, frozenset())

    class SkillRenderer:
        def __init__(self):
            self.detail = None

        def skill_detail(self, **kwargs):
            self.detail = kwargs

        def warning(self, _text):
            raise AssertionError("unexpected warning")

    ui = SkillUI()
    renderer = SkillRenderer()
    applied = []
    session = SimpleNamespace(
        project_root=tmp_path,
        update_skill_access=lambda disabled: applied.append(disabled),
        skills=lambda: (
            (
                {
                    "name": "unslop",
                    "skill_id": skill_id,
                    "source": source,
                    "source_id": source,
                    "sources": [
                        {"source_id": source, "source": source},
                        {"source_id": alternate, "source": alternate},
                    ],
                    "description": "First line\nsecond line",
                    "available": True,
                },
            ),
            (),
        ),
    )
    context = SimpleNamespace(session=session, ui=ui, renderer=renderer)

    asyncio.run(skill_command(context, ""))

    assert ui.call[0] == "skills (1)"
    assert ui.call[1] == [
        MenuItem(
            skill_id,
            "unslop",
            description="First line second line",
            search_terms=(source, alternate),
        )
    ]
    assert ui.call[2]["disabled_keys"] == frozenset()
    assert applied == [frozenset()]
    assert renderer.detail == {
        "name": "unslop",
        "source": ".agents/skills/unslop/SKILL.md",
        "alternate_sources": (".cursor/skills/unslop/SKILL.md",),
        "description": "First line\nsecond line",
    }


def test_skill_detail_keeps_full_description_and_reflows():
    store = TranscriptStore(width=56)
    sink = TuiRenderSink(
        store,
        invalidate=lambda: None,
        prompts=NullPromptDriver(),
    )
    description = (
        "Use **plain language** and retain every relevant detail in the final text."
    )

    sink.skill_detail(
        name="unslop",
        source="~/.agents/skills/unslop/SKILL.md",
        alternate_sources=("~/.cursor/skills/unslop/SKILL.md",),
        description=description,
    )

    assert "skill · unslop" in store.plain_text()
    assert "source       ~/.agents/skills/unslop/SKILL.md" in store.plain_text()
    assert "also found   ~/.cursor/skills/unslop/SKILL.md" in store.plain_text()
    assert "retain every relevant detail" in store.plain_text()
    sink.reflow_assistants(28)
    assert "retain" in store.plain_text() and "detail" in store.plain_text()


def test_file_index_filters_hidden_dirs_and_ranks_name_matches(tmp_path, monkeypatch):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("", encoding="utf-8")
    (tmp_path / "myapp.py").write_text("", encoding="utf-8")
    (tmp_path / ".ness").mkdir()
    (tmp_path / ".ness" / "secret.py").write_text("", encoding="utf-8")
    index = FileIndex(tmp_path)
    monkeypatch.setattr(index, "_git_files", lambda: None)
    results = index.search("app.py")
    assert [item.key for item in results] == ["src/app.py", "myapp.py"]
    assert all("secret" not in item.key for item in index.search(""))


class Renderer:
    def __init__(self):
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def error(self, value):
        self.errors.append(value)

    def warning(self, value):
        self.warnings.append(value)


def command_context(tmp_path):
    renderer = Renderer()
    submitted: list[str] = []

    async def submit(value):
        submitted.append(value)

    return (
        SimpleNamespace(
            renderer=renderer,
            runtime=SimpleNamespace(paths=SimpleNamespace(ness_dir=tmp_path / ".ness")),
            threads=SimpleNamespace(submit=submit),
        ),
        renderer,
        submitted,
    )


def test_command_registry_dispatches_alias_and_enforces_busy_safety(tmp_path):
    calls = []

    async def handler(context, args):
        calls.append(args)

    registry = CommandRegistry(
        [CommandSpec("test", "summary", "group", handler, aliases=("t",))]
    )
    context, renderer, _ = command_context(tmp_path)
    assert not asyncio.run(registry.dispatch(context, "plain text", busy=False))
    assert asyncio.run(registry.dispatch(context, "/t value", busy=False))
    assert calls == ["value"]
    assert asyncio.run(registry.dispatch(context, "/test", busy=True))
    assert renderer.warnings and calls == ["value"]


def test_command_registry_reports_unknown_and_loads_project_prompts(tmp_path):
    registry = CommandRegistry([])
    context, renderer, submitted = command_context(tmp_path)
    commands = tmp_path / ".ness" / "commands"
    commands.mkdir(parents=True)
    (commands / "review.md").write_text("Review {{args}}", encoding="utf-8")
    assert registry.matches("rev", commands_dir=commands)[0].name == "review"
    assert asyncio.run(registry.dispatch(context, "/review module.py", busy=False))
    assert submitted == ["Review module.py"]
    assert asyncio.run(registry.dispatch(context, "/missing", busy=False))
    assert "Unknown command" in renderer.errors[-1]


def test_builtin_commands_have_unique_names_and_release_critical_entries():
    specs = command_registry().specs
    names = [spec.name for spec in specs]
    assert len(names) == len(set(names))
    assert {"save", "new", "threads", "rollback", "fork", "export", "compact"} <= set(names)
    by_name = {spec.name: spec for spec in specs}
    assert by_name["new"].busy_safe
    assert by_name["threads"].busy_safe


class Session:
    def __init__(self, thread_id="t"):
        self.thread_id = thread_id
        self.mode = "act"
        self.cancelled = 0

    def cancel(self):
        self.cancelled += 1

    def toggle_mode(self):
        self.mode = "plan" if self.mode == "act" else "act"
        return self.mode


def test_controller_cancel_clears_queue_and_arms_backstop():
    async def exercise():
        session = Session()
        controller = TuiController(SimpleNamespace(), SimpleNamespace(), changed=lambda: None)
        waiter = asyncio.create_task(asyncio.Event().wait())
        selected = ThreadRuntime(session, task=waiter, prompts=deque([TurnRequest("later")]))
        controller._selected = selected
        try:
            assert controller.cancel()
            assert session.cancelled == 1 and selected.cancelling and not selected.prompts
            assert selected.cancel_backstop is not None
            assert controller.toggle_mode() == "plan"
        finally:
            if selected.cancel_backstop:
                selected.cancel_backstop.cancel()
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter

    asyncio.run(exercise())


def test_controller_requires_initialization_and_idle_operations():
    controller = TuiController(SimpleNamespace(), SimpleNamespace(), changed=lambda: None)
    with pytest.raises(RuntimeError, match="not initialized"):
        _ = controller.session
    session = Session()
    controller._selected = ThreadRuntime(session)
    assert not controller.cancel()


def test_controller_refreshes_todos_after_todo_tool_events():
    class TodoSession(Session):
        def __init__(self):
            super().__init__("todo-thread")
            self.todos = []

        async def stream(self, _request):
            yield SessionEvent(
                "tool_start",
                {"name": "todo", "args": {"todos": []}},
            )
            self.todos = [
                {"id": "1", "content": "verify migration", "status": "in_progress"}
            ]
            yield SessionEvent("tool_end", {"name": "todo", "content": "Updated 1 todo"})
            yield SessionEvent("assistant_final", {"content": "Working on it."})

        async def get_todos(self):
            return list(self.todos)

        def history(self):
            return ()

        def install_interaction_handlers(self, **_handlers):
            pass

    class TodoRuntime:
        resume_thread_id = None

        def __init__(self, session):
            self.session = session

        async def initial_session(self):
            return self.session

    async def exercise():
        store = TranscriptStore()
        sink = TuiRenderSink(
            store,
            invalidate=lambda: None,
            prompts=NullPromptDriver(),
        )
        controller = TuiController(
            TodoRuntime(TodoSession()),
            sink,
            changed=lambda: None,
        )
        await controller.initialize()
        assert await controller.submit("continue")
        text = store.plain_text()
        assert "verify migration" in text
        assert text.count("todos") == 1
        assert "Updated 1 todo" not in text

    asyncio.run(exercise())


class LiveSession(Session):
    def __init__(self, thread_id):
        super().__init__(thread_id)
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.save_calls = 0
        self.handlers = None

    async def stream(self, request):
        self.started.set()
        await self.release.wait()
        yield SessionEvent(
            "assistant_final",
            {"content": f"answer from {self.thread_id}"},
        )

    async def save(self):
        self.save_calls += 1

    def history(self):
        return ()

    def list_threads(self):
        return []

    def usage_summary(self):
        return {"turns": 0, "cost_usd": 0.0}

    def install_interaction_handlers(self, **handlers):
        self.handlers = handlers

    def cancel(self):
        super().cancel()
        self.release.set()


class LiveRuntime:
    resume_thread_id = None

    def __init__(self, *sessions):
        self.sessions = deque(sessions)
        self.resume_calls = []

    async def initial_session(self):
        return self.sessions.popleft()

    async def new_session(self, **_kwargs):
        return self.sessions.popleft()

    async def resume_session(self, thread_id):
        self.resume_calls.append(thread_id)
        raise AssertionError("live threads must not be replayed")


def _sink_factory(sinks):
    def create(thread_id):
        sink = TuiRenderSink(
            TranscriptStore(),
            invalidate=lambda: None,
            prompts=NullPromptDriver(),
        )
        sinks[thread_id] = sink
        return sink

    return create


def test_option_question_returns_optional_note():
    class Prompts:
        async def choose(self, *_args, **_kwargs):
            return "second"

        async def request_input(self, label, **_kwargs):
            assert "optional" in label
            return "Only if the migration is reversible."

    sink = TuiRenderSink(
        TranscriptStore(),
        invalidate=lambda: None,
        prompts=Prompts(),
    )

    answers = asyncio.run(
        sink.ask_questions(
            [
                {
                    "id": "strategy",
                    "prompt": "Which strategy?",
                    "allow_note": True,
                    "options": [
                        {"id": "first", "label": "First"},
                        {"id": "second", "label": "Second"},
                    ],
                }
            ]
        )
    )

    assert answers == [
        {
            "id": "strategy",
            "selected": {"id": "second", "label": "Second"},
            "note": "Only if the migration is reversible.",
        }
    ]


def test_option_question_uses_one_integrated_choice_and_note_form():
    class Prompts:
        def __init__(self):
            self.calls = []

        async def choose_with_note(
            self,
            title,
            items,
            *,
            initial_key=None,
            allow_note=True,
        ):
            self.calls.append((title, items, initial_key, allow_note))
            return "second", "Only if the migration is reversible."

        async def choose(self, *_args, **_kwargs):
            raise AssertionError("the option picker must remain open with the note")

        async def request_input(self, *_args, **_kwargs):
            raise AssertionError("the note must not open a second prompt")

    prompts = Prompts()
    sink = TuiRenderSink(
        TranscriptStore(),
        invalidate=lambda: None,
        prompts=prompts,
    )

    answers = asyncio.run(
        sink.ask_questions(
            [
                {
                    "id": "strategy",
                    "prompt": "Which strategy?",
                    "allow_note": True,
                    "options": [
                        {"id": "first", "label": "First"},
                        {
                            "id": "second",
                            "label": "Second",
                            "recommended": True,
                        },
                    ],
                }
            ]
        )
    )

    assert prompts.calls[0][0] == "Which strategy?"
    assert prompts.calls[0][2:] == ("second", True)
    assert answers[0]["selected"] == {"id": "second", "label": "Second"}
    assert answers[0]["note"] == "Only if the migration is reversible."


def test_free_form_question_does_not_cancel_later_questions():
    class Prompts:
        async def choose(self, *_args, **_kwargs):
            return "yes"

        async def request_input(self, *_args, **_kwargs):
            return "Because it preserves history."

    sink = TuiRenderSink(
        TranscriptStore(),
        invalidate=lambda: None,
        prompts=Prompts(),
    )

    answers = asyncio.run(
        sink.ask_questions(
            [
                {"id": "why", "prompt": "Why?"},
                {
                    "id": "confirm",
                    "prompt": "Proceed?",
                    "allow_note": False,
                    "options": [{"id": "yes", "label": "Yes"}],
                },
            ]
        )
    )

    assert answers == [
        {
            "id": "why",
            "selected": None,
            "note": "Because it preserves history.",
        },
        {
            "id": "confirm",
            "selected": {"id": "yes", "label": "Yes"},
            "note": "",
        },
    ]


def test_idle_ctrl_c_clears_a_draft_without_exiting():
    class DraftBuffer:
        text = "unfinished prompt"

        def reset(self):
            self.text = ""

    exits = []
    invalidations = []
    ui = SimpleNamespace(
        _menu=None,
        _prompt_future=None,
        controller=SimpleNamespace(busy=False),
        input_buffer=DraftBuffer(),
        _pending_images={1: "data"},
        _selected_skills={"review"},
        _image_counter=1,
        request_exit=lambda: exits.append(True),
        invalidate=lambda: invalidations.append(True),
    )

    TuiApp.interrupt_or_exit(ui)

    assert ui.input_buffer.text == ""
    assert ui._pending_images == {}
    assert ui._selected_skills == set()
    assert ui._image_counter == 0
    assert invalidations == [True]
    assert exits == []

    TuiApp.interrupt_or_exit(ui)
    assert exits == []


def test_shell_escape_uses_legacy_timeout_and_output_cap(monkeypatch, tmp_path):
    observed = {}

    class Process:
        returncode = 0

        async def communicate(self):
            return b"x" * 20_001, b""

    async def create_process(command, **kwargs):
        observed.update(command=command, kwargs=kwargs)
        return Process()

    async def wait_for(awaitable, *, timeout):
        observed["timeout"] = timeout
        return await awaitable

    class Sink:
        def __init__(self):
            self.result = None

        def tool_call(self, *_args, **_kwargs):
            pass

        def tool_result(self, name, content, *, exit_status=None):
            self.result = (name, content, exit_status)

        def warning(self, text):
            raise AssertionError(text)

        def error(self, text):
            raise AssertionError(text)

    monkeypatch.setattr(asyncio, "create_subprocess_shell", create_process)
    monkeypatch.setattr(asyncio, "wait_for", wait_for)
    sink = Sink()
    ui = SimpleNamespace(
        runtime=SimpleNamespace(paths=SimpleNamespace(project_root=tmp_path)),
        _SHELL_TIMEOUT=TuiApp._SHELL_TIMEOUT,
        _SHELL_OUTPUT_CAP=TuiApp._SHELL_OUTPUT_CAP,
    )

    assert asyncio.run(TuiApp._run_shell(ui, "generate", sink=sink))
    assert observed["timeout"] == 60.0
    assert observed["command"] == "generate"
    assert observed["kwargs"]["cwd"] == tmp_path
    assert sink.result is not None
    name, content, status = sink.result
    assert name == "shell" and status == "ok"
    first_line, message = content.splitlines()
    assert len(first_line) == 20_000
    assert message == "... (truncated, 20001 chars total)"


def test_switching_threads_keeps_source_turn_running_and_output_isolated():
    async def exercise():
        first = LiveSession("thread-1")
        second = LiveSession("thread-2")
        runtime = LiveRuntime(first, second)
        sinks = {}
        activated = []
        controller = TuiController(
            runtime,
            sink_factory=_sink_factory(sinks),
            selected_changed=activated.append,
            changed=lambda: None,
        )
        await controller.initialize()
        assert activated == [sinks["thread-1"]]

        first_task = asyncio.create_task(controller.submit("first prompt"))
        await first.started.wait()
        await controller.new_thread()
        assert controller.thread_id == "thread-2"
        assert activated[-1] is sinks["thread-2"]
        assert not first_task.done()
        assert first.save_calls == 0
        assert first.handlers is not second.handlers

        second_task = asyncio.create_task(controller.submit("second prompt"))
        await second.started.wait()
        first.release.set()
        assert await first_task
        assert "answer from thread-1" in sinks["thread-1"].store.plain_text()
        assert "answer from thread-1" not in sinks["thread-2"].store.plain_text()

        await controller.resume_thread("thread-1")
        assert controller.sink is sinks["thread-1"]
        assert activated[-1] is sinks["thread-1"]
        assert runtime.resume_calls == []
        second.release.set()
        assert await second_task
        assert "answer from thread-2" in sinks["thread-2"].store.plain_text()
        assert "answer from thread-2" not in sinks["thread-1"].store.plain_text()

    asyncio.run(exercise())


def test_background_interaction_waits_until_source_thread_is_selected():
    async def exercise():
        first = LiveSession("thread-1")
        second = LiveSession("thread-2")
        sinks = {}
        controller = TuiController(
            LiveRuntime(first, second),
            sink_factory=_sink_factory(sinks),
            changed=lambda: None,
        )
        await controller.initialize()
        await controller.new_thread()

        waiting = asyncio.create_task(controller.begin_interaction("thread-1"))
        await asyncio.sleep(0)
        assert not waiting.done()
        rows = controller.list_threads()
        first_row = next(row for row in rows if row["thread_id"] == "thread-1")
        assert first_row["live_status"] == "waiting for input"

        await controller.resume_thread("thread-1")
        await waiting
        controller.finish_interaction("thread-1")
        assert not controller._runtimes["thread-1"].waiting_for_input

    asyncio.run(exercise())


def test_controller_close_cancels_every_live_thread():
    async def exercise():
        first = LiveSession("thread-1")
        second = LiveSession("thread-2")
        controller = TuiController(
            LiveRuntime(first, second),
            sink_factory=_sink_factory({}),
            changed=lambda: None,
        )
        await controller.initialize()
        first_task = asyncio.create_task(controller.submit("first"))
        await first.started.wait()
        await controller.new_thread()
        second_task = asyncio.create_task(controller.submit("second"))
        await second.started.wait()

        await controller.close()
        await asyncio.gather(first_task, second_task)
        assert first.cancelled == 1
        assert second.cancelled == 1

    asyncio.run(exercise())


def _input_app(tmp_path):
    runtime = SimpleNamespace(
        paths=SimpleNamespace(
            cli_history=tmp_path / "cache" / "cli_history",
            project_root=tmp_path,
            ness_dir=tmp_path / ".ness",
        )
    )
    app = TuiApp(runtime)
    app.controller = SimpleNamespace(mode="act")
    return app


def test_end_edits_input_while_ctrl_end_and_focused_end_navigate_transcript():
    transcript_ends = []
    exits = []
    invalidations = []
    ui = SimpleNamespace(
        menu_open=False,
        transcript_focused=False,
        transcript_has_selection=False,
        menu_horizontal_enabled=False,
        reasoning_toggle_available=True,
        follow_transcript_end=lambda: transcript_ends.append(True),
        request_exit=lambda: exits.append(True),
        move_input_cursor_vertical=lambda _delta: False,
    )
    bindings = build_key_bindings(ui).bindings

    end_bindings = [binding for binding in bindings if binding.keys == (Keys.End,)]
    input_end = next(binding for binding in end_bindings if binding.filter())
    ctrl_end = next(
        binding for binding in bindings if binding.keys == (Keys.ControlEnd,)
    )
    quit_app = next(
        binding for binding in bindings if binding.keys == (Keys.ControlQ,)
    )
    buffer = Buffer(multiline=True)
    buffer.text = "first line\nsecond line"
    buffer.cursor_position = 2
    event = SimpleNamespace(
        arg=1,
        current_buffer=buffer,
        app=SimpleNamespace(invalidate=lambda: invalidations.append(True)),
    )

    input_end.handler(event)
    assert buffer.cursor_position == len("first line")
    assert transcript_ends == []

    ctrl_end.handler(event)
    ui.transcript_focused = True
    transcript_end = next(binding for binding in end_bindings if binding.filter())
    transcript_end.handler(event)
    quit_app.handler(event)

    assert transcript_ends == [True, True]
    assert len(invalidations) == 3
    assert exits == [True]


def test_input_vertical_navigation_uses_wrapped_display_rows(tmp_path):
    app = _input_app(tmp_path)
    app._terminal_size = lambda: (20, 24)
    app.input_buffer.text = "abcdefghijklmnopqrstuvwxyz012345"
    app.input_buffer.cursor_position = 24

    assert app.move_input_cursor_vertical(-1)
    assert app.input_buffer.cursor_position == 4

    assert app.move_input_cursor_vertical(1)
    assert app.input_buffer.cursor_position == 24


def test_input_vertical_navigation_counts_tabs_and_wide_characters(tmp_path):
    app = _input_app(tmp_path)
    app._terminal_size = lambda: (16, 24)
    app.input_buffer.text = "ab\t界界界界界界"
    app.input_buffer.cursor_position = len(app.input_buffer.text)

    assert app.move_input_cursor_vertical(-1)
    # The lower row ends at cell 12; the closest position on the first row
    # is the tab at offset two, cell eight after the six-cell prompt.
    assert app.input_buffer.cursor_position == 2
    assert app.move_input_cursor_vertical(1)
    assert app.input_buffer.cursor_position == len(app.input_buffer.text)


def test_prompt_history_is_file_backed_and_multiline_paste_round_trips(tmp_path):
    app = _input_app(tmp_path)
    assert isinstance(app.input_buffer.history, FileHistory)

    payload = "α first\nβ second\nthird"
    app.input_buffer.text = "before "
    app.input_buffer.cursor_position = len(app.input_buffer.text)
    app.insert_paste(payload)

    assert app.input_buffer.text == "before [pasted 3 lines]"
    assert app._expand_paste(app.input_buffer.text) == f"before {payload}"
    app._append_history(app._expand_paste(app.input_buffer.text))
    history = app.runtime.paths.cli_history.read_text(encoding="utf-8")
    assert "α first" in history and "β second" in history

    app.input_buffer.text = "before "
    assert app._pending_paste is None


def test_input_height_counts_unicode_wrapping_and_keeps_transcript_room(tmp_path):
    app = _input_app(tmp_path)
    app._terminal_size = lambda: (20, 24)
    app.input_buffer.text = "界" * 20

    assert app._input_row_count() == 3
    assert app.input_height().preferred == 3

    app.input_buffer.text = "/a/very/long/path/" * 12 + "\nnext"
    assert app._input_row_count() > 6
    assert app.input_height().preferred <= 6


def test_slash_menu_keeps_the_command_buffer_editable(tmp_path):
    app = _input_app(tmp_path)
    app._terminal_size = lambda: (120, 24)

    app.input_buffer.insert_text("/")

    assert app._menu is not None and app._menu.kind == "slash"
    assert not app._input_read_only()
    assert "".join(text for _, text in app.prompt_fragments()) == "act > "
    assert app.menu_height().preferred == 12

    app.input_buffer.insert_text("he")
    assert app.input_buffer.text == "/he"
    assert app._menu is not None
    assert [item.key for item in app._menu.items] == ["help"]

    app.input_buffer.delete_before_cursor()
    app.input_buffer.delete_before_cursor()
    app.input_buffer.delete_before_cursor()
    assert app.input_buffer.text == ""
    assert app._menu is None


def test_slash_menu_matches_legacy_full_width_rows(tmp_path):
    app = _input_app(tmp_path)
    app._terminal_size = lambda: (80, 24)
    app.input_buffer.insert_text("/")

    fragments = app.menu_fragments()
    text = "".join(part for _, part in fragments)
    first_row = text.splitlines()[0]

    assert len(first_row) == 80
    assert first_row.startswith("-> help")
    assert "Show the command reference" in first_row
    assert "commands" not in first_row
    assert "Esc close" not in text
    assert any(style == "class:chrome.menu.label.current" for style, _ in fragments)


def test_question_form_preserves_note_selection_and_transcript_space(tmp_path):
    async def exercise():
        app = _input_app(tmp_path)
        app._terminal_size = lambda: (80, 16)
        items = [MenuItem(str(index), f"Option {index}") for index in range(20)]
        app._menu = MenuState(
            "question",
            "Which deployment strategy?",
            items,
            note_allowed=True,
            hint="Up/Down option · Tab note · Enter submit · Esc cancel",
        )
        app._menu_future = asyncio.get_running_loop().create_future()

        assert app.menu_height().preferred <= 5
        first = "".join(text for _, text in app.menu_fragments())
        assert "Which deployment strategy?" in first
        assert "1-2/20 ↓" in first
        assert "note (saved)  optional" in first

        app.toggle_question_note()
        assert app._menu.note_active and not app._input_read_only()
        app.insert_paste("Only after canary\nand smoke tests.")
        app.move_menu(1)
        app.toggle_question_note()

        assert not app._menu.note_active and app._input_read_only()
        assert app._menu.selected == items[1]
        assert app._menu.note_text == "Only after canary\nand smoke tests."
        saved = "".join(text for _, text in app.menu_fragments())
        assert "Option 1" in saved and "Only after canary" in saved

        app._menu.index = 10
        middle = "".join(text for _, text in app.menu_fragments())
        assert "↑ " in middle and " ↓" in middle

        app._resolve_question()
        assert await app._menu_future == (
            "10",
            "Only after canary\nand smoke tests.",
        )

    asyncio.run(exercise())


def test_menu_filter_ranking_and_live_refresh_preserve_selection():
    items = [
        MenuItem(f"vendor/model-{index}", f"Model {index}")
        for index in range(497)
    ] + [
        MenuItem("vendor/atlas-pro", "Atlas Pro", search_terms=("Vendor",)),
        MenuItem("vendor/atlas", "Atlas", search_terms=("Vendor",)),
        MenuItem("vendor/x-atlas", "X Atlas", search_terms=("Vendor",)),
    ]
    menu = MenuState("picker", "models", items, filterable=True)
    menu.apply_filter("atlas")

    assert [item.key for item in menu.items] == [
        "vendor/atlas",
        "vendor/atlas-pro",
        "vendor/x-atlas",
    ]
    menu.index = 1
    selected = menu.selected
    refreshed = [MenuItem("new", "New"), *items]
    menu.replace_items(refreshed)
    assert menu.selected == selected
    assert menu.query == "atlas"


def test_model_items_show_current_capabilities_context_and_pricing():
    spec = next(spec for spec in SPECS if spec.key == "model_name")
    provider = SimpleNamespace(id="openrouter", display_name="OpenRouter")
    context = SimpleNamespace(
        runtime=SimpleNamespace(
            config=SimpleNamespace(
                model=SimpleNamespace(
                    provider_id="openrouter",
                    model_name="vendor/vision",
                )
            ),
            providers=SimpleNamespace(active=lambda: provider),
        )
    )
    model = SimpleNamespace(
        id="vendor/vision",
        name="Vision Model",
        context_window=128_000,
        input_price=1.25,
        output_price=5.0,
        supports_vision=True,
    )

    item = _model_items(context, spec, (model,))[0]

    assert item.suffix == "current"
    assert "vision" in item.description
    assert "128,000 context" in item.description
    assert "$1.25/$5 per 1M" in item.description
    assert item.search_terms == ("Vision Model", "OpenRouter", "openrouter")


def test_reasoning_is_collapsed_and_toggle_rebuilds_tracked_blocks():
    store = TranscriptStore(width=48)
    sink = TuiRenderSink(
        store,
        invalidate=lambda: None,
        prompts=NullPromptDriver(),
    )
    handle = sink.start_assistant()
    sink.assistant_delta(handle, "Final answer")
    sink.reasoning(handle, "**private working**\n\n- check one", elapsed=1.25)
    sink.finish_turn()

    collapsed = store.plain_text()
    assert "+ reasoning 1.2s" in collapsed
    assert "private working" not in collapsed
    assert "Final answer" in collapsed

    assert sink.toggle_reasoning()
    expanded = store.plain_text()
    assert "- reasoning 1.2s" in expanded
    assert "private working" in expanded
    assert "check one" in expanded
    assert not sink.toggle_reasoning()
    assert "private working" not in store.plain_text()


def test_approval_detail_keeps_decision_visible_and_scrolls(tmp_path):
    app = _input_app(tmp_path)
    app._terminal_size = lambda: (80, 24)
    app._approval_name = "write"
    app._approval_args = {
        "path": str(tmp_path / "sample.txt"),
        "content": "\n".join(f"line {index}" for index in range(40)),
    }
    app._approval_scope_key = "session"
    app._menu = MenuState(
        "approval",
        "approval needed: write",
        [
            MenuItem("yes", "Approve once"),
            MenuItem("session", "Approve session"),
            MenuItem("no", "Deny once"),
            MenuItem("diff", "Show diff"),
            MenuItem("show", "Show args"),
        ],
        summary_lines=["sample.txt"],
    )

    app._inspect_approval("diff")
    rendered = "".join(text for _, text in app.menu_fragments())
    assert "decision  Approve session" in rendered
    assert "proposed diff" in rendered
    assert len(app._menu.detail_lines) > app._menu_limits()[2]
    assert app.scroll_menu_detail(1)
    assert app._menu.detail_scroll > 0

    app._inspect_approval("show")
    assert '"content"' in "\n".join(app._menu.detail_lines)


def test_sink_approval_uses_inspectable_driver_without_logging_raw_args():
    class ApprovalPrompts:
        def __init__(self):
            self.calls = []

        async def approve(self, name, args):
            self.calls.append((name, args))
            return "session"

        async def choose(self, *_args, **_kwargs):
            raise AssertionError("approval should use the dedicated prompt")

    prompts = ApprovalPrompts()
    store = TranscriptStore()
    sink = TuiRenderSink(store, invalidate=lambda: None, prompts=prompts)

    answer = asyncio.run(
        sink.ask_approval("edit", {"path": "secret.py", "new_string": "token"})
    )

    assert answer == "session"
    assert prompts.calls == [
        ("edit", {"path": "secret.py", "new_string": "token"})
    ]
    assert "token" not in store.plain_text()


def test_status_chrome_shows_work_queue_usage_cost_and_context(tmp_path):
    app = _input_app(tmp_path)
    session = SimpleNamespace(
        usage_summary=lambda: {
            "input_tokens": 12_345,
            "output_tokens": 678,
            "cost_usd": 0.4321,
        },
        context_summary=lambda: {"used": 24_000, "total": 128_000},
    )
    app.controller = SimpleNamespace(
        mode="act",
        cancelling=False,
        busy=True,
        working_elapsed=65.4,
        queued_prompts=(TurnRequest("run the migration after tests"),),
        thread_id="thread-123456789",
        session=session,
    )
    app.runtime.config = SimpleNamespace(
        model=SimpleNamespace(
            provider_id="openrouter",
            model_name="vendor/model",
            reasoning_effort="high",
        )
    )
    app.runtime.providers = SimpleNamespace(
        active=lambda: SimpleNamespace(billing_label="metered")
    )
    app._terminal_size = lambda: (120, 30)

    status = "".join(text for _, text in app.status_fragments())
    queue = "".join(text for _, text in app.queue_fragments())
    stats = "".join(text for _, text in app.stats_fragments())
    path = "".join(text for _, text in app.path_fragments())

    assert "working 1m 05s" in status
    assert "vendor/model" not in status and "high" not in status
    assert "queued (1)" in queue and "run the migration" in queue
    assert "12,345" in stats and "678" in stats and "$0.4321" in stats
    assert "24k/128k used" in stats and "██" in stats
    assert str(tmp_path) in path
    assert path.endswith("model - high")
    assert len(path) == 120

    app.controller.busy = False
    assert app.status_height().preferred == 0
    assert app.status_fragments() == []

    app._terminal_size = lambda: (40, 24)
    narrow = "".join(text for _, text in app.stats_fragments())
    narrow_path = "".join(text for _, text in app.path_fragments())
    assert "↑12k" in narrow and "ctx 24k/128k" in narrow
    assert len(narrow) <= 40
    assert len(narrow_path) <= 40
