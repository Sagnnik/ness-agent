"""Basic full-screen layout and key bindings."""

from __future__ import annotations

import string

from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.filters import Condition
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.processors import (
    BeforeInput,
    ConditionalProcessor,
    PasswordProcessor,
)

from ness_cli.tui.transcript import TranscriptView


def build_layout(ui) -> Layout:
    transcript = TranscriptView(
        ui.transcript,
        on_resize=ui.on_transcript_resize,
        focus=ui.focus_transcript,
        invalidate=ui.invalidate,
    )
    ui.transcript_view = transcript
    ui.transcript_window = Window(
        content=transcript,
        height=Dimension(weight=1),
        wrap_lines=False,
        always_hide_cursor=True,
        style="class:screen",
        get_vertical_scroll=lambda _window: transcript.vertical_scroll,
    )
    ui.input_window = Window(
        content=BufferControl(
            buffer=ui.input_buffer,
            focusable=True,
            focus_on_click=True,
            input_processors=[
                BeforeInput(ui.prompt_fragments),
                ConditionalProcessor(
                    PasswordProcessor(),
                    Condition(lambda: ui.secret_input),
                ),
            ],
        ),
        height=ui.input_height,
        wrap_lines=True,
        style="class:screen",
    )
    status = Window(
        FormattedTextControl(ui.status_fragments),
        height=ui.status_height,
        style="class:screen",
    )
    queue = Window(
        FormattedTextControl(ui.queue_fragments),
        height=ui.queue_height,
        style="class:screen",
    )
    stats = Window(
        FormattedTextControl(ui.stats_fragments),
        height=Dimension.exact(1),
        style="class:screen",
    )
    menu = Window(
        FormattedTextControl(ui.menu_fragments),
        height=ui.menu_height,
        style="class:screen",
    )
    path = Window(
        FormattedTextControl(ui.path_fragments),
        height=Dimension.exact(1),
        style="class:screen",
    )
    input_top_rule = Window(
        char="─",
        height=Dimension.exact(1),
        style="class:chrome.rule",
    )
    input_bottom_rule = Window(
        char="─",
        height=Dimension.exact(1),
        style="class:chrome.rule",
    )
    ui.status_window = status
    ui.queue_window = queue
    ui.menu_window = menu
    ui.stats_window = stats
    ui.path_window = path
    ui.input_top_rule = input_top_rule
    ui.input_bottom_rule = input_bottom_rule
    root = HSplit(
        [
            ui.transcript_window,
            status,
            input_top_rule,
            ui.input_window,
            input_bottom_rule,
            queue,
            menu,
            stats,
            path,
        ],
        style="class:screen",
    )
    layout = Layout(root, focused_element=ui.input_window)
    return layout


def build_key_bindings(ui) -> KeyBindings:
    keys = KeyBindings()
    menu_open = Condition(lambda: ui.menu_open)
    transcript_focused = Condition(lambda: ui.transcript_focused)
    transcript_selected = Condition(
        lambda: ui.transcript_focused and ui.transcript_has_selection
    )
    transcript_navigation = Condition(lambda: not ui.menu_open)
    input_navigation = Condition(
        lambda: not ui.menu_open and not ui.transcript_focused
    )
    menu_horizontal = Condition(lambda: ui.menu_horizontal_enabled)
    reasoning_toggle = Condition(lambda: ui.reasoning_toggle_available)

    @keys.add("up", filter=menu_open, eager=True)
    def menu_up(event) -> None:
        ui.move_menu(-1)
        event.app.invalidate()

    @keys.add("down", filter=menu_open, eager=True)
    def menu_down(event) -> None:
        ui.move_menu(1)
        event.app.invalidate()

    @keys.add("up", filter=input_navigation, eager=True)
    def input_up(event) -> None:
        buffer = event.current_buffer
        if buffer.complete_state:
            buffer.complete_previous(count=event.arg)
        elif (
            not ui.move_input_cursor_vertical(-event.arg)
            and not buffer.selection_state
        ):
            buffer.history_backward(count=event.arg)
        event.app.invalidate()

    @keys.add("down", filter=input_navigation, eager=True)
    def input_down(event) -> None:
        buffer = event.current_buffer
        if buffer.complete_state:
            buffer.complete_next(count=event.arg)
        elif (
            not ui.move_input_cursor_vertical(event.arg)
            and not buffer.selection_state
        ):
            buffer.history_forward(count=event.arg)
        event.app.invalidate()

    @keys.add("tab", filter=menu_open, eager=True)
    def menu_complete(event) -> None:
        ui.complete_menu()
        event.app.invalidate()

    @keys.add("left", filter=menu_horizontal, eager=True)
    @keys.add("right", filter=menu_horizontal, eager=True)
    def menu_horizontal_change(event) -> None:
        ui.change_menu_horizontal()
        event.app.invalidate()

    @keys.add("pageup", filter=menu_open, eager=True)
    def menu_detail_up(event) -> None:
        if not ui.scroll_menu_detail(-1):
            ui.move_menu(-8)
        event.app.invalidate()

    @keys.add("pagedown", filter=menu_open, eager=True)
    def menu_detail_down(event) -> None:
        if not ui.scroll_menu_detail(1):
            ui.move_menu(8)
        event.app.invalidate()

    @keys.add("pageup", filter=transcript_navigation, eager=True)
    def transcript_page_up(event) -> None:
        ui.scroll_transcript_page(-1)
        event.app.invalidate()

    @keys.add("pagedown", filter=transcript_navigation, eager=True)
    def transcript_page_down(event) -> None:
        ui.scroll_transcript_page(1)
        event.app.invalidate()

    @keys.add("c-home", filter=transcript_navigation, eager=True)
    def transcript_top(event) -> None:
        ui.scroll_transcript_to_top()
        event.app.invalidate()

    @keys.add("c-end", filter=transcript_navigation, eager=True)
    @keys.add("end", filter=transcript_focused, eager=True)
    def transcript_end(event) -> None:
        ui.follow_transcript_end()
        event.app.invalidate()

    @keys.add("end", filter=input_navigation, eager=True)
    def input_end(event) -> None:
        buffer = event.current_buffer
        buffer.cursor_position += buffer.document.get_end_of_line_position()
        event.app.invalidate()

    @keys.add("enter", filter=transcript_focused, eager=True)
    def transcript_enter(event) -> None:
        ui.refocus_input()
        event.app.invalidate()

    @keys.add("backspace", filter=transcript_focused, eager=True)
    def transcript_backspace(event) -> None:
        ui.backspace_from_transcript()
        event.app.invalidate()

    for character in string.ascii_letters + string.digits + string.punctuation + " ":

        @keys.add(character, filter=transcript_focused, eager=True)
        def transcript_type(event, value=character) -> None:
            ui.insert_from_transcript(value)
            event.app.invalidate()

    @keys.add("enter", filter=~transcript_focused)
    def submit(event) -> None:
        ui.submit_input()
        event.app.invalidate()

    @keys.add("c-j", filter=~menu_open)
    def newline(event) -> None:
        event.app.current_buffer.insert_text("\n")

    @keys.add(Keys.BracketedPaste, eager=True)
    def bracketed_paste(event) -> None:
        ui.insert_paste(event.data)
        event.app.invalidate()

    @keys.add("c-g")
    def paste_image(event) -> None:
        ui.paste_clipboard_image()
        event.app.invalidate()

    @keys.add("s-tab")
    def toggle_mode(event) -> None:
        ui.toggle_mode()
        event.app.invalidate()

    @keys.add("c-t", filter=reasoning_toggle, eager=True)
    def toggle_reasoning(event) -> None:
        ui.toggle_reasoning()
        event.app.invalidate()

    @keys.add("c-c", filter=transcript_selected, eager=True)
    def copy_transcript(event) -> None:
        ui.copy_transcript_selection()
        event.app.invalidate()

    @keys.add("c-c", filter=~transcript_selected)
    def interrupt(event) -> None:
        ui.interrupt_or_exit()
        event.app.invalidate()

    @keys.add("escape", filter=transcript_focused, eager=True)
    def leave_transcript(event) -> None:
        ui.refocus_input()
        event.app.invalidate()

    @keys.add("escape", filter=~transcript_focused, eager=True)
    def escape(event) -> None:
        ui.cancel_prompt()
        event.app.invalidate()

    @keys.add("c-d")
    def exit_app(event) -> None:
        if not event.app.current_buffer.text:
            ui.request_exit()

    @keys.add("c-q", eager=True)
    def quit_app(event) -> None:
        ui.request_exit()

    return keys
