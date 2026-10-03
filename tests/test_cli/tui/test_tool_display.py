import pytest

from ness_cli.tui.tool_display import format_tool_args


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ({"paths": ["one.txt"]}, "one.txt"),
        ({"paths": ["one.txt", "two.txt"]}, "2 files"),
        ({"path": "old-history.txt"}, "old-history.txt"),
    ],
)
def test_delete_summary_supports_bulk_calls_and_legacy_history(arguments, expected):
    assert format_tool_args("delete", arguments) == expected
