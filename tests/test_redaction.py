import pytest

from ness_agent.mcp import redact_text, redact_url


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (
            "https://user:password@example.com:8443/mcp?token=secret#fragment",
            "https://example.com:8443/mcp",
        ),
        ("https://example.com/mcp?first=one&second=two", "https://example.com/mcp"),
        ("http://example.com/mcp", "http://example.com/mcp"),
        ("https://example.com:bad/mcp", "[invalid URL]"),
        ("https://[broken/mcp", "[invalid URL]"),
    ],
)
def test_redact_url(value, expected):
    assert redact_url(value) == expected


@pytest.mark.parametrize(
    ("value", "secrets", "expected"),
    [
        ("plain text", ("", "missing"), "plain text"),
        ("secret secret", ("secret", "secret", ""), "[redacted] [redacted]"),
        ("secret-token secret", ("secret", "secret-token"), "[redacted] [redacted]"),
        ("TOKEN token", ("token",), "TOKEN [redacted]"),
        ("literal a.b* plus", ("a.b*",), "literal [redacted] plus"),
        ("abcd", ("abcd",), "[redacted]"),
        ("failed with abc", ("abc",), "[redacted]"),
        ("failed with ab", ("ab",), "[redacted]"),
        ("failed with z", ("z",), "[redacted]"),
        ("all clear", ("xy",), "all clear"),
    ],
)
def test_redact_literal_text(value, secrets, expected):
    assert redact_text(value, secrets) == expected


def test_matching_short_secret_uses_custom_whole_message_fallback():
    assert (
        redact_text("secret and xy", ("secret", "xy"), fallback="ValueError")
        == "ValueError"
    )
