from __future__ import annotations

import pytest

from ness_agent.options import NessAgentOptions


def test_ness_agent_options_defaults_are_valid():
    opts = NessAgentOptions()
    assert opts.compaction_token_budget == 120_000
    assert opts.reflection_token_ratio == 0.0
    assert opts.recursion_limit == 500


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"context_window": 0}, "context_window must be positive"),
        ({"context_window": -1}, "context_window must be positive"),
        ({"compaction_token_budget": 0}, "compaction_token_budget must be positive"),
        ({"compaction_buffer_tokens": 0}, "compaction_buffer_tokens must be positive"),
        ({"compaction_summary_max_tokens": 0}, "compaction_summary_max_tokens must be positive"),
        (
            {"compaction_buffer_tokens": 4_096, "compaction_summary_max_tokens": 4_096},
            "compaction_summary_max_tokens must be smaller than compaction_buffer_tokens",
        ),
        (
            {"context_window": 10_000, "compaction_buffer_tokens": 10_000},
            "context limit must be larger than compaction_buffer_tokens",
        ),
        ({"reflection_token_ratio": -0.1}, "reflection_token_ratio must be between 0 and 1"),
        ({"reflection_token_ratio": 1.1}, "reflection_token_ratio must be between 0 and 1"),
        ({"recursion_limit": 0}, "recursion_limit must be at least 1"),
        ({"shell_default_timeout": 0}, "shell_default_timeout"),
        ({"shell_default_timeout": None}, "shell_default_timeout"),
        ({"shell_default_timeout": float("nan")}, "shell_default_timeout"),
        ({"shell_max_timeout": -1}, "shell_max_timeout"),
        ({"shell_max_timeout": float("inf")}, "shell_max_timeout"),
        ({"shell_max_timeout": True}, "shell_max_timeout"),
    ],
)
def test_ness_agent_options_rejects_invalid_values(kwargs, match: str):
    with pytest.raises(ValueError, match=match):
        NessAgentOptions(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"reflection_token_ratio": 0.0},
        {"reflection_token_ratio": 1.0},
        {"recursion_limit": 1},
        {"shell_default_timeout": 0.1, "shell_max_timeout": None},
        {"shell_default_timeout": 900, "shell_max_timeout": 1200},
        {"context_window": None, "compaction_token_budget": 50_000},
    ],
)
def test_ness_agent_options_accepts_boundary_values(kwargs):
    NessAgentOptions(**kwargs)
