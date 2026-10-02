from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from ness_cli.paths import NessPaths, resolve_paths


_CREDENTIAL_VARIABLES = (
    "OPENAI_API_KEY",
    "OPENCODE_API_KEY",
    "OPENCODE_GO_API_KEY",
    "EXA_API_KEY",
    "CODEX_HOME",
    "OPENAI_BASE_URL",
    "OPENROUTER_SESSION_ID",
)


@dataclass(frozen=True, slots=True)
class IsolatedCliEnv:
    root: Path
    project: Path
    config: Path
    cache: Path
    ness: Path
    paths: NessPaths


@pytest.fixture(autouse=True)
def isolated_cli_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> IsolatedCliEnv:
    """Keep CLI state, credentials, and cwd inside one temporary directory."""
    project = tmp_path / "project"
    config = tmp_path / "config"
    cache = tmp_path / "cache"
    ness = tmp_path / "project-data"
    project.mkdir()

    monkeypatch.setenv("NESS_AGENT_CONFIG_DIR", str(config))
    monkeypatch.setenv("NESS_AGENT_CACHE_DIR", str(cache))
    monkeypatch.setenv("NESS_DIR", str(ness))
    if request.node.get_closest_marker("live") is None:
        for name in _CREDENTIAL_VARIABLES:
            monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(project)

    paths = resolve_paths(project_root=project)
    return IsolatedCliEnv(tmp_path, project, config, cache, ness, paths)
