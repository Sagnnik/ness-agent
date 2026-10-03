from __future__ import annotations

import asyncio
import json
import stat
from types import SimpleNamespace

import pytest

from ness_agent.workspace import setup_ness_structure
from ness_cli.instructions import INSTRUCTION_FILES
from ness_cli.paths import (
    ensure_global_config,
    ensure_project_runtime,
    project_hash,
    resolve_paths,
    resolve_project_slug,
    sanitize_slug,
)
from ness_cli.tui.commands.context import init_command


def test_sanitize_slug() -> None:
    assert sanitize_slug("My App!") == "my-app"
    assert sanitize_slug("___") == "project"


def test_all_paths_respect_isolated_overrides(isolated_cli_env) -> None:
    paths = isolated_cli_env.paths

    assert paths.project_root == isolated_cli_env.project.resolve()
    assert paths.ness_dir == isolated_cli_env.ness.resolve()
    assert paths.config_dir == isolated_cli_env.config.resolve()
    assert paths.cache_dir == (
        isolated_cli_env.cache.resolve() / project_hash(isolated_cli_env.project)
    )
    assert paths.user_file == paths.config_dir / "USER.md"
    assert paths.configs_file == paths.config_dir / "configs.json"
    assert paths.secrets_file == paths.config_dir / "secrets.json"
    assert paths.skill_state_file == paths.config_dir / "skill-state.json"
    assert paths.instructions_dir == paths.config_dir / "instructions"
    assert paths.plans_dir == paths.config_dir / "plans" / "project"
    assert paths.sessions_dir == paths.ness_dir / "runtime" / "sessions"
    assert paths.shells_dir == paths.ness_dir / "runtime" / "shells"
    assert paths.threads_dir == paths.ness_dir / "threads"
    assert paths.cli_history == paths.cache_dir / "cli_history"


def test_relative_ness_dir_is_resolved_from_project(
    isolated_cli_env, monkeypatch
) -> None:
    monkeypatch.setenv("NESS_DIR", "local-state")

    paths = resolve_paths(project_root=isolated_cli_env.project)

    assert paths.ness_dir == isolated_cli_env.project / "local-state"


def test_slug_collision_appends_hash(isolated_cli_env) -> None:
    first = isolated_cli_env.root / "apps" / "demo"
    second = isolated_cli_env.root / "other" / "demo"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    plans_root = isolated_cli_env.config / "plans"
    marker = plans_root / "demo" / ".project"
    marker.parent.mkdir(parents=True)
    marker.write_text(f"{first.resolve()}\n", encoding="utf-8")

    assert resolve_project_slug(first, plans_root) == "demo"
    assert resolve_project_slug(second, plans_root).startswith("demo-")


def test_global_and_project_initialization_is_idempotent(isolated_cli_env) -> None:
    paths = isolated_cli_env.paths

    first_global = ensure_global_config(paths)
    first_project = ensure_project_runtime(paths)
    second_global = ensure_global_config(paths)
    second_project = ensure_project_runtime(paths)

    assert first_global
    assert first_project
    assert second_global == []
    assert second_project == []
    assert paths.user_file.is_file()
    assert (paths.plans_dir / ".project").read_text(encoding="utf-8") == (
        f"{paths.project_root}\n"
    )
    assert json.loads(paths.secrets_file.read_text(encoding="utf-8")) == {}
    assert stat.S_IMODE(paths.secrets_file.stat().st_mode) == 0o600
    assert not paths.configs_file.exists()
    assert all((paths.instructions_dir / name).is_file() for name in INSTRUCTION_FILES)


def test_global_initialization_preserves_user_instructions(isolated_cli_env) -> None:
    paths = isolated_cli_env.paths
    ensure_global_config(paths)
    target = paths.instructions_dir / "l0_harness.md"
    target.write_text("CUSTOM\n", encoding="utf-8")

    ensure_global_config(paths)

    assert target.read_text(encoding="utf-8") == "CUSTOM\n"


@pytest.mark.parametrize("initialize", ["bootstrap", "init"])
def test_initialization_does_not_create_skill_directories(isolated_cli_env, initialize):
    paths = isolated_cli_env.paths
    if initialize == "bootstrap":
        setup_ness_structure(paths.ness_dir)
        assert setup_ness_structure(paths.ness_dir) == []
    else:
        context = SimpleNamespace(
            runtime=SimpleNamespace(paths=paths),
            renderer=SimpleNamespace(notice=lambda *_args: None),
        )
        asyncio.run(init_command(context, ""))
        asyncio.run(init_command(context, ""))

    assert paths.ness_dir.is_dir()
    assert (paths.ness_dir / "NESS.md").is_file()
    assert (paths.ness_dir / "agents").is_dir()
    assert not (paths.ness_dir / "skills").exists()
    assert not (paths.project_root / ".agents").exists()
