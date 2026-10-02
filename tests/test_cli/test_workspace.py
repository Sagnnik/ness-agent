from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

from ness_cli import main
from ness_cli import workspace


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["-w", "short"], "short"),
        (["--worktree", "long"], "long"),
        (["--worktree=joined"], "joined"),
    ],
)
def test_worktree_argument_forms(argv, expected) -> None:
    assert main._worktree_name(argv) == expected


def test_bootstrap_changes_directory_and_sets_identity(
    isolated_cli_env, monkeypatch
) -> None:
    target = isolated_cli_env.root / "selected-worktree"
    target.mkdir()
    monkeypatch.setattr(workspace, "ensure_worktree", lambda _name: target)

    main.bootstrap_worktree(["--worktree", "feature"])

    assert os.getcwd() == str(target)
    assert os.environ["NESS_AGENT_WORKTREE"] == "feature"
    assert os.environ["NESS_AGENT_WORKTREE_PATH"] == str(target)


def test_main_bootstraps_before_dispatching_cli(monkeypatch) -> None:
    calls: list[str] = []

    def bootstrap(_argv) -> None:
        calls.append("bootstrap")

    def app(*, args) -> None:
        calls.append(f"cli:{args}")

    monkeypatch.setattr(main, "bootstrap_worktree", bootstrap)
    monkeypatch.setitem(sys.modules, "ness_cli.cli", SimpleNamespace(app=app))

    main.main(["--worktree=feature", "--version"])

    assert calls == ["bootstrap", "cli:['--worktree=feature', '--version']"]


def test_invalid_worktree_name_has_a_useful_error() -> None:
    with pytest.raises(workspace.WorktreeError, match="Invalid worktree name"):
        workspace.slugify("!!!")


def test_non_git_project_has_a_useful_error(monkeypatch) -> None:
    monkeypatch.setattr(workspace, "_main_repo_root", lambda *_args: None)

    with pytest.raises(workspace.WorktreeError, match="requires git"):
        workspace.ensure_worktree("feature")


def test_bootstrap_reports_worktree_errors(monkeypatch, capsys) -> None:
    def fail(_name):
        raise workspace.WorktreeError("bad repository")

    monkeypatch.setattr(workspace, "ensure_worktree", fail)

    with pytest.raises(SystemExit) as raised:
        main.bootstrap_worktree(["--worktree=feature"])

    assert raised.value.code == 1
    assert "worktree error: bad repository" in capsys.readouterr().err
