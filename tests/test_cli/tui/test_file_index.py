from __future__ import annotations

import subprocess
import os
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from ness_cli.tui.input.files import FileIndex
from ness_cli.tui.input import files
from ness_cli.tui.app import TuiApp


def _git(root: Path, *args, input=None):
    return subprocess.run(
        ["git", *args],
        cwd=root,
        input=input,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    return root


def _write(root, name, content="example"):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def _keys(index, query="", limit=100):
    return [item.key for item in index.search(query, limit=limit)]


def test_mentions_include_tracked_and_untracked_files(repo):
    _write(repo, "tracked.py")
    _write(repo, "src/new.py")
    _git(repo, "add", "tracked.py")
    assert set(_keys(FileIndex(repo))) == {"tracked.py", "src/new.py"}


def test_git_ignores_are_respected(repo, tmp_path):
    _write(repo, ".gitignore", "*.log\nnode_modules/\n.venv/\n.ness/\n")
    _write(repo, "tracked.py")
    _git(repo, "add", "tracked.py", ".gitignore")
    for name in (
        "secret.log",
        "node_modules/package.js",
        ".venv/library.py",
        ".ness/state.json",
        "local-only.txt",
        "globally-ignored.txt",
    ):
        _write(repo, name)
    _write(repo, ".git/info/exclude", "local-only.txt\n")
    global_excludes = _write(tmp_path, "excludes", "globally-ignored.txt\n")
    _git(repo, "config", "core.excludesfile", str(global_excludes))
    assert set(_keys(FileIndex(repo))) == {"tracked.py", ".gitignore"}


def test_empty_git_result_does_not_reintroduce_ignored_files(repo):
    _write(repo, ".git/info/exclude", "*\n")
    _write(repo, "ignored.txt")
    assert _keys(FileIndex(repo)) == []


def test_git_results_exclude_runtime_and_dependency_directories(repo):
    _write(repo, "visible.py")
    _git(repo, "add", "visible.py")
    for name in (
        ".ness/state.json",
        "node_modules/package.js",
        ".venv/library.py",
        "src/__pycache__/app.pyc",
        "build/output.py",
        "dist/archive.txt",
    ):
        _write(repo, name)
    assert _keys(FileIndex(repo)) == ["visible.py"]


def test_merge_stages_are_deduplicated_before_index_cap(repo):
    _write(repo, "a.py")
    _write(repo, "z.py")
    _git(repo, "add", "z.py")
    oid = _git(repo, "hash-object", "-w", "--stdin", input="conflicting content\n")
    _git(
        repo,
        "update-index",
        "--index-info",
        input="".join(f"100644 {oid} {stage}\ta.py\n" for stage in (1, 2, 3)),
    )
    assert set(_keys(FileIndex(repo, limit=2))) == {"a.py", "z.py"}


def test_project_subdirectory_keeps_paths_local(repo):
    _write(repo, "outside.py")
    _write(repo, "project/tracked.py")
    _write(repo, "project/new.py")
    _git(repo, "add", "outside.py", "project/tracked.py")
    assert set(_keys(FileIndex(repo / "project"))) == {"tracked.py", "new.py"}


def test_tracked_files_remain_available_when_ignore_patterns_match(repo):
    _write(repo, ".gitignore", "*.log\nnode_modules/\n")
    _write(repo, "tracked.log")
    _write(repo, "node_modules/tracked.js")
    _write(repo, "node_modules/untracked.js")
    _git(repo, "add", "-f", "tracked.log", "node_modules/tracked.js")
    assert set(_keys(FileIndex(repo))) == {
        ".gitignore",
        "tracked.log",
        "node_modules/tracked.js",
    }


@pytest.mark.parametrize(
    "name",
    [
        "folder/file with spaces.py",
        "folder/line\nbreak.py",
        "folder/café.py",
        ".github/workflows/check.yml",
    ],
)
def test_git_filename_boundaries_and_hidden_project_config(repo, name):
    _write(repo, name)
    _write(repo, "tracked.py")
    _git(repo, "add", "tracked.py")
    assert name in _keys(FileIndex(repo))


@pytest.mark.parametrize("limit", [0, 1, 5])
def test_index_limit_applies_to_combined_unique_files(repo, limit):
    for name in ("tracked.py", "new.py", "other.py"):
        _write(repo, name)
    _git(repo, "add", "tracked.py")
    result = _keys(FileIndex(repo, limit=limit))
    assert len(result) == min(limit, 3)
    assert len(set(result)) == len(result)


def test_search_ranking_and_result_limit_remain_intact(repo):
    for name in (
        "app.py",
        "src/app.py",
        "myapp.py",
        "src/app_config.py",
        "src/folder/app.py/notes.txt",
    ):
        path = _write(repo, name)
        os.utime(path, (100, 100))
    _git(repo, "add", "myapp.py")
    index = FileIndex(repo)
    assert _keys(index, "app.py", limit=3) == ["app.py", "src/app.py", "myapp.py"]
    assert _keys(index, "APP.PY", limit=3) == _keys(index, "app.py", limit=3)


def test_empty_git_result_is_cached_until_expiry(repo, monkeypatch):
    _write(repo, ".git/info/exclude", "*\n")
    clock = [0.0]
    monkeypatch.setattr(files, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    index = FileIndex(repo)
    assert _keys(index) == []
    _write(repo, "new.py")
    _git(repo, "add", "-f", "new.py")
    clock[0] = 1.0
    assert _keys(index) == []
    clock[0] = 31.0
    assert _keys(index) == ["new.py"]


def test_cached_index_picks_up_new_untracked_files_at_expiry(repo, monkeypatch):
    _write(repo, "tracked.py")
    _git(repo, "add", "tracked.py")
    clock = [0.0]
    monkeypatch.setattr(files, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    index = FileIndex(repo)
    assert _keys(index) == ["tracked.py"]
    _write(repo, "new.py")
    clock[0] = 1.0
    assert _keys(index) == ["tracked.py"]
    clock[0] = 31.0
    assert set(_keys(index)) == {"tracked.py", "new.py"}


def test_non_git_project_keeps_filesystem_fallback(tmp_path):
    root = tmp_path / "plain"
    root.mkdir()
    _write(root, "visible.py")
    _write(root, "src/new.py")
    _write(root, "node_modules/package.js")
    _write(root, ".ness/state.json")
    assert set(_keys(FileIndex(root))) == {"visible.py", "src/new.py"}


@pytest.mark.parametrize("failure", ["missing", "timeout", "status"])
def test_git_failure_keeps_filesystem_fallback(tmp_path, monkeypatch, failure):
    root = tmp_path / "plain"
    root.mkdir()
    _write(root, "visible.py")

    def run(*_args, **_kwargs):
        if failure == "missing":
            raise FileNotFoundError("git unavailable")
        if failure == "timeout":
            raise subprocess.TimeoutExpired("git", 10)
        return SimpleNamespace(returncode=128, stdout=b"")

    monkeypatch.setattr(files.subprocess, "run", run)
    assert _keys(FileIndex(root)) == ["visible.py"]


def test_tui_completes_an_untracked_file(repo, tmp_path):
    _write(repo, "tracked.py")
    _write(repo, "new.py")
    _git(repo, "add", "tracked.py")

    async def exercise():
        app = TuiApp(
            SimpleNamespace(
                paths=SimpleNamespace(
                    project_root=repo,
                    ness_dir=repo / ".ness",
                    cli_history=tmp_path / "cache" / "cli_history",
                )
            )
        )
        app.controller = SimpleNamespace(mode="act")
        app.input_buffer.text = "@new"
        app.input_buffer.cursor_position = len(app.input_buffer.text)
        app._on_input_changed(app.input_buffer)
        assert app._menu.kind == "mention"
        assert [item.key for item in app._menu.items] == ["new.py"]
        app.complete_menu()
        assert app.input_buffer.text == "@new.py "

    asyncio.run(exercise())
