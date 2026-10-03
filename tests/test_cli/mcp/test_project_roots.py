from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from ness_cli.mcp.config import ProjectMCPConfig
from ness_cli.mcp.manager import ProjectMCPManager


@pytest.fixture(params=[ProjectMCPConfig, ProjectMCPManager], ids=["config", "manager"])
def constructor(request):
    return request.param


def _config_file(project: Path) -> Path:
    project.mkdir(exist_ok=True)
    (project / "server.env").write_text(f"PROJECT_LABEL={project.name}\n")
    path = project / "mcp.json"
    path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "local": {
                        "command": "test-server",
                        "args": ["${workspaceFolder}", "${workspaceFolderBasename}"],
                        "envFile": "server.env",
                    },
                    "relative": {"command": "test-server", "cwd": "tools"},
                }
            }
        )
    )
    return path


def test_fresh_import_then_directory_change_uses_construction_directory(
    tmp_path, constructor
):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    probe = """
import os
import sys
from pathlib import Path
from ness_cli.mcp.config import ProjectMCPConfig
from ness_cli.mcp.manager import ProjectMCPManager

constructor = globals()[sys.argv[1]]
first = Path.cwd()
original = constructor()
assert original.project_root == first
os.chdir(sys.argv[2])
second = Path.cwd()
assert constructor().project_root == second
assert constructor(project_root=None).project_root == second
assert original.project_root == first
instance = constructor()
if isinstance(instance, ProjectMCPManager):
    assert instance._config.project_root == second
print('ok')
"""
    result = subprocess.run(
        [sys.executable, "-c", probe, constructor.__name__, str(second)],
        cwd=first,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "ok"


def test_default_project_stays_pinned_for_lazy_config_loading(
    tmp_path, monkeypatch, constructor
):
    first = tmp_path / "first"
    second = tmp_path / "second"
    path = _config_file(first)
    _config_file(second)
    monkeypatch.chdir(first)
    original = constructor(path)
    monkeypatch.chdir(second)

    current = constructor(second / "mcp.json")
    for instance, expected in ((original, first), (current, second)):
        local = instance.server_spec("local").connection
        relative = instance.server_spec("relative").connection
        assert instance.project_root == expected
        assert local.cwd == expected
        assert local.args == (str(expected), expected.name)
        assert dict(local.env)["PROJECT_LABEL"] == expected.name
        assert relative.cwd == expected / "tools"
    assert original.trust_preview.fingerprint != current.trust_preview.fingerprint


@pytest.mark.parametrize("relative", [False, True], ids=["absolute", "relative"])
def test_explicit_root_controls_config_after_directory_change(
    tmp_path, monkeypatch, constructor, relative
):
    project = tmp_path / "project-root"
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    path = _config_file(project)
    monkeypatch.chdir(elsewhere)
    root = Path("../project-root") if relative else project

    instance = constructor(path, project_root=root)
    monkeypatch.chdir(tmp_path)
    spec = instance.server_spec("local").connection

    assert instance.project_root == project.resolve()
    assert spec.cwd == project.resolve()
    assert spec.args == (str(project.resolve()), project.name)
    assert dict(spec.env)["PROJECT_LABEL"] == project.name
    if isinstance(instance, ProjectMCPManager):
        assert instance._config.project_root == instance.project_root
