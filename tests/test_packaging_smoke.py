"""Build/install smoke, including an offline saved and resumed headless turn.

Opt-in (slow / network for dependency install):

    PACKAGING_SMOKE=1 uv run pytest -m packaging
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

from ness_cli.config.settings import _ENV_ALIASES

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.packaging
@pytest.mark.skipif(
    os.environ.get("PACKAGING_SMOKE") != "1",
    reason="set PACKAGING_SMOKE=1 to build and install a clean wheel",
)
def test_wheel_clean_install_smoke(tmp_path: Path) -> None:
    supplied = os.environ.get("PACKAGING_DIST_DIR")
    dist = Path(supplied).resolve() if supplied else tmp_path / "dist"
    venv = tmp_path / "venv"
    check_cwd = tmp_path / "cwd"
    check_cwd.mkdir()
    config_home = tmp_path / "config"
    config_home.mkdir()

    if not supplied:
        subprocess.run(
            ["uv", "build", "-o", str(dist)],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    wheels = list(dist.glob("ness_agent-*.whl"))
    assert len(wheels) == 1, f"expected one wheel, got {wheels}"
    wheel = wheels[0]
    sdists = list(dist.glob("ness_agent-*.tar.gz"))
    assert len(sdists) == 1, f"expected one sdist, got {sdists}"
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        assert "ness_cli/runtime.py" in names
        assert "ness_cli/main.py" in names
        assert not any("ness_cli_next" in Path(name).parts for name in names)
    with tarfile.open(sdists[0]) as archive:
        names = archive.getnames()
        assert any(name.endswith("/src/ness_cli/runtime.py") for name in names)
        assert not any(
            {"ness_cli_next", "test_cli_next"} & set(Path(name).parts)
            for name in names
        )

    subprocess.run(
        [sys.executable, "-m", "venv", str(venv)],
        check=True,
        capture_output=True,
        text=True,
    )
    venv_python = venv / ("Scripts" if os.name == "nt" else "bin") / "python"
    venv_ness = venv / ("Scripts" if os.name == "nt" else "bin") / "ness"

    subprocess.run(
        ["uv", "pip", "install", "--python", str(venv_python), str(wheel)],
        check=True,
        capture_output=True,
        text=True,
    )

    excluded = {alias for aliases in _ENV_ALIASES.values() for alias in aliases}
    excluded.update({"PYTHONPATH", "VIRTUAL_ENV"})
    env = {
        key: value
        for key, value in os.environ.items()
        if key.upper() not in excluded
    }
    env["NESS_AGENT_CONFIG_DIR"] = str(config_home)
    env["NESS_AGENT_CACHE_DIR"] = str(tmp_path / "cache")
    env["NESS_DIR"] = str(check_cwd / ".ness")
    env["HOME"] = str(tmp_path / "home")
    Path(env["HOME"]).mkdir(exist_ok=True)

    probe = r"""
import sys
from pathlib import Path

import importlib.util

import ness_cli
import ness_cli.cli
import ness_cli.main
import ness_cli.runtime

assert ness_cli.__name__ == "ness_cli"
assert Path(ness_cli.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
assert importlib.util.find_spec("ness_cli_next") is None

from ness_agent.defaults import default_agent_profiles
from ness_cli.config.settings import Settings
from ness_cli.instructions import default_instruction_files

import ness_agent
assert ness_agent.__name__ == "ness_agent"
assert Path(ness_agent.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())

settings = Settings()
assert settings.model_name == "deepseek/deepseek-v4-flash", settings.model_name

profiles = default_agent_profiles()
assert "explore.md" in profiles, sorted(profiles)
assert profiles["explore.md"].strip(), "explore.md body is empty"

instructions = default_instruction_files()
assert "l0_harness.md" in instructions, sorted(instructions)
assert instructions["l0_harness.md"].strip(), "l0_harness.md body is empty"
print("ok")
"""
    result = subprocess.run(
        [str(venv_python), "-c", probe],
        cwd=check_cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    assert "ok" in result.stdout

    # Copy the pytest-free test driver outside the checkout. Its SDK
    # and CLI imports must come from the clean wheel installation above.
    driver = check_cwd / "headless_probe.py"
    shutil.copy2(ROOT / "tests/test_cli/fakes/headless.py", driver)
    result = subprocess.run(
        [str(venv_python), str(driver), str(check_cwd)],
        cwd=check_cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    assert "headless save/resume/close ok" in result.stdout

    if os.name != "nt":
        assert "from ness_cli.main import main" in venv_ness.read_text(
            encoding="utf-8"
        )

    for arguments in (["--help"], ["--version"], ["mcp", "--help"]):
        result = subprocess.run(
            [str(venv_ness), *arguments],
            cwd=check_cwd,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, (
            f"{venv_ness.name} {' '.join(arguments)} failed\n"
            f"{result.stdout}\n{result.stderr}"
        )

    result = subprocess.run(
        [
            str(venv_python),
            str(ROOT / "scripts" / "fetch_openrouter_models.py"),
            "--help",
        ],
        cwd=check_cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    assert "--refresh" in result.stdout
