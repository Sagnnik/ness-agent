from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as dist_version

from typer.testing import CliRunner

from ness_cli import cli


def _installed_version() -> str | None:
    try:
        return dist_version("ness-agent")
    except PackageNotFoundError:
        return None


def test_version_prints_the_installed_version() -> None:
    result = CliRunner().invoke(cli.app, ["--version"])

    assert result.exit_code == 0, result.output
    installed = _installed_version()
    if installed is None:
        assert "version unknown" in result.output
    else:
        assert result.output.strip() == f"ness {installed}"


def test_version_exits_before_runtime_construction(monkeypatch) -> None:
    def fail(*_args, **_kwargs):
        raise AssertionError("runtime options must not be built")

    monkeypatch.setattr(cli, "_build_overrides", fail)

    result = CliRunner().invoke(cli.app, ["--version", "--print", "hello"])

    assert result.exit_code == 0, result.output
    assert result.output.startswith("ness ")
