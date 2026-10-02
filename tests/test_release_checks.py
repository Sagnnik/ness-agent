"""Release checks must fail before exporting unchecked distributions."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from scripts import check_release


@pytest.mark.parametrize("failure", ["source", "wheel", None])
def test_release_outputs_only_packages_that_pass_verification(
    tmp_path, monkeypatch, failure
):
    output = tmp_path / "verified"
    checked = {}
    wheel = "ness_agent-0.2.4-py3-none-any.whl"
    sdist = "ness_agent-0.2.4.tar.gz"

    def run(args, *, cwd, env, check):
        if failure == "source" and "not live and not packaging" in args:
            raise subprocess.CalledProcessError(1, args)
        if args[:2] == ("uv", "build"):
            dist = Path(args[-1])
            dist.mkdir()
            (dist / wheel).write_bytes(b"wheel checked by the probe")
            (dist / sdist).write_bytes(b"source distribution checked by the probe")
        if "tests/test_packaging_smoke.py" in args:
            assert env["PACKAGING_SMOKE"] == "1"
            dist = Path(env["PACKAGING_DIST_DIR"])
            checked.update({path.name: path.read_bytes() for path in dist.iterdir()})
            if failure == "wheel":
                raise subprocess.CalledProcessError(1, args)

    monkeypatch.setattr(check_release.subprocess, "run", run)
    monkeypatch.setattr(sys, "argv", ["check_release.py", "--dist-dir", str(output)])
    if failure:
        with pytest.raises(subprocess.CalledProcessError):
            check_release.main()
        assert not output.exists()
        if failure == "source":
            assert not checked
    else:
        check_release.main()
        assert {path.name: path.read_bytes() for path in output.iterdir()} == checked
        assert set(checked) == {wheel, sdist}


def test_release_check_refuses_to_mix_with_existing_packages(tmp_path, monkeypatch):
    package = tmp_path / "old.whl"
    package.write_bytes(b"existing package")
    monkeypatch.setattr(sys, "argv", ["check_release.py", "--dist-dir", str(tmp_path)])
    with pytest.raises(SystemExit) as error:
        check_release.main()
    assert error.value.code == 2
    assert package.read_bytes() == b"existing package"
