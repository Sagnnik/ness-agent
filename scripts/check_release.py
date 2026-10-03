"""Run the same offline release checks used by CI. Never publishes packages."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dist-dir",
        type=Path,
        help="copy verified distributions into this empty output directory",
    )
    options = parser.parse_args()
    output = options.dist_dir.resolve() if options.dist_dir is not None else None
    if output is not None and output.exists() and (
        not output.is_dir() or any(output.iterdir())
    ):
        parser.error("--dist-dir must be absent or an empty directory")

    def run(*args: str, env: dict[str, str] | None = None) -> None:
        subprocess.run(args, cwd=ROOT, env=env, check=True)

    run(sys.executable, "-m", "compileall", "-q", "src", "scripts", "evals/codex")
    run(
        sys.executable, "-m", "ruff", "check", "--select", "F",
        "src", "scripts", "evals/codex",
    )
    run(sys.executable, "-m", "pytest", "-q", "-m", "not live and not packaging")
    with tempfile.TemporaryDirectory(prefix="ness-release-") as temporary:
        dist = Path(temporary) / "dist"
        run("uv", "build", "--out-dir", str(dist))
        run(
            sys.executable, "-m", "pytest", "-q", "tests/test_packaging_smoke.py",
            env={**os.environ, "PACKAGING_SMOKE": "1", "PACKAGING_DIST_DIR": str(dist)},
        )
        if output is not None:
            output.mkdir(parents=True, exist_ok=True)
            for package in dist.iterdir():
                shutil.copy2(package, output / package.name)
            print(f"Verified distributions: {output}")


if __name__ == "__main__":
    main()
