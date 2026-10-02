"""Git worktree helpers for isolated Ness Agent sessions.

Stdlib-only: must not import config or permissions (cwd is set before those load).
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


class WorktreeError(RuntimeError):
    """Raised when worktree setup fails."""


def slugify(name: str) -> str:
    """Return a safe branch/directory slug from a user-provided name."""
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    if not slug:
        raise WorktreeError(f"Invalid worktree name: {name!r}")
    return slug


def repo_root(cwd: Path | None = None) -> Path | None:
    """Return the git repository root for *cwd*, or None if not in a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=cwd or Path.cwd(),
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, OSError):
        return None
    line = result.stdout.strip()
    return Path(line).resolve() if line else None


def _main_repo_root(cwd: Path | None = None) -> Path | None:
    """Return the primary checkout root (parent of linked worktrees under .ness/worktrees)."""
    root = repo_root(cwd)
    if root is None:
        return None
    parts = root.parts
    if len(parts) >= 3 and parts[-3:] == (".ness", "worktrees", parts[-1]):
        nested = Path(*parts[:-3])
        if (nested / ".git").exists() or (nested / ".git").is_file():
            return nested.resolve()
    return root


def worktree_path(name: str, cwd: Path | None = None) -> Path:
    """Absolute path where worktree *name* lives under the main repo."""
    root = _main_repo_root(cwd)
    if root is None:
        raise WorktreeError("Not inside a git repository.")
    return (root / ".ness" / "worktrees" / slugify(name)).resolve()


def _run_git(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def _registered_worktree_paths(repo: Path) -> set[Path]:
    result = _run_git(["worktree", "list", "--porcelain"], cwd=repo)
    if result.returncode != 0:
        return set()
    paths: set[Path] = set()
    for line in result.stdout.splitlines():
        if line.startswith("worktree "):
            paths.add(Path(line[len("worktree ") :].strip()).resolve())
    return paths


def _branch_exists(repo: Path, branch: str) -> bool:
    result = _run_git(["rev-parse", "--verify", f"refs/heads/{branch}"], cwd=repo)
    return result.returncode == 0


def ensure_worktree(name: str) -> Path:
    """Create or reuse an isolated worktree and return its absolute path."""
    repo = _main_repo_root()
    if repo is None:
        raise WorktreeError("Not inside a git repository; --worktree requires git.")

    slug = slugify(name)
    path = (repo / ".ness" / "worktrees" / slug).resolve()
    branch = f"worktree-{slug}"

    registered = _registered_worktree_paths(repo)
    if path in registered:
        return path

    if path.exists():
        raise WorktreeError(
            f"Path exists but is not a git worktree: {path}. "
            "Remove it or choose another name."
        )

    path.parent.mkdir(parents=True, exist_ok=True)

    if _branch_exists(repo, branch):
        add_args = ["worktree", "add", str(path), branch]
    else:
        add_args = ["worktree", "add", str(path), "-b", branch]

    result = _run_git(add_args, cwd=repo)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise WorktreeError(f"git worktree add failed: {detail}")

    return path
