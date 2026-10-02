"""Project-scoped file completion for @mentions."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from ness_cli.tui.input.menus import MenuItem

_SKIP = frozenset(
    {
        ".git",
        ".ness",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        "dist",
        "build",
    }
)


class FileIndex:
    def __init__(self, project_root: Path, *, limit: int = 2_000) -> None:
        self._root = project_root.resolve()
        self._limit = max(0, limit)
        self._files: tuple[Path, ...] = ()
        self._expires_at = 0.0

    def search(self, query: str, *, limit: int = 8) -> list[MenuItem]:
        files = self._index()
        normalized = query.casefold()
        scored: list[tuple[int, int, float, Path]] = []
        for path in files:
            relative = path.relative_to(self._root).as_posix()
            name = path.name.casefold()
            haystack = relative.casefold()
            if normalized and normalized not in haystack:
                continue
            if not normalized:
                rank = 0
            elif name == normalized:
                rank = 0
            elif name.startswith(normalized):
                rank = 1
            elif "/" not in normalized and normalized in name:
                rank = 2
            else:
                rank = 3
            try:
                modified = path.stat().st_mtime
            except OSError:
                modified = 0.0
            scored.append((rank, relative.count("/"), -modified, path))
        scored.sort()
        return [
            MenuItem(
                path.relative_to(self._root).as_posix(),
                path.relative_to(self._root).as_posix(),
                description=(
                    ""
                    if path.parent == self._root
                    else path.parent.relative_to(self._root).as_posix()
                ),
            )
            for _, _, _, path in scored[:limit]
        ]

    def _index(self) -> tuple[Path, ...]:
        now = time.monotonic()
        if now < self._expires_at:
            return self._files
        if not self._limit:
            return ()
        files = self._git_files()
        if files is None:
            files = self._walk_files()
        self._files = tuple(files)
        self._expires_at = now + 30.0
        return self._files

    def _git_files(self) -> list[Path] | None:
        try:
            result = subprocess.run(
                [
                    "git",
                    "ls-files",
                    "--cached",
                    "--others",
                    "--exclude-standard",
                    "-z",
                    # Prune untracked runtime/dependency directories in Git.
                    *(f"--exclude={directory}/" for directory in sorted(_SKIP)),
                ],
                cwd=self._root,
                capture_output=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if result.returncode != 0:
            return None
        files: list[Path] = []
        seen: set[Path] = set()
        for raw in result.stdout.split(b"\0"):
            if not raw:
                continue
            path = self._root / os.fsdecode(raw)
            if path in seen:
                continue
            seen.add(path)
            files.append(path)
            if len(files) >= self._limit:
                break
        return files

    def _walk_files(self) -> list[Path]:
        files: list[Path] = []
        for directory, children, names in os.walk(self._root):
            children[:] = [
                child
                for child in children
                if child not in _SKIP and not child.startswith(".")
            ]
            for name in names:
                files.append(Path(directory) / name)
                if len(files) >= self._limit:
                    return files
        return files
