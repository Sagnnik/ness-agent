from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path.cwd()
_SNIPPET_LIMIT = 1200

IGNORED_DIR_NAMES: frozenset[str] = frozenset(
    {
        ".git",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        "dist",
        "build",
        "target",
        "vendor",
        ".gradle",
        "bin",
        "obj",
        ".cargo",
    }
)

MANIFEST_FILES: tuple[str, ...] = (
    "pyproject.toml",
    "package.json",
    "requirements.txt",
    "Cargo.toml",
    "go.mod",
    "CMakeLists.txt",
    "Makefile",
    "meson.build",
    "build.zig",
    "pom.xml",
    "build.gradle",
    "build.gradle.kts",
    "settings.gradle",
    "pubspec.yaml",
    "mix.exs",
    "flake.nix",
    "MODULE.bazel",
    "WORKSPACE",
    "Package.swift",
    "tsconfig.json",
)


def is_ignored_dir(name: str) -> bool:
    if name in IGNORED_DIR_NAMES:
        return True
    return name.startswith("cmake-build")


def discover_manifest_files() -> list[Path]:
    """Return a few repo-root .NET manifests without scanning the whole tree."""
    found: list[Path] = []
    for pattern in ("*.csproj", "*.sln"):
        found.extend(sorted(PROJECT_ROOT.glob(pattern))[:2])
    return found


def get_project_context(max_files: int = 80) -> str:
    """Return a compact project tree and key manifest snippets."""
    files = _git_files(max_files)
    if not files:
        files = _walk_files(max_files)

    tree = _render_tree(files)
    summaries = []
    for label, path in _manifest_paths():
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8")[:_SNIPPET_LIMIT]
            summaries.append(f"--- {label} ---\n{text}\n")
        except (OSError, UnicodeDecodeError):
            continue

    return f"Project structure (top {max_files} files):\n{tree}\n\n{''.join(summaries)}"


def _manifest_paths() -> list[tuple[str, Path]]:
    paths = [(name, PROJECT_ROOT / name) for name in MANIFEST_FILES]
    paths.extend((str(p.relative_to(PROJECT_ROOT)), p) for p in discover_manifest_files())
    return paths


def _git_files(max_files: int) -> list[str]:
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return []
    if result.returncode != 0:
        return []
    return [line for line in result.stdout.splitlines() if line and not _is_runtime_path(line)][:max_files]


def _walk_files(max_files: int) -> list[str]:
    files = []
    for dirpath, dirs, names in os.walk(PROJECT_ROOT):
        dirs[:] = [d for d in dirs if not is_ignored_dir(d) and not d.startswith(".")]
        for name in names:
            files.append(str((Path(dirpath) / name).relative_to(PROJECT_ROOT)))
            if len(files) >= max_files:
                return files
    return files


def _render_tree(files: Iterable[str]) -> str:
    tree: dict = {}
    for file in files:
        parts = Path(file).parts
        current = tree
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current.setdefault("__files__", []).append(parts[-1])

    def render(node: dict, prefix: str = "") -> list[str]:
        lines = []
        for key, value in sorted(node.items(), key=lambda item: (item[0] == "__files__", item[0])):
            if key == "__files__":
                lines.extend(f"{prefix}- {name}" for name in sorted(value))
            else:
                lines.append(f"{prefix}- {key}/")
                lines.extend(render(value, prefix + "  "))
        return lines

    return "\n".join(render(tree))


def _is_runtime_path(path: str) -> bool:
    normalized = path.replace("\\", "/")
    return normalized.startswith(
        (
            ".ness/threads/",
            ".ness/runtime/",
            ".ness/sessions/",
            ".ness/shells/",
        )
    )
