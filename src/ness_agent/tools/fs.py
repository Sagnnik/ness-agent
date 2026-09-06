from __future__ import annotations

import difflib
import fnmatch
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from ness_agent.media import (
    ImageNormalizationError,
    ImageNotRecognized,
    normalize_image,
    png_data_url,
)
from ness_agent.session_context import get_session_context

READ_FILE_DEFAULT_LIMIT = 400
READ_FILE_MAX_LIMIT = 2000
READ_FILE_MAX_BYTES = 50 * 1024 * 1024
PROTECTED_WRITE_DIRS = frozenset({".git", ".ness"})


def _validate_path(path: str) -> str:
    return get_session_context().permissions.validate_path(path)


def _relative_to_root(path: str) -> str:
    return get_session_context().permissions.relative_to_root(path)


def _project_root() -> Path:
    return get_session_context().project_root

def _unsupported_binary_message(path: Path, raw: bytes) -> str:
    suffix = path.suffix.lower()
    
    if suffix == ".pdf" or raw.startswith(b"%PDF-"):
        return (
            "Unsupported PDF file. Render selected pages to PNG/JPEG with PyMuPDF, "
            "then pass them as base64 images in the image_url content block."
        )
    
    if suffix in {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm", ".flv", ".ts", ".3gp", ".ogv"}:
        return (
            "Unsupported video file. Extract keyframes with ffmpeg or OpenCV, "
            "then pass them as base64 images in the image_url content block."
        )
    
    return (
        f"Unsupported binary file: {path.name}. Convert it to UTF-8 text or a raster "
        "image (PNG/JPEG) for submission."
    )


@tool
def read(path: str, offset: int = 1, limit: int | None = None) -> str | list[dict[str, Any]]:
    """Read a UTF-8 text file or raster image from the local filesystem.

    Defaults to 400 lines from ``offset`` (1-based); cap is 2000 lines per call.
    For PDFs or videos, use shell tools to render pages or extract frames first.
    """
    try:
        abs_path = _validate_path(path)
        target = Path(abs_path)
        if not target.is_file():
            return f"Error: {path} is not a regular file"
        size = target.stat().st_size
        if size > READ_FILE_MAX_BYTES:
            return (
                f"Error: {target.name} is {size} bytes; read() supports files up to "
                f"{READ_FILE_MAX_BYTES} bytes"
            )

        raw = target.read_bytes()
        try:
            normalized, width, height = normalize_image(raw)
        except ImageNotRecognized:
            pass
        except ImageNormalizationError as exc:
            return f"Error: cannot read image {target.name}: {exc}"
        else:
            description = f"Read image: {target.name}, {width}x{height}"
            if get_session_context().vision is False:
                return f"{description}\n[image omitted: model is text-only]"
            return [
                {"type": "text", "text": description},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": png_data_url(normalized),
                        "detail": "high",
                    },
                },
            ]

        try:
            decoded = raw.decode("utf-8")
        except UnicodeDecodeError:
            return f"Error: {_unsupported_binary_message(target, raw)}"
        if "\x00" in decoded:
            return f"Error: {_unsupported_binary_message(target, raw)}"
        lines = decoded.splitlines()
        start = max(0, int(offset) - 1)
        requested_limit = READ_FILE_DEFAULT_LIMIT if limit is None else max(0, int(limit))
        effective_limit = min(requested_limit, READ_FILE_MAX_LIMIT)
        end = start + effective_limit
        chunk = lines[start:end]
        if not chunk:
            return "(empty file)"
        output = "\n".join(f"{i + start + 1:4d}| {line}" for i, line in enumerate(chunk))
        if end < len(lines):
            output += (
                f"\n... (truncated; showing {len(chunk)} of {len(lines)} lines, "
                f"use offset={end + 1} to continue)"
            )
        return output
    except Exception as exc:
        return f"Error: {exc}"


def is_protected_write_path(path: str) -> bool:
    """Return True for project-relative paths under reserved writable state dirs."""
    rel = path.replace("\\", "/")
    while rel.startswith("./"):
        rel = rel[2:]
    return any(rel == name or rel.startswith(f"{name}/") for name in PROTECTED_WRITE_DIRS)


def _reject_protected_write(rel_path: str, action: str) -> str | None:
    if is_protected_write_path(rel_path):
        return f"Error: refusing to {action} protected path {rel_path}"
    return None


@tool
def delete(path: str) -> str:
    """Delete a file from the filesystem."""
    try:
        abs_path = _validate_path(path)
        rel = _relative_to_root(abs_path)
        if error := _reject_protected_write(rel, "delete"):
            return error
        p = Path(abs_path)
        if not p.exists():
            return f"Error: {rel} does not exist"
        if p.is_dir():
            return f"Error: {rel} is a directory; delete only removes files"
        p.unlink()
        return f"Deleted {rel}"
    except Exception as exc:
        return f"Error: {exc}"


@tool
def write(path: str, content: str) -> str:
    """Write a file to the local filesystem."""
    try:
        abs_path = _validate_path(path)
        rel = _relative_to_root(abs_path)
        if error := _reject_protected_write(rel, "write"):
            return error
        p = Path(abs_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        original = p.read_text(encoding="utf-8") if p.exists() else ""
        _atomic_write(p, content)
        auto_format(abs_path)
        written = p.read_text(encoding="utf-8")
        summary = f"Wrote {len(content)} chars to {rel}"
        return _with_diff(summary, _unified_diff(rel, original, written))
    except Exception as exc:
        return f"Error: {exc}"


@tool
def edit(
    path: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
) -> str:
    """Edit a file with one string replacement.

    Provide exact-text ``old_string`` / ``new_string`` (optional ``replace_all``).
    When ``replace_all`` is false, ``old_string`` must match exactly once; if it
    matches multiple times the file is left unchanged. For multiple independent
    replacements, call ``edit`` once per change (or set ``replace_all=True``).

    Uses exact match first with a conservative fuzzy fallback. If no match is
    found, the file is left unchanged.

    WARNING: When an exact match fails, a near-match (>= 0.95 similarity) may be
    rewritten in your place. Treat any 'FUZZY MATCH' result as suspicious and
    verify the change before relying on it; re-issue the edit with a more
    specific old_string if the wrong region was matched.
    """
    try:
        abs_path = _validate_path(path)
        rel = _relative_to_root(abs_path)
        if error := _reject_protected_write(rel, "edit"):
            return error
        p = Path(abs_path)
        original = p.read_text(encoding="utf-8")
        content, count, fuzzy = _replace_content(
            original, old_string, new_string, replace_all
        )
        if count < 0:
            n = -count
            return (
                f"Error: found {n} matches for old_string in {rel}; "
                "provide more surrounding context to make the match unique, "
                "or set replace_all=True; file was not changed"
            )
        if count == 0:
            return f"Error: no match for edit in {rel}; file was not changed"
        _atomic_write(p, content)
        auto_format(abs_path)
        written = p.read_text(encoding="utf-8")
        if fuzzy:
            summary = (
                f"WARNING: FUZZY MATCH applied in {rel} — "
                f"verify the result before continuing. "
                f"Applied 1 edit ({count} replacement{'s' if count != 1 else ''})."
            )
        else:
            summary = (
                f"Applied 1 edit ({count} replacement{'s' if count != 1 else ''}) to {rel}"
            )
        return _with_diff(summary, _unified_diff(rel, original, written))
    except Exception as exc:
        return f"Error: {exc}"


@tool
def glob(pattern: str) -> str:
    """Find files matching a glob pattern."""
    try:
        if not pattern:
            return "Error: pattern is empty. Provide a glob pattern to match files."
        matches = _git_glob(pattern) if is_git_repo(str(_project_root())) else []
        if not matches:
            matches = _filesystem_glob(pattern)
        return "\n".join(sorted(matches)[:300]) or "No matches"
    except Exception as exc:
        return f"Error: {exc}"


def _replace_content(content: str, old: str, new: str, replace_all: bool) -> tuple[str, int, bool]:
    if not old:
        return content, 0, False
    if old in content:
        occurrences = content.count(old)
        if not replace_all and occurrences > 1:
            return content, -occurrences, False
        count = occurrences if replace_all else 1
        return content.replace(old, new, -1 if replace_all else 1), count, False

    # fuzzy match (conservative — high threshold to avoid false positives):
    # 1. split file and old into lines
    # 2. Slide a window of len(old_lines) through the file
    # 3. At each position, join those lines into a block and compare to old using SequenceMatcher
    # 4. Keep the block with the highest similarity ratio
    # 5. Replace the block with the new lines only if score >= 0.95
    # 6. Otherwise fail (count = 0) — caller must re-issue with a precise old_string
    lines = content.splitlines()
    old_lines = old.splitlines()
    if replace_all or not old_lines or len(old_lines) > len(lines):
        return content, 0, False

    best_ratio = 0.0
    best_idx = -1
    for idx in range(len(lines) - len(old_lines) + 1):
        block = "\n".join(lines[idx : idx + len(old_lines)])
        ratio = difflib.SequenceMatcher(None, old, block).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_idx = idx

    if best_ratio < 0.95:
        return content, 0, False

    replacement_lines = new.splitlines()
    new_lines = lines[:best_idx] + replacement_lines + lines[best_idx + len(old_lines) :]
    return "\n".join(new_lines) + ("\n" if content.endswith("\n") else ""), 1, True


def _unified_diff(rel: str, old: str, new: str) -> str:
    """Return a unified diff string for the file change, or empty if identical."""
    if old == new:
        return ""
    fromfile = f"a/{rel}" if old else "/dev/null"
    tofile = f"b/{rel}" if new else "/dev/null"
    diff = difflib.unified_diff(
        old.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile=fromfile,
        tofile=tofile,
    )
    return "".join(diff).rstrip()


def _with_diff(summary: str, diff_text: str) -> str:
    if not diff_text or not diff_text.strip():
        return summary
    return f"{summary}\ndiff:\n{diff_text}"


def _atomic_write(path: Path, content: str) -> None:
    # does atomic replace of the temporary file with the content
    # create the temp file in parent directory of the path
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent), text=True)
    try:
        # write the content to the temporary file
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        # replace the original file with the temporary file
        os.replace(tmp, path)
    # clean up the temporary file
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _run_formatter(cmd: list[str]) -> None:
    if not cmd or not shutil.which(cmd[0]):
        return
    try:
        subprocess.run(
            cmd,
            cwd=_project_root(),
            capture_output=True,
            text=True,
            timeout=20,
        )
    except Exception:
        pass


def auto_format(path: str) -> None:
    """Run a formatter for known file types when the tool is available on PATH."""
    if not get_session_context().options.format_on_write:
        return

    p = Path(path)
    suffix = p.suffix.lower()
    path_str = str(p)

    prettier = _local_bin("prettier")
    prettier_cmd = [prettier, "--write", path_str] if prettier else None

    formatters: dict[str, list[str] | None] = {
        ".py": ["python", "-m", "black", "--quiet", "--", path_str],
        ".ts": prettier_cmd,
        ".tsx": prettier_cmd,
        ".js": prettier_cmd,
        ".jsx": prettier_cmd,
        ".json": prettier_cmd,
        ".md": prettier_cmd,
        ".rs": ["rustfmt", path_str],
        ".go": ["gofmt", "-w", path_str],
        ".c": ["clang-format", "-i", path_str],
        ".h": ["clang-format", "-i", path_str],
        ".cpp": ["clang-format", "-i", path_str],
        ".cc": ["clang-format", "-i", path_str],
        ".cxx": ["clang-format", "-i", path_str],
        ".hpp": ["clang-format", "-i", path_str],
        ".hh": ["clang-format", "-i", path_str],
    }
    cmd = formatters.get(suffix)
    if cmd:
        _run_formatter(cmd)


def _local_bin(name: str) -> str | None:
    """Return a project-local binary path if present, else fall back to PATH."""
    local = _project_root() / "node_modules" / ".bin" / name
    if local.exists():
        return str(local)
    return shutil.which(name)


def is_git_repo(path: str = ".") -> bool:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=path,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return False
    return result.returncode == 0 and result.stdout.strip() == "true"


def _filesystem_glob(pattern: str) -> list[str]:
    root = _project_root()
    return [str(p.relative_to(root)) for p in root.glob(pattern) if p.is_file()]


def _git_glob(pattern: str) -> list[str]:
    # git ls-files is a lot faster than the filesystem glob
    # if the repo is not a git repo then fallback to the filesystem glob
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=_project_root(),
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return []
    if result.returncode != 0:
        return []
    return [line for line in result.stdout.splitlines() if fnmatch.fnmatch(line, pattern)]
