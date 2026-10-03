"""Expand project file mentions for model input."""

from __future__ import annotations

import re
from pathlib import Path

from ness_agent import PermissionStore

_MENTION_TOKEN = re.compile(r"(?<![\w])@([\w./\-]+)")
MAX_INLINE_FILE_BYTES = 256 * 1024


def extract_mentions(text: str) -> tuple[str, tuple[str, ...]]:
    """Return the unchanged text and mentioned paths in input order."""
    source = text or ""
    return source, tuple(match.group(1) for match in _MENTION_TOKEN.finditer(source))


def expand_documents(text: str, permission_store: PermissionStore, *, yolo_mode: bool = False) -> str:
    """Prepend one document block per mention while retaining the raw text."""
    source, mentions = extract_mentions(text)
    if not source or not mentions:
        return source
    blocks = [_render_document(path, permission_store, yolo_mode=yolo_mode) for path in mentions]
    return "\n\n".join((*blocks, source))


def _document(path: str, content: str) -> str:
    return (
        "<document>\n"
        "  <document_content>\n"
        f"  {content}\n"
        "  </document_content>\n"
        f"  <source>{path}</source>\n"
        "</document>"
    )


def _render_document(path: str, permission_store: PermissionStore, *, yolo_mode: bool = False) -> str:
    try:
        resolved = Path(permission_store.validate_path(path, yolo_mode=yolo_mode))
    except Exception as error:
        return _document(path, f"Error: {error}")

    try:
        if not resolved.exists():
            return _document(path, f"Error: {path} does not exist")
        if resolved.is_dir():
            return _document(path, f"Error: {path} is a directory")
        size = resolved.stat().st_size
        if size > MAX_INLINE_FILE_BYTES:
            return _document(
                path,
                (
                    f"Error: {path} is too large ({size} bytes) to inline as "
                    "a mention; use the read tool with an offset to inspect it."
                ),
            )
        content = resolved.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return _document(
            path,
            f"Error: {path} is not valid UTF-8 (likely binary)",
        )
    except Exception as error:
        return _document(path, f"Error: {error}")

    return _document(path, content)
