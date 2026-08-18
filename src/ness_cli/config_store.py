"""Global JSON config/secrets storage for the CLI adapter.

Values live in the global config dir alongside ``USER.md`` and ``plans/``:

- ``configs.json``: non-secret adapter settings (0644)
- ``secrets.json``: API keys and other secrets (0600)

Files are written lazily: only values the user explicitly sets are
persisted. Defaults stay on the ``Settings`` class (see
:mod:`ness_cli.config`), which layers the JSON files between
process env vars and its own defaults.
"""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO, Iterator

from ness_cli.paths import config_dir_from_env

# Settings field names routed to ``secrets.json`` instead of ``configs.json``.
SECRET_KEYS: frozenset[str] = frozenset(
    {"openai_api_key", "opencode_api_key", "exa_api_key"}
)

_CONFIGS_NAME = "configs.json"
_SECRETS_NAME = "secrets.json"


def _lock_file(handle: BinaryIO) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        return

    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)


def _unlock_file(handle: BinaryIO) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return

    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def locked_path(path: Path, *, secret: bool = False) -> Iterator[None]:
    """Hold an advisory lock that remains stable when ``path`` is replaced."""
    lock_path = path.with_name(f"{path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    fd = os.open(lock_path, flags, 0o600 if secret else 0o666)
    with os.fdopen(fd, "r+b") as handle:
        if secret and hasattr(os, "fchmod"):
            os.fchmod(handle.fileno(), 0o600)
        _lock_file(handle)
        try:
            yield
        finally:
            _unlock_file(handle)


def configs_path(config_dir: Path | None = None) -> Path:
    return (config_dir or config_dir_from_env()) / _CONFIGS_NAME


def secrets_path(config_dir: Path | None = None) -> Path:
    return (config_dir or config_dir_from_env()) / _SECRETS_NAME


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def read_json_document(path: Path) -> dict[str, Any]:
    """Read an object-valued JSON document, returning ``{}`` on failure."""
    return _read_json(path)


def load_configs(config_dir: Path | None = None) -> dict[str, Any]:
    """Read ``configs.json`` (missing/corrupt file -> ``{}``)."""
    return _read_json(configs_path(config_dir))


def load_secrets(config_dir: Path | None = None) -> dict[str, Any]:
    """Read ``secrets.json`` (missing/corrupt file -> ``{}``)."""
    return _read_json(secrets_path(config_dir))


def _atomic_write(path: Path, data: dict[str, Any], *, secret: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
            handle.write("\n")
        if secret:
            os.chmod(tmp_name, 0o600)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def atomic_write_json(path: Path, data: dict[str, Any], *, secret: bool = False) -> None:
    """Atomically replace a JSON object, optionally enforcing mode ``0600``."""
    _atomic_write(path, data, secret=secret)


def _write_value(path: Path, key: str, value: Any, *, secret: bool) -> None:
    with locked_path(path, secret=secret):
        data = _read_json(path)
        if value is None:
            if key not in data:
                return
            del data[key]
        else:
            data[key] = value
        _atomic_write(path, data, secret=secret)


def write_config(key: str, value: Any, config_dir: Path | None = None) -> None:
    """Persist a non-secret value; ``None`` deletes the key."""
    _write_value(configs_path(config_dir), key, value, secret=False)


def write_secret(key: str, value: Any, config_dir: Path | None = None) -> None:
    """Persist a secret value (0600); ``None`` deletes the key."""
    _write_value(secrets_path(config_dir), key, value, secret=True)


def ensure_secrets_file(config_dir: Path | None = None) -> Path | None:
    """Create an empty ``secrets.json`` (0600) if missing. Returns path if created."""
    path = secrets_path(config_dir)
    if path.exists():
        return None
    _atomic_write(path, {}, secret=True)
    return path
