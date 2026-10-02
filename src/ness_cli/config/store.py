from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterator, Callable, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO

SECRET_KEYS: frozenset[str] = frozenset(
    {"openai_api_key", "opencode_api_key", "exa_api_key"}
)

_CONFIGS_NAME = "configs.json"
_SECRETS_NAME = "secrets.json"


class ConfigWriteError(RuntimeError):
    """Identify a failed document write while preserving its original cause."""

    def __init__(self, document: str) -> None:
        super().__init__(f"Error while saving {document}.")
        self.document = document


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
def locked_path(
    path: Path,
    *,
    secret: bool = False,
) -> Iterator[None]:
    lock_path = path.with_name(f"{path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    flags = os.O_CREAT | os.O_RDWR

    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW

    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY

    mode = 0o600 if secret else 0o666
    fd = os.open(lock_path, flags, mode)

    with os.fdopen(fd, "r+b") as handle:
        if secret and hasattr(os, "fchmod"):
            os.fchmod(handle.fileno(), 0o600)

        _lock_file(handle)

        try:
            yield
        finally:
            _unlock_file(handle)


def _read_json(path: Path, *, strict: bool = False) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError, TypeError):
        if strict:
            raise
        return {}

    if strict and not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value if isinstance(value, dict) else {}


def read_json_document(path: Path) -> dict[str, Any]:
    return _read_json(path)


def _atomic_write(
    path: Path,
    value: Mapping[str, Any],
    *,
    secret: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f"{path.name}.",
        suffix=".tmp",
    )

    temporary = Path(temporary_name)

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                dict(value),
                handle,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        os.chmod(temporary, 0o600 if secret else 0o644)
        os.replace(temporary, path)

        if secret:
            os.chmod(path, 0o600)
    except BaseException:
        try:
            temporary.unlink()
        except OSError:
            pass

        raise


def atomic_write_json(
    path: Path,
    value: Mapping[str, Any],
    *,
    secret: bool = False,
) -> None:
    _atomic_write(path, value, secret=secret)


class ConfigStore:
    def __init__(self, config_dir: Path) -> None:
        self.config_dir = Path(config_dir).expanduser().resolve()
        self.configs_file = self.config_dir / _CONFIGS_NAME
        self.secrets_file = self.config_dir / _SECRETS_NAME

    def load_config(self, *, strict: bool = False) -> dict[str, Any]:
        return _read_json(self.configs_file, strict=strict)

    def load_secrets(self, *, strict: bool = False) -> dict[str, Any]:
        return _read_json(self.secrets_file, strict=strict)

    def mutate_documents(
        self,
        operation: Callable[
            [dict[str, Any], dict[str, Any]],
            tuple[dict[str, Any], dict[str, Any]],
        ],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Prepare and validate a patch against both locked documents.

        Locks always use config-then-secrets order and match single-document
        writers. Replacements remain separate: a later failure can leave an
        earlier document saved. Return the actual committed documents.
        """
        with (
            locked_path(self.configs_file),
            locked_path(self.secrets_file, secret=True),
        ):
            config = _read_json(self.configs_file)
            secrets = _read_json(self.secrets_file)
            current_config, current_secrets = operation(dict(config), dict(secrets))
            if not isinstance(current_config, dict) or not isinstance(
                current_secrets, dict
            ):
                raise TypeError("config mutation must return two dictionaries")

            for path, previous, current, secret in (
                (self.configs_file, config, current_config, False),
                (self.secrets_file, secrets, current_secrets, True),
            ):
                if current != previous:
                    try:
                        _atomic_write(path, current, secret=secret)
                    except Exception as error:
                        raise ConfigWriteError(path.name) from error

            return current_config, current_secrets

    def mutate_config(
        self,
        operation: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> dict[str, Any]:
        return self._mutate(
            self.configs_file,
            operation,
            secret=False,
        )

    def mutate_secrets(
        self,
        operation: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> dict[str, Any]:
        return self._mutate(
            self.secrets_file,
            operation,
            secret=True,
        )

    def _mutate(
        self,
        path: Path,
        operation: Callable[[dict[str, Any]], dict[str, Any]],
        *,
        secret: bool,
    ) -> dict[str, Any]:
        with locked_path(path, secret=secret):
            previous = _read_json(path)
            current = operation(dict(previous))

            if not isinstance(current, dict):
                raise TypeError("config mutation must return a dictionary")

            if current != previous:
                _atomic_write(path, current, secret=secret)

            return current

    def write_config(self, key: str, value: Any) -> None:
        self._write_value(key, value, secret=False)

    def write_secret(self, key: str, value: Any) -> None:
        self._write_value(key, value, secret=True)

    def _write_value(self, key: str, value: Any, *, secret: bool) -> None:
        def update(document: dict[str, Any]) -> dict[str, Any]:
            if value is None:
                document.pop(key, None)
            else:
                document[key] = value

            return document

        mutate = self.mutate_secrets if secret else self.mutate_config
        mutate(update)

    def provider_profile(self, provider_id: str) -> dict[str, Any]:
        profiles = self.load_config().get("provider_profiles")

        if not isinstance(profiles, dict):
            return {}

        profile = profiles.get(provider_id)
        return dict(profile) if isinstance(profile, dict) else {}

    def update_provider_profile(
        self,
        provider_id: str,
        values: Mapping[str, Any],
    ) -> None:
        def update(document: dict[str, Any]) -> dict[str, Any]:
            raw_profiles = document.get("provider_profiles")
            profiles = dict(raw_profiles) if isinstance(raw_profiles, dict) else {}

            raw_profile = profiles.get(provider_id)
            profile = dict(raw_profile) if isinstance(raw_profile, dict) else {}

            for key, value in values.items():
                if value is None:
                    profile.pop(key, None)
                else:
                    profile[key] = value

            if profile:
                profiles[provider_id] = profile
            else:
                profiles.pop(provider_id, None)

            if profiles:
                document["provider_profiles"] = profiles
            else:
                document.pop("provider_profiles", None)

            return document

        self.mutate_config(update)

    def ensure_secrets_file(self) -> Path | None:
        path = self.secrets_file

        with locked_path(path, secret=True):
            if path.exists():
                try:
                    os.chmod(path, 0o600)
                except OSError:
                    pass

                return None

            _atomic_write(path, {}, secret=True)
            return path


def ensure_secrets_file(config_dir: Path) -> Path | None:
    return ConfigStore(config_dir).ensure_secrets_file()
