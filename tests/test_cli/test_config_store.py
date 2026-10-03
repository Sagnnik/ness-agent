from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from ness_cli.config.settings import load_settings
from ness_cli.config.store import ConfigStore, locked_path


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_defaults_load_without_creating_documents(isolated_cli_env) -> None:
    store = ConfigStore(isolated_cli_env.config)

    settings = load_settings(store, {})

    assert settings.model_provider == "openrouter"
    assert settings.model_name == "deepseek/deepseek-v4-flash"
    assert not store.configs_file.exists()
    assert not store.secrets_file.exists()


def test_config_round_trip_is_lazy_and_preserves_unrelated_keys(
    isolated_cli_env,
) -> None:
    store = ConfigStore(isolated_cli_env.config)

    store.write_config("model_name", "openai/gpt-test")
    store.write_config("enable_approval", False)

    assert store.load_config() == {
        "enable_approval": False,
        "model_name": "openai/gpt-test",
    }
    assert not store.secrets_file.exists()


def test_secrets_use_a_separate_restrictive_file(isolated_cli_env) -> None:
    store = ConfigStore(isolated_cli_env.config)

    store.write_secret("openai_api_key", "sk-test")

    assert store.load_secrets() == {"openai_api_key": "sk-test"}
    assert store.load_config() == {}
    assert _mode(store.secrets_file) == 0o600


def test_provider_profiles_are_separate(isolated_cli_env) -> None:
    store = ConfigStore(isolated_cli_env.config)
    store.write_config("enable_approval", False)

    store.update_provider_profile("codex", {"model_name": "gpt-test"})
    store.update_provider_profile("openrouter", {"model_name": "openai/gpt-test"})

    assert store.provider_profile("codex") == {"model_name": "gpt-test"}
    assert store.provider_profile("openrouter") == {"model_name": "openai/gpt-test"}
    assert store.load_config()["enable_approval"] is False


def test_none_deletes_a_key_without_creating_a_document(isolated_cli_env) -> None:
    store = ConfigStore(isolated_cli_env.config)
    store.write_config("openai_base_url", None)
    assert not store.configs_file.exists()

    store.write_config("openai_base_url", "https://example.test/v1")
    store.write_config("openai_base_url", None)
    assert store.load_config() == {}


@pytest.mark.parametrize("secret", [False, True])
@pytest.mark.parametrize("value", [None, False, ""])
def test_single_key_writes_preserve_public_mutation_overrides(
    isolated_cli_env, secret, value
) -> None:
    class MemoryStore(ConfigStore):
        def __init__(self):
            super().__init__(isolated_cli_env.config)
            self.config = {"keep": "config", "target": "before"}
            self.secrets = {"keep": "secret", "target": "before"}
            self.calls = []

        def mutate_config(self, operation):
            self.calls.append("config")
            self.config = operation(dict(self.config))
            return self.config

        def mutate_secrets(self, operation):
            self.calls.append("secrets")
            self.secrets = operation(dict(self.secrets))
            return self.secrets

    store = MemoryStore()
    writer = store.write_secret if secret else store.write_config
    writer("target", value)

    selected = store.secrets if secret else store.config
    untouched = store.config if secret else store.secrets
    expected = {"keep": "secret" if secret else "config"}
    if value is not None:
        expected["target"] = value
    assert selected == expected
    assert untouched == {"keep": "config" if secret else "secret", "target": "before"}
    assert store.calls == ["secrets" if secret else "config"]
    assert not store.configs_file.exists()
    assert not store.secrets_file.exists()


def test_corrupt_documents_recover_on_the_next_write(isolated_cli_env) -> None:
    store = ConfigStore(isolated_cli_env.config)
    isolated_cli_env.config.mkdir()
    store.configs_file.write_text("{bad json", encoding="utf-8")
    store.secrets_file.write_text("[]", encoding="utf-8")

    assert store.load_config() == {}
    assert store.load_secrets() == {}

    store.write_config("model_name", "model-a")
    assert json.loads(store.configs_file.read_text(encoding="utf-8")) == {
        "model_name": "model-a"
    }


def test_ensure_secrets_file_is_idempotent(isolated_cli_env) -> None:
    store = ConfigStore(isolated_cli_env.config)

    created = store.ensure_secrets_file()

    assert created == store.secrets_file
    assert json.loads(created.read_text(encoding="utf-8")) == {}
    assert _mode(created) == 0o600
    assert store.ensure_secrets_file() is None


@pytest.mark.skipif(not hasattr(os, "O_NOFOLLOW"), reason="O_NOFOLLOW unavailable")
def test_locked_path_refuses_a_sidecar_symlink(isolated_cli_env) -> None:
    destination = isolated_cli_env.root / "credentials.json"
    target = isolated_cli_env.root / "unrelated"
    target.write_text("keep", encoding="utf-8")
    destination.with_name("credentials.json.lock").symlink_to(target)

    with pytest.raises(OSError):
        with locked_path(destination, secret=True):
            pass

    assert target.read_text(encoding="utf-8") == "keep"


def test_atomic_replace_failure_keeps_the_previous_document(
    isolated_cli_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ConfigStore(isolated_cli_env.config)
    store.write_config("model_name", "before")
    original = store.configs_file.read_bytes()

    def fail_replace(_source, _destination) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        store.write_config("model_name", "after")

    assert store.configs_file.read_bytes() == original
    assert not list(isolated_cli_env.config.glob("*.tmp"))


@pytest.mark.skipif(os.name == "nt", reason="POSIX lock probe")
def test_document_preparation_holds_both_locks_before_writing(isolated_cli_env) -> None:
    store = ConfigStore(isolated_cli_env.config)
    store.write_config("format_on_write", True)
    store.write_secret("openai_api_key", "old-test-key")
    probe = """
import fcntl
import sys
for name in sys.argv[1:]:
    with open(name, 'rb') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            continue
        raise AssertionError('document lock was not held')
"""

    def prepare(config, secrets):
        subprocess.run(
            [
                sys.executable,
                "-c",
                probe,
                str(store.configs_file.with_name("configs.json.lock")),
                str(store.secrets_file.with_name("secrets.json.lock")),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert store.load_config()["format_on_write"] is True
        assert store.load_secrets()["openai_api_key"] == "old-test-key"
        config["format_on_write"] = False
        secrets["openai_api_key"] = "new-test-key"
        return config, secrets

    config, secrets = store.mutate_documents(prepare)

    assert config == store.load_config() == {"format_on_write": False}
    assert secrets == store.load_secrets() == {"openai_api_key": "new-test-key"}
    assert _mode(store.secrets_file) == 0o600
