"""Tests for the global JSON config/secrets store and Settings sourcing."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from ness_cli.config_store import (
    ensure_secrets_file,
    load_configs,
    load_secrets,
    locked_path,
    write_config,
    write_secret,
)
from ness_cli.provider.profile import provider_profile, update_provider_profile


@pytest.fixture
def config_dir(tmp_path: Path, monkeypatch) -> Path:
    cfg = tmp_path / "cfg"
    monkeypatch.setenv("NESS_AGENT_CONFIG_DIR", str(cfg))
    return cfg


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_write_config_roundtrip_and_lazy_creation(config_dir: Path):
    assert not (config_dir / "configs.json").exists()
    write_config("model_name", "openai/gpt-4o")
    write_config("enable_approval", False)
    assert load_configs() == {"enable_approval": False, "model_name": "openai/gpt-4o"}


def test_write_secret_permissions(config_dir: Path):
    write_secret("openai_api_key", "sk-test")
    path = config_dir / "secrets.json"
    assert load_secrets() == {"openai_api_key": "sk-test"}
    assert _mode(path) == 0o600


def test_provider_profiles_are_nested_and_preserve_other_config(config_dir: Path):
    write_config("enable_approval", False)
    update_provider_profile("codex", {"model_name": "gpt-test"})
    update_provider_profile("openrouter", {"model_name": "openai/gpt-test"})
    assert provider_profile("codex") == {"model_name": "gpt-test"}
    assert load_configs()["enable_approval"] is False


def test_write_none_deletes_key(config_dir: Path):
    write_config("openai_base_url", "https://example.com/v1")
    write_config("openai_base_url", None)
    assert load_configs() == {}
    # Deleting a missing key is a no-op and does not create the file.
    (config_dir / "configs.json").unlink()
    write_config("openai_base_url", None)
    assert not (config_dir / "configs.json").exists()


def test_corrupt_json_tolerated(config_dir: Path):
    config_dir.mkdir(parents=True)
    (config_dir / "configs.json").write_text("{not json", encoding="utf-8")
    (config_dir / "secrets.json").write_text("[1, 2]", encoding="utf-8")
    assert load_configs() == {}
    assert load_secrets() == {}
    # A corrupt file is replaced wholesale on next write.
    write_config("model_name", "x")
    assert load_configs() == {"model_name": "x"}


def test_ensure_secrets_file_creates_once(config_dir: Path):
    created = ensure_secrets_file()
    assert created is not None and created.is_file()
    assert json.loads(created.read_text()) == {}
    assert _mode(created) == 0o600
    assert ensure_secrets_file() is None


@pytest.mark.skipif(not hasattr(os, "O_NOFOLLOW"), reason="O_NOFOLLOW is unavailable")
def test_locked_path_refuses_a_sidecar_symlink(tmp_path: Path):
    destination = tmp_path / "credentials.json"
    target = tmp_path / "unrelated"
    target.write_text("keep", encoding="utf-8")
    destination.with_name("credentials.json.lock").symlink_to(target)

    with pytest.raises(OSError):
        with locked_path(destination, secret=True):
            pass

    assert target.read_text(encoding="utf-8") == "keep"


def test_settings_reads_global_json(config_dir: Path, monkeypatch):
    monkeypatch.delenv("MODEL_NAME", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    write_config("model_name", "openai/gpt-4o")
    write_secret("openai_api_key", "sk-json")
    from ness_cli.config import Settings

    fresh = Settings()
    assert fresh.model_name == "openai/gpt-4o"
    assert fresh.openai_api_key == "sk-json"


def test_settings_reads_opencode_key_aliases_and_saved_secret(
    config_dir: Path, monkeypatch
):
    monkeypatch.delenv("OPENCODE_GO_API_KEY", raising=False)
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
    write_secret("opencode_api_key", "sk-json-go")
    from ness_cli.config import Settings

    assert Settings().opencode_api_key == "sk-json-go"
    monkeypatch.setenv("OPENCODE_API_KEY", "sk-env-go")
    assert Settings().opencode_api_key == "sk-env-go"


def test_settings_env_overrides_json(config_dir: Path, monkeypatch):
    write_config("model_name", "openai/gpt-4o")
    monkeypatch.setenv("MODEL_NAME", "deepseek/deepseek-chat")
    from ness_cli.config import Settings

    assert Settings().model_name == "deepseek/deepseek-chat"


def test_settings_defaults_without_json(config_dir: Path, monkeypatch):
    monkeypatch.delenv("MODEL_NAME", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from ness_cli.config import Settings

    fresh = Settings()
    assert fresh.model_name == "deepseek/deepseek-v4-flash"
    assert fresh.openai_api_key is None


def test_project_dotenv_is_not_loaded_or_migrated(
    config_dir: Path, tmp_path: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MODEL_NAME", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "MODEL_NAME=dotenv/model\nOPENAI_API_KEY=dotenv-secret\n",
        encoding="utf-8",
    )
    from ness_cli.config import Settings
    from ness_cli.factory import prepare_paths

    fresh = Settings()
    prepare_paths(project_root=tmp_path)

    assert fresh.model_name == "deepseek/deepseek-v4-flash"
    assert fresh.openai_api_key is None
    assert load_configs() == {}
    assert load_secrets() == {}
    assert "dotenv-secret" in dotenv.read_text(encoding="utf-8")
