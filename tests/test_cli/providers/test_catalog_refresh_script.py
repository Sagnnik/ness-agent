from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict
from pathlib import Path

import pytest

from ness_cli.config import ConfigManager
from ness_cli.providers.openrouter.catalog import (
    CATALOG_VERSION,
    OPENROUTER_CATALOG_CACHE_NAME,
    ModelRecord,
)
from ness_cli.providers.registry import ProviderRegistry
from scripts.fetch_openrouter_models import main


def _seed_cache(path: Path) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "version": CATALOG_VERSION,
                "fetched_at": time.time(),
                "models": [asdict(ModelRecord(id="old/model", name="Old model"))],
            }
        ),
        encoding="utf-8",
    )
    return path.read_bytes()


def test_refresh_script_updates_the_runtime_cache(
    isolated_cli_env, monkeypatch, capsys
):
    cache_path = isolated_cli_env.cache / OPENROUTER_CATALOG_CACHE_NAME
    _seed_cache(cache_path)
    monkeypatch.setattr(
        "ness_cli.providers.openrouter.catalog.fetch_catalog",
        lambda: [ModelRecord(id="fresh/model", name="Fresh", context_length=12345)],
    )

    assert main(["--refresh"]) == 0

    output = capsys.readouterr()
    assert output.out == f"cached 1 models at {cache_path}\n"
    assert output.err == ""
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    assert payload["version"] == CATALOG_VERSION
    assert [item["id"] for item in payload["models"]] == ["fresh/model"]
    config = ConfigManager.load(isolated_cli_env.paths, environment={})
    registry = ProviderRegistry.load(paths=isolated_cli_env.paths, config=config)
    try:
        info = registry.model_info("fresh/model", provider_id="openrouter")
        assert info is not None and info.context_window == 12345
    finally:
        asyncio.run(registry.close())


def test_refresh_help_describes_explicit_listing_policy(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    assert "does not refresh automatically" in capsys.readouterr().out


@pytest.mark.parametrize("failure", ["fetch", "write"])
def test_refresh_script_reports_failure_and_preserves_the_cache(
    isolated_cli_env, monkeypatch, capsys, failure
):
    cache_path = isolated_cli_env.cache / OPENROUTER_CATALOG_CACHE_NAME
    previous = _seed_cache(cache_path)

    def fail(*_args):
        raise OSError(f"{failure} failed")

    if failure == "fetch":
        monkeypatch.setattr("ness_cli.providers.openrouter.catalog.fetch_catalog", fail)
    else:
        monkeypatch.setattr(
            "ness_cli.providers.openrouter.catalog.fetch_catalog",
            lambda: [ModelRecord(id="fresh/model", name="Fresh")],
        )
        monkeypatch.setattr("ness_cli.providers.openrouter.catalog.os.replace", fail)

    assert main(["--refresh"]) == 1

    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == f"error: catalog refresh failed: {failure} failed\n"
    assert cache_path.read_bytes() == previous
    assert not tuple(cache_path.parent.glob(f"{cache_path.name}.*.tmp"))
