from __future__ import annotations

import base64
import io
from types import SimpleNamespace

import platformdirs
import pytest
from PIL import Image
from prompt_toolkit.buffer import Buffer

import ness_cli.paths as cli_paths
from ness_cli.paths import project_hash, resolve_paths
from ness_cli.tui.app import TuiApp
from ness_cli.tui.input import images


@pytest.fixture
def clipboard_png(monkeypatch):
    buffer = io.BytesIO()
    Image.new("RGB", (3, 2), "red").save(buffer, format="PNG")
    monkeypatch.setattr(images, "grab_clipboard_image", buffer.getvalue)


@pytest.fixture
def platform_cache(tmp_path, monkeypatch):
    # Keep even the old implementation's failing probe out of the user's cache.
    directory = tmp_path / "platform-cache"
    monkeypatch.setattr(platformdirs, "user_cache_dir", lambda _app: str(directory))
    monkeypatch.setattr(cli_paths, "user_cache_dir", lambda _app: str(directory))
    return directory


def test_save_uses_configured_project_cache(
    isolated_cli_env, clipboard_png, platform_cache
):
    path, data_url = images.save_clipboard_image()

    assert path.parent == isolated_cli_env.paths.cache_dir / "images"
    assert base64.b64decode(data_url.split(",", 1)[1]) == path.read_bytes()
    assert not platform_cache.exists()


def test_save_uses_platform_cache_when_override_is_absent(
    isolated_cli_env, monkeypatch, clipboard_png, platform_cache
):
    monkeypatch.delenv("NESS_AGENT_CACHE_DIR")
    path, _data_url = images.save_clipboard_image()

    assert (
        path.parent
        == platform_cache / project_hash(isolated_cli_env.project) / "images"
    )
    assert not isolated_cli_env.cache.exists()


@pytest.mark.parametrize("override", ["relative-cache", "~/image-cache"])
def test_save_resolves_relative_and_home_cache_overrides(
    isolated_cli_env, monkeypatch, clipboard_png, platform_cache, override
):
    monkeypatch.setenv("HOME", str(isolated_cli_env.root))
    monkeypatch.setenv("NESS_AGENT_CACHE_DIR", override)
    expected = (
        isolated_cli_env.project / override
        if not override.startswith("~")
        else isolated_cli_env.root / "image-cache"
    )
    path, _data_url = images.save_clipboard_image()

    assert path.parent == expected / isolated_cli_env.paths.project_hash / "images"
    assert not platform_cache.exists()


def test_each_project_has_its_own_scratch_directory(
    isolated_cli_env, monkeypatch, clipboard_png, platform_cache
):
    first, _ = images.save_clipboard_image()
    other = isolated_cli_env.root / "other-project"
    other.mkdir()
    monkeypatch.chdir(other)
    second, _ = images.save_clipboard_image()

    assert first.parent == isolated_cli_env.paths.cache_dir / "images"
    assert second.parent == isolated_cli_env.cache / project_hash(other) / "images"
    assert first.parent != second.parent
    assert first.is_file()
    assert not platform_cache.exists()


def test_tui_uses_startup_paths_after_cwd_and_environment_change(
    isolated_cli_env, monkeypatch, clipboard_png, platform_cache
):
    paths = isolated_cli_env.paths
    other = isolated_cli_env.root / "other-project"
    other.mkdir()
    monkeypatch.chdir(other)
    redirected_cache = isolated_cli_env.root / "redirected-cache"
    monkeypatch.setenv("NESS_AGENT_CACHE_DIR", str(redirected_cache))
    warnings = []
    app = SimpleNamespace(
        runtime=SimpleNamespace(paths=paths),
        sink=SimpleNamespace(warning=warnings.append),
        _image_counter=0,
        _pending_images={},
        input_buffer=Buffer(),
    )

    TuiApp.paste_clipboard_image(app)

    files = list((paths.cache_dir / "images").glob("*.png"))
    assert len(files) == 1
    assert (
        base64.b64decode(app._pending_images[1].split(",", 1)[1])
        == files[0].read_bytes()
    )
    assert app.input_buffer.text == "[Image #1] "
    assert warnings == []
    assert not redirected_cache.exists()
    assert not platform_cache.exists()

    data_url = app._pending_images[1]
    files[0].unlink()
    assert TuiApp._images_for_text(app, app.input_buffer.text) == [data_url]


def test_cache_write_failure_warns_without_using_another_cache(
    isolated_cli_env, clipboard_png, platform_cache
):
    paths = isolated_cli_env.paths
    paths.cache_dir.mkdir(parents=True)
    blocker = paths.cache_dir / "images"
    blocker.write_text("A file prevents creation of the scratch directory")
    warnings = []
    app = SimpleNamespace(
        runtime=SimpleNamespace(paths=paths),
        sink=SimpleNamespace(warning=warnings.append),
        _image_counter=0,
        _pending_images={},
        input_buffer=Buffer(),
    )

    TuiApp.paste_clipboard_image(app)

    assert len(warnings) == 1
    assert warnings[0].startswith("Image paste failed:")
    assert str(blocker) in warnings[0]
    assert app.input_buffer.text == ""
    assert app._pending_images == {}
    assert app._image_counter == 0
    assert blocker.is_file()
    assert not platform_cache.exists()


@pytest.mark.parametrize("kind", ["empty", "invalid", "oversized"])
def test_unsuccessful_paste_does_not_create_cache(
    isolated_cli_env, monkeypatch, platform_cache, kind
):
    if kind == "empty":

        def no_image():
            raise images.NoClipboardImage()

        monkeypatch.setattr(images, "grab_clipboard_image", no_image)
        assert images.save_clipboard_image() is None
    elif kind == "invalid":
        from ness_agent.media import ImageNormalizationError

        monkeypatch.setattr(images, "grab_clipboard_image", lambda: b"not an image")
        with pytest.raises(ImageNormalizationError):
            images.save_clipboard_image()
    else:
        import ness_agent.media as media

        buffer = io.BytesIO()
        Image.new("RGB", (3, 2), "red").save(buffer, format="PNG")
        monkeypatch.setattr(images, "grab_clipboard_image", buffer.getvalue)
        monkeypatch.setattr(media, "MAX_NORMALIZED_IMAGE_BYTES", 1)
        with pytest.raises(images.ImageTooLarge):
            images.save_clipboard_image()

    assert not isolated_cli_env.cache.exists()
    assert not platform_cache.exists()


def test_explicit_paths_keep_scratch_writes_under_the_resolved_root(
    isolated_cli_env, clipboard_png, platform_cache
):
    other = isolated_cli_env.root / "other-project"
    other.mkdir()
    paths = resolve_paths(project_root=other)

    first, _ = images.save_clipboard_image(paths=paths)
    second, _ = images.save_clipboard_image(paths=paths)

    assert first.parent == paths.cache_dir / "images"
    assert second.parent == first.parent
    assert first != second
    assert first.is_file() and second.is_file()
    assert not (isolated_cli_env.paths.cache_dir / "images").exists()
    assert not platform_cache.exists()
