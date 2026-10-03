from __future__ import annotations

import io
import base64
from types import SimpleNamespace

import pytest
from PIL import Image, ImageGrab
from prompt_toolkit.buffer import Buffer

import ness_agent.media as media
from ness_agent.media import ImageNormalizationError, ImageTooLarge

from ness_cli.paths import resolve_paths
from ness_cli.tui.app import TuiApp
from ness_cli.tui.input import images


def _file(tmp_path, format_name="PNG", name="image", *, size=(12, 8)):
    path = tmp_path / name
    Image.new("RGB", size, "red").save(path, format=format_name)
    return path


def _clipboard(monkeypatch, value):
    monkeypatch.setattr(ImageGrab, "grabclipboard", lambda: value)


def test_native_image_object_remains_supported(monkeypatch):
    source = Image.new("RGB", (12, 8), "red")
    _clipboard(monkeypatch, source)
    with Image.open(io.BytesIO(images._grab_imagegrab())) as image:
        assert image.format == "PNG"
        assert image.size == source.size
        assert image.getpixel((0, 0)) == (255, 0, 0)


@pytest.mark.parametrize("value", [None, [], "unexpected string", 42, [None, 42, {}]])
def test_empty_or_unsupported_clipboard_reports_no_image(monkeypatch, value):
    _clipboard(monkeypatch, value)
    with pytest.raises(images.NoClipboardImage):
        images._grab_imagegrab()


def test_file_list_uses_first_readable_image(tmp_path, monkeypatch):
    notes = tmp_path / "notes.txt"
    notes.write_text("Not an image")
    first = _file(tmp_path, name="first.png")
    second = _file(tmp_path, name="second.png", size=(3, 2))
    _clipboard(
        monkeypatch,
        [str(tmp_path / "missing.png"), str(notes), str(first), str(second)],
    )
    assert images._grab_imagegrab() == first.read_bytes()


@pytest.mark.parametrize("kind", ["corrupt", "permission"])
def test_unreadable_file_list_reports_no_image(tmp_path, monkeypatch, kind):
    path = tmp_path / "image.png"
    if kind == "corrupt":
        path.write_bytes(b"\x89PNG\r\n\x1a\ncorrupt")
    elif kind == "permission":
        path = _file(tmp_path)

        def denied(self):
            raise PermissionError("test access denied")

        monkeypatch.setattr(type(path), "read_bytes", denied)
    _clipboard(monkeypatch, [str(path)])
    with pytest.raises(images.NoClipboardImage):
        images._grab_imagegrab()


def test_native_clipboard_error_is_reported_as_no_image(monkeypatch):
    def fail():
        raise RuntimeError("clipboard unavailable")

    monkeypatch.setattr(ImageGrab, "grabclipboard", fail)
    with pytest.raises(images.NoClipboardImage, match="clipboard unavailable") as error:
        images._grab_imagegrab()
    assert isinstance(error.value.__cause__, RuntimeError)


def _save_from_native_clipboard(tmp_path, monkeypatch, value):
    _clipboard(monkeypatch, value)
    monkeypatch.setattr(images, "grab_clipboard_image", images._grab_imagegrab)
    return images.save_clipboard_image()


@pytest.mark.parametrize("format_name", ["PNG", "JPEG", "GIF"])
def test_copied_file_is_normalized_saved_and_encoded(tmp_path, monkeypatch, format_name):
    original = _file(tmp_path, format_name, name="copied image without extension", size=(2400, 1200))
    original_bytes = original.read_bytes()
    result = _save_from_native_clipboard(tmp_path, monkeypatch, [str(original)])
    scratch, data_url = result
    assert scratch != original
    assert original.read_bytes() == original_bytes
    assert base64.b64decode(data_url.split(",", 1)[1]) == scratch.read_bytes()
    with Image.open(scratch) as image:
        assert image.format == "PNG"
        assert image.size == (2000, 1000)


def test_copied_file_preserves_alpha(tmp_path, monkeypatch):
    original = tmp_path / "transparent.png"
    Image.new("RGBA", (2, 1), (255, 0, 0, 64)).save(original)
    scratch, _url = _save_from_native_clipboard(tmp_path, monkeypatch, [str(original)])
    with Image.open(scratch) as image:
        assert image.getpixel((0, 0)) == (255, 0, 0, 64)


def test_copied_file_applies_exif_orientation(tmp_path, monkeypatch):
    original = tmp_path / "oriented.jpg"
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (3, 2), "red").save(original, exif=exif)
    scratch, _url = _save_from_native_clipboard(tmp_path, monkeypatch, [str(original)])
    with Image.open(scratch) as image:
        assert image.size == (2, 3)


def test_copied_truncated_image_reports_pipeline_decode_error(tmp_path, monkeypatch):
    original = _file(tmp_path, "JPEG", size=(40, 30))
    original.write_bytes(original.read_bytes()[:-30])
    cache = resolve_paths().cache_dir / "images"
    assert not cache.exists()
    with pytest.raises(ImageNormalizationError, match="invalid or corrupt image"):
        _save_from_native_clipboard(tmp_path, monkeypatch, [str(original)])
    assert not cache.exists()


@pytest.mark.parametrize("pixel_limit", [100, 250])
def test_copied_file_uses_pipeline_pixel_limit(tmp_path, monkeypatch, pixel_limit):
    original = _file(tmp_path, size=(20, 20))
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", pixel_limit)
    cache = resolve_paths().cache_dir / "images"
    assert not cache.exists()
    with pytest.raises(ImageNormalizationError, match="safe pixel limit"):
        _save_from_native_clipboard(tmp_path, monkeypatch, [str(original)])
    assert not cache.exists()


def test_copied_file_uses_pipeline_output_limit(tmp_path, monkeypatch):
    original = _file(tmp_path)
    monkeypatch.setattr(media, "MAX_NORMALIZED_IMAGE_BYTES", 10)
    cache = resolve_paths().cache_dir / "images"
    assert not cache.exists()
    with pytest.raises(ImageTooLarge):
        _save_from_native_clipboard(tmp_path, monkeypatch, [str(original)])
    assert not cache.exists()


@pytest.mark.parametrize("valid", [False, True])
def test_tui_file_paste_stages_image_or_warns(tmp_path, monkeypatch, valid):
    original = _file(tmp_path)
    if not valid:
        original.write_text("Not an image")
    _clipboard(monkeypatch, [str(original)])
    monkeypatch.setattr(images, "grab_clipboard_image", images._grab_imagegrab)
    warnings = []
    app = SimpleNamespace(
        runtime=SimpleNamespace(paths=resolve_paths()),
        sink=SimpleNamespace(warning=warnings.append),
        _image_counter=0,
        _pending_images={},
        input_buffer=Buffer(),
    )
    TuiApp.paste_clipboard_image(app)
    if valid:
        assert app.input_buffer.text == "[Image #1] "
        assert app._pending_images[1].startswith("data:image/png;base64,")
        assert warnings == []
    else:
        assert app.input_buffer.text == ""
        assert app._pending_images == {}
        assert warnings == ["No image found on the clipboard."]
