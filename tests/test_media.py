from __future__ import annotations

import io

import pytest
from PIL import Image

import ness_agent.media as media
from ness_agent.media import (
    ImageNormalizationError,
    ImageTooLarge,
    normalize_image,
    png_data_url,
)


def _encoded_image(format_name: str, size: tuple[int, int] = (12, 8)) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", size, (20, 80, 140)).save(output, format=format_name)
    return output.getvalue()


@pytest.mark.parametrize("format_name", ["PNG", "JPEG", "PPM"])
def test_normalize_supported_formats_to_png(format_name: str) -> None:
    normalized, width, height = normalize_image(_encoded_image(format_name))

    assert normalized.startswith(b"\x89PNG\r\n\x1a\n")
    assert (width, height) == (12, 8)
    with Image.open(io.BytesIO(normalized)) as image:
        assert image.format == "PNG"
        assert image.size == (12, 8)


def test_normalize_limits_long_edge() -> None:
    normalized, width, height = normalize_image(_encoded_image("PNG", (3000, 1200)))

    assert (width, height) == (2000, 800)
    with Image.open(io.BytesIO(normalized)) as image:
        assert image.size == (2000, 800)


def test_normalize_uses_first_animated_frame() -> None:
    output = io.BytesIO()
    frames = [Image.new("RGB", (3, 2), color) for color in ("red", "blue")]
    frames[0].save(output, format="GIF", save_all=True, append_images=frames[1:])

    normalized, width, height = normalize_image(output.getvalue())

    assert (width, height) == (3, 2)
    with Image.open(io.BytesIO(normalized)) as image:
        assert image.convert("RGB").getpixel((0, 0)) == (255, 0, 0)


def test_normalize_applies_exif_orientation() -> None:
    output = io.BytesIO()
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (3, 2), "yellow").save(output, format="JPEG", exif=exif)

    _normalized, width, height = normalize_image(output.getvalue())

    assert (width, height) == (2, 3)


def test_normalize_rejects_corrupt_image() -> None:
    with pytest.raises(ImageNormalizationError):
        normalize_image(b"\x89PNG\r\n\x1a\ncorrupt")


def test_normalize_rejects_decompression_bomb(monkeypatch) -> None:
    raw = _encoded_image("PNG", (20, 20))
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)

    with pytest.raises(ImageNormalizationError, match="safe pixel limit"):
        normalize_image(raw)


def test_normalize_enforces_five_megabyte_output_limit(monkeypatch) -> None:
    assert media.MAX_NORMALIZED_IMAGE_BYTES == 5 * 1024 * 1024
    monkeypatch.setattr(media, "MAX_NORMALIZED_IMAGE_BYTES", 10)

    with pytest.raises(ImageTooLarge):
        normalize_image(_encoded_image("PNG"))


def test_png_data_url() -> None:
    assert png_data_url(b"png") == "data:image/png;base64,cG5n"
