from __future__ import annotations

import base64
import io
import warnings

from PIL import Image, ImageOps, UnidentifiedImageError

MAX_IMAGE_LONG_EDGE = 2000
MAX_NORMALIZED_IMAGE_BYTES = 5 * 1024 * 1024
PNG_MIME_TYPE = "image/png"


class ImageNormalizationError(ValueError):
    """Raised when Pillow cannot safely decode and normalize an image."""


class ImageNotRecognized(ImageNormalizationError):
    """Raised when the input is not an image format known to Pillow."""


class ImageTooLarge(ImageNormalizationError):
    """Raised when the normalized PNG exceeds the output-size limit."""


def normalize_image(raw: bytes) -> tuple[bytes, int, int]:
    """Decode an image, normalize its first frame, and return PNG bytes and size."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as source:
                source.seek(0)  # move the stream pointer to 0 (used for gifs)
                source.load()  # load the image into memory as .open only does lazy loading
                image = ImageOps.exif_transpose(source)

                has_alpha = "A" in image.getbands() or "transparency" in image.info
                image = image.convert("RGBA" if has_alpha else "RGB")
                width, height = image.size
                long_edge = max(width, height)
                if long_edge > MAX_IMAGE_LONG_EDGE:
                    scale = MAX_IMAGE_LONG_EDGE / long_edge
                    image = image.resize(
                        (
                            max(1, round(width * scale)),
                            max(1, round(height * scale)),
                        ),
                        Image.Resampling.LANCZOS,
                    )

                width, height = image.size
                output = io.BytesIO()
                image.save(output, format="PNG")
    except UnidentifiedImageError as exc:
        raise ImageNotRecognized("input is not an image Pillow can decode") from exc
    except Image.DecompressionBombError as exc:
        raise ImageNormalizationError(f"image exceeds Pillow's safe pixel limit: {exc}") from exc
    except Image.DecompressionBombWarning as exc:
        raise ImageNormalizationError(f"image exceeds Pillow's safe pixel limit: {exc}") from exc
    except (OSError, SyntaxError, ValueError) as exc:
        raise ImageNormalizationError(f"invalid or corrupt image: {exc}") from exc

    normalized = output.getvalue()
    if len(normalized) > MAX_NORMALIZED_IMAGE_BYTES:
        raise ImageTooLarge(
            f"image is {len(normalized)} bytes after normalization; "
            f"max is {MAX_NORMALIZED_IMAGE_BYTES}"
        )
    return normalized, width, height


def png_data_url(data: bytes) -> str:
    """Build a base64 PNG data URL."""
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{PNG_MIME_TYPE};base64,{encoded}"
