"""Clipboard image grab, resize, and data-URL helpers for the CLI.

Triggered by the Ctrl+G keybinding in the TUI: reads the OS image clipboard,
resizes the image to a max 2000px long edge (aspect preserved), re-encodes as
PNG, rejects if the result exceeds 5 MB, and writes it to the platform cache
directory. The data URL (base64-encoded) is returned alongside the scratch
path so the caller can persist it directly into the events DB without
re-reading the file.
"""

from __future__ import annotations

import io
import os
import re
import shutil
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path

from ness_agent.media import ImageTooLarge, normalize_image, png_data_url


class NoClipboardImage(Exception):
    """Raised when the clipboard has no image or the grab tool is missing."""


def image_cache_dir() -> Path:
    """Return the per-user cache directory for pasted images (created on call)."""
    from platformdirs import user_cache_dir

    cache = Path(user_cache_dir("ness-agent")) / "images"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def grab_clipboard_image() -> bytes:
    """Read raw image bytes from the OS clipboard.

    WSL2: ``powershell.exe`` reads the Windows clipboard (the Linux/Wayland
    clipboard under WSLg does not mirror Windows snipping-tool images).
    Linux: ``wl-paste`` (Wayland) or ``xclip`` (X11).
    macOS / Windows: ``PIL.ImageGrab.grabclipboard()`` re-encoded to PNG.

    Raises :class:`NoClipboardImage` when the clipboard is empty or the
    required tool/library is unavailable.
    """
    if _is_wsl():
        return _grab_wsl()
    if sys.platform.startswith("linux"):
        return _grab_linux()
    if sys.platform == "darwin":
        return _grab_imagegrab()
    if sys.platform == "win32":
        return _grab_imagegrab()
    raise NoClipboardImage(f"unsupported platform: {sys.platform}")


def _is_wsl() -> bool:
    """Detect WSL2 via ``/proc/version`` (kernel string mentions microsoft)."""
    try:
        with open("/proc/version", "r") as f:
            return "microsoft" in f.read().lower()
    except OSError:
        return False


def _grab_wsl() -> bytes:
    """Grab the Windows clipboard image via ``powershell.exe``.

    PowerShell saves the clipboard image to a temp PNG in the Windows temp
    dir; the path is converted to a WSL path via ``wslpath`` and read back.
    """
    if shutil.which("powershell.exe") is None:
        raise NoClipboardImage("powershell.exe not found")
    ps_script = (
        "Add-Type -AssemblyName System.Windows.Forms;"
        "$img = [System.Windows.Forms.Clipboard]::GetImage();"
        "if ($img -eq $null) { exit 1 }"
        "$tmp = [System.IO.Path]::Combine("
        "[System.IO.Path]::GetTempPath(),"
        "'lh_clip_' + [System.Guid]::NewGuid().ToString() + '.png');"
        "$img.Save($tmp);"
        "Write-Output $tmp"
    )
    try:
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", ps_script],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as exc:
        raise NoClipboardImage(f"powershell.exe failed: {exc}") from exc
    if result.returncode != 0 or not result.stdout.strip():
        raise NoClipboardImage("clipboard has no image")
    win_path = result.stdout.strip()
    wsl_path = _win_to_wsl_path(win_path)
    try:
        data = Path(wsl_path).read_bytes()
    except OSError as exc:
        raise NoClipboardImage(f"failed to read {wsl_path}: {exc}") from exc
    finally:
        try:
            Path(wsl_path).unlink(missing_ok=True)
        except OSError:
            pass
    return data


def _win_to_wsl_path(win_path: str) -> str:
    """Convert a Windows path (``C:\\Users\\...``) to a WSL path (``/mnt/c/...``)."""
    if shutil.which("wslpath") is None:
        raise NoClipboardImage("wslpath not found")
    try:
        result = subprocess.run(
            ["wslpath", win_path], capture_output=True, text=True, timeout=5
        )
    except Exception as exc:
        raise NoClipboardImage(f"wslpath failed: {exc}") from exc
    if result.returncode != 0 or not result.stdout.strip():
        raise NoClipboardImage(f"wslpath conversion failed for {win_path}")
    return result.stdout.strip()


def _grab_linux() -> bytes:
    if os.environ.get("WAYLAND_DISPLAY"):
        return _run_paste_tool(["wl-paste", "--type", "image/png"])
    return _run_paste_tool(["xclip", "-selection", "clipboard", "-t", "image/png", "-o"])


def _run_paste_tool(cmd: list[str]) -> bytes:
    exe = shutil.which(cmd[0])
    if exe is None:
        raise NoClipboardImage(f"{cmd[0]} not installed")
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=5)
    except Exception as exc:
        raise NoClipboardImage(f"{cmd[0]} failed: {exc}") from exc
    if result.returncode != 0 or not result.stdout:
        raise NoClipboardImage("clipboard has no image")
    return result.stdout


def _grab_imagegrab() -> bytes:
    from PIL import ImageGrab

    try:
        img = ImageGrab.grabclipboard()
    except Exception as exc:
        raise NoClipboardImage(f"ImageGrab failed: {exc}") from exc
    if img is None:
        raise NoClipboardImage("clipboard has no image")
    if isinstance(img, list):
        if not img:
            raise NoClipboardImage("clipboard has no image")
        img = img[0]
    buf = io.BytesIO()
    img = img.convert("RGB")
    img.save(buf, format="PNG")
    return buf.getvalue()


def process_image(raw: bytes) -> bytes:
    """Resize (long edge <= 2000px) and re-encode to PNG. Reject if > 5 MB."""
    processed, _width, _height = normalize_image(raw)
    return processed


def save_clipboard_image() -> tuple[Path, str] | None:
    """Grab, process, and save the clipboard image.

    Returns ``(scratch_path, data_url)`` or ``None`` when the clipboard is
    empty (a benign condition the caller renders as a warning).
    """
    try:
        raw = grab_clipboard_image()
    except NoClipboardImage:
        return None
    processed = process_image(raw)
    stamp = re.sub(r"[-:.TZ]", "", datetime.utcnow().isoformat(timespec="seconds"))
    name = f"{stamp}-{uuid.uuid4().hex[:8]}.png"
    path = image_cache_dir() / name
    path.write_bytes(processed)
    data_url = _bytes_to_data_url(processed)
    return path, data_url


def _bytes_to_data_url(data: bytes) -> str:
    return png_data_url(data)
