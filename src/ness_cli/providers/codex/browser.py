from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse


def _is_wsl() -> bool:
    if os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"):
        return True
    try:
        release = Path("/proc/sys/kernel/osrelease").read_text(encoding="utf-8")
        return "microsoft" in release.lower()
    except OSError:
        return False


def open_auth_url(url: str) -> bool:
    """Open an auth URL without attaching browser output to the Ness TUI.

    Desktop browser launchers inherit stdout/stderr by default. Chromium then
    writes DBus, dconf, and GPU diagnostics into prompt_toolkit's alternate
    screen, corrupting the UI. Both streams and stdin are therefore detached.
    The URL remains visible in the transcript when no launcher is available.
    """
    if urlparse(url).scheme not in {"http", "https"}:
        return False

    if sys.platform == "win32":
        try:
            os.startfile(url)  # type: ignore[attr-defined]
        except OSError:
            return False
        return True

    if _is_wsl():
        # Neither WSLg's xdg-open nor explorer.exe is reliable here: xdg-open
        # can launch a broken Linux Chrome session, while explorer.exe can
        # interpret the URL as a filesystem target. Keep the URL in the Ness
        # transcript and let the user open it manually.
        return False
    elif sys.platform == "darwin":
        command = ["open", url]
    else:
        # Avoid launching a noisy/failed browser in a genuinely headless or
        # SSH-only Linux session. The printed URL and device flow still work.
        if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            return False
        launcher = shutil.which("xdg-open")
        if launcher is None:
            return False
        command = [launcher, url]

    try:
        subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            start_new_session=True,
        )
    except OSError:
        return False
    return True
