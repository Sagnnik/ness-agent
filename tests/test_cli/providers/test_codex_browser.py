from __future__ import annotations

import subprocess
from unittest.mock import Mock

from ness_cli.providers.codex import browser


def test_linux_browser_launch_is_detached(monkeypatch) -> None:
    monkeypatch.setattr(browser.sys, "platform", "linux")
    monkeypatch.setattr(browser, "_is_wsl", lambda: False)
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setattr(browser.shutil, "which", lambda _name: "/usr/bin/xdg-open")
    popen = Mock()
    monkeypatch.setattr(browser.subprocess, "Popen", popen)

    assert browser.open_auth_url("https://auth.openai.com/example") is True
    popen.assert_called_once_with(
        ["/usr/bin/xdg-open", "https://auth.openai.com/example"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        start_new_session=True,
    )


def test_headless_linux_returns_manual_fallback(monkeypatch) -> None:
    monkeypatch.setattr(browser.sys, "platform", "linux")
    monkeypatch.setattr(browser, "_is_wsl", lambda: False)
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    popen = Mock()
    monkeypatch.setattr(browser.subprocess, "Popen", popen)

    assert browser.open_auth_url("https://auth.openai.com/example") is False
    popen.assert_not_called()


def test_wsl_never_launches_a_browser(monkeypatch) -> None:
    monkeypatch.setattr(browser.sys, "platform", "linux")
    monkeypatch.setattr(browser, "_is_wsl", lambda: True)
    popen = Mock()
    monkeypatch.setattr(browser.subprocess, "Popen", popen)

    assert browser.open_auth_url("https://auth.openai.com/example") is False
    popen.assert_not_called()


def test_browser_rejects_non_http_urls(monkeypatch) -> None:
    popen = Mock()
    monkeypatch.setattr(browser.subprocess, "Popen", popen)

    assert browser.open_auth_url("file:///tmp/not-auth") is False
    popen.assert_not_called()
