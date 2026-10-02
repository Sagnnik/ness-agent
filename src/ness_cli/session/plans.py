"""Capture and save plan-mode output."""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

_INTERRUPTED_SUFFIX = " … [interrupted]"


def final_plan_text(texts: list[str]) -> str | None:
    cleaned = [text.strip() for text in texts if text.strip()]
    return cleaned[-1] if cleaned else None


class PlanStore:
    def __init__(self, plans_dir: Path) -> None:
        self._plans_dir = plans_dir

    def save(self, thread_id: str, text: str) -> Path:
        self._plans_dir.mkdir(parents=True, exist_ok=True)
        stamp = re.sub(
            r"[-:.TZ]",
            "",
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
        ).replace("+0000", "")
        path = self._plans_dir / f"{stamp}-{thread_id}.md"
        path.write_text(text.strip() + "\n", encoding="utf-8")
        return path


class PlanCapture:
    """Collect SDK plan hooks for one turn and save one final plan."""

    def __init__(
        self,
        store: PlanStore,
        *,
        thread_id: Callable[[], str],
        in_plan_mode: Callable[[], bool],
    ) -> None:
        self._store = store
        self._thread_id = thread_id
        self._in_plan_mode = in_plan_mode
        self._texts: list[str] = []

    def begin_turn(self) -> None:
        self._texts.clear()

    def on_plan_turn(self, text: str) -> None:
        cleaned = str(text or "").strip()
        if cleaned:
            self._texts.append(cleaned)

    def on_interrupt(self, partial_text: str) -> str:
        cleaned = str(partial_text or "").strip()
        if self._in_plan_mode() and cleaned:
            self._texts.append(cleaned + _INTERRUPTED_SUFFIX)
        return partial_text

    def finish_turn(self) -> Path | None:
        text = final_plan_text(self._texts)
        self._texts.clear()
        if text is None:
            return None
        return self._store.save(self._thread_id(), text)
