"""CLI-owned access to the SDK thread store."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any
from pathlib import Path

from ness_agent import ThreadStore

from ness_cli.session.events import (
    DurableEvent,
    RollbackCheckpoint,
    UserTurn,
    compact_event,
    parse_events,
    user_event,
)


class SessionRepository:
    """Wrap the session's SDK persistence view, including its autosave policy."""

    def __init__(self, store: ThreadStore) -> None:
        self._store = store

    @property
    def auto_save(self) -> bool:
        return bool(self._store.auto_save)

    @auto_save.setter
    def auto_save(self, value: bool) -> None:
        self._store.auto_save = bool(value)

    def exists(self, thread_id: str) -> bool:
        return self._store.thread_exists(thread_id)

    def events(self, thread_id: str) -> list[DurableEvent]:
        return parse_events(self._store.load_thread_events(thread_id))

    def raw_events(self, thread_id: str) -> list[dict[str, Any]]:
        return [event.as_dict() for event in self.events(thread_id)]

    def events_since(
        self,
        thread_id: str,
        start_seq: int,
    ) -> list[DurableEvent]:
        return parse_events(self._store.load_thread_events_since(thread_id, start_seq))

    def append(
        self,
        thread_id: str,
        event: Mapping[str, Any],
    ) -> int | None:
        return self._store.append_event(thread_id, dict(event))

    def append_user(
        self,
        thread_id: str,
        content: str,
        *,
        images: Sequence[str] = (),
    ) -> int | None:
        return self.append(
            thread_id,
            user_event(content, images=images),
        )

    def append_compaction(
        self,
        thread_id: str,
        data: Mapping[str, Any],
    ) -> int | None:
        return self.append(thread_id, compact_event(data))

    def save_checkpoint(
        self,
        thread_id: str,
        user_seq: int,
        *,
        git_hash: str | None,
        memory_snapshot: str,
    ) -> None:
        self._store.save_checkpoint(
            thread_id,
            user_seq,
            git_hash,
            memory_snapshot,
        )

    def checkpoint(
        self,
        thread_id: str,
        user_seq: int,
    ) -> RollbackCheckpoint | None:
        row = self._store.get_checkpoint(thread_id, user_seq)
        if row is None:
            return None
        paths: tuple[str, ...] = ()
        raw_paths = row.get("modified_paths")
        if raw_paths:
            try:
                decoded = json.loads(str(raw_paths))
            except (TypeError, ValueError):
                decoded = []
            if isinstance(decoded, list):
                paths = tuple(str(path) for path in decoded if str(path))
        return RollbackCheckpoint(
            thread_id=str(row.get("thread_id") or thread_id),
            user_seq=int(row.get("user_seq", user_seq)),
            git_hash=(str(row["git_hash"]) if row.get("git_hash") else None),
            modified_paths=paths,
            memory_snapshot=str(row.get("mem_snapshot") or ""),
        )

    def add_modified_path(
        self,
        thread_id: str,
        user_seq: int,
        path: str,
    ) -> None:
        self._store.add_modified_path(thread_id, user_seq, path)

    def truncate(self, thread_id: str, user_seq: int) -> None:
        self._store.truncate_after(thread_id, user_seq)

    def user_turns(self, thread_id: str) -> tuple[UserTurn, ...]:
        return tuple(
            UserTurn(
                seq=int(row.get("seq", -1)),
                content=str(row.get("content") or ""),
            )
            for row in self._store.list_user_turns(thread_id)
        )

    def list_threads(self, limit: int = 100) -> list[dict[str, Any]]:
        return self._store.list_threads(limit)

    def rename(self, thread_id: str, name: str) -> bool:
        return self._store.set_thread_name(thread_id, name)

    def first_user_message(self, thread_id: str) -> str | None:
        return self._store.first_user_message(thread_id)

    def archive(self, thread_id: str) -> str:
        return self._store.archive_thread(thread_id)

    def subagents(self, thread_id: str) -> list[dict[str, Any]]:
        return self._store.list_subagents(thread_id)

    def copy_prefix(
        self,
        source_thread_id: str,
        target_thread_id: str,
        before_seq: int,
    ) -> list[dict[str, Any]]:
        return self._store.copy_thread_prefix(
            source_thread_id,
            target_thread_id,
            before_seq,
        )

    def export_html(
        self,
        thread_id: str,
        *,
        project_root: Path,
        destination: Path,
    ):
        from ness_cli.session.export import export_thread_html

        return export_thread_html(
            thread_store=self._store,
            thread_id=thread_id,
            project_root=project_root,
            destination=destination,
        )
