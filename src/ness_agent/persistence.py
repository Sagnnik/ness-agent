from __future__ import annotations
import json, sqlite3, threading
from contextlib import contextmanager
from copy import copy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

SESSION_THREAD_PREFIX = "session-"
SUBAGENT_THREAD_PREFIX = "subagent-"
_FULL_TREE_SENTINEL = "*"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS threads (
    thread_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    archived_at TEXT,
    turn_count INTEGER NOT NULL DEFAULT 0,
    model TEXT NOT NULL DEFAULT '',
    name TEXT NOT NULL DEFAULT '',
    summary TEXT NOT NULL DEFAULT '',
    total_cost_usd REAL NOT NULL DEFAULT 0.0,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    cached_input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    fork_root_id TEXT,
    fork_parent_id TEXT,
    forked_from_seq INTEGER
);

CREATE TABLE IF NOT EXISTS events (
    thread_id TEXT NOT NULL REFERENCES threads(thread_id) ON DELETE CASCADE,
    seq INTEGER NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY (thread_id, seq)
);

CREATE TABLE IF NOT EXISTS subagents (
    subagent_thread_id TEXT PRIMARY KEY,
    parent_thread_id TEXT NOT NULL REFERENCES threads(thread_id),
    agent_name TEXT NOT NULL,
    label TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'running',
    started_at TEXT NOT NULL,
    completed_at TEXT,
    duration_ms INTEGER NOT NULL DEFAULT 0,
    output TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS checkpoints (
    thread_id TEXT NOT NULL REFERENCES threads(thread_id) ON DELETE CASCADE,
    user_seq INTEGER NOT NULL,
    git_hash TEXT,
    modified_paths TEXT NOT NULL DEFAULT '',
    mem_snapshot TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    PRIMARY KEY (thread_id, user_seq)
);

CREATE INDEX IF NOT EXISTS idx_threads_updated ON threads(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_threads_fork_root ON threads(fork_root_id);
CREATE INDEX IF NOT EXISTS idx_subagents_parent ON subagents(parent_thread_id, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_checkpoints_thread ON checkpoints(thread_id, user_seq ASC);
"""

class ThreadStore:
    def __init__(
        self,
        threads_dir: Path | None = None,
        *,
        auto_save: bool = True,
        default_model: str = "",
    ) -> None:
        """Persistent SQLite-backed store for conversation threads, events, subagents, and rollback checkpoints.

        Args:
            threads_dir: Directory for the threads.db file. Defaults to ``.ness/threads``.
            auto_save: When False, all write operations silently no-op.
            default_model: Model name written to new thread rows when no usage event has set it yet.
        """
        self.threads_dir = threads_dir or Path(".ness/threads")
        self.threads_db = self.threads_dir / "threads.db"
        self.auto_save = auto_save
        self.default_model = default_model or ""
        self._write_lock = threading.Lock()
        self._validate_existing_schema()

    def fork_for_session(self, *, auto_save: bool | None = None) -> ThreadStore:
        """Share the database and writer lock with an independent autosave flag.

        Keep this view's policy unless the session supplies an explicit value.
        Forking opens no connection and does not create or alter the database.
        """
        fork = copy(self)
        if auto_save is not None:
            fork.auto_save = bool(auto_save)
        return fork

    def _validate_existing_schema(self) -> None:
        """Reject pre-name databases without changing or deleting them."""
        if not self.threads_db.exists():
            return
        with sqlite3.connect(self.threads_db) as conn:
            columns = {
                str(row[1])
                for row in conn.execute("PRAGMA table_info(threads)").fetchall()
            }
        if columns and "name" not in columns:
            raise RuntimeError(
                "Incompatible thread database schema. Back up or remove "
                f"{self.threads_db} so Ness can create the new schema; "
                "automatic SQLite migrations are not supported."
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self.threads_dir.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.threads_db, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=5000")
        try:
            yield conn
        finally:
            conn.close()

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        conn.executescript(_SCHEMA)

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def append_event(self, thread_id, event):
        """Append an event to the durable log. Returns the assigned seq, or None when skipped.

        Skipped when autosave is off or when the target is a subagent thread (those
        only roll up usage). Returning the seq lets callers (e.g. the rollback
        checkpoint hook) key side tables off the exact user-message event.
        """
        if not self.auto_save: 
            return None
        
        if thread_id.startswith(SUBAGENT_THREAD_PREFIX):
            if event.get("kind") == "usage": 
                self._rollup_subagent_usage(thread_id, event)
            return None
        
        payload = dict(event)
        payload.setdefault("t", self._now())
        
        with self._write_lock:
            with self._connect() as conn:
                self._ensure_schema(conn)
                now = self._now()
                conn.execute(
                    """
                    INSERT INTO threads (
                        thread_id, started_at, updated_at, model, fork_root_id
                    )
                    VALUES (?, ?, ?, ?, ?) ON CONFLICT(thread_id) DO NOTHING
                    """,
                    (thread_id, now, now, self.default_model, thread_id),
                )
                
                seq = conn.execute(
                    "SELECT COALESCE(MAX(seq), -1) + 1 FROM events WHERE thread_id = ?",
                    (thread_id,),
                ).fetchone()[0]
                
                conn.execute(
                    "INSERT INTO events (thread_id, seq, payload) VALUES (?, ?, ?)",
                    (thread_id, seq, json.dumps(payload, ensure_ascii=False)),
                )
                
                self._apply_event_to_thread(conn, thread_id, payload, now)
                conn.commit()
                return seq

    def append_compaction_checkpoint(
        self,
        thread_id: str,
        event: dict[str, Any],
        *,
        active_turn: bool,
    ) -> int | None:
        """Atomically persist a summary, active suffix, and source boundary."""
        if not self.auto_save or thread_id.startswith(SUBAGENT_THREAD_PREFIX):
            return None
        payload = dict(event)
        payload["kind"] = "compaction_llm"
        payload.setdefault("t", self._now())
        with self._write_lock:
            with self._connect() as conn:
                self._ensure_schema(conn)
                now = self._now()
                conn.execute(
                    """
                    INSERT INTO threads (thread_id, started_at, updated_at, model, fork_root_id)
                    VALUES (?, ?, ?, ?, ?) ON CONFLICT(thread_id) DO NOTHING
                    """,
                    (thread_id, now, now, self.default_model, thread_id),
                )
                seq = int(conn.execute(
                    "SELECT COALESCE(MAX(seq), -1) + 1 FROM events WHERE thread_id = ?",
                    (thread_id,),
                ).fetchone()[0])
                if active_turn:
                    row = conn.execute(
                        """
                        SELECT MAX(seq) FROM events
                        WHERE thread_id = ? AND json_extract(payload, '$.kind') = 'user'
                        """,
                        (thread_id,),
                    ).fetchone()
                    active_user_seq = int(row[0]) if row and row[0] is not None else None
                    # The checkpoint carries the complete active semantic
                    # suffix.  Its boundary can therefore consume every raw
                    # event written before this atomic snapshot.  This also
                    # makes SDK replay safe when no separate user event was
                    # persisted (or the latest user row belongs to an older
                    # CLI-driven turn).
                    source_seq = seq - 1
                else:
                    active_user_seq = None
                    source_seq = seq - 1
                payload["source_event_seq"] = source_seq
                payload["active_user_seq"] = active_user_seq
                conn.execute(
                    "INSERT INTO events (thread_id, seq, payload) VALUES (?, ?, ?)",
                    (thread_id, seq, json.dumps(payload, ensure_ascii=False)),
                )
                self._apply_event_to_thread(conn, thread_id, payload, now)
                conn.commit()
                return seq

    def list_threads(self, n: int = 10) -> list[dict[str, Any]]:
        """Return the *n* most recently updated session threads."""
        with self._connect() as conn:
            self._ensure_schema(conn)
            rows = conn.execute(
                """
                SELECT
                    t.thread_id, t.started_at, t.updated_at, t.archived_at,
                    t.turn_count, t.model, t.name, t.summary,
                    t.total_cost_usd, t.input_tokens, t.cached_input_tokens,
                    t.output_tokens, t.fork_root_id, t.fork_parent_id,
                    t.forked_from_seq,
                    MAX(0, (
                        SELECT COUNT(*) - 1 FROM threads f
                        WHERE COALESCE(f.fork_root_id, f.thread_id)
                            = COALESCE(t.fork_root_id, t.thread_id)
                    )) AS fork_count,
                    CASE
                        WHEN t.fork_parent_id IS NULL THEN 0
                        ELSE (
                            SELECT COUNT(*) FROM threads f
                            WHERE COALESCE(f.fork_root_id, f.thread_id)
                                = COALESCE(t.fork_root_id, t.thread_id)
                              AND f.fork_parent_id IS NOT NULL
                              AND (
                                    f.started_at < t.started_at
                                    OR (
                                        f.started_at = t.started_at
                                        AND f.thread_id <= t.thread_id
                                    )
                              )
                        )
                    END AS fork_index
                FROM threads t
                WHERE t.thread_id LIKE ?
                ORDER BY t.updated_at DESC
                LIMIT ?
                """,
                (f"{SESSION_THREAD_PREFIX}%", n),
            ).fetchall()

        return [
            {
                "thread_id": row["thread_id"],
                "started_at": row["started_at"],
                "updated_at": row["updated_at"],
                "turn_count": int(row["turn_count"]),
                "model": row["model"],
                "name": row["name"],
                "summary": row["summary"],
                "total_cost_usd": float(row["total_cost_usd"]),
                "input_tokens": int(row["input_tokens"]),
                "cached_input_tokens": int(row["cached_input_tokens"]),
                "output_tokens": int(row["output_tokens"]),
                "fork_root_id": row["fork_root_id"] or row["thread_id"],
                "fork_parent_id": row["fork_parent_id"],
                "forked_from_seq": row["forked_from_seq"],
                "fork_count": int(row["fork_count"]),
                "fork_index": int(row["fork_index"]),
                **({"archived_at": row["archived_at"]} if row["archived_at"] else {}),
            }
            for row in rows
        ]

    def thread_exists(self, thread_id: str) -> bool:
        """Return whether a persisted metadata row exists for *thread_id*."""
        if not self.threads_db.exists():
            return False
        with self._connect() as conn:
            self._ensure_schema(conn)
            row = conn.execute(
                "SELECT 1 FROM threads WHERE thread_id = ?",
                (thread_id,),
            ).fetchone()
        return row is not None

    def set_thread_name(self, thread_id: str, name: str) -> bool:
        """Set the explicit display name for a thread.

        The metadata row is created when necessary so callers can name a new
        session before its first user turn. Returns ``False`` only when thread
        autosave is disabled.
        """
        if not self.auto_save:
            return False
        cleaned = " ".join(str(name).split()).strip()
        if not cleaned:
            raise ValueError("Session name cannot be empty.")
        if len(cleaned) > 80:
            raise ValueError("Session name must be 80 characters or fewer.")

        with self._write_lock:
            with self._connect() as conn:
                self._ensure_schema(conn)
                now = self._now()
                conn.execute(
                    """
                    INSERT INTO threads (
                        thread_id, started_at, updated_at, model, name, fork_root_id
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(thread_id) DO UPDATE SET
                        name = excluded.name,
                        updated_at = excluded.updated_at
                    """,
                    (
                        thread_id,
                        now,
                        now,
                        self.default_model,
                        cleaned,
                        thread_id,
                    ),
                )
                conn.commit()
        return True

    def load_thread_events(self, thread_id: str) -> list[dict[str, Any]]:
        """Return every stored event for *thread_id* in sequence order, or an empty list.

        Returns the raw deserialized payload dicts — caller must interpret the ``kind`` key.
        """
        if not self.threads_db.exists():
            return []

        with self._connect() as conn:
            self._ensure_schema(conn)
            rows = conn.execute(
                "SELECT payload FROM events WHERE thread_id = ? ORDER BY seq",
                (thread_id,),
            ).fetchall()

        return [json.loads(row[0]) for row in rows]

    def load_thread_events_since(
        self, thread_id: str, start_seq: int
    ) -> list[dict[str, Any]]:
        """Return events with ``seq >= start_seq``, each payload tagged with ``seq``.

        Used by ``/goal`` to feed the judge the conversation from the goal-start
        boundary without including earlier thread history.
        """
        if not self.threads_db.exists():
            return []

        with self._connect() as conn:
            self._ensure_schema(conn)
            rows = conn.execute(
                """
                SELECT seq, payload FROM events
                WHERE thread_id = ? AND seq >= ?
                ORDER BY seq
                """,
                (thread_id, int(start_seq)),
            ).fetchall()

        events: list[dict[str, Any]] = []
        for row in rows:
            payload = json.loads(row[1])
            payload["seq"] = int(row[0])
            events.append(payload)
        return events

    def copy_thread_prefix(
        self,
        source_thread_id: str,
        target_thread_id: str,
        before_seq: int,
    ) -> list[dict[str, Any]]:
        """Create a fork containing events strictly before one user event."""
        if not self.auto_save:
            raise ValueError("Thread autosave must be enabled to fork")
        with self._write_lock:
            with self._connect() as conn:
                self._ensure_schema(conn)
                boundary = conn.execute(
                    "SELECT payload FROM events WHERE thread_id = ? AND seq = ?",
                    (source_thread_id, before_seq),
                ).fetchone()
                if boundary is None or json.loads(boundary[0]).get("kind") != "user":
                    raise ValueError(f"Fork target seq {before_seq} is not a user message")
                source = conn.execute(
                    """
                    SELECT model, fork_root_id FROM threads WHERE thread_id = ?
                    """,
                    (source_thread_id,),
                ).fetchone()
                if source is None:
                    raise ValueError(f"Unknown source thread: {source_thread_id}")
                now = self._now()
                root_id = source["fork_root_id"] or source_thread_id
                conn.execute(
                    """
                    INSERT INTO threads (
                        thread_id, started_at, updated_at, model,
                        fork_root_id, fork_parent_id, forked_from_seq
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        target_thread_id,
                        now,
                        now,
                        source["model"] or self.default_model,
                        root_id,
                        source_thread_id,
                        before_seq,
                    ),
                )
                event_rows = conn.execute(
                    """
                    SELECT seq, payload FROM events
                    WHERE thread_id = ? AND seq < ? ORDER BY seq
                    """,
                    (source_thread_id, before_seq),
                ).fetchall()
                copied: list[dict[str, Any]] = []
                for row in event_rows:
                    payload = json.loads(row["payload"])
                    if payload.get("kind") == "usage":
                        payload["inherited"] = True
                    copied.append(payload)
                    conn.execute(
                        "INSERT INTO events (thread_id, seq, payload) VALUES (?, ?, ?)",
                        (
                            target_thread_id,
                            int(row["seq"]),
                            json.dumps(payload, ensure_ascii=False),
                        ),
                    )
                    self._apply_event_to_thread(conn, target_thread_id, payload, now)
                conn.execute(
                    """
                    INSERT INTO checkpoints (
                        thread_id, user_seq, git_hash, modified_paths,
                        mem_snapshot, created_at
                    )
                    SELECT ?, user_seq, git_hash, modified_paths,
                           mem_snapshot, created_at
                    FROM checkpoints
                    WHERE thread_id = ? AND user_seq < ?
                    """,
                    (target_thread_id, source_thread_id, before_seq),
                )
                conn.commit()
                return copied

    def archive_thread(self, thread_id: str) -> str:
        """Mark a thread archived in the threads table.

        Idempotent: skips when the thread was already archived with no new events since.
        Headline is derived from the first user message.
        """
        if not self.auto_save:
            return "Thread autosave disabled"

        with self._write_lock:
            with self._connect() as conn:
                self._ensure_schema(conn)
                row = conn.execute(
                    "SELECT updated_at, archived_at, name FROM threads WHERE thread_id = ?",
                    (thread_id,),
                ).fetchone()

                if row is None:
                    return f"No thread to archive: {thread_id}"

                if row["archived_at"] and str(row["archived_at"]) >= str(row["updated_at"]):
                    return f"Thread already archived: {thread_id}"

                event_rows = conn.execute(
                    "SELECT payload FROM events WHERE thread_id = ? ORDER BY seq",
                    (thread_id,),
                ).fetchall()
                if not event_rows and not str(row["name"] or "").strip():
                    return f"No events to archive: {thread_id}"

                events = [json.loads(item[0]) for item in event_rows]
                headline = ""
                for event in events:
                    if event.get("kind") != "user":
                        continue
                    content = event.get("content")
                    text = content if isinstance(content, str) else str(content)
                    headline = " ".join(text.split())[:200]
                    break
                raw = headline or ""
                cleaned = " ".join(raw.split()).strip().strip('"').strip(".")
                summary = cleaned[:80].strip()
                archived_at = self._now()
                conn.execute(
                    "UPDATE threads SET summary = ?, archived_at = ?, updated_at = ? WHERE thread_id = ?",
                    (summary, archived_at, archived_at, thread_id),
                )
                conn.commit()

        return f"Archived thread {thread_id}"

    def _rollup_subagent_usage(self, subagent_thread_id: str, event: dict[str, Any]) -> None:
        with self._write_lock:
            with self._connect() as conn:
                self._ensure_schema(conn)
                row = conn.execute(
                    "SELECT parent_thread_id FROM subagents WHERE subagent_thread_id = ?",
                    (subagent_thread_id,),
                ).fetchone()
                if row is None:
                    return
                self._apply_event_to_thread(conn, row[0], event, self._now())
                conn.commit()

    def register_subagent(
        self,
        parent_thread_id: str,
        subagent_thread_id: str,
        *,
        agent_name: str,
        label: str = "",
    ) -> None:
        """Insert a subagent row linked to *parent_thread_id* with status ``running``.

        Also ensures a parent thread row exists (idempotent upsert).
        """
        if not self.auto_save:
            return

        with self._write_lock:
            with self._connect() as conn:
                self._ensure_schema(conn)
                now = self._now()
                conn.execute(
                    """
                    INSERT INTO threads (
                        thread_id, started_at, updated_at, model, fork_root_id
                    )
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(thread_id) DO NOTHING
                    """,
                    (
                        parent_thread_id,
                        now,
                        now,
                        self.default_model,
                        parent_thread_id,
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO subagents (
                        subagent_thread_id, parent_thread_id, agent_name, label,
                        status, started_at
                    ) VALUES (?, ?, ?, ?, 'running', ?)
                    """,
                    (subagent_thread_id, parent_thread_id, agent_name, label, now),
                )
                conn.commit()

    def complete_subagent(
        self,
        subagent_thread_id: str,
        *,
        status: str,
        output: str = "",
        duration_ms: int = 0,
    ) -> None:
        """Mark a subagent run as finished with *status*, *output*, and *duration_ms*."""
        if not self.auto_save:
            return

        with self._write_lock:
            with self._connect() as conn:
                self._ensure_schema(conn)
                completed_at = self._now()
                conn.execute(
                    """
                    UPDATE subagents
                    SET status = ?, completed_at = ?, duration_ms = ?, output = ?
                    WHERE subagent_thread_id = ?
                    """,
                    (status, completed_at, duration_ms, output, subagent_thread_id),
                )
                conn.commit()

    def list_subagents(self, parent_thread_id: str) -> list[dict[str, Any]]:
        """Return all subagent rows for *parent_thread_id*, oldest-first."""
        with self._connect() as conn:
            self._ensure_schema(conn)
            rows = conn.execute(
                """
                SELECT
                    subagent_thread_id, parent_thread_id, agent_name, label,
                    status, started_at, completed_at, duration_ms, output
                FROM subagents
                WHERE parent_thread_id = ?
                ORDER BY started_at ASC
                """,
                (parent_thread_id,),
            ).fetchall()

        return [
            {
                "subagent_thread_id": row[0],
                "parent_thread_id": row[1],
                "agent_name": row[2],
                "label": row[3],
                "status": row[4],
                "started_at": row[5],
                "completed_at": row[6],
                "duration_ms": int(row[7]),
                "output": row[8],
            }
            for row in rows
        ]

    def save_checkpoint(
        self,
        thread_id: str,
        user_seq: int,
        git_hash: str | None,
        mem_snapshot: str = "",
    ) -> None:
        """Persist a per-user-turn rollback checkpoint row.

        Called right after the user message is appended (so ``user_seq`` is known).
        Overwrites any existing row for the same (thread_id, user_seq) so a re-run
        of the same turn replaces the stale snapshot instead of stranding two.
        """
        if not self.auto_save:
            return

        with self._write_lock:
            with self._connect() as conn:
                self._ensure_schema(conn)
                conn.execute(
                    """
                    INSERT INTO checkpoints (thread_id, user_seq, git_hash, modified_paths, mem_snapshot, created_at)
                    VALUES (?, ?, ?, '', ?, ?)
                    ON CONFLICT(thread_id, user_seq) DO UPDATE SET
                        git_hash = excluded.git_hash,
                        mem_snapshot = excluded.mem_snapshot,
                        modified_paths = '',
                        created_at = excluded.created_at
                    """,
                    (thread_id, user_seq, git_hash, mem_snapshot, self._now()),
                )
                conn.commit()

    def add_modified_path(self, thread_id: str, user_seq: int, path: str) -> None:
        """Record a filesystem path the agent mutated during one user turn.

        Called from the tools node for destructive fs/shell calls. Empty path or
        the ``"*"`` sentinel marks the turn as full-tree restore (used for shell
        commands where mutated paths cannot be enumerated). The set is
        expanded idempotently; once ``*`` is set, per-path entries are dropped.
        """
        if not self.auto_save:
            return
        if not path:
            return

        with self._write_lock:
            with self._connect() as conn:
                self._ensure_schema(conn)
                row = conn.execute(
                    "SELECT modified_paths FROM checkpoints WHERE thread_id = ? AND user_seq = ?",
                    (thread_id, user_seq),
                ).fetchone()
                if row is None:
                    return
                raw = row[0] or ""
                paths: set[str] = set()
                if raw:
                    try:
                        paths = set(json.loads(raw))
                    except ValueError:
                        paths = set()
                # Once the "*" full-tree sentinel is set, no further per-path
                # recording can refine it; adding "*" again is also a no-op.
                if _FULL_TREE_SENTINEL in paths:
                    return
                if path == _FULL_TREE_SENTINEL:
                    paths = {_FULL_TREE_SENTINEL}
                else:
                    paths.add(path)
                conn.execute(
                    "UPDATE checkpoints SET modified_paths = ? WHERE thread_id = ? AND user_seq = ?",
                    (json.dumps(sorted(paths)), thread_id, user_seq),
                )
                conn.commit()

    def get_checkpoint(self, thread_id: str, user_seq: int) -> dict[str, Any] | None:
        """Return the rollback checkpoint for *(thread_id, user_seq)*, or None if absent."""
        with self._connect() as conn:
            self._ensure_schema(conn)
            row = conn.execute(
                """
                SELECT thread_id, user_seq, git_hash, modified_paths, mem_snapshot, created_at
                FROM checkpoints WHERE thread_id = ? AND user_seq = ?
                """,
                (thread_id, user_seq),
            ).fetchone()
        if row is None:
            return None
        return {
            "thread_id": row[0],
            "user_seq": int(row[1]),
            "git_hash": row[2],
            "modified_paths": row[3] or "",
            "mem_snapshot": row[4] or "",
            "created_at": row[5],
        }

    def first_user_message(self, thread_id: str) -> str | None:
        """Return the headline (truncated first user message) for a thread.

        Mirrors the summary derivation in archive_thread so the /threads table
        can show a useful label for threads that have not been archived yet.
        """
        if not self.threads_db.exists():
            return None
        with self._connect() as conn:
            self._ensure_schema(conn)
            row = conn.execute(
                """
                SELECT payload FROM events
                WHERE thread_id = ? AND json_extract(payload, '$.kind') = 'user'
                ORDER BY seq ASC
                LIMIT 1
                """,
                (thread_id,),
            ).fetchone()
        if row is None:
            return None
        payload = json.loads(row[0])
        content = payload.get("content")
        text = content if isinstance(content, str) else str(content)
        headline = " ".join(text.split())[:200]
        cleaned = " ".join(headline.split()).strip().strip('"').strip(".")
        return cleaned[:80].strip() or None

    def list_user_turns(self,thread_id: str) -> list[dict[str, Any]]:
        """Return every persisted user-message event for a thread, oldest-first.

        Used to populate the /rollback picker. seq is the event seq (the
        /rollback target); content is the (possibly truncated) user text.
        """
        if not self.threads_db.exists():
            return []
        with self._connect() as conn:
            self._ensure_schema(conn)
            rows = conn.execute(
                """
                SELECT seq, payload FROM events
                WHERE thread_id = ? AND json_extract(payload, '$.kind') = 'user'
                ORDER BY seq ASC
                """,
                (thread_id,),
            ).fetchall()
        turns: list[dict[str, Any]] = []
        for row in rows:
            payload = json.loads(row[1])
            content = payload.get("content")
            text = content if isinstance(content, str) else str(content)
            turns.append({"seq": int(row[0]), "content": text})
        return turns

    def truncate_after(self, thread_id: str, user_seq: int) -> None:
        """Delete events with seq >= user_seq and re-derive thread rollup columns.

        Also drops checkpoint rows at/after user_seq (those snapshots describe the
        abandoned tail). Hard truncate per design decision: no undo of the undo.
        """
        if not self.auto_save:
            return

        with self._write_lock:
            with self._connect() as conn:
                self._ensure_schema(conn)
                # Delete the abandoned conversation tail and its checkpoints.
                conn.execute(
                    "DELETE FROM events WHERE thread_id = ? AND seq >= ?",
                    (thread_id, user_seq),
                )
                conn.execute(
                    "DELETE FROM checkpoints WHERE thread_id = ? AND user_seq >= ?",
                    (thread_id, user_seq),
                )
                # Re-derive threads rollup from the remaining events so /status
                # matches the new (shorter) conversation. Cost tracker stays
                # process-global per product decision.
                remaining = conn.execute(
                    "SELECT payload FROM events WHERE thread_id = ? ORDER BY seq",
                    (thread_id,),
                ).fetchall()
                turn_count = 0
                total_cost = 0.0
                input_tokens = 0
                cached_tokens = 0
                output_tokens = 0
                for item in remaining:
                    event = json.loads(item[0])
                    if event.get("kind") == "user":
                        turn_count += 1
                    elif event.get("kind") == "usage" and not event.get("inherited"):
                        total_cost += float(event.get("cost_usd") or 0.0)
                        input_tokens += int(event.get("input_tokens") or 0)
                        cached_tokens += int(event.get("cached_input_tokens") or 0)
                        output_tokens += int(event.get("output_tokens") or 0)
                conn.execute(
                    """
                    UPDATE threads SET
                        updated_at = ?,
                        turn_count = ?,
                        total_cost_usd = ?,
                        input_tokens = ?,
                        cached_input_tokens = ?,
                        output_tokens = ?
                    WHERE thread_id = ?
                    """,
                    (self._now(), turn_count, total_cost, input_tokens, cached_tokens, output_tokens, thread_id),
                )
                conn.commit()

    def _apply_event_to_thread(
        self,
        conn: sqlite3.Connection,
        thread_id: str,
        event: dict[str, Any],
        updated_at: str,
    ) -> None:
        turn_delta = 1 if event.get("kind") == "user" else 0
        cost_delta = 0.0
        input_delta = 0
        cached_delta = 0
        output_delta = 0

        if event.get("kind") == "usage" and not event.get("inherited"):
            cost_delta = float(event.get("cost_usd") or 0.0)
            input_delta = int(event.get("input_tokens") or 0)
            cached_delta = int(event.get("cached_input_tokens") or 0)
            output_delta = int(event.get("output_tokens") or 0)
            model = str(event.get("model") or "").strip()
            if model:
                conn.execute(
                    """
                    UPDATE threads SET model = ?
                    WHERE thread_id = ? AND (model = '' OR model IS NULL)
                    """,
                    (model, thread_id),
                )

        conn.execute(
            """
            UPDATE threads SET
                updated_at = ?,
                turn_count = turn_count + ?,
                total_cost_usd = total_cost_usd + ?,
                input_tokens = input_tokens + ?,
                cached_input_tokens = cached_input_tokens + ?,
                output_tokens = output_tokens + ?
            WHERE thread_id = ?
            """,
            (
                updated_at,
                turn_delta,
                cost_delta,
                input_delta,
                cached_delta,
                output_delta,
                thread_id,
            ),
        )
