"""Record Git-visible workspace changes around CLI tool execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ness_agent import Hook, HookRunner

from ness_cli.session.persistence import SessionRepository
from ness_cli.session.rollback import (
    create_file_checkpoint,
    mutated_paths,
    tree_entries,
    workspace_path,
)


@dataclass
class _PendingMutation:
    user_seq: int
    before_tree: str | None
    overlapping: bool = False
    start_seq: int | None = None
    error: str | None = None


class WorkspaceMutations:
    """One recorder shared by the sessions using a project's hook runner.

    Changes observed during a tool's execution window belong to that tool.
    Overlapping managed tool windows are ambiguous and cannot be rolled back
    automatically. Later edits are protected by comparison with the final tree.
    """

    def __init__(self, root: Path, repository: SessionRepository) -> None:
        self.root = root
        self.repository = repository
        self._turns: dict[str, int] = {}
        self._pending: dict[str, _PendingMutation] = {}
        self._restoring = False

    def begin_restore(self, thread_id: str) -> bool:
        if self._restoring or self._pending or thread_id in self._turns:
            return False
        self._restoring = True
        return True

    def end_restore(self) -> None:
        self._restoring = False

    @classmethod
    def attach(
        cls,
        hooks: HookRunner,
        root: Path,
        repository: SessionRepository,
    ) -> WorkspaceMutations:
        recorder = getattr(hooks, "_ness_workspace_mutations", None)
        if recorder is None:
            recorder = cls(root, repository)
            hooks.register(Hook("preToolUse", handler=recorder.before_tool))
            hooks.register(Hook("postToolUse", handler=recorder.after_tool))
            hooks._ness_workspace_mutations = recorder
        return recorder

    def begin_turn(self, thread_id: str, user_seq: int) -> None:
        self._turns[thread_id] = user_seq
        self.repository.append(
            thread_id, {"kind": "workspace_tracking", "user_seq": user_seq}
        )

    def _record(self, thread_id: str, pending: _PendingMutation, **data: Any) -> None:
        self.repository.append(
            thread_id,
            {
                "kind": "workspace_mutation",
                "user_seq": pending.user_seq,
                "before_tree": pending.before_tree,
                "start_seq": pending.start_seq,
                **data,
            },
        )

    def before_tool(self, payload: dict[str, Any]) -> tuple[bool, str]:
        thread_id = str(payload.get("thread_id") or "")
        tool = str(payload.get("tool") or "")
        if not mutated_paths(tool, payload.get("args") or {}):
            return True, ""
        if self._restoring:
            return (
                False,
                "Workspace rollback is in progress; retry this tool afterward.",
            )
        # Include unbound subagent/sibling calls in overlap detection, so they
        # cannot silently become part of the parent's filesystem delta.
        try:
            before = create_file_checkpoint(self.root)
        except Exception:
            before = None
        overlapping = bool(self._pending)
        for pending in self._pending.values():
            pending.overlapping = True
        self._pending[thread_id] = _PendingMutation(
            self._turns.get(thread_id, -1),
            before,
            overlapping,
        )
        pending = self._pending[thread_id]
        if before is not None:
            try:
                entries = tree_entries(before, self.root)
                for raw_path in mutated_paths(tool, payload.get("args") or {}):
                    if raw_path == "*":
                        continue
                    path, target = workspace_path(raw_path, self.root)
                    if (target.exists() or target.is_symlink()) and path not in entries:
                        raise ValueError(
                            f"File is not covered by the Git snapshot: {path}."
                        )
            except Exception as error:
                pending.error = str(error)
        if pending.user_seq >= 0:
            pending.start_seq = self.repository.append(
                thread_id,
                {
                    "kind": "workspace_mutation_start",
                    "user_seq": pending.user_seq,
                },
            )
        return True, ""

    def after_tool(self, payload: dict[str, Any]) -> tuple[bool, str]:
        thread_id = str(payload.get("thread_id") or "")
        pending = self._pending.pop(thread_id, None)
        if pending is None or pending.user_seq < 0:
            return True, ""
        try:
            after = create_file_checkpoint(self.root)
            if pending.overlapping:
                raise ValueError(
                    "Overlapping tool mutations have ambiguous file ownership."
                )
            if pending.error:
                raise ValueError(pending.error)
            if (
                payload.get("tool") == "shell"
                and (payload.get("args") or {}).get("action") == "start"
            ):
                raise ValueError(
                    "Background shell mutations cannot be attributed to a completed tool window."
                )
            if (
                pending.before_tree is None
                or after is None
                or pending.start_seq is None
            ):
                raise ValueError("Tool mutation snapshot is unavailable.")
            entries = tree_entries(after, self.root)
            for raw_path in mutated_paths(
                str(payload.get("tool")), payload.get("args") or {}
            ):
                if raw_path == "*":
                    continue
                path, target = workspace_path(raw_path, self.root)
                if (target.exists() or target.is_symlink()) and path not in entries:
                    raise ValueError(
                        f"File is not covered by the Git snapshot: {path}."
                    )
            self._record(thread_id, pending, after_tree=after)
        except Exception as error:
            self._record(thread_id, pending, error=str(error))
        return True, ""

    def finish_turn(self, thread_id: str) -> None:
        self._turns.pop(thread_id, None)
        pending = self._pending.pop(thread_id, None)
        if pending is not None:
            self._record(
                thread_id, pending, error="Tool mutation recording did not complete."
            )
