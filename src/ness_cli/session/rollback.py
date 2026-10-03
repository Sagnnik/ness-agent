"""Workspace checkpoints and durable session rollback."""

from __future__ import annotations

import asyncio
import os
import subprocess
import tempfile
from typing import TYPE_CHECKING
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from ness_agent import MemoryBackend

from ness_cli.session.events import RollbackCheckpoint, ToolMutationEvent
from ness_cli.session.persistence import SessionRepository

if TYPE_CHECKING:
    from ness_cli.session.mutations import WorkspaceMutations

_FULL_TREE = "*"


def _run_git(
    arguments: list[str],
    *,
    cwd: Path,
    environment: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    process_environment = dict(os.environ)
    if environment:
        process_environment.update(environment)
    return subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        env=process_environment,
    )


def is_git_workspace(cwd: Path) -> bool:
    result = _run_git(
        ["rev-parse", "--is-inside-work-tree"],
        cwd=cwd,
    )
    return result.returncode == 0 and result.stdout.strip() == "true"


def create_file_checkpoint(cwd: Path) -> str | None:
    """Return an immutable Git tree without touching the user's index.

    Repositories without an initial commit have no baseline and return None.
    """
    if not is_git_workspace(cwd):
        return None

    head = _run_git(["rev-parse", "--verify", "HEAD^{tree}"], cwd=cwd)
    if head.returncode != 0:
        return None
    head_tree = head.stdout.strip()
    if not head_tree:
        return None

    status = _run_git(["status", "--porcelain"], cwd=cwd)
    if status.returncode == 0 and not status.stdout.strip():
        return head_tree

    temporary_index = tempfile.NamedTemporaryFile(
        prefix="ness-shadow-",
        suffix=".idx",
        delete=False,
    )
    temporary_index.close()
    try:
        environment = {"GIT_INDEX_FILE": temporary_index.name}
        if (
            _run_git(
                ["read-tree", head_tree],
                cwd=cwd,
                environment=environment,
            ).returncode
            != 0
        ):
            return None
        if (
            _run_git(
                ["add", "-A"],
                cwd=cwd,
                environment=environment,
            ).returncode
            != 0
        ):
            return None
        result = _run_git(
            ["write-tree"],
            cwd=cwd,
            environment=environment,
        )
        if result.returncode != 0:
            return None
        return result.stdout.strip() or None
    finally:
        try:
            os.unlink(temporary_index.name)
        except OSError:
            pass


TreeEntry = tuple[str, str]
IgnoredFileFingerprint = tuple[int, int, int, int]


def ignored_file_fingerprints(
    cwd: Path, *, excluded_paths: tuple[Path, ...] = ()
) -> dict[str, IgnoredFileFingerprint]:
    """Inspect ignored files without reading or storing their contents in Git."""
    result = _run_git(
        ["ls-files", "--others", "--ignored", "--exclude-standard", "-z"], cwd=cwd
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "could not inspect ignored files")
    fingerprints = {}
    for raw_path in result.stdout.split("\0"):
        if not raw_path:
            continue
        path, target = workspace_path(raw_path, cwd)
        if any(target.is_relative_to(excluded) for excluded in excluded_paths):
            continue
        info = target.lstat()
        fingerprints[path] = (
            info.st_mode,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
        )
    return fingerprints


@dataclass(frozen=True, slots=True)
class RestoreResult:
    ok: bool
    warnings: tuple[str, ...] = ()


def tree_entries(tree: str, cwd: Path) -> dict[str, TreeEntry]:
    result = _run_git(["ls-tree", "-rz", tree], cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "could not read checkpoint tree")
    entries = {}
    for row in result.stdout.split("\0"):
        if row:
            metadata, path = row.split("\t", 1)
            mode, kind, object_id = metadata.split()
            if kind not in {"blob", "commit"}:
                raise ValueError(f"unsupported Git entry: {path}")
            entries[path] = (mode, object_id)
    return entries


def workspace_path(path: str, cwd: Path) -> tuple[str, Path]:
    root = cwd.resolve()
    target = Path(os.path.abspath(root / path))
    relative = target.relative_to(root)
    if not relative.parts or ".git" in relative.parts:
        raise ValueError(f"unsafe rollback path: {path}")
    # Resolve parents, but do not follow a leaf symlink that must be restored.
    target.parent.resolve().relative_to(root)
    return relative.as_posix(), target


def restore_paths(
    git_hash: str,
    paths: tuple[str, ...],
    cwd: Path,
    *,
    owned_paths: tuple[str, ...] | None = None,
    expected: Mapping[str, TreeEntry | None] | None = None,
) -> RestoreResult:
    """Restore owned working files and preserve staging and unrelated files.

    A full-tree request requires an explicit mutation set. Expected final
    entries protect edits made after mutation recording. Preflight checks run
    before any file is changed; an I/O failure can still leave a partial restore.
    """
    try:
        baseline = tree_entries(git_hash, cwd)
        if not paths or _FULL_TREE in paths:
            if owned_paths is None:
                raise ValueError("Full-tree rollback has no recorded file ownership.")
            paths = owned_paths
        selected = dict(workspace_path(path, cwd) for path in paths)
        if expected is not None:
            current_tree = create_file_checkpoint(cwd)
            if current_tree is None:
                raise RuntimeError(
                    "Could not inspect current workspace before rollback."
                )
            current = tree_entries(current_tree, cwd)
            for path in selected:
                if path not in expected:
                    raise ValueError(f"No mutation fingerprint for {path}.")
                if current.get(path) not in (expected[path], baseline.get(path)):
                    raise ValueError(
                        f"File changed after the recorded operation: {path}."
                    )
        for path, target in selected.items():
            if target.is_dir() and not target.is_symlink():
                raise ValueError(f"Cannot roll back a directory as a file: {path}.")
            if baseline.get(path, ("", ""))[0] == "160000":
                raise ValueError(f"Submodule rollback is unsupported: {path}.")
    except Exception as error:
        return RestoreResult(False, (f"File restore failed: {error}",))

    errors: list[str] = []
    for path, target in selected.items():
        try:
            if path in baseline:
                result = _run_git(
                    [
                        "--literal-pathspecs",
                        "restore",
                        "--source",
                        git_hash,
                        "--worktree",
                        "--overlay",
                        "--",
                        path,
                    ],
                    cwd=cwd,
                )
                if result.returncode != 0:
                    raise OSError(
                        (result.stderr or result.stdout).strip() or "git restore failed"
                    )
            elif target.is_symlink() or target.is_file():
                target.unlink()
        except Exception as error:
            errors.append(f"Could not restore {path}: {error}")
    return RestoreResult(not errors, tuple(errors))


def mutated_paths(tool: str, arguments: Mapping[str, object]) -> tuple[str, ...]:
    name = tool.casefold()
    paths = arguments.get("paths")
    if name == "delete" and isinstance(paths, list):
        return tuple(str(path) for path in paths)
    if name in {"edit", "write", "delete", "auto_format"}:
        path = arguments.get("path")
        return (str(path),) if path else ()
    if name in {"shell", "spawn_subagent"}:
        return (_FULL_TREE,)
    return ()


@dataclass(frozen=True, slots=True)
class PendingCheckpoint:
    git_hash: str | None
    memory_snapshot: str


@dataclass(frozen=True, slots=True)
class RollbackResult:
    ok: bool
    user_seq: int
    message: str
    warnings: tuple[str, ...] = ()


ReplayCallback = Callable[[bool, int | None], Awaitable[bool]]


class RollbackService:
    def __init__(
        self,
        *,
        project_root: Path,
        repository: SessionRepository,
        memory: MemoryBackend,
        mutations: WorkspaceMutations | None = None,
    ) -> None:
        self._project_root = project_root
        self._repository = repository
        self._memory = memory
        self._mutations = mutations

    async def snapshot(self, thread_id: str) -> PendingCheckpoint:
        git_hash, memory_snapshot = await asyncio.gather(
            asyncio.to_thread(create_file_checkpoint, self._project_root),
            asyncio.to_thread(self._memory.read_session_raw, thread_id),
        )
        return PendingCheckpoint(git_hash, memory_snapshot)

    def save(
        self,
        thread_id: str,
        user_seq: int | None,
        checkpoint: PendingCheckpoint,
    ) -> None:
        if user_seq is None:
            return
        self._repository.save_checkpoint(
            thread_id,
            user_seq,
            git_hash=checkpoint.git_hash,
            memory_snapshot=checkpoint.memory_snapshot,
        )
        if self._mutations is not None:
            self._mutations.begin_turn(thread_id, user_seq)

    def record_mutations(self, thread_id: str, user_seq: int | None) -> None:
        if user_seq is None:
            return
        if self._mutations is not None:
            self._mutations.finish_turn(thread_id)
        for event in self._repository.events_since(thread_id, user_seq + 1):
            tool_event = ToolMutationEvent.parse(event)
            if tool_event is None:
                continue
            if tool_event.exit_status in {"denied", "mode_gated"}:
                continue
            if tool_event.result.startswith("Error:"):
                continue
            for path in mutated_paths(tool_event.tool, tool_event.arguments):
                self._repository.add_modified_path(thread_id, user_seq, path)

    async def rollback(
        self,
        thread_id: str,
        user_seq: int,
        *,
        replay: ReplayCallback,
    ) -> RollbackResult:
        if self._mutations is not None and not self._mutations.begin_restore(thread_id):
            return RollbackResult(
                False,
                user_seq,
                "Workspace mutation or rollback is still active; history was preserved.",
            )
        try:
            return await self._rollback(thread_id, user_seq, replay=replay)
        finally:
            if self._mutations is not None:
                self._mutations.end_restore()

    async def _rollback(
        self,
        thread_id: str,
        user_seq: int,
        *,
        replay: ReplayCallback,
    ) -> RollbackResult:
        if user_seq < 0:
            return RollbackResult(False, user_seq, "Invalid rollback target.")
        if not self._repository.auto_save:
            return RollbackResult(False, user_seq, "Rollback requires thread autosave.")
        checkpoint = self._repository.checkpoint(thread_id, user_seq)
        if checkpoint is None:
            return RollbackResult(
                False,
                user_seq,
                f"No checkpoint for seq {user_seq} in this thread.",
            )

        restored = await self._restore(thread_id, checkpoint)
        if not restored.ok:
            return RollbackResult(
                False,
                user_seq,
                "\n".join(
                    (*restored.warnings, "Rollback failed; history was preserved.")
                ),
                restored.warnings,
            )
        try:
            if not await replay(False, user_seq):
                raise RuntimeError("Conversation replay did not complete.")
            if not self._repository.auto_save:
                raise RuntimeError("Thread autosave was disabled during rollback.")
            self._repository.truncate(thread_id, user_seq)
        except Exception as error:
            return RollbackResult(
                False,
                user_seq,
                f"Conversation restore failed: {error}\nHistory was preserved.",
                (f"Conversation restore failed: {error}",),
            )
        return RollbackResult(True, user_seq, f"Rolled back to turn at seq {user_seq}.")

    def _mutation_plan(
        self,
        thread_id: str,
        checkpoint: RollbackCheckpoint,
    ) -> tuple[tuple[str, ...], dict[str, TreeEntry | None] | None]:
        events = self._repository.events_since(thread_id, checkpoint.user_seq)
        tracked = any(event.kind == "workspace_tracking" for event in events)
        if not tracked:
            paths = {path for path in checkpoint.modified_paths}
            for event in events:
                tool = ToolMutationEvent.parse(event)
                if tool is not None and tool.exit_status not in {
                    "denied",
                    "mode_gated",
                }:
                    paths.update(mutated_paths(tool.tool, tool.arguments))
            if paths:
                raise ValueError(
                    "Rollback has no recorded file ownership for these mutations."
                )
            return (), None

        baseline = tree_entries(checkpoint.git_hash, self._project_root)
        expected: dict[str, TreeEntry | None] = {}
        turns = {
            int(event.get("user_seq"))
            for event in events
            if event.kind == "workspace_tracking"
        }
        started = {
            event.seq for event in events if event.kind == "workspace_mutation_start"
        }
        completed = {
            event.get("start_seq")
            for event in events
            if event.kind == "workspace_mutation"
        }
        if started - completed:
            raise ValueError("Tool mutation recording did not complete.")
        for event in events:
            if event.kind != "workspace_mutation":
                continue
            if event.get("error"):
                raise ValueError(str(event.get("error")))
            before = tree_entries(str(event.get("before_tree")), self._project_root)
            after = tree_entries(str(event.get("after_tree")), self._project_root)
            for path in before.keys() | after.keys():
                if before.get(path) == after.get(path):
                    continue
                prior = expected.get(path, baseline.get(path))
                if before.get(path) != prior:
                    raise ValueError(
                        f"File changed outside the recorded operation: {path}."
                    )
                expected[path] = after.get(path)
        recorded_counts: dict[int, int] = {}
        for event in events:
            if event.kind == "workspace_mutation_start":
                seq = int(event.get("user_seq"))
                recorded_counts[seq] = recorded_counts.get(seq, 0) + 1
        # Every mutating turn in the discarded suffix must have tracking data.
        user_seq = checkpoint.user_seq
        for event in events:
            if event.kind == "user":
                user_seq = event.seq
            tool = ToolMutationEvent.parse(event)
            if (
                tool is not None
                and tool.exit_status not in {"denied", "mode_gated"}
                and mutated_paths(tool.tool, tool.arguments)
            ):
                if user_seq not in turns or recorded_counts.get(user_seq, 0) <= 0:
                    raise ValueError("A tool mutation has no recorded file ownership.")
                recorded_counts[user_seq] -= 1
        return tuple(sorted(expected)), expected

    async def _restore(
        self,
        thread_id: str,
        checkpoint: RollbackCheckpoint,
    ) -> RestoreResult:
        if checkpoint.git_hash is None:
            return RestoreResult(
                False,
                ("No Git snapshot for this checkpoint; files were not restored.",),
            )
        try:
            paths, expected = self._mutation_plan(thread_id, checkpoint)
            files = await asyncio.to_thread(
                restore_paths,
                checkpoint.git_hash,
                paths,
                self._project_root,
                owned_paths=paths,
                expected=expected,
            )
            if not files.ok:
                return files
            await asyncio.to_thread(
                self._memory.write_session_raw,
                thread_id,
                checkpoint.memory_snapshot,
            )
        except Exception as error:
            return RestoreResult(False, (f"Rollback restore failed: {error}",))
        return RestoreResult(True)
