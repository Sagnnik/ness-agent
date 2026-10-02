from __future__ import annotations

import subprocess
import asyncio
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from ness_agent import HookRunner, ThreadStore

from ness_cli.session import rollback
from ness_cli.session.rollback import (
    RollbackService,
    create_file_checkpoint,
    restore_paths,
)
from ness_cli.session.mutations import WorkspaceMutations
from ness_cli.session.persistence import SessionRepository


def git(project: Path, *arguments: str) -> str:
    result = subprocess.run(
        [
            "git",
            "-c",
            "user.name=Ness rollback test",
            "-c",
            "user.email=rollback-test@example.invalid",
            *arguments,
        ],
        cwd=project,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


@pytest.fixture
def git_project(tmp_path: Path) -> Path:
    project = tmp_path / "rollback-repo"
    project.mkdir()
    git(project, "init", "--quiet")
    return project


@pytest.mark.parametrize("dirty", [False, True], ids=["clean", "dirty"])
def test_checkpoint_restores_original_contents_after_head_moves(git_project, dirty):
    path = git_project / "app.txt"
    path.write_text("committed contents\n")
    git(git_project, "add", "app.txt")
    git(git_project, "commit", "--quiet", "-m", "initial")

    if dirty:
        path.write_text("staged contents\n")
        git(git_project, "add", "app.txt")
        path.write_text("checkpoint contents\n")
    expected = path.read_text()
    index_before = git(git_project, "ls-files", "--stage")

    checkpoint = create_file_checkpoint(git_project)

    assert git(git_project, "ls-files", "--stage") == index_before
    path.write_text("later commit\n")
    git(git_project, "add", "app.txt")
    git(git_project, "commit", "--quiet", "-m", "later")
    assert checkpoint is not None
    assert restore_paths(checkpoint, ("app.txt",), git_project).ok
    assert path.read_text() == expected
    assert git(git_project, "cat-file", "-t", checkpoint) == "tree"


@pytest.mark.parametrize("dirty", [False, True], ids=["empty", "untracked-file"])
def test_checkpoint_without_initial_commit_is_unavailable(git_project, dirty):
    if dirty:
        (git_project / "app.txt").write_text("not committed\n")
    index_before = git(git_project, "ls-files", "--stage")

    assert create_file_checkpoint(git_project) is None

    assert git(git_project, "ls-files", "--stage") == index_before


def test_checkpoint_outside_git_is_unavailable(isolated_cli_env):
    assert create_file_checkpoint(isolated_cli_env.project) is None


@pytest.fixture
def committed_project(git_project: Path) -> Path:
    (git_project / "app.txt").write_text("committed contents\n")
    git(git_project, "add", "app.txt")
    git(git_project, "commit", "--quiet", "-m", "initial")
    return git_project


@pytest.mark.parametrize(
    "paths",
    [("app.txt",), (), ("*",)],
    ids=["targeted", "full-default", "full-shell"],
)
@pytest.mark.parametrize("stage_after_snapshot", [False, True])
def test_restore_preserves_staged_contents(
    committed_project, paths, stage_after_snapshot
):
    project = committed_project
    path = project / "app.txt"
    path.write_text("staged contents\n")
    git(project, "add", "app.txt")
    path.write_text("checkpoint contents\n")
    checkpoint = create_file_checkpoint(project)
    assert checkpoint is not None

    staged_contents = "staged contents"
    if stage_after_snapshot:
        staged_contents = "later staged contents"
        path.write_text(staged_contents + "\n")
        git(project, "add", "app.txt")
    path.write_text("agent changes\n")
    unrelated = project / "unrelated.txt"
    unrelated.write_text("user-created file\n")
    git(project, "add", "unrelated.txt")
    index_before = git(project, "ls-files", "--stage")

    output = restore_paths(checkpoint, paths, project, owned_paths=("app.txt",))

    assert output.ok and not output.warnings
    assert path.read_text() == "checkpoint contents\n"
    assert unrelated.read_text() == "user-created file\n"
    assert git(project, "show", ":app.txt") == staged_contents
    assert git(project, "ls-files", "--stage") == index_before


@pytest.mark.parametrize(
    "paths",
    [("app.txt",), (), ("*",)],
    ids=["targeted", "full-default", "full-shell"],
)
def test_restore_preserves_staged_deletion(committed_project, paths):
    project = committed_project
    checkpoint = create_file_checkpoint(project)
    assert checkpoint is not None
    git(project, "rm", "--cached", "app.txt")
    path = project / "app.txt"
    path.write_text("agent changes\n")
    index_before = git(project, "ls-files", "--stage")

    output = restore_paths(checkpoint, paths, project, owned_paths=("app.txt",))

    assert output.ok and not output.warnings
    assert path.read_text() == "committed contents\n"
    assert git(project, "ls-files", "--stage") == index_before


def test_restore_removes_created_file_without_unstaging_it(committed_project):
    project = committed_project
    checkpoint = create_file_checkpoint(project)
    assert checkpoint is not None
    path = project / "created.txt"
    path.write_text("staged creation\n")
    git(project, "add", "created.txt")
    path.write_text("agent changes\n")
    index_before = git(project, "ls-files", "--stage")

    assert restore_paths(checkpoint, ("created.txt",), project).ok

    assert not path.exists()
    assert git(project, "show", ":created.txt") == "staged creation"
    assert git(project, "ls-files", "--stage") == index_before


@pytest.fixture
def rollback_env(committed_project, tmp_path):
    class Memory:
        value = "before turn"

        def read_session_raw(self, thread_id):
            return self.value

        def write_session_raw(self, thread_id, value):
            self.value = value

    repository = SessionRepository(
        ThreadStore(threads_dir=tmp_path / "threads", auto_save=True)
    )
    hooks = HookRunner(project_root=committed_project)
    mutations = WorkspaceMutations.attach(hooks, committed_project, repository)
    memory = Memory()
    return SimpleNamespace(
        project=committed_project,
        repository=repository,
        hooks=hooks,
        mutations=mutations,
        memory=memory,
        thread="session-rollback",
        service=RollbackService(
            project_root=committed_project,
            repository=repository,
            memory=memory,
            mutations=mutations,
        ),
    )


def start_turn(env):
    snapshot = asyncio.run(env.service.snapshot(env.thread))
    seq = env.repository.append_user(env.thread, "change workspace")
    env.service.save(env.thread, seq, snapshot)
    return seq


@contextmanager
def tool_window(env, *, thread=None, tool="shell", args=None):
    payload = {
        "thread_id": thread or env.thread,
        "tool": tool,
        "args": args or {"command": "test"},
    }
    assert env.hooks.run("preToolUse", payload)[0]
    yield
    assert env.hooks.run("postToolUse", payload)[0]
    env.repository.append(
        payload["thread_id"],
        {
            "kind": "tool",
            "tool": tool,
            "args": payload["args"],
            "result": "ok",
        },
    )


async def replay_ok(replay_cost, before_seq):
    assert replay_cost is False
    return True


def test_shell_rollback_restores_owned_changes_and_keeps_unrelated_files(rollback_env):
    env = rollback_env
    project = env.project
    (project / "deleted.txt").write_text("restore me")
    (project / "unrelated.txt").write_text("original user file")
    git(project, "add", ".")
    git(project, "commit", "--quiet", "-m", "more files")
    (project / "preexisting.txt").write_text("untracked before checkpoint")
    seq = start_turn(env)
    with tool_window(env):
        (project / "app.txt").write_text("agent edit")
        (project / "deleted.txt").unlink()
        (project / "nested").mkdir()
        (project / "nested" / "created.txt").write_text("agent creation")
    env.service.record_mutations(env.thread, seq)
    (project / "unrelated.txt").write_text("later user edit")
    (project / "later.txt").write_text("later user creation")
    git(project, "add", "later.txt", "nested/created.txt")
    index_before = git(project, "ls-files", "--stage")

    async def replay(replay_cost, before_seq):
        assert before_seq == seq
        assert env.repository.raw_events(env.thread)  # No premature truncate.
        return True

    result = asyncio.run(env.service.rollback(env.thread, seq, replay=replay))

    assert result.ok and not result.warnings
    assert (project / "app.txt").read_text() == "committed contents\n"
    assert (project / "deleted.txt").read_text() == "restore me"
    assert not (project / "nested" / "created.txt").exists()
    assert (project / "unrelated.txt").read_text() == "later user edit"
    assert (project / "preexisting.txt").read_text() == "untracked before checkpoint"
    assert (project / "later.txt").read_text() == "later user creation"
    assert git(project, "ls-files", "--stage") == index_before
    assert env.repository.raw_events(env.thread) == []


def test_rollback_collects_mutations_from_later_turns(rollback_env):
    env = rollback_env
    first = start_turn(env)
    with tool_window(env):
        (env.project / "first.txt").write_text("first")
    env.service.record_mutations(env.thread, first)
    second = start_turn(env)
    with tool_window(env):
        (env.project / "second.txt").write_text("second")
        (env.project / "app.txt").write_text("later change")
    env.service.record_mutations(env.thread, second)

    # A new service has no in-memory ownership state; durable records suffice.
    env.service = RollbackService(
        project_root=env.project,
        repository=env.repository,
        memory=env.memory,
        mutations=WorkspaceMutations(env.project, env.repository),
    )

    assert asyncio.run(env.service.rollback(env.thread, first, replay=replay_ok)).ok
    assert not (env.project / "first.txt").exists()
    assert not (env.project / "second.txt").exists()
    assert (env.project / "app.txt").read_text() == "committed contents\n"


@pytest.mark.parametrize("later_edit", ["created", "modified", "deleted"])
def test_rollback_refuses_to_overwrite_later_edits(rollback_env, later_edit):
    env = rollback_env
    deleted = env.project / "deleted.txt"
    deleted.write_text("original")
    git(env.project, "add", "deleted.txt")
    git(env.project, "commit", "--quiet", "-m", "deletable")
    seq = start_turn(env)
    with tool_window(env):
        (env.project / "created.txt").write_text("agent")
        (env.project / "app.txt").write_text("agent")
        deleted.unlink()
    env.service.record_mutations(env.thread, seq)
    target = (
        env.project
        / {"created": "created.txt", "modified": "app.txt", "deleted": "deleted.txt"}[
            later_edit
        ]
    )
    target.write_text("later user/thread edit")
    events = env.repository.raw_events(env.thread)

    result = asyncio.run(env.service.rollback(env.thread, seq, replay=replay_ok))

    assert not result.ok and "changed after" in result.message
    assert target.read_text() == "later user/thread edit"
    assert (env.project / "app.txt").read_text() != "committed contents\n"
    assert env.repository.raw_events(env.thread) == events


def test_rollback_rejects_overlapping_thread_mutations(rollback_env):
    env = rollback_env
    seq = start_turn(env)
    sibling = {
        "tool": "shell",
        "args": {"command": "other"},
        "thread_id": "session-sibling",
    }
    with tool_window(env):
        env.hooks.run("preToolUse", sibling)
        (env.project / "created.txt").write_text("ambiguous creation")
        env.hooks.run("postToolUse", sibling)
    env.service.record_mutations(env.thread, seq)
    events = env.repository.raw_events(env.thread)

    result = asyncio.run(env.service.rollback(env.thread, seq, replay=replay_ok))

    assert not result.ok and "Overlapping" in result.message
    assert (env.project / "created.txt").exists()
    assert env.repository.raw_events(env.thread) == events


@pytest.mark.parametrize(
    "failure", ["git", "memory", "replay-false", "replay-error", "truncate"]
)
def test_failed_rollback_preserves_history_and_checkpoint(
    rollback_env, monkeypatch, failure
):
    env = rollback_env
    seq = start_turn(env)
    with tool_window(env):
        (env.project / "app.txt").write_text("agent change")
    env.service.record_mutations(env.thread, seq)
    env.memory.value = "after turn"
    events = env.repository.raw_events(env.thread)
    checkpoint = env.repository.checkpoint(env.thread, seq)
    calls = []

    def fail(*args, **kwargs):
        raise OSError("injected failure")

    if failure == "git":
        run_git = rollback._run_git

        def fail_restore(arguments, **kwargs):
            if "restore" in arguments:
                return subprocess.CompletedProcess(arguments, 1, "", "injected failure")
            return run_git(arguments, **kwargs)

        monkeypatch.setattr(rollback, "_run_git", fail_restore)
    elif failure == "memory":
        monkeypatch.setattr(env.memory, "write_session_raw", fail)
    elif failure == "truncate":
        monkeypatch.setattr(env.repository, "truncate", fail)

    async def replay(replay_cost, before_seq):
        calls.append(before_seq)
        assert env.repository.raw_events(env.thread) == events
        if failure == "replay-error":
            raise RuntimeError("injected replay failure")
        return failure != "replay-false"

    result = asyncio.run(env.service.rollback(env.thread, seq, replay=replay))

    assert not result.ok
    assert "preserved" in result.message.lower()
    assert env.repository.raw_events(env.thread) == events
    assert env.repository.checkpoint(env.thread, seq) == checkpoint
    if failure in {"git", "memory"}:
        assert calls == []


def test_invalid_snapshot_does_not_delete_files(committed_project):
    target = committed_project / "created.txt"
    target.write_text("keep me")
    result = restore_paths("not-a-git-object", ("created.txt",), committed_project)
    assert not result.ok
    assert target.read_text() == "keep me"


def test_full_restore_without_ownership_is_rejected(committed_project):
    checkpoint = create_file_checkpoint(committed_project)
    (committed_project / "new.txt").write_text("unattributed")
    assert not restore_paths(checkpoint, ("*",), committed_project).ok
    assert (committed_project / "new.txt").exists()


def test_incomplete_mutation_recording_preserves_history(rollback_env):
    env = rollback_env
    seq = start_turn(env)
    payload = {"tool": "shell", "args": {}, "thread_id": env.thread}
    env.hooks.run("preToolUse", payload)
    (env.project / "created.txt").write_text("unfinished operation")
    env.mutations.finish_turn(env.thread)
    events = env.repository.raw_events(env.thread)
    result = asyncio.run(env.service.rollback(env.thread, seq, replay=replay_ok))
    assert not result.ok and "did not complete" in result.message
    assert env.repository.raw_events(env.thread) == events
    assert (env.project / "created.txt").exists()


def test_restore_blocks_managed_writes_and_releases_guard_after_failure(rollback_env):
    env = rollback_env
    seq = start_turn(env)
    with tool_window(env):
        (env.project / "created.txt").write_text("owned")
    env.service.record_mutations(env.thread, seq)
    payload = {"tool": "shell", "args": {}, "thread_id": "session-sibling"}

    async def replay(replay_cost, before_seq):
        allowed, message = env.hooks.run("preToolUse", payload)
        assert not allowed and "rollback is in progress" in message
        return False

    assert not asyncio.run(env.service.rollback(env.thread, seq, replay=replay)).ok
    assert env.hooks.run("preToolUse", payload)[0]
    env.hooks.run("postToolUse", payload)


def test_rollback_refuses_a_still_active_turn(rollback_env):
    env = rollback_env
    seq = start_turn(env)
    events = env.repository.raw_events(env.thread)
    result = asyncio.run(env.service.rollback(env.thread, seq, replay=replay_ok))
    assert not result.ok and "still active" in result.message
    assert env.repository.raw_events(env.thread) == events


def test_historical_shell_checkpoint_without_ownership_preserves_history(rollback_env):
    env = rollback_env
    seq = env.repository.append_user(env.thread, "old shell turn")
    env.repository.save_checkpoint(
        env.thread,
        seq,
        git_hash=create_file_checkpoint(env.project),
        memory_snapshot="old",
    )
    env.repository.add_modified_path(env.thread, seq, "*")
    (env.project / "unattributed.txt").write_text("keep")
    events = env.repository.raw_events(env.thread)
    result = asyncio.run(env.service.rollback(env.thread, seq, replay=replay_ok))
    assert not result.ok and "no recorded file ownership" in result.message
    assert env.repository.raw_events(env.thread) == events
    assert (env.project / "unattributed.txt").exists()


def test_rollback_removes_created_symlink_without_touching_its_target(
    rollback_env, tmp_path
):
    env = rollback_env
    outside = tmp_path / "outside.txt"
    outside.write_text("keep outside contents")
    seq = start_turn(env)
    with tool_window(env):
        (env.project / "created-link").symlink_to(outside)
    env.service.record_mutations(env.thread, seq)
    assert asyncio.run(env.service.rollback(env.thread, seq, replay=replay_ok)).ok
    assert not (env.project / "created-link").is_symlink()
    assert outside.read_text() == "keep outside contents"


def test_ignored_file_tool_cannot_claim_a_complete_rollback(rollback_env):
    env = rollback_env
    (env.project / ".gitignore").write_text("ignored.txt\n")
    seq = start_turn(env)
    with tool_window(env, tool="write", args={"path": "ignored.txt"}):
        (env.project / "ignored.txt").write_text("not covered")
    env.service.record_mutations(env.thread, seq)
    events = env.repository.raw_events(env.thread)
    result = asyncio.run(env.service.rollback(env.thread, seq, replay=replay_ok))
    assert not result.ok and "not covered" in result.message
    assert env.repository.raw_events(env.thread) == events


@pytest.mark.parametrize("path", ["../outside.txt", ".git/config"])
def test_restore_rejects_paths_outside_the_workspace(committed_project, path):
    checkpoint = create_file_checkpoint(committed_project)
    result = restore_paths(checkpoint, (path,), committed_project)
    assert not result.ok


def test_user_edit_before_tool_execution_is_not_rolled_back(rollback_env):
    env = rollback_env
    seq = start_turn(env)
    path = env.project / "app.txt"
    path.write_text("user edit after checkpoint")
    with tool_window(env):
        path.write_text("agent edit based on user's edit")
    env.service.record_mutations(env.thread, seq)
    events = env.repository.raw_events(env.thread)
    result = asyncio.run(env.service.rollback(env.thread, seq, replay=replay_ok))
    assert not result.ok and "outside the recorded operation" in result.message
    assert path.read_text() == "agent edit based on user's edit"
    assert env.repository.raw_events(env.thread) == events


def test_memory_failure_can_be_retried_after_partial_file_restore(
    rollback_env, monkeypatch
):
    env = rollback_env
    seq = start_turn(env)
    with tool_window(env):
        (env.project / "created.txt").write_text("owned")
        (env.project / "app.txt").write_text("agent edit")
    env.service.record_mutations(env.thread, seq)

    def fail(*args):
        raise OSError("memory unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(env.memory, "write_session_raw", fail)
        assert not asyncio.run(
            env.service.rollback(env.thread, seq, replay=replay_ok)
        ).ok
    assert env.repository.checkpoint(env.thread, seq) is not None
    assert asyncio.run(env.service.rollback(env.thread, seq, replay=replay_ok)).ok
    assert not (env.project / "created.txt").exists()
    assert env.repository.raw_events(env.thread) == []


def test_background_shell_turn_refuses_automatic_rollback(rollback_env):
    env = rollback_env
    seq = start_turn(env)
    with tool_window(env, args={"command": "test", "action": "start"}):
        (env.project / "background.txt").write_text("background output")
    env.service.record_mutations(env.thread, seq)
    events = env.repository.raw_events(env.thread)
    result = asyncio.run(env.service.rollback(env.thread, seq, replay=replay_ok))
    assert not result.ok and "Background shell" in result.message
    assert env.repository.raw_events(env.thread) == events
    assert (env.project / "background.txt").exists()


def test_autosave_disabled_during_replay_does_not_report_success(rollback_env):
    env = rollback_env
    seq = start_turn(env)
    env.service.record_mutations(env.thread, seq)
    events = env.repository.raw_events(env.thread)

    async def replay(replay_cost, before_seq):
        env.repository.auto_save = False
        return True

    result = asyncio.run(env.service.rollback(env.thread, seq, replay=replay))
    assert not result.ok and "autosave was disabled" in result.message
    assert env.repository.raw_events(env.thread) == events
    assert env.repository.checkpoint(env.thread, seq) is not None


def test_cli_stream_records_shell_creation_and_rolls_it_back(isolated_cli_env):
    from langchain_core.messages import AIMessage
    from ness_agent import NessAgent, NessAgentOptions, NoOverlay, PromptLayersConfig
    from ness_cli.config import ConfigManager
    from ness_cli.session import CodingSession

    class Model:
        model = "offline-rollback-test"
        calls = 0

        def bind_tools(self, tools):
            return self

        async def ainvoke(self, messages, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "shell",
                            "args": {"command": "touch generated.txt"},
                            "id": "create-file",
                            "type": "tool_call",
                        }
                    ],
                )
            return AIMessage(content="Created the file.")

    paths = isolated_cli_env.paths
    git(paths.project_root, "init", "--quiet")
    (paths.project_root / "original.txt").write_text("original")
    git(paths.project_root, "add", ".")
    git(paths.project_root, "commit", "--quiet", "-m", "initial")
    agent = NessAgent(
        model=Model(),
        tools=["shell"],
        overlay=NoOverlay(),
        skills_dirs=[],
        prompt=PromptLayersConfig(l0="Offline test", persona="Test"),
        options=NessAgentOptions(
            project_root=paths.project_root,
            ness_dir=paths.ness_dir,
            yolo_mode=True,
            enable_approval=False,
            reflection_token_ratio=0.0,
        ),
    )
    sdk_session = agent.session(thread_id="session-cli-rollback", git_available=True)
    repository = SessionRepository(sdk_session.config.thread_store)
    coding = CodingSession.from_sdk_session(
        sdk_session,
        paths=paths,
        runtime_config=ConfigManager.load(paths, environment={}).current,
        repository=repository,
        model_loader=lambda _: None,
        vision=False,
    )

    async def exercise():
        events = [event async for event in coding.run_turn("create a file")]
        assert not any(event.kind == "error" for event in events)
        assert (paths.project_root / "generated.txt").exists()
        checkpoint = repository.checkpoint(coding.thread_id, 0)
        assert checkpoint is not None and checkpoint.modified_paths == ("*",)
        assert any(
            event.kind == "workspace_mutation"
            for event in repository.events(coding.thread_id)
        )
        result = await coding.rollback_to(0)
        assert result.ok, result.message
        assert not (paths.project_root / "generated.txt").exists()
        assert repository.raw_events(coding.thread_id) == []
        assert await coding.get_messages() == []
        await coding.close()

    asyncio.run(exercise())
