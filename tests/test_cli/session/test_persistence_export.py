from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest
from ness_agent import HookRunner, PermissionStore, ThreadStore

from ness_cli.session.coding_session import CodingSession
from ness_cli.session.export import (
    ExportError,
    export_thread_html,
    normalize_events,
    resolve_export_path,
)
from ness_cli.session.persistence import SessionRepository
from ness_cli.session.rollback import create_file_checkpoint


def repository(tmp_path):
    store = ThreadStore(threads_dir=tmp_path / ".ness" / "threads", auto_save=True)
    return store, SessionRepository(store)


class Memory:
    def __init__(self):
        self.values = {}

    def read_session_raw(self, thread_id):
        return self.values.get(thread_id, "")

    def write_session_raw(self, thread_id, value):
        self.values[thread_id] = value


class SdkSession:
    def __init__(self, tmp_path, thread_id="session-test"):
        self.thread_id = thread_id
        self.mode = "act"
        self.turn_count = 1
        self.finalize_calls = 0
        self.bootstrap_messages = []
        self.refresh_calls = 0
        self.cost_tracker = type("Cost", (), {"restore": lambda self, *args: None})()
        self.config = type(
            "Config",
            (),
            {
                "permission_store": PermissionStore(
                    ness_dir=tmp_path / ".ness", project_root=tmp_path
                ),
                "memory_store": Memory(),
            },
        )()

    async def finalize_reflection(self):
        self.finalize_calls += 1

    def reset_checkpointer(self):
        self.bootstrap_messages = []

    def bootstrap(self, messages):
        self.bootstrap_messages = list(messages)

    async def refresh_context_snapshot(self):
        self.refresh_calls += 1


def coding_session(tmp_path, isolated_cli_env, thread_id="session-test"):
    store = ThreadStore(threads_dir=tmp_path / "threads", auto_save=True)
    sdk = SdkSession(tmp_path, thread_id)
    coding = CodingSession(
        sdk,
        paths=isolated_cli_env.paths,
        runtime_config=object(),
        repository=SessionRepository(store),
        model_loader=lambda _: None,
        vision=False,
    )
    return sdk, store, coding


def test_repository_round_trip_checkpoint_truncate_and_prefix_copy(tmp_path):
    _, repo = repository(tmp_path)
    thread = "source"
    first = repo.append_user(thread, "first", images=("image",))
    assert first is not None
    repo.append(thread, {"kind": "assistant", "content": "answer"})
    second = repo.append_user(thread, "second")
    assert second is not None
    repo.save_checkpoint(thread, first, git_hash="HEAD", memory_snapshot="memory")
    repo.add_modified_path(thread, first, "src/app.py")

    checkpoint = repo.checkpoint(thread, first)
    assert checkpoint is not None
    assert checkpoint.git_hash == "HEAD"
    assert checkpoint.modified_paths == ("src/app.py",)
    assert [turn.content for turn in repo.user_turns(thread)] == ["first", "second"]

    copied = repo.copy_prefix(thread, "fork", second)
    assert [event["kind"] for event in copied] == ["user", "assistant"]
    assert [event["kind"] for event in repo.raw_events("fork")] == ["user", "assistant"]
    repo.truncate(thread, first)
    assert repo.events(thread) == []


def test_coding_session_save_archives_without_finalizing_reflection(
    tmp_path, isolated_cli_env
):
    sdk, store, coding = coding_session(tmp_path, isolated_cli_env)
    store.append_event(coding.thread_id, {"kind": "user", "content": "hello"})
    first = asyncio.run(coding.save())
    second = asyncio.run(coding.save())
    assert sdk.finalize_calls == 0
    assert first.resume_thread_id == coding.thread_id
    assert second.thread_id == first.thread_id


def test_coding_session_end_finalization_is_idempotent_for_unchanged_turn_count(
    tmp_path, isolated_cli_env
):
    sdk, store, coding = coding_session(tmp_path, isolated_cli_env)
    store.append_event(coding.thread_id, {"kind": "user", "content": "hello"})

    first = asyncio.run(coding.finalize_and_save())
    second = asyncio.run(coding.finalize_and_save())

    assert sdk.finalize_calls == 1
    assert first.resume_thread_id == coding.thread_id
    assert second.thread_id == first.thread_id


def test_coding_session_resume_replays_history_and_cost_once(
    tmp_path, isolated_cli_env
):
    sdk, store, coding = coding_session(tmp_path, isolated_cli_env)
    store.append_event(coding.thread_id, {"kind": "user", "content": "hello"})
    store.append_event(coding.thread_id, {"kind": "assistant", "content": "answer"})
    assert asyncio.run(coding.resume())
    assert [message.content for message in sdk.bootstrap_messages] == ["hello", "answer"]
    assert sdk.refresh_calls == 1


def test_coding_session_fork_copies_prefix_and_checkpoint_memory(
    tmp_path, isolated_cli_env
):
    sdk, store, coding = coding_session(tmp_path, isolated_cli_env)
    user_seq = store.append_event(coding.thread_id, {"kind": "user", "content": "fork prompt"})
    store.append_event(coding.thread_id, {"kind": "assistant", "content": "old answer"})
    store.save_checkpoint(coding.thread_id, user_seq, None, "checkpoint memory")
    result = asyncio.run(coding.fork_before(user_seq))
    assert result.prompt == "fork prompt" and result.copied_events == 0
    assert sdk.config.memory_store.values[result.thread_id] == "checkpoint memory"


def test_coding_session_rollback_without_git_preserves_memory_and_history(
    tmp_path, isolated_cli_env
):
    sdk, store, coding = coding_session(tmp_path, isolated_cli_env)
    user_seq = store.append_event(coding.thread_id, {"kind": "user", "content": "remove me"})
    store.append_event(coding.thread_id, {"kind": "assistant", "content": "also removed"})
    store.save_checkpoint(coding.thread_id, user_seq, None, "before turn")
    result = asyncio.run(coding.rollback_to(user_seq))
    assert not result.ok and "No Git snapshot" in result.warnings[0]
    assert coding.thread_id not in sdk.config.memory_store.values
    assert len(store.load_thread_events(coding.thread_id)) == 2
    assert sdk.refresh_calls == 0


@pytest.mark.parametrize("replay_fails", [False, True])
def test_coding_session_replays_retained_prefix_before_history_is_truncated(
    tmp_path, isolated_cli_env, replay_fails, monkeypatch,
):
    import subprocess

    project = isolated_cli_env.project
    subprocess.run(["git", "init", "--quiet"], cwd=project, check=True)
    (project / "app.txt").write_text("original")
    subprocess.run(["git", "add", "."], cwd=project, check=True)
    subprocess.run([
        "git", "-c", "user.name=Test", "-c", "user.email=test@example.test",
        "commit", "--quiet", "-m", "initial",
    ], cwd=project, check=True)
    sdk, store, coding = coding_session(tmp_path, isolated_cli_env)
    store.append_event(coding.thread_id, {"kind": "user", "content": "keep question"})
    store.append_event(coding.thread_id, {"kind": "assistant", "content": "keep answer"})
    seq = store.append_event(coding.thread_id, {"kind": "user", "content": "discard question"})
    store.append_event(coding.thread_id, {"kind": "assistant", "content": "discard answer"})
    store.save_checkpoint(coding.thread_id, seq, create_file_checkpoint(project), "before")
    events_before = store.load_thread_events(coding.thread_id)

    async def refresh():
        assert store.load_thread_events(coding.thread_id) == events_before
        if replay_fails:
            raise RuntimeError("replay failed")

    monkeypatch.setattr(sdk, "refresh_context_snapshot", refresh)
    result = asyncio.run(coding.rollback_to(seq))

    assert result.ok is not replay_fails
    assert [message.content for message in sdk.bootstrap_messages] == ["keep question", "keep answer"]
    assert store.load_thread_events(coding.thread_id) == (events_before if replay_fails else events_before[:2])
    assert (store.get_checkpoint(coding.thread_id, seq) is not None) is replay_fails


def test_export_keeps_pre_compaction_history_and_omits_image_payload(tmp_path):
    store, _ = repository(tmp_path)
    thread = "export"
    store.set_thread_name(thread, "Audit export")
    store.append_event(thread, {"kind": "user", "content": "old", "images": ["data:image/png;base64,SECRET"]})
    store.append_event(thread, {"kind": "assistant", "content": "</script><script>alert(1)</script>"})
    store.append_compaction_checkpoint(
        thread,
        {"response": "summary", "trigger": "automatic", "active_suffix": [{"type": "human", "data": {"content": "DUPLICATE"}}]},
        active_turn=False,
    )
    store.append_event(thread, {"kind": "user", "content": "new"})
    destination = tmp_path / "report.html"
    result = export_thread_html(
        thread_store=store,
        thread_id=thread,
        project_root=tmp_path,
        destination=destination,
        generated_at=datetime(2026, 8, 20, tzinfo=timezone.utc),
    )
    document = destination.read_text(encoding="utf-8")
    assert result.event_count == 4
    assert "old" in document and "new" in document
    assert "SECRET" not in document and "DUPLICATE" not in document
    assert "Content-Security-Policy" in document
    assert "<script>alert(1)</script>" not in document


def test_export_normalizes_domain_events_and_subagent_details():
    records = normalize_events(
        [
            {"kind": "approval", "tool": "shell", "decision": "yes"},
            {"kind": "reflection", "response": {"new_bullet_points": ["Remember"]}},
            {"kind": "goal", "phase": "start", "goal": "Ship"},
            {"kind": "tool", "tool": "spawn_subagent", "result": "done"},
        ],
        subagents=[{"agent_name": "review", "status": "ok", "output": "Looks good"}],
    )
    assert records[1].content == "• Remember"
    assert records[2].title == "Goal · start"
    assert records[3].details["subagents"][0]["output"] == "Looks good"
    assert all(record.seq == index for index, record in enumerate(records))


def test_export_path_and_safety_failures(tmp_path):
    assert resolve_export_path('"reports/My session.html"', tmp_path) == (
        tmp_path / "reports" / "My session.html"
    ).resolve()
    with pytest.raises(ExportError, match="must end in .html"):
        resolve_export_path("report.json", tmp_path)

    store, _ = repository(tmp_path)
    store.append_event("t", {"kind": "user", "content": "hello"})
    destination = tmp_path / "existing.html"
    destination.write_text("keep", encoding="utf-8")
    with pytest.raises(ExportError, match="Refusing to overwrite"):
        export_thread_html(thread_store=store, thread_id="t", project_root=tmp_path, destination=destination)
    assert destination.read_text(encoding="utf-8") == "keep"
    store.auto_save = False
    with pytest.raises(ExportError, match="autosave is disabled"):
        export_thread_html(thread_store=store, thread_id="t", project_root=tmp_path, destination=tmp_path / "new.html")


def test_rollback_restores_bulk_deletes_alongside_writes(tmp_path):
    import subprocess
    from ness_cli.session.rollback import RollbackService
    from ness_cli.session.mutations import WorkspaceMutations

    project = tmp_path / "rollback-repo"
    project.mkdir()

    def git(*args):
        subprocess.run(["git", *args], cwd=project, check=True, capture_output=True)

    git("init")
    for name in ("edited.txt", "deleted-a.txt", "deleted-b.txt", "unrelated.txt"):
        (project / name).write_text("before", encoding="utf-8")
    git("add", ".")
    git("-c", "user.name=Test", "-c", "user.email=test@example.test", "commit", "-m", "initial")
    store = ThreadStore(threads_dir=tmp_path / "threads", auto_save=True)
    repo = SessionRepository(store)
    memory = Memory()
    hooks = HookRunner(project_root=project)
    mutations = WorkspaceMutations.attach(hooks, project, repo)
    service = RollbackService(project_root=project, repository=repo, memory=memory, mutations=mutations)

    async def exercise():
        snapshot = await service.snapshot("thread")
        seq = repo.append_user("thread", "edit one file and delete two")
        service.save("thread", seq, snapshot)
        write_payload = {"tool": "write", "args": {"path": "edited.txt"}, "thread_id": "thread"}
        assert hooks.run("preToolUse", write_payload)[0]
        (project / "edited.txt").write_text("after", encoding="utf-8")
        assert hooks.run("postToolUse", write_payload)[0]
        deleted = ["deleted-a.txt", "deleted-b.txt"]
        delete_payload = {"tool": "delete", "args": {"paths": deleted}, "thread_id": "thread"}
        assert hooks.run("preToolUse", delete_payload)[0]
        for name in deleted:
            (project / name).unlink()
        assert hooks.run("postToolUse", delete_payload)[0]
        repo.append("thread", {
            "kind": "tool", "tool": "write", "args": {"path": "edited.txt"}, "result": "ok"
        })
        repo.append("thread", {
            "kind": "tool", "tool": "delete", "args": {"paths": deleted}, "result": "ok"
        })
        service.record_mutations("thread", seq)
        # A later unrelated user edit must survive the targeted restore.
        (project / "unrelated.txt").write_text("user edit", encoding="utf-8")

        async def replay(replay_cost, before_seq):
            assert replay_cost is False
            assert before_seq == seq
            return True

        result = await service.rollback("thread", seq, replay=replay)
        assert result.ok and not result.warnings
        for name in ("edited.txt", *deleted):
            assert (project / name).read_text(encoding="utf-8") == "before"
        assert (project / "unrelated.txt").read_text(encoding="utf-8") == "user edit"
        assert repo.raw_events("thread") == []

    asyncio.run(exercise())
