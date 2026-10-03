from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from ness_agent import NessAgent, NessAgentOptions, PromptLayersConfig, ThreadStore


@pytest.mark.parametrize("initial", [True, False])
def test_store_views_keep_independent_policy_and_share_existing_data(tmp_path, initial):
    store = ThreadStore(tmp_path / "threads", auto_save=initial, default_model="test")
    inherited = store.fork_for_session()
    enabled = store.fork_for_session(auto_save=True)
    disabled = enabled.fork_for_session(auto_save=False)

    assert not store.threads_db.exists()
    assert inherited.auto_save is initial
    assert inherited.default_model == "test"
    assert (
        enabled.append_event("session-shared", {"kind": "user", "content": "saved"})
        == 0
    )
    assert (
        disabled.append_event("session-shared", {"kind": "user", "content": "skipped"})
        is None
    )
    assert store.auto_save is initial
    assert len(disabled.load_thread_events("session-shared")) == 1
    disabled.auto_save = True
    assert (
        disabled.append_event(
            "session-shared", {"kind": "assistant", "content": "answer"}
        )
        == 1
    )
    enabled.auto_save = False
    assert inherited.auto_save is initial
    assert disabled.auto_save is True
    assert len(store.load_thread_events("session-shared")) == 2


def test_concurrent_store_views_share_writer_lock_and_allocate_unique_sequences(
    tmp_path,
):
    store = ThreadStore(tmp_path / "threads")
    views = [store.fork_for_session() for _ in range(4)]
    assert all(view._write_lock is store._write_lock for view in views)

    def append(index):
        return views[index % len(views)].append_event(
            "session-shared", {"kind": "user", "content": str(index)}
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        sequences = list(executor.map(append, range(20)))

    assert sorted(sequences) == list(range(20))
    assert {row["content"] for row in store.load_thread_events("session-shared")} == {
        str(index) for index in range(20)
    }


@pytest.mark.parametrize("initial", [True, False])
def test_sdk_sessions_snapshot_autosave_defaults_without_mutating_open_sessions(
    tmp_path, initial
):
    agent = NessAgent(
        model=FakeListChatModel(responses=["ok"]),
        tools=[],
        prompt=PromptLayersConfig(l0="Test"),
        skills_dirs=[],
        options=NessAgentOptions(
            project_root=tmp_path,
            ness_dir=tmp_path / ".ness",
            auto_save_threads=initial,
        ),
    )
    first = agent.session(thread_id="session-first")
    agent.config.options.auto_save_threads = not initial
    second = agent.session(thread_id="session-second")

    assert first.config.options.auto_save_threads is initial
    assert first.config.thread_store.auto_save is initial
    assert second.config.options.auto_save_threads is (not initial)
    assert second.config.thread_store.auto_save is (not initial)
    assert agent.config.thread_store.auto_save is initial
    assert (
        first.config.thread_store.append_event(first.thread_id, {"kind": "user"})
        is not None
    ) is initial
    assert (
        second.config.thread_store.append_event(second.thread_id, {"kind": "user"})
        is not None
    ) is (not initial)
