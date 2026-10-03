import asyncio
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage

from ness_agent import (
    AgentSpec,
    ApprovalHandler,
    NessAgent,
    NessAgentOptions,
    PromptLayersConfig,
)
from ness_agent.permissions import PermissionStore
from ness_agent.tools.fs import glob, read
from ness_agent.tools.search import grep
from tests.sdk_fixtures import installed_session_context


class CallsModel:
    def __init__(self, batches):
        self.responses = []
        for batch in batches:
            self.responses.extend(
                [AIMessage(content="", tool_calls=batch), AIMessage(content="done")]
            )

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages, **kwargs):
        return self.responses.pop(0)


class Approvals(ApprovalHandler):
    def __init__(self, decisions, before=None):
        self.decisions = list(decisions)
        self.calls = []
        self.before = before

    async def __call__(self, name, args):
        self.calls.append((name, args))
        if self.before:
            self.before()
        return self.decisions.pop(0)


def call(name, path, *, id="call", **args):
    if name == "delete":
        args["paths"] = [str(path)]
    else:
        args["path"] = str(path)
    return {"name": name, "args": args, "id": id}


def agent(project, batches, handler=None, *, yolo=False, enable_approval=True):
    return NessAgent.from_spec(
        AgentSpec(
            model=CallsModel(batches),
            prompt=PromptLayersConfig(l0="test"),
            tools=["read", "write", "edit", "delete", "grep", "glob"],
            approval_handler=handler,
            options=NessAgentOptions(
                project_root=project,
                ness_dir=project / ".ness",
                yolo_mode=yolo,
                enable_approval=enable_approval,
                format_on_write=False,
                auto_save_threads=False,
            ),
        )
    )


def results(turn):
    assert not any(event.kind == "error" for event in turn.events)
    return "\n".join(
        event.data["content"] for event in turn.events if event.kind == "tool_end"
    )


@pytest.mark.parametrize(
    "operation", ["read", "write", "edit", "delete", "grep", "glob"]
)
def test_external_file_tools_request_approval_and_then_execute(tmp_path, operation):
    project = tmp_path / "app"
    project.mkdir()
    outside = tmp_path / "reference"
    outside.mkdir()
    source = outside / "source.txt"
    source.write_text("original")
    target = outside if operation in {"grep", "glob"} else source
    extra = {
        "write": {"content": "changed"},
        "edit": {"old_string": "original", "new_string": "changed"},
        "grep": {"pattern": "original"},
        "glob": {"pattern": "*.txt"},
    }.get(operation, {})
    handler = Approvals(["yes"])
    session = agent(project, [[call(operation, target, **extra)]], handler).session(
        thread_id="external"
    )
    try:
        output = results(asyncio.run(session.run("use file")))
        assert "Error:" not in output
        assert len(handler.calls) == 1
        assert handler.calls[0][1]["filesystem_access"] == [
            {
                "path": str(target),
                "write": operation in {"write", "edit", "delete"},
                "recursive": operation in {"grep", "glob"},
            }
        ]
        if operation in {"write", "edit"}:
            assert source.read_text() == "changed"
        elif operation == "delete":
            assert not source.exists()
        elif operation == "glob":
            assert str(source) in output
        else:
            assert "original" in output
        # The approval was for this call, not an enduring path grant.
        with pytest.raises(PermissionError):
            session.config.permission_store.validate_path(str(target))
    finally:
        asyncio.run(session.close())


def test_once_approval_does_not_allow_next_external_write(tmp_path):
    project = tmp_path / "app"
    project.mkdir()
    target = tmp_path / "outside.txt"
    handler = Approvals(["yes", "no"])
    session = agent(
        project,
        [
            [call("write", target, content="first")],
            [call("write", target, content="second", id="second")],
        ],
        handler,
    ).session(thread_id="once")
    try:
        assert "Wrote" in results(asyncio.run(session.run("first")))
        assert "Denied by user" in results(asyncio.run(session.run("second")))
        assert target.read_text() == "first"
        assert len(handler.calls) == 2
    finally:
        asyncio.run(session.close())


def test_session_approval_remembers_exact_path_and_does_not_leak_to_sibling(tmp_path):
    project = tmp_path / "app"
    project.mkdir()
    source = tmp_path / "outside.txt"
    source.write_text("outside")
    handler = Approvals(["session", "no"])
    shared = agent(
        project,
        [
            [call("read", source)],
            [call("read", source, id="again")],
            [call("read", source, id="sibling")],
        ],
        handler,
    )
    first = shared.session(thread_id="first")
    second = shared.session(thread_id="second")
    try:
        assert "outside" in results(asyncio.run(first.run("first")))
        assert "outside" in results(asyncio.run(first.run("again")))
        assert len(handler.calls) == 1
        assert "Denied by user" in results(asyncio.run(second.run("sibling")))
        with pytest.raises(PermissionError):
            first.config.permission_store.validate_path(str(tmp_path / "other.txt"))
    finally:
        asyncio.run(first.close())
        asyncio.run(second.close())


def test_read_grant_does_not_allow_external_mutation(tmp_path):
    project = tmp_path / "app"
    project.mkdir()
    source = tmp_path / "outside.txt"
    source.write_text("keep")
    handler = Approvals(["session", "no"])
    session = agent(
        project,
        [[call("read", source)], [call("write", source, content="bad", id="write")]],
        handler,
    ).session(thread_id="readonly")
    try:
        results(asyncio.run(session.run("read")))
        assert "Denied by user" in results(asyncio.run(session.run("write")))
        assert source.read_text() == "keep"
        assert len(handler.calls) == 2
    finally:
        asyncio.run(session.close())


@pytest.mark.parametrize("decision", ["always", "never"])
def test_remembered_filesystem_decision_survives_new_agent(tmp_path, decision):
    project = tmp_path / "app"
    project.mkdir()
    source = tmp_path / "outside[1].txt"
    source.write_text("outside")
    first = agent(project, [[call("read", source)]], Approvals([decision])).session(
        thread_id="first"
    )
    try:
        results(asyncio.run(first.run("first")))
    finally:
        asyncio.run(first.close())
    handler = Approvals([])
    second = agent(project, [[call("read", source, offset=1)]], handler).session(
        thread_id="new-agent"
    )
    try:
        output = results(asyncio.run(second.run("again")))
        assert ("Denied by permission" in output) == (decision == "never")
        assert not handler.calls
    finally:
        asyncio.run(second.close())


def test_denied_or_unhandled_access_cannot_modify_external_file(tmp_path):
    project = tmp_path / "app"
    project.mkdir()
    target = tmp_path / "outside.txt"
    for handler in (None, Approvals(["no"])):
        session = agent(
            project, [[call("write", target, content="bad")]], handler
        ).session(thread_id="denied")
        try:
            output = results(asyncio.run(session.run("write")))
            assert "Approval required" in output or "Denied by user" in output
            assert not target.exists()
        finally:
            asyncio.run(session.close())


def test_approval_disabled_does_not_implicitly_grant_external_access(tmp_path):
    project = tmp_path / "app"
    project.mkdir()
    target = tmp_path / "outside.txt"
    session = agent(
        project, [[call("write", target, content="bad")]], enable_approval=False
    ).session(thread_id="no-prompts")
    try:
        assert "requires approval" in results(asyncio.run(session.run("write")))
        assert not target.exists()
    finally:
        asyncio.run(session.close())


def test_yolo_bypasses_external_and_protected_file_access_without_approval(tmp_path):
    project = tmp_path / "app"
    project.mkdir()
    outside = tmp_path / "outside.txt"
    runtime = project / ".ness" / "test-state.txt"
    git_state = project / ".git" / "test-state.txt"
    handler = Approvals([])
    batches = [
        [call("write", target, content="initial", id=str(i))]
        for i, target in enumerate((outside, runtime, git_state))
    ]
    batches.extend(
        [
            [call("read", outside)],
            [call("edit", runtime, old_string="initial", new_string="edited")],
            [call("delete", git_state)],
        ]
    )
    session = agent(project, batches, handler, yolo=True).session(thread_id="yolo")
    session.config.permission_store.persist_rule("write:*", "deny", scope="session")
    try:
        for _ in batches:
            assert "Error:" not in results(asyncio.run(session.run("use file")))
        assert outside.read_text() == "initial"
        assert runtime.read_text() == "edited"
        assert not git_state.exists()
        assert not handler.calls
    finally:
        asyncio.run(session.close())


def test_symlink_retargeting_does_not_reuse_old_approval(tmp_path):
    project = tmp_path / "app"
    project.mkdir()
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("first")
    second.write_text("second")
    link = project / "link.txt"
    link.symlink_to(first)

    def retarget():
        link.unlink()
        link.symlink_to(second)

    handler = Approvals(["yes"], before=retarget)
    session = agent(project, [[call("write", link, content="bad")]], handler).session(
        thread_id="retarget"
    )
    try:
        assert "requires approval" in results(asyncio.run(session.run("write")))
        assert first.read_text() == "first"
        assert second.read_text() == "second"
        assert handler.calls[0][1]["filesystem_access"][0]["path"] == str(first)
    finally:
        asyncio.run(session.close())


def test_internal_paths_work_normally_and_protected_state_remains_blocked(tmp_path):
    handler = Approvals([])
    session = agent(
        tmp_path,
        [
            [call("write", tmp_path / "file.txt", content="ok")],
            [call("write", tmp_path / ".ness" / "state.txt", content="bad")],
        ],
        handler,
    ).session(thread_id="normal")
    try:
        assert "Wrote" in results(asyncio.run(session.run("write")))
        assert "protected" in results(asyncio.run(session.run("state")))
        assert not handler.calls
    finally:
        asyncio.run(session.close())


def test_temporary_grant_cleans_up_after_exception_and_recursive_read_stays_read_only(
    tmp_path,
):
    project = tmp_path / "app"
    project.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    source = external / "source.txt"
    source.write_text("outside")
    store = PermissionStore(project_root=project, ness_dir=project / ".ness")
    request = store.file_access_requests("grep", {"path": str(external)})
    with pytest.raises(RuntimeError):
        with store.file_access(request):
            assert store.validate_path(str(source)) == str(source)
            with pytest.raises(PermissionError):
                store.validate_path(str(source), write=True)
            raise RuntimeError("failed tool")
    with pytest.raises(PermissionError):
        store.validate_path(str(source))


def test_yolo_allows_symlink_reads_and_python_search(tmp_path):
    project = tmp_path / "app"
    project.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside")
    (project / "escape.txt").symlink_to(outside)
    with installed_session_context(project) as ctx:
        ctx.options.yolo_mode = True
        assert "outside" in read.invoke({"path": "escape.txt"})
        assert str(outside) in glob.invoke({"pattern": "*.txt"})
        with patch("ness_agent.tools.search.shutil.which", return_value=None):
            assert "outside" in grep.invoke({"pattern": "outside"})


def test_plan_batch_prompts_for_external_read_and_blocks_write(tmp_path):
    project = tmp_path / "app"
    project.mkdir()
    source = tmp_path / "outside.txt"
    source.write_text("reference")
    target = project / "blocked.txt"
    handler = Approvals(["yes"])
    session = agent(
        project,
        [[call("read", source), call("write", target, content="bad", id="write")]],
        handler,
    ).session(thread_id="plan", mode="plan")
    try:
        assert "reference" in results(asyncio.run(session.run("inspect")))
        assert len(handler.calls) == 1
        assert handler.calls[0][0] == "read"
        assert not target.exists()
    finally:
        asyncio.run(session.close())


def test_batch_approval_does_not_leak_to_declined_sibling(tmp_path):
    project = tmp_path / "app"
    project.mkdir()
    target = tmp_path / "outside.txt"
    handler = Approvals(["yes", "no"])
    session = agent(
        project,
        [
            [
                call("write", target, content="first"),
                call("write", target, content="bad", id="denied"),
            ]
        ],
        handler,
    ).session(thread_id="siblings")
    try:
        output = results(asyncio.run(session.run("write")))
        assert "Wrote" in output and "Denied by user" in output
        assert target.read_text() == "first"
        assert len(handler.calls) == 2
    finally:
        asyncio.run(session.close())


def test_external_mentions_follow_yolo_and_remembered_read_grants(tmp_path):
    from ness_cli.session.mentions import expand_documents
    from ness_cli.session.replay import events_to_messages

    project = tmp_path / "app"
    project.mkdir()
    source = tmp_path / "outside.txt"
    source.write_text("reference content")
    store = PermissionStore(project_root=project, ness_dir=project / ".ness")
    text = f"inspect @{source}"
    assert "requires approval" in expand_documents(text, store)
    assert "reference content" in expand_documents(text, store, yolo_mode=True)
    replay = events_to_messages(
        [{"kind": "user", "content": text}], permission_store=store, yolo_mode=True
    )
    assert "reference content" in replay[0].content
    store.remember_file_access(
        store.file_access_requests("read", {"path": str(source)}),
        allow=True,
        scope="session",
    )
    assert "reference content" in expand_documents(text, store)


def test_exact_path_grant_does_not_authorize_directory_search(tmp_path):
    project = tmp_path / "app"
    project.mkdir()
    directory = tmp_path / "external"
    directory.mkdir()
    store = PermissionStore(project_root=project, ness_dir=project / ".ness")
    store.remember_file_access(
        store.file_access_requests("read", {"path": str(directory)}),
        allow=True,
        scope="session",
    )
    assert store.file_access_requests("grep", {"path": str(directory)})[0]["recursive"]
    with pytest.raises(PermissionError):
        store.validate_path(str(directory), recursive=True)


def test_recursive_search_respects_previously_denied_child(tmp_path):
    project = tmp_path / "app"
    project.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    visible = external / "visible.txt"
    denied = external / "denied.txt"
    visible.write_text("search-visible")
    denied.write_text("search-secret")
    handler = Approvals(["yes"])
    session = agent(
        project, [[call("grep", external, pattern="search")]], handler
    ).session(thread_id="search")
    store = session.config.permission_store
    store.remember_file_access(
        store.file_access_requests("read", {"path": str(denied)}),
        allow=False,
        scope="session",
    )
    try:
        output = results(asyncio.run(session.run("search")))
        assert "search-visible" in output
        assert "search-secret" not in output
    finally:
        asyncio.run(session.close())
