from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage

from ness_agent import NessAgent, NessAgentOptions, PromptLayers, PromptLayersConfig
from ness_agent.skills import SkillLoader, default_skill_search_dirs, merge_skill_dirs


def _write_skill(skill_dir: Path, name: str, description: str, body: str = "Body") -> None:
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n{body}\n"
    )


def _make_agent(tmp_path: Path, **kwargs) -> NessAgent:
    return NessAgent(
        model=FakeListChatModel(responses=["ok"]),
        prompt=PromptLayers(PromptLayersConfig(l0="L0")),
        options=NessAgentOptions(project_root=tmp_path, ness_dir=tmp_path / ".ness"),
        **kwargs,
    )


def test_skill_loader_skips_invalid_name(tmp_path: Path):
    """Missing or empty frontmatter name is skipped; dir name is not a fallback."""
    skills_root = tmp_path / ".ness" / "skills"
    missing = skills_root / "no_name"
    missing.mkdir(parents=True)
    (missing / "SKILL.md").write_text("---\ndescription: No name here\n---\nBody\n")
    empty = skills_root / "empty_name"
    empty.mkdir(parents=True)
    (empty / "SKILL.md").write_text("---\nname: ''\ndescription: Empty name\n---\nBody\n")
    dir_only = skills_root / "dir_name_skill"
    dir_only.mkdir(parents=True)
    (dir_only / "SKILL.md").write_text(
        "---\ndescription: Dir name should not become name\n---\nBody\n"
    )

    loader = SkillLoader(skills_root)
    assert loader.load() == {}


def test_skill_loader_skips_invalid_description(tmp_path: Path):
    """Missing/empty description is skipped; body heading is not a fallback."""
    skills_root = tmp_path / ".ness" / "skills"
    missing = skills_root / "no_desc"
    missing.mkdir(parents=True)
    (missing / "SKILL.md").write_text("---\nname: no_desc\n---\nBody\n")
    empty = skills_root / "empty_desc"
    empty.mkdir(parents=True)
    (empty / "SKILL.md").write_text("---\nname: empty_desc\ndescription: ''\n---\nBody\n")
    body_heading = skills_root / "skill_x"
    body_heading.mkdir(parents=True)
    (body_heading / "SKILL.md").write_text(
        "---\nname: skill_x\n---\n# Heading should not become description\n"
    )

    loader = SkillLoader(skills_root)
    assert loader.load() == {}


def test_skill_loader_loads_multiple_skills_skips_invalid(tmp_path: Path):
    skills_root = tmp_path / ".ness" / "skills"
    (skills_root / "valid").mkdir(parents=True)
    (skills_root / "valid" / "SKILL.md").write_text(
        "---\nname: good\ndescription: Good skill\n---\nGood body\n"
    )
    (skills_root / "invalid").mkdir(parents=True)
    (skills_root / "invalid" / "SKILL.md").write_text(
        "---\ndescription: Missing name\n---\nBad body\n"
    )

    loader = SkillLoader(skills_root)
    skills = loader.load()
    assert list(skills) == ["good"]


def test_render_catalog_empty_when_no_skills():
    loader = SkillLoader()
    assert loader.render_catalog({}) == ""


def test_skill_loader_disabled_when_no_dirs():
    assert SkillLoader().load() == {}
    assert SkillLoader(skills_dirs=None).load() == {}
    assert SkillLoader(skills_dirs=[]).load() == {}


def test_skill_loader_multi_root(tmp_path: Path):
    ness = tmp_path / ".ness" / "skills"
    agents = tmp_path / ".agents" / "skills"
    _write_skill(ness / "alpha", "alpha", "From ness")
    _write_skill(agents / "beta", "beta", "From agents")

    loader = SkillLoader(skills_dirs=[ness, agents])
    skills = loader.load()
    assert set(skills) == {"alpha", "beta"}
    assert "ness" in skills["alpha"]["source"]
    assert "agents" in skills["beta"]["source"]


def test_skill_loader_nested_category(tmp_path: Path):
    root = tmp_path / ".agents" / "skills"
    _write_skill(root / "product-a" / "skill-one", "skill_one", "Nested one")
    _write_skill(root / "product-a" / "skill-two", "skill_two", "Nested two")
    _write_skill(root / "flat-skill", "flat", "Flat skill")

    skills = SkillLoader(root).load()
    assert set(skills) == {"skill_one", "skill_two", "flat"}


def test_skill_loader_shadowing_does_not_descend(tmp_path: Path):
    root = tmp_path / ".agents" / "skills"
    foo = root / "foo"
    _write_skill(foo, "outer", "Outer skill", body="Outer body")
    _write_skill(foo / "inner", "inner", "Inner skill", body="Inner body")
    (foo / "scripts").mkdir()
    (foo / "scripts" / "run.sh").write_text("echo hi")

    skills = SkillLoader(root).load()
    assert set(skills) == {"outer"}
    assert "Outer body" in skills["outer"]["body"]


def test_skill_loader_user_dir_wins_name_collision(tmp_path: Path):
    ness = tmp_path / ".ness" / "skills"
    agents = tmp_path / ".agents" / "skills"
    _write_skill(ness / "shared", "shared", "Ness wins", body="from-ness")
    _write_skill(agents / "shared", "shared", "Agents loses", body="from-agents")

    skills = SkillLoader(skills_dirs=[ness, agents]).load()
    assert skills["shared"]["body"] == "from-ness"
    assert "ness" in skills["shared"]["source"]


def test_skill_loader_snapshot_preserves_duplicates_and_falls_back(tmp_path: Path):
    first = tmp_path / ".agents" / "skills"
    second = tmp_path / ".claude" / "skills"
    _write_skill(first / "shared", "shared", "First", body="first")
    _write_skill(second / "shared", "shared", "Second", body="second")

    loader = SkillLoader(skills_dirs=[first, second])
    discovered = loader.discover()

    assert [skill["name"] for skill in discovered] == ["shared", "shared"]
    loader.set_snapshot(
        discovered,
        disabled_skill_ids=[SkillLoader.skill_id(discovered[0])],
    )
    assert loader.load()["shared"]["body"] == "second"


def test_skill_loader_collapses_exact_bundle_copies(tmp_path: Path):
    first = tmp_path / ".codex" / "skills" / "browser-use"
    second = tmp_path / ".cursor" / "skills" / "browser-use"
    _write_skill(first, "browser-use", "Browser control", body="instructions")
    _write_skill(second, "browser-use", "Browser control", body="instructions")
    (first / "scripts").mkdir()
    (second / "scripts").mkdir()
    (first / "scripts" / "run.py").write_text("print('run')\n")
    (second / "scripts" / "run.py").write_text("print('run')\n")

    loader = SkillLoader(skills_dirs=[first.parent, second.parent])
    discovered = loader.discover()

    assert len(discovered) == 1
    assert discovered[0]["skill_id"].startswith("sha256:")
    assert [
        source["source"] for source in SkillLoader.sources(discovered[0])
    ] == [str(first / "SKILL.md"), str(second / "SKILL.md")]
    loader.set_snapshot(
        discovered,
        disabled_skill_ids=[SkillLoader.skill_id(discovered[0])],
    )
    assert loader.load() == {}


def test_skill_loader_keeps_same_markdown_with_different_resources(tmp_path: Path):
    first = tmp_path / ".codex" / "skills" / "browser-use"
    second = tmp_path / ".cursor" / "skills" / "browser-use"
    _write_skill(first, "browser-use", "Browser control", body="instructions")
    _write_skill(second, "browser-use", "Browser control", body="instructions")
    (first / "scripts").mkdir()
    (second / "scripts").mkdir()
    (first / "scripts" / "run.py").write_text("print('first')\n")
    (second / "scripts" / "run.py").write_text("print('second')\n")

    discovered = SkillLoader(
        skills_dirs=[first.parent, second.parent]
    ).discover()

    assert len(discovered) == 2
    assert len({skill["skill_id"] for skill in discovered}) == 2


def test_skill_catalog_moves_to_one_shot_l3_without_changing_system_prefix(
    tmp_path: Path,
):
    class RecordingModel:
        model = "recording"

        def __init__(self) -> None:
            self.calls = []

        def bind_tools(self, _tools):
            return self

        async def ainvoke(self, messages, **_kwargs):
            self.calls.append(list(messages))
            return AIMessage(content="ok")

    skills_root = tmp_path / ".agents" / "skills"
    _write_skill(skills_root / "alpha", "alpha", "Alpha skill")
    model = RecordingModel()
    agent = NessAgent(
        model=model,
        tools=[],
        prompt=PromptLayers(
            PromptLayersConfig(
                l0="L0",
                persona="P",
                include_skill_catalog=False,
            )
        ),
        options=NessAgentOptions(
            project_root=tmp_path,
            ness_dir=tmp_path / ".ness",
        ),
        skills_dir=skills_root,
    )
    session = agent.session(thread_id="session-skills", git_available=False)

    asyncio.run(session.run("one"))
    asyncio.run(session.run("two"))

    assert "Skill catalog" not in str(model.calls[0][0].content)
    first_catalogs = [
        message
        for message in model.calls[0]
        if "Current effective skill catalog" in str(message.content)
    ]
    second_catalogs = [
        message
        for message in model.calls[1]
        if "Current effective skill catalog" in str(message.content)
    ]
    assert len(first_catalogs) == len(second_catalogs) == 1
    assert "alpha" in str(first_catalogs[0].content)

    skill_id = SkillLoader.skill_id(session.config.skill_loader.all_skills()[0])
    session.set_skill_access([skill_id])
    asyncio.run(session.run("three"))

    third_catalogs = [
        message
        for message in model.calls[2]
        if "Current effective skill catalog" in str(message.content)
    ]
    assert len(third_catalogs) == 2
    assert "No skills are currently available" in str(third_catalogs[-1].content)
    assert [call[0].content for call in model.calls] == [model.calls[0][0].content] * 3
    assert model.calls[2][: len(model.calls[1])] == model.calls[1]


def test_skill_loader_project_wins_over_global(tmp_path: Path, monkeypatch):
    project = tmp_path / "proj"
    home = tmp_path / "home"
    project.mkdir()
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    project_agents = project / ".agents" / "skills"
    global_agents = home / ".agents" / "skills"
    _write_skill(project_agents / "shared", "shared", "Project", body="project")
    _write_skill(global_agents / "shared", "shared", "Global", body="global")
    _write_skill(global_agents / "only_global", "only_global", "Global only")

    dirs = default_skill_search_dirs(project)
    skills = SkillLoader(skills_dirs=dirs).load()
    assert skills["shared"]["body"] == "project"
    assert "only_global" in skills


@pytest.mark.parametrize("link_scope", ["root", "skill"])
def test_skill_loader_symlink_dedupes_resolved_path(tmp_path: Path, link_scope):
    canonical = tmp_path / ".agents" / "skills"
    linked_root = tmp_path / ".claude" / "skills"
    _write_skill(canonical / "dup", "dup", "Canonical", body="once")
    linked_root.parent.mkdir(parents=True, exist_ok=True)
    if link_scope == "root":
        linked_root.symlink_to(canonical, target_is_directory=True)
    else:
        linked_root.mkdir()
        (linked_root / "dup").symlink_to(canonical / "dup", target_is_directory=True)

    skills = SkillLoader(skills_dirs=[canonical, linked_root]).load()
    assert list(skills) == ["dup"]


def test_merge_skill_dirs_order_and_dedupe(tmp_path: Path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    project = tmp_path / "proj"
    project.mkdir()
    user = project / ".ness" / "skills"

    dirs = merge_skill_dirs(project, user)
    assert dirs[0] == user
    assert dirs[1] == (project / ".agents" / "skills")
    assert (home / ".agents" / "skills") in dirs
    # No duplicate resolved paths
    resolved = [p.resolve() for p in dirs]
    assert len(resolved) == len(set(resolved))


def test_merge_skill_dirs_global_rels_filter(tmp_path: Path, monkeypatch):
    """global_rels restricts user-global roots; project roots stay complete."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    project = tmp_path / "proj"
    project.mkdir()
    user = project / ".ness" / "skills"

    dirs = merge_skill_dirs(project, user, global_rels=(".agents/skills",))
    assert dirs[0] == user
    # Project-local well-known roots are all still present
    for rel in (".agents", ".claude", ".codex", ".cursor"):
        assert (project / rel / "skills") in dirs
    # Only the chosen global root is included
    assert (home / ".agents" / "skills") in dirs
    for rel in (".claude", ".codex", ".cursor"):
        assert (home / rel / "skills") not in dirs


def test_merge_skill_dirs_project_rels_filter(tmp_path: Path, monkeypatch):
    """project_rels restricts project-local roots; global roots stay complete."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    project = tmp_path / "proj"
    project.mkdir()
    user = project / ".ness" / "skills"

    dirs = merge_skill_dirs(
        project, user, project_rels=(".agents/skills", "skills"), global_rels=()
    )
    assert dirs[0] == user
    # Only the chosen project-local roots are included
    assert (project / ".agents" / "skills") in dirs
    assert (project / "skills") in dirs
    for rel in (".claude", ".codex", ".cursor"):
        assert (project / rel / "skills") not in dirs
    # Empty global_rels opts out of user-global roots entirely
    for rel in (".agents", ".claude", ".codex", ".cursor"):
        assert (home / rel / "skills") not in dirs


def test_agent_skills_dir_scans_exactly_that_dir(tmp_path: Path):
    """A bare skills_dir is exact — the SDK adds no well-known roots."""
    mine = tmp_path / "my-skills"
    _write_skill(mine / "mine", "mine", "Explicit dir skill")
    _write_skill(tmp_path / ".agents" / "skills" / "sneaky", "sneaky", "Well-known root skill")

    agent = _make_agent(tmp_path, skills_dir=mine)

    assert agent.config.skills_dir == mine
    assert agent.config.skills_dirs == [mine]
    assert agent.config.skill_loader.skills_dirs == [mine]
    assert set(agent.config.skill_loader.load()) == {"mine"}


def test_agent_skills_dirs_used_verbatim(tmp_path: Path):
    """skills_dirs is the exhaustive root list — nothing is appended."""
    a = tmp_path / "a-skills"
    b = tmp_path / "b-skills"
    _write_skill(a / "one", "one", "From a")
    _write_skill(b / "two", "two", "From b")
    _write_skill(tmp_path / ".claude" / "skills" / "sneaky", "sneaky", "Well-known root skill")

    agent = _make_agent(tmp_path, skills_dirs=[a, b])

    assert agent.config.skills_dir == a  # primary root = first entry
    assert agent.config.skills_dirs == [a, b]
    assert agent.config.skill_loader.skills_dirs == [a, b]
    assert set(agent.config.skill_loader.load()) == {"one", "two"}


def test_agent_skills_dir_and_skills_dirs_conflict(tmp_path: Path):
    with pytest.raises(ValueError, match="skills_dir"):
        _make_agent(tmp_path, skills_dir=tmp_path / "x", skills_dirs=[tmp_path / "y"])


def test_agent_skills_disabled_by_default(tmp_path: Path):
    """No skills_dir / skills_dirs → no skill scanning at all."""
    _write_skill(tmp_path / ".agents" / "skills" / "sneaky", "sneaky", "Well-known root skill")

    agent = _make_agent(tmp_path)

    assert agent.config.skills_dir is None
    assert agent.config.skills_dirs is None
    assert agent.config.skill_loader.load() == {}


def test_skill_view_returns_skill_content(tmp_path: Path):
    from ness_agent.session_context import SessionContext, set_session_context, reset_session_context

    skill_dir = tmp_path / ".ness" / "skills" / "view_me"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: view_me\ndescription: View this\n---\n# View Me\n\nContent here.\n"
    )

    loader = SkillLoader(tmp_path / ".ness" / "skills")
    all_skills = loader.load()

    rt = SessionContext(
        permissions=MagicMock(),
        options=MagicMock(),
        thread_store=MagicMock(),
        ness_dir=tmp_path / ".ness",
        project_root=tmp_path,
        agent_config=MagicMock(),
        available_skills=all_skills,
    )
    token = set_session_context(rt)
    try:
        from ness_agent.tools.skill import skill_view

        result = skill_view.invoke({"name": "view_me"})
        data = json.loads(result)
        assert "View Me" in data["content"]
        assert "Content here." in data["content"]
        assert isinstance(data["linked_files"], dict)
        assert data["usage_hint"] == "To view linked files, call read(path=...) tool"
    finally:
        reset_session_context(token)


def test_skill_view_returns_linked_files(tmp_path: Path):
    from ness_agent.session_context import SessionContext, set_session_context, reset_session_context

    skill_dir = tmp_path / ".ness" / "skills" / "linked"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: linked\ndescription: Has linked files\n---\nBody\n")

    ref_dir = skill_dir / "references"
    ref_dir.mkdir()
    (ref_dir / "guide.md").write_text("# Guide")
    (ref_dir / "sources.md").write_text("# Sources")

    script_dir = skill_dir / "scripts"
    script_dir.mkdir()
    (script_dir / "build.sh").write_text("echo build")

    loader = SkillLoader(tmp_path / ".ness" / "skills")
    all_skills = loader.load()

    rt = SessionContext(
        permissions=MagicMock(),
        options=MagicMock(),
        thread_store=MagicMock(),
        ness_dir=tmp_path / ".ness",
        project_root=tmp_path,
        agent_config=MagicMock(),
        available_skills=all_skills,
    )
    token = set_session_context(rt)
    try:
        from ness_agent.tools.skill import skill_view

        result = skill_view.invoke({"name": "linked"})
        data = json.loads(result)
        lf = data["linked_files"]
        assert "references" in lf
        assert "scripts" in lf
        assert "templates" not in lf
        assert "assets" not in lf
        assert len(lf["references"]) == 2
        assert any("guide.md" in p for p in lf["references"])
        assert any("sources.md" in p for p in lf["references"])
        assert any("build.sh" in p for p in lf["scripts"])
        for p in lf["references"]:
            assert Path(p).is_absolute()
    finally:
        reset_session_context(token)


def test_skill_view_unknown_skill(tmp_path: Path):
    from ness_agent.session_context import SessionContext, set_session_context, reset_session_context

    rt = SessionContext(
        permissions=MagicMock(),
        options=MagicMock(),
        thread_store=MagicMock(),
        ness_dir=tmp_path / ".ness",
        project_root=tmp_path,
        agent_config=MagicMock(),
        available_skills={},
    )
    token = set_session_context(rt)
    try:
        from ness_agent.tools.skill import skill_view

        result = skill_view.invoke({"name": "nonexistent"})
        assert result.startswith("Error: unknown skill")
    finally:
        reset_session_context(token)


def test_skill_view_registered():
    from ness_agent.tools import BUILTIN_TOOLS, ALWAYS_ON, READ_ONLY_TOOLS, TOOL_NAMES
    assert any(t.name == "skill_view" for t in BUILTIN_TOOLS)
    assert "skill_view" in ALWAYS_ON
    assert "skill_view" in READ_ONLY_TOOLS
    assert "skill_view" in TOOL_NAMES


class _SkillRecordingModel:
    model = "skill-recording"

    def __init__(self, responses=()):
        self.calls = []
        self.responses = iter(responses)

    def bind_tools(self, _tools):
        return self

    async def ainvoke(self, messages, **_kwargs):
        self.calls.append(list(messages))
        return next(self.responses, AIMessage(content="ok"))


def _recording_skill_session(tmp_path, model, roots, *, include_catalog=False):
    from ness_agent import MemoryConfig

    agent = NessAgent(
        model=model,
        tools=["skill_view"],
        prompt=PromptLayersConfig(l0="L0", persona="P", include_skill_catalog=include_catalog),
        options=NessAgentOptions(
            project_root=tmp_path,
            ness_dir=tmp_path / ".ness",
            enable_approval=False,
            auto_save_threads=False,
        ),
        memory=MemoryConfig(disabled=True),
        skills_dirs=roots,
    )
    return agent.session(thread_id="skill-contract", git_available=False)


def test_preview_shows_pending_catalog_without_consuming_it(tmp_path):
    root = tmp_path / "skills"
    _write_skill(root / "alpha", "alpha", "Alpha procedure")
    model = _SkillRecordingModel()
    session = _recording_skill_session(tmp_path, model, [root])

    async def run():
        first = await session.preview_context()
        second = await session.preview_context()
        assert first.overlay_sections["skill_catalog"] == second.overlay_sections["skill_catalog"]
        assert "alpha" in first.overlay_sections["skill_catalog"]
        await session.run("one")
        assert first.overlay_sections["skill_catalog"] in str(model.calls[0][-1].content)
        assert "skill_catalog" not in (await session.preview_context()).overlay_sections

        skill_id = session.config.skill_loader.all_skills()[0]["skill_id"]
        session.set_skill_access([skill_id])
        refreshed = await session.preview_context()
        assert "No skills are currently available" in refreshed.overlay_sections["skill_catalog"]
        await session.run("two")
        assert refreshed.overlay_sections["skill_catalog"] in str(model.calls[1][-1].content)

    asyncio.run(run())


@pytest.mark.parametrize("include_catalog", [False, True])
def test_disabled_skill_is_omitted_from_current_reminders(tmp_path, include_catalog):
    from langchain_core.messages import ToolMessage

    root = tmp_path / "skills"
    _write_skill(root / "alpha", "alpha", "Alpha procedure", body="Original instructions")
    call = AIMessage(content="", tool_calls=[{"name": "skill_view", "args": {"name": "alpha"}, "id": "load-alpha"}])
    model = _SkillRecordingModel([call, AIMessage(content="loaded")])
    session = _recording_skill_session(tmp_path, model, [root], include_catalog=include_catalog)

    async def run():
        await session.run("load alpha")
        assert "loaded_skills" not in (await session.get_state())
        session.set_skill_access([session.config.skill_loader.all_skills()[0]["skill_id"]])
        session.stage_skills(["alpha"])
        preview = await session.preview_context()
        assert "loaded_skills" not in preview.overlay_sections
        assert "skill_request" not in preview.overlay_sections

        await session.run("continue", requested_skills=["alpha"])
        current_turn = str(model.calls[-1][-1].content)
        assert "LOADED SKILLS" not in current_turn
        assert "SKILL REQUEST" not in current_turn
        assert any(
            isinstance(message, ToolMessage) and "Original instructions" in str(message.content)
            for message in model.calls[-1]
        )

    asyncio.run(run())


def test_same_name_fallback_returns_current_body_without_tracking(tmp_path):
    from langchain_core.messages import ToolMessage
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_skill(first / "alpha", "alpha", "First procedure", body="First body")
    _write_skill(second / "alpha", "alpha", "Second procedure", body="Second body")
    calls = [
        AIMessage(content="", tool_calls=[{"name": "skill_view", "args": {"name": "alpha"}, "id": call_id}])
        for call_id in ("first-load", "second-load")
    ]
    model = _SkillRecordingModel([calls[0], AIMessage(content="first"), calls[1], AIMessage(content="second")])
    session = _recording_skill_session(tmp_path, model, [first, second])
    records = session.config.skill_loader.discover()
    session.configure_skills(records)

    async def run():
        await session.run("load alpha")
        session.set_skill_access([records[0]["skill_id"]])
        assert "loaded_skills" not in (await session.preview_context()).overlay_sections
        await session.run("reload alpha", requested_skills=["alpha"])
        state = await session.get_state()
        assert "loaded_skills" not in state
        results = [message for message in state["messages"] if isinstance(message, ToolMessage)]
        assert json.loads(results[0].content)["content"] == "First body"
        assert json.loads(results[-1].content)["content"] == "Second body"

    asyncio.run(run())


@pytest.mark.parametrize("include_catalog", [False, True])
def test_child_receives_effective_catalog_without_changing_parent_prompts(tmp_path, include_catalog):
    from ness_agent.session_context import reset_session_context
    from ness_agent.tools.skill import skill_view
    from ness_agent.tools.subagents import PreparedTask, SubagentTask, _invoke_subagent

    root = tmp_path / "skills"
    _write_skill(root / "alpha", "alpha", "Alpha procedure")
    _write_skill(root / "hidden", "hidden", "Hidden procedure")
    model = _SkillRecordingModel()
    session = _recording_skill_session(tmp_path, model, [root], include_catalog=include_catalog)
    records = session.config.skill_loader.discover()
    hidden_id = next(skill["skill_id"] for skill in records if skill["name"] == "hidden")
    session.configure_skills(records, disabled_skill_ids=[hidden_id])
    token = session._install_session_runtime()
    try:
        prepared = PreparedTask(SubagentTask(name="probe", prompt="inspect"), "inspect", [skill_view])
        asyncio.run(_invoke_subagent(prepared, model, "skill-child"))
    finally:
        reset_session_context(token)

    system = str(model.calls[0][0].content)
    assert "- alpha: Alpha procedure:" in system
    assert "Hidden procedure" not in system
    assert session.config.prompts.config.include_skill_catalog is include_catalog


@pytest.mark.parametrize("skills_enabled", [False, True])
@pytest.mark.parametrize("include_catalog", [False, True])
def test_compaction_uses_general_skill_reminder_without_tracking(tmp_path, skills_enabled, include_catalog):
    root = tmp_path / "skills"
    _write_skill(root / "alpha", "alpha", "Alpha procedure")
    model = _SkillRecordingModel()
    session = _recording_skill_session(
        tmp_path, model, [root] if skills_enabled else [], include_catalog=include_catalog,
    )

    async def run():
        await session.run("first task")
        assert "reload its instructions" not in str(model.calls[-1][-1].content)
        session.request_compact()
        await session.run("continue")
        current_turn = str(model.calls[-1][-1].content)
        assert ("reload its instructions with skill_view" in current_turn) is skills_enabled
        assert "LOADED SKILLS" not in current_turn
        assert "loaded_skills" not in (await session.get_state())

    asyncio.run(run())
