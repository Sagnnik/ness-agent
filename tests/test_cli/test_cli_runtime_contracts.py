from __future__ import annotations

import asyncio
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from ness_cli import cli, main, runtime


def test_cli_rejects_prompt_without_print_mode():
    result = CliRunner().invoke(cli.app, ["hello"])
    assert result.exit_code == 2
    assert "requires --print" in result.stderr


def test_cli_print_requires_input(monkeypatch):
    monkeypatch.setattr(cli, "_read_stdin", lambda: "")
    result = CliRunner().invoke(cli.app, ["--print"])
    assert result.exit_code == 2
    assert "requires a prompt" in result.stderr


def test_cli_print_passes_overrides_and_exit_code(monkeypatch):
    observed = {}

    async def fake_run(prompt, *, options):
        observed.update(prompt=prompt, options=options)
        return 7

    monkeypatch.setattr("ness_cli.headless.run_headless", fake_run)
    monkeypatch.setattr(cli, "_read_stdin", lambda: "stdin")
    result = CliRunner().invoke(
        cli.app,
        ["--print", "--model", "chosen", "--yolo", "argument"],
    )
    assert result.exit_code == 7
    assert observed["prompt"] == "stdin\n\nargument"
    assert observed["options"].yolo
    assert observed["options"].overrides.model_name == "chosen"


def test_worktree_argument_parser_uses_last_value():
    assert main._worktree_name(["-w", "one", "--worktree=two"]) == "two"
    assert main._worktree_name(["--worktree"]) is None
    assert main._worktree_name([]) is None


@pytest.mark.parametrize(
    ("repository", "expected", "vision"),
    [(Path("/repo"), True, True), (None, False, False), (None, False, None)],
)
def test_session_factory_passes_detected_git_state(
    monkeypatch,
    repository,
    expected,
    vision,
):
    captured = {}
    sdk_session = SimpleNamespace(
        config=SimpleNamespace(
            options=SimpleNamespace(context_window=None),
            permission_store=SimpleNamespace(clear_session_rules=lambda: None),
            thread_store=object(),
        )
    )

    class Agent:
        @staticmethod
        def new_thread_id():
            return "thread-1"

        @staticmethod
        def session(**kwargs):
            captured.update(kwargs)
            return sdk_session

    models = SimpleNamespace(
        model=object(),
        reflection_model=object(),
        context_window=128_000,
        vision=vision,
        runtime_config=object(),
    )
    result = SimpleNamespace(
        thread_id="thread-1", apply_runtime_config=lambda _config: None
    )
    monkeypatch.setattr(runtime, "repo_root", lambda _path: repository)
    monkeypatch.setattr(runtime, "_session_models", lambda *_args, **_kwargs: models)
    monkeypatch.setattr(runtime, "SessionRepository", lambda _store: object())
    monkeypatch.setattr(
        runtime,
        "CodingSession",
        SimpleNamespace(from_sdk_session=lambda *_args, **_kwargs: result),
    )
    factory = runtime._SessionFactory(
        paths=SimpleNamespace(project_root=Path("/project")),
        config=SimpleNamespace(current=object()),
        providers=object(),
        agent=Agent(),
    )

    assert asyncio.run(factory.new()) is result
    assert captured["git_available"] is expected
    assert captured["vision"] is vision


@pytest.mark.parametrize("yolo,update_selected", [(False, False), (True, True)])
def test_new_sessions_apply_current_behavior_settings(
    isolated_cli_env, monkeypatch, yolo, update_selected
):
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from ness_agent import NessAgent, NessAgentOptions, PromptLayers, PromptLayersConfig
    from ness_cli.config import ConfigManager, ConfigPatch

    paths = isolated_cli_env.paths
    manager = ConfigManager.load(paths, environment={})
    manager.apply(ConfigPatch({"enable_approval": False}))
    model = FakeListChatModel(responses=["ok"])
    agent = NessAgent(
        model=model,
        prompt=PromptLayers(PromptLayersConfig(l0="Test")),
        options=NessAgentOptions(
            project_root=paths.project_root,
            ness_dir=paths.ness_dir,
            enable_approval=False,
            yolo_mode=yolo,
        ),
        skills_dirs=[],
    )
    monkeypatch.setattr(
        runtime,
        "_session_models",
        lambda _providers, config, **_kwargs: runtime.SessionModels(
            model, None, 128_000, False, config
        ),
    )
    factory = runtime._SessionFactory(
        paths=paths, config=manager, providers=None, agent=agent
    )
    facade = runtime.InteractiveRuntime(
        SimpleNamespace(config=manager), resume_thread_id=None
    )
    changes = {
        "enable_approval": True,
        "format_on_write": False,
        "auto_save_threads": False,
        "session_end_reflection": True,
        "reflection_token_ratio": 0.2,
        "compaction_token_budget": 90_000,
        "compaction_buffer_tokens": 8_192,
        "compaction_summary_max_tokens": 2_048,
        "exa_api_key": "test-only-key",
    }

    async def exercise():
        first = await factory.new(thread_id="first")
        await facade.apply_config(
            ConfigPatch(changes), selected=first if update_selected else None
        )
        second = await factory.new(thread_id="second")
        options = second._session.config.options
        for key, value in changes.items():
            expected = not yolo if key == "enable_approval" else value
            assert getattr(options, key) == expected, key
        assert options.yolo_mode is yolo
        assert options.context_window == 128_000
        assert not second._repository.auto_save
        assert second.runtime_config == manager.current
        # Existing sessions keep their own options unless selected for update.
        assert first._session.config.options.format_on_write is (not update_selected)

    asyncio.run(exercise())


@pytest.fixture
def build_behavior_runtime(isolated_cli_env, monkeypatch):
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from ness_cli.config import ConfigManager, ConfigPatch

    paths = isolated_cli_env.paths
    manager = ConfigManager.load(paths, environment={})
    model = FakeListChatModel(responses=["ok"])
    monkeypatch.setattr(
        runtime,
        "_session_models",
        lambda _providers, config, **_kwargs: runtime.SessionModels(
            model, None, 12_000, False, config
        ),
    )
    providers = SimpleNamespace(bind_pricing=lambda _pricing: None)

    def build(settings, options):
        manager.apply(ConfigPatch(settings))
        agent = runtime._build_agent(
            paths=paths, config=manager, providers=providers, options=options
        )
        factory = runtime._SessionFactory(
            paths=paths, config=manager, providers=providers, agent=agent
        )
        return manager, agent, factory

    return build


@pytest.mark.parametrize(
    "options_type", [runtime.InteractiveOptions, runtime.HeadlessOptions]
)
def test_cli_discovers_shared_skill_roots_with_project_precedence(
    build_behavior_runtime, isolated_cli_env, monkeypatch, options_type
):
    project = isolated_cli_env.project
    home = isolated_cli_env.root / "home"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    def write_skill(root, name, body):
        path = root / name / "SKILL.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"---\nname: {name}\ndescription: Test skill\n---\n{body}\n")

    for scope, root in (("project", project), ("global", home)):
        for directory in (".agents", ".claude", ".codex", ".cursor"):
            skills = root / directory / "skills"
            body = f"{scope} {directory}"
            write_skill(skills, "shared", body)
            write_skill(skills, scope + "-shared", body)
            write_skill(skills, scope + directory, body)
    write_skill(project / ".claude/skills", "project-over-global", "project fallback")
    write_skill(home / ".agents/skills", "project-over-global", "global primary")
    write_skill(project / ".ness/skills", "legacy-project", "ignored")
    write_skill(isolated_cli_env.ness / "skills", "legacy-override", "ignored")

    _, agent, _ = build_behavior_runtime(
        {"compaction_buffer_tokens": 4096, "compaction_summary_max_tokens": 1024},
        options_type(),
    )
    available = agent.config.skill_loader.load()

    assert available["shared"]["body"] == "project .agents"
    assert available["project-shared"]["body"] == "project .agents"
    assert available["global-shared"]["body"] == "global .agents"
    assert available["project-over-global"]["body"] == "project fallback"
    assert set(available) == {
        "shared",
        "project-shared",
        "global-shared",
        "project-over-global",
        *(
            scope + directory
            for scope in ("project", "global")
            for directory in (".agents", ".claude", ".codex", ".cursor")
        ),
    }


@pytest.mark.parametrize(
    "options_type,yolo,approval",
    [
        (runtime.InteractiveOptions, False, False),
        (runtime.InteractiveOptions, True, True),
        (runtime.HeadlessOptions, False, True),
        (runtime.HeadlessOptions, True, False),
    ],
)
def test_startup_and_live_updates_apply_all_behavior_settings(
    build_behavior_runtime, isolated_cli_env, options_type, yolo, approval
):
    from ness_cli.config import ConfigPatch

    initial = {
        "enable_approval": approval,
        "auto_save_threads": False,
        "session_end_reflection": True,
        "reflection_token_ratio": 0.25,
        "compaction_token_budget": 90_000,
        "compaction_buffer_tokens": 8_192,
        "compaction_summary_max_tokens": 2_048,
        "format_on_write": False,
        "exa_api_key": "initial-test-key",
    }
    manager, agent, factory = build_behavior_runtime(initial, options_type(yolo=yolo))

    def assert_settings(options, settings):
        for name, value in settings.items():
            expected = value and not yolo if name == "enable_approval" else value
            assert getattr(options, name) == expected, name
        assert options.yolo_mode is yolo
        assert options.context_window == 12_000
        assert options.project_root == isolated_cli_env.paths.project_root
        assert options.ness_dir == isolated_cli_env.paths.ness_dir

    assert_settings(agent.config.options, initial)

    async def exercise():
        session = await factory.new(thread_id="behavior")
        options = session._session.config.options
        options.interruption_marker = "custom interruption"
        options.recursion_limit = 80
        updated = {
            "enable_approval": not approval,
            "auto_save_threads": True,
            "session_end_reflection": False,
            "reflection_token_ratio": 0.5,
            "compaction_token_budget": 80_000,
            "compaction_buffer_tokens": 4_096,
            "compaction_summary_max_tokens": 1_024,
            "format_on_write": True,
            "exa_api_key": None,
        }
        manager.apply(ConfigPatch(updated))
        session.apply_runtime_config(manager.current)

        assert_settings(options, updated)
        assert session._session.config.options is options
        assert options.interruption_marker == "custom interruption"
        assert options.recursion_limit == 80
        assert session._repository.auto_save is True
        assert session.runtime_config == manager.current
        assert_settings(agent.config.options, initial)

    asyncio.run(exercise())


@pytest.mark.parametrize(
    ("invalid", "message"),
    [
        ({"compaction_buffer_tokens": 0}, "compaction_buffer_tokens must be positive"),
        (
            {"compaction_summary_max_tokens": 8_192},
            "compaction_summary_max_tokens must be smaller than compaction_buffer_tokens",
        ),
        (
            {"compaction_buffer_tokens": 12_000},
            "context limit must be larger than compaction_buffer_tokens",
        ),
    ],
)
def test_startup_and_live_updates_share_sdk_validation(
    build_behavior_runtime, invalid, message
):
    from ness_cli.config import ConfigPatch

    initial = {
        "compaction_buffer_tokens": 8_192,
        "compaction_summary_max_tokens": 2_048,
    }
    manager, _agent, factory = build_behavior_runtime(initial, runtime.InteractiveOptions())

    async def exercise():
        session = await factory.new(thread_id="validation")
        options = session._session.config.options
        before = asdict(options)
        before_config = session.runtime_config
        manager.apply(
            ConfigPatch({
                "enable_approval": False,
                "auto_save_threads": False,
                "format_on_write": False,
                **invalid,
            })
        )
        with pytest.raises(ValueError, match=message):
            session.apply_runtime_config(manager.current)

        assert session._session.config.options is options
        assert asdict(options) == before
        assert session.runtime_config is before_config
        assert session._repository.auto_save is True
        with pytest.raises(ValueError, match=message):
            build_behavior_runtime({}, runtime.InteractiveOptions())

    asyncio.run(exercise())
