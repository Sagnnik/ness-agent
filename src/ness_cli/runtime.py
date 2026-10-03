"""Composition root for CLI runtimes.

Provider, SDK, and session objects are constructed here. Headless and
interactive callers receive small runtime facades over the same core.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ness_agent import (
    CostTracker,
    MemoryConfig,
    NessAgent,
    NessAgentOptions,
    default_skill_search_dirs,
)

from ness_cli.cleanup import cleanup_all, cleanup_on_exit
from ness_cli.config import (
    CliOverrides,
    ConfigApplyError,
    ConfigManager,
    ConfigPatch,
    ConfigUpdate,
    RuntimeConfig,
)
from ness_cli.config.settings import Settings, sdk_behavior_values
from ness_cli.paths import (
    NessPaths,
    ensure_global_config,
    ensure_project_runtime,
    resolve_paths,
)
from ness_cli.mcp import MCPOAuthService, ProjectMCPManager
from ness_cli.mcp.trust import (
    authorize_mcp_interactively,
    is_mcp_trusted,
)
from ness_cli.providers import ModelRequest, ProviderRegistry
from ness_cli.session import CodingSession, SessionModels, SessionRepository
from ness_cli.workspace import repo_root


@dataclass(frozen=True, slots=True)
class InteractiveOptions:
    overrides: CliOverrides | None = None
    project_root: Path | None = None
    resume_thread_id: str | None = None
    yolo: bool = False
    approval_handler: Any = None
    question_handler: Any = None


@dataclass(frozen=True, slots=True)
class HeadlessOptions:
    overrides: CliOverrides | None = None
    project_root: Path | None = None
    resume_thread_id: str | None = None
    yolo: bool = False
    approval_handler: Any = None
    question_handler: Any = None


class ProviderAuthenticationError(RuntimeError):
    """Raised when a runtime cannot call its selected provider."""


def _model_request(
    config: RuntimeConfig,
    *,
    thread_id: str,
    reflection: bool,
) -> ModelRequest:
    selection = config.model
    return ModelRequest(
        thread_id=thread_id,
        purpose="reflection" if reflection else "main",
        model_name=(
            selection.reflection_model_name if reflection else selection.model_name
        ),
        reasoning_effort=selection.reasoning_effort,
        provider_id=selection.provider_id,
    )


def _session_models(
    providers: ProviderRegistry,
    config: RuntimeConfig,
    *,
    thread_id: str,
) -> SessionModels:
    selection = config.model
    info = providers.model_info(
        selection.model_name,
        provider_id=selection.provider_id,
    )
    return SessionModels(
        model=providers.create_model(
            _model_request(config, thread_id=thread_id, reflection=False)
        ),
        reflection_model=providers.create_model(
            _model_request(config, thread_id=thread_id, reflection=True)
        ),
        context_window=info.context_window if info is not None else None,
        vision=info.supports_vision if info is not None else None,
        runtime_config=config,
    )


def _build_agent(
    *,
    paths: NessPaths,
    config: ConfigManager,
    providers: ProviderRegistry,
    options: InteractiveOptions | HeadlessOptions,
) -> NessAgent:
    from ness_cli.prompts import (
        default_aux_prompts,
        default_prompt_layers,
        plan_act_modes,
    )

    current = config.current
    cost_tracker = CostTracker()
    providers.bind_pricing(cost_tracker.pricing)
    bootstrap = _session_models(
        providers,
        current,
        thread_id="runtime-default",
    )
    sdk_options = NessAgentOptions(
        context_window=bootstrap.context_window,
        yolo_mode=options.yolo,
        project_root=paths.project_root,
        ness_dir=paths.ness_dir,
        **sdk_behavior_values(current.settings, yolo_mode=options.yolo),
    )

    return NessAgent(
        model=bootstrap.model,
        reflection_model=bootstrap.reflection_model,
        prompt=default_prompt_layers(instructions_dir=paths.instructions_dir),
        aux_prompts=default_aux_prompts(instructions_dir=paths.instructions_dir),
        modes=plan_act_modes(
            plans_dir=paths.plans_dir,
            instructions_dir=paths.instructions_dir,
        ),
        memory=MemoryConfig(
            user_memory=paths.user_file,
            session_memory_dir=paths.sessions_dir,
        ),
        hooks_config=paths.ness_dir / "hooks.json",
        skills_dirs=default_skill_search_dirs(paths.project_root),
        options=sdk_options,
        approval_handler=options.approval_handler,
        question_handler=options.question_handler,
        cost_tracker=cost_tracker,
    )


class _SessionFactory:
    """Create thread-bound sessions without exposing provider internals."""

    def __init__(
        self,
        *,
        paths: NessPaths,
        config: ConfigManager,
        providers: ProviderRegistry,
        agent: NessAgent,
    ) -> None:
        self._paths = paths
        self._config = config
        self._providers = providers
        self._agent = agent
        self._sessions: dict[str, CodingSession] = {}
        self._git_available = repo_root(paths.project_root) is not None

    @property
    def has_sessions(self) -> bool:
        return bool(self._sessions)

    def _models_for(self, thread_id: str) -> SessionModels:
        return _session_models(
            self._providers,
            self._config.current,
            thread_id=thread_id,
        )

    async def new(
        self,
        *,
        thread_id: str | None = None,
        mode: str = "act",
    ) -> CodingSession:
        selected_thread = thread_id or self._agent.new_thread_id()
        existing = self._sessions.get(selected_thread)
        if existing is not None:
            return existing

        models = self._models_for(selected_thread)
        sdk_session = self._agent.session(
            thread_id=selected_thread,
            mode=mode,
            git_available=self._git_available,
            vision=models.vision,
            model=models.model,
            reflection_model=models.reflection_model,
        )
        sdk_session.config.options.context_window = models.context_window
        sdk_session.config.permission_store.clear_session_rules()

        session = CodingSession.from_sdk_session(
            sdk_session,
            paths=self._paths,
            runtime_config=models.runtime_config,
            repository=SessionRepository(sdk_session.config.thread_store),
            model_loader=self._models_for,
            vision=models.vision,
        )
        session.apply_runtime_config(models.runtime_config)
        self._sessions[selected_thread] = session
        return session

    async def resume(self, thread_id: str) -> CodingSession:
        # Session construction persists a skill snapshot. Reject missing threads
        # before creating any session-owned resources or state.
        if not SessionRepository(self._agent.config.thread_store).exists(thread_id):
            raise LookupError(thread_id)
        session = await self.new(thread_id=thread_id)
        if not await session.resume():
            self._sessions.pop(thread_id, None)
            async with cleanup_on_exit(f"session {thread_id}", session.close):
                raise LookupError(thread_id)
        return session

    async def close(self) -> None:
        sessions = tuple(self._sessions.items())
        self._sessions.clear()
        await cleanup_all(
            (f"session {thread_id}", session.close)
            for thread_id, session in sessions
        )


class _RuntimeCore:
    def __init__(
        self,
        *,
        paths: NessPaths,
        config: ConfigManager,
        providers: ProviderRegistry,
        mcp: ProjectMCPManager,
        oauth: MCPOAuthService,
        warnings: tuple[str, ...],
        agent: NessAgent,
        sessions: _SessionFactory,
    ) -> None:
        self.paths = paths
        self.config = config
        self.providers = providers
        self.mcp = mcp
        self.oauth = oauth
        self.warnings = warnings
        self.agent = agent
        self.sessions = sessions
        self._closed = False

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await cleanup_all(
            (
                ("sessions", self.sessions.close),
                ("MCP", self.mcp.stop),
                ("providers", self.providers.close),
            )
        )


class InteractiveRuntime:
    def __init__(
        self,
        core: _RuntimeCore,
        *,
        resume_thread_id: str | None,
    ) -> None:
        self._core = core
        self._resume_thread_id = resume_thread_id

    @property
    def paths(self) -> NessPaths:
        return self._core.paths

    @property
    def config(self) -> RuntimeConfig:
        return self._core.config.current

    @property
    def settings(self) -> Settings:
        return self._core.config.settings

    @property
    def providers(self) -> ProviderRegistry:
        return self._core.providers

    def provider_profile(self, provider_id: str) -> dict[str, Any]:
        return self._core.config.provider_profile(provider_id)

    def goal_judge_model(self, thread_id: str) -> Any:
        current = self._core.config.current
        model_name = (
            current.settings.goal_judge_model or current.model.reflection_model_name
        )
        return self._core.providers.create_model(
            ModelRequest(
                thread_id=thread_id,
                purpose="goal",
                model_name=model_name,
                reasoning_effort=current.model.reasoning_effort,
                provider_id=current.model.provider_id,
            )
        )

    @property
    def mcp(self) -> ProjectMCPManager:
        return self._core.mcp

    @property
    def oauth(self) -> MCPOAuthService:
        return self._core.oauth

    @property
    def warnings(self) -> tuple[str, ...]:
        return self._core.warnings

    @property
    def approval_state(self) -> str:
        options = self._core.agent.config.options
        if options.yolo_mode:
            return "yolo"
        return "on" if options.enable_approval else "off"

    @property
    def resume_thread_id(self) -> str | None:
        return self._resume_thread_id

    def install_interaction_handlers(
        self,
        *,
        approval_handler: Any,
        question_handler: Any,
    ) -> None:
        """Bind TUI handlers before the first SDK session is forked."""
        if self._core.sessions.has_sessions:
            raise RuntimeError(
                "interaction handlers must be installed before creating sessions"
            )
        if self._core.agent.config.approval_handler is None:
            self._core.agent.config.approval_handler = approval_handler
        if self._core.agent.config.question_handler is None:
            self._core.agent.config.question_handler = question_handler

    async def initial_session(self) -> CodingSession:
        if self._resume_thread_id:
            return await self.resume_session(self._resume_thread_id)
        return await self.new_session()

    async def new_session(
        self,
        *,
        thread_id: str | None = None,
        mode: str = "act",
    ) -> CodingSession:
        return await self._core.sessions.new(thread_id=thread_id, mode=mode)

    async def resume_session(self, thread_id: str) -> CodingSession:
        return await self._core.sessions.resume(thread_id)

    async def apply_config(
        self,
        patch: ConfigPatch,
        *,
        selected: CodingSession | None = None,
    ) -> ConfigUpdate:
        try:
            update = self._core.config.apply(patch)
        except ConfigApplyError as error:
            if error.update is not None:
                try:
                    await self._apply_selected_update(
                        error.update,
                        selected,
                        autosave_requested="auto_save_threads" in (error.saved_keys or ()),
                    )
                except Exception as refresh_error:
                    diagnostic = (
                        "Could not apply saved configuration to the selected "
                        f"session ({type(refresh_error).__name__})."
                    )
                    error.args = (f"{error} {diagnostic}",)
                    error.add_note(diagnostic)
            raise
        await self._apply_selected_update(
            update,
            selected,
            autosave_requested="auto_save_threads" in patch.values,
        )
        return update

    @staticmethod
    async def _apply_selected_update(
        update: ConfigUpdate,
        selected: CodingSession | None,
        *,
        autosave_requested: bool = False,
    ) -> None:
        if selected is not None and update.options_changed:
            selected.apply_runtime_config(update.current)
        elif selected is not None and autosave_requested:
            # The saved default can already match the request while the
            # selected session retains a different autosave setting.
            selected.set_autosave(update.current.settings.auto_save_threads)
        if selected is not None and update.reload_selected:
            await selected.reload_model()

    async def close(self) -> None:
        await self._core.close()


class HeadlessRuntime:
    def __init__(
        self,
        core: _RuntimeCore,
        *,
        resume_thread_id: str | None,
    ) -> None:
        self._core = core
        self._resume_thread_id = resume_thread_id

    @property
    def warnings(self) -> tuple[str, ...]:
        return self._core.warnings

    @property
    def mcp(self) -> ProjectMCPManager:
        return self._core.mcp

    async def session(self) -> CodingSession:
        if self._resume_thread_id:
            return await self._core.sessions.resume(self._resume_thread_id)
        return await self._core.sessions.new()

    async def close(self) -> None:
        await self._core.close()


async def _open_core(
    options: InteractiveOptions | HeadlessOptions,
) -> _RuntimeCore:
    paths = resolve_paths(project_root=options.project_root)
    ensure_global_config(paths)
    config = ConfigManager.load(paths, options.overrides)
    paths = resolve_paths(
        project_root=paths.project_root,
        ness_dir=config.current.settings.ness_dir,
    )
    ensure_project_runtime(paths)

    providers = ProviderRegistry.load(paths=paths, config=config)
    mcp: ProjectMCPManager | None = None
    warnings: list[str] = []
    try:
        oauth = MCPOAuthService(
            project_root=paths.project_root,
            config_dir=paths.config_dir,
        )
        mcp = ProjectMCPManager(
            paths.ness_dir / "mcp.json",
            project_root=paths.project_root,
            http_auth_factory=oauth.startup_auth,
        )
        active_provider = providers.active()
        if (
            isinstance(options, InteractiveOptions)
            and not active_provider.is_authenticated()
        ):
            warnings.append(
                f"{active_provider.display_name} is not authenticated; use /login to connect"
            )
        if isinstance(options, InteractiveOptions):
            trusted = authorize_mcp_interactively(
                mcp,
                config_dir=paths.config_dir,
            )
        else:
            trusted = is_mcp_trusted(mcp, config_dir=paths.config_dir)
            if not trusted:
                mcp.mark_untrusted()
                warnings.append(
                    "MCP configuration is not trusted; run interactive Ness "
                    "once to review and approve it"
                )
        if trusted:
            await mcp.start()

        agent = _build_agent(
            paths=paths,
            config=config,
            providers=providers,
            options=options,
        )
        agent.config.tool_registry.register_dynamic(mcp.tools.values())
        agent.config.tool_registry.set_mcp_catalog(mcp.catalog())
        summary, level = mcp.startup_summary()
        if level == "warn":
            warnings.append(summary)
        warnings.extend(oauth.warnings)

        return _RuntimeCore(
            paths=paths,
            config=config,
            providers=providers,
            mcp=mcp,
            oauth=oauth,
            warnings=tuple(dict.fromkeys(warnings)),
            agent=agent,
            sessions=_SessionFactory(
                paths=paths,
                config=config,
                providers=providers,
                agent=agent,
            ),
        )
    except BaseException as error:
        operations = []
        if mcp is not None:
            operations.append(("MCP", mcp.stop))
        operations.append(("providers", providers.close))
        await cleanup_all(operations, primary_error=error)
        raise


def _require_provider_authentication(core: _RuntimeCore) -> None:
    provider = core.providers.active()
    if provider.is_authenticated():
        return
    raise ProviderAuthenticationError(
        f"{provider.display_name} is not authenticated; "
        "provide --api-key or configure a provider credential"
    )


@asynccontextmanager
async def open_interactive_runtime(options: InteractiveOptions):
    core = await _open_core(options)
    runtime = InteractiveRuntime(
        core,
        resume_thread_id=options.resume_thread_id,
    )
    async with cleanup_on_exit("interactive runtime", runtime.close):
        yield runtime


@asynccontextmanager
async def open_headless_runtime(options: HeadlessOptions):
    core = await _open_core(options)
    runtime = HeadlessRuntime(
        core,
        resume_thread_id=options.resume_thread_id,
    )
    async with cleanup_on_exit("headless runtime", runtime.close):
        _require_provider_authentication(core)
        yield runtime
