"""Resolve global, project, and cache paths for the CLI."""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path

from platformdirs import user_cache_dir, user_config_dir

_APP_NAME = "ness-agent"
_PROJECT_MARKER = ".project"
_SLUG_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True, slots=True)
class NessPaths:
    """Concrete filesystem locations for one project checkout."""

    project_root: Path
    ness_dir: Path
    config_dir: Path
    cache_dir: Path
    user_file: Path
    configs_file: Path
    secrets_file: Path
    skill_state_file: Path
    instructions_dir: Path
    plans_dir: Path
    sessions_dir: Path
    shells_dir: Path
    threads_dir: Path
    cli_history: Path
    project_slug: str
    project_hash: str


def project_hash(project_root: Path) -> str:
    """Return a stable short hash for a resolved project root."""
    root = project_root.expanduser().resolve()
    return hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:12]


def sanitize_slug(name: str) -> str:
    """Convert a directory name to a safe lowercase slug."""
    return _SLUG_RE.sub("-", name.lower()).strip("-") or "project"


def resolve_project_slug(
    project_root: Path,
    plans_root: Path,
    *,
    hash_hex: str | None = None,
) -> str:
    """Resolve the plan directory slug without crossing project roots."""
    root = project_root.expanduser().resolve()
    digest = hash_hex or project_hash(root)
    base = sanitize_slug(root.name)
    candidate = plans_root / base
    marker = candidate / _PROJECT_MARKER

    if not candidate.exists():
        return base

    if marker.is_file():
        try:
            if marker.read_text(encoding="utf-8").strip() == str(root):
                return base
        except OSError:
            pass

    return f"{base}-{digest[:6]}"


def config_dir_from_env() -> Path:
    override = os.environ.get("NESS_AGENT_CONFIG_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return Path(user_config_dir(_APP_NAME))


def cache_dir_from_env() -> Path:
    override = os.environ.get("NESS_AGENT_CACHE_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return Path(user_cache_dir(_APP_NAME))


def resolve_paths(
    *,
    project_root: Path | None = None,
    ness_dir: Path | str | None = None,
) -> NessPaths:
    """Resolve all paths once at CLI startup."""
    root = (project_root or Path.cwd()).expanduser().resolve()

    raw_ness_dir = (
        Path(ness_dir)
        if ness_dir is not None
        else Path(os.environ.get("NESS_DIR", ".ness"))
    )
    ness = (
        raw_ness_dir.expanduser().resolve()
        if raw_ness_dir.is_absolute()
        else (root / raw_ness_dir).resolve()
    )

    config_dir = config_dir_from_env()
    digest = project_hash(root)
    plans_root = config_dir / "plans"
    slug = resolve_project_slug(root, plans_root, hash_hex=digest)
    project_cache = cache_dir_from_env() / digest

    return NessPaths(
        project_root=root,
        ness_dir=ness,
        config_dir=config_dir,
        cache_dir=project_cache,
        user_file=config_dir / "USER.md",
        configs_file=config_dir / "configs.json",
        secrets_file=config_dir / "secrets.json",
        skill_state_file=config_dir / "skill-state.json",
        instructions_dir=config_dir / "instructions",
        plans_dir=plans_root / slug,
        sessions_dir=ness / "runtime" / "sessions",
        shells_dir=ness / "runtime" / "shells",
        threads_dir=ness / "threads",
        cli_history=project_cache / "cli_history",
        project_slug=slug,
        project_hash=digest,
    )


def ensure_global_config(paths: NessPaths) -> list[str]:
    """Create required global files without overwriting user content."""
    created: list[str] = []

    if not paths.config_dir.exists():
        paths.config_dir.mkdir(parents=True, exist_ok=True)
        created.append(str(paths.config_dir))

    if not paths.user_file.exists():
        paths.user_file.parent.mkdir(parents=True, exist_ok=True)
        paths.user_file.write_text("", encoding="utf-8")
        created.append(str(paths.user_file))

    from ness_cli.config.store import ensure_secrets_file

    secrets_created = ensure_secrets_file(paths.config_dir)
    if secrets_created is not None:
        created.append(str(secrets_created))

    if not paths.plans_dir.exists():
        paths.plans_dir.mkdir(parents=True, exist_ok=True)
        created.append(str(paths.plans_dir))

    marker = paths.plans_dir / _PROJECT_MARKER
    root_text = f"{paths.project_root.resolve()}\n"
    try:
        current_marker = marker.read_text(encoding="utf-8")
    except OSError:
        current_marker = None

    if current_marker != root_text:
        marker.write_text(root_text, encoding="utf-8")
        if current_marker is None:
            created.append(str(marker))

    created.extend(_ensure_instruction_files(paths))
    return created


def _ensure_instruction_files(paths: NessPaths) -> list[str]:
    from ness_cli.instructions import default_instruction_files

    created: list[str] = []

    if not paths.instructions_dir.exists():
        paths.instructions_dir.mkdir(parents=True, exist_ok=True)
        created.append(str(paths.instructions_dir))

    for filename, content in default_instruction_files().items():
        path = paths.instructions_dir / filename
        if path.exists():
            continue
        path.write_text(content, encoding="utf-8")
        created.append(str(path))

    return created


def ensure_project_runtime(paths: NessPaths) -> list[str]:
    """Create mutable project runtime directories."""
    created: list[str] = []

    for directory in (
        paths.sessions_dir,
        paths.shells_dir,
        paths.threads_dir,
    ):
        if directory.exists():
            continue
        directory.mkdir(parents=True, exist_ok=True)
        created.append(str(directory))

    return created
