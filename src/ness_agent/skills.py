from __future__ import annotations

import hashlib
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import yaml

# Max directory depth below a skills root for discovery
# (root/skill, root/category/skill, root/a/b/skill).
MAX_SKILL_DEPTH = 3

_SKIP_DIR_NAMES = frozenset({".git", "node_modules"})

_PROJECT_SKILL_RELS = (
    ".agents/skills",
    ".claude/skills",
    ".codex/skills",
    ".cursor/skills",
)

_GLOBAL_SKILL_RELS = (
    ".agents/skills",
    ".claude/skills",
    ".codex/skills",
    ".cursor/skills",
)


def default_skill_search_dirs(
    project_root: Path,
    *,
    project_rels: Sequence[str] | None = None,
    global_rels: Sequence[str] | None = None,
) -> list[Path]:
    """Well-known project-local and user-global skill roots.

    Opt-in helper for host applications — the SDK never scans these
    unless the caller passes them in via ``AgentSpec.skills_dirs``.
    Project roots precede user-global roots, with ``.agents/skills``
    first in each scope.

    ``project_rels`` / ``global_rels`` restrict which project-local
    (relative to ``project_root``) and user-global (relative to the home
    directory) roots are included; ``None`` means all well-known ones.

    Example::

        from pathlib import Path
        from ness_agent import NessAgent, default_skill_search_dirs

        project_root = Path("/path/to/repo")
        agent = NessAgent(
            model=model,
            prompt=prompt,
            skills_dirs=default_skill_search_dirs(project_root),
        )
    """
    root = project_root.resolve()
    home = Path.home()
    dirs: list[Path] = [
        root / rel for rel in (_PROJECT_SKILL_RELS if project_rels is None else project_rels)
    ]
    dirs.extend(home / rel for rel in (_GLOBAL_SKILL_RELS if global_rels is None else global_rels))
    return dirs


def merge_skill_dirs(
    project_root: Path,
    skills_dir: Path,
    *,
    project_rels: Sequence[str] | None = None,
    global_rels: Sequence[str] | None = None,
) -> list[Path]:
    """User/CLI dir first, then known roots; dedupe by resolved path.

    Convenience for host applications (e.g. the Ness CLI) that want to
    opt into the well-known agent skill roots in addition to their own
    directory. Pass the result as ``AgentSpec.skills_dirs``.

    ``project_rels`` / ``global_rels`` restrict which project-local
    (relative to ``project_root``) and user-global (relative to the home
    directory) roots are included; ``None`` means all well-known ones.

    Example::

        from pathlib import Path
        from ness_agent import NessAgent, merge_skill_dirs

        project_root = Path("/path/to/repo")
        agent = NessAgent(
            model=model,
            prompt=prompt,
            skills_dirs=merge_skill_dirs(
                project_root,
                project_root / "my-app-skills",
            ),
        )
    """
    ordered: list[Path] = [
        skills_dir,
        *default_skill_search_dirs(
            project_root, project_rels=project_rels, global_rels=global_rels
        ),
    ]
    seen: set[Path] = set()
    result: list[Path] = []
    for path in ordered:
        try:
            key = path.resolve()
        except OSError:
            key = path
        if key in seen:
            continue
        seen.add(key)
        result.append(path)
    return result


def _iter_skill_md_paths(root: Path) -> Iterator[Path]:
    """Yield ``SKILL.md`` paths under *root* with category nesting support.

    If a directory contains ``SKILL.md``, it is a skill and we do not
    descend. Otherwise it is treated as a category and we recurse into
    subdirectories up to :data:`MAX_SKILL_DEPTH`.
    """
    if not root.is_dir():
        return

    def visit(directory: Path, depth: int) -> Iterator[Path]:
        skill_md = directory / "SKILL.md"
        if skill_md.is_file():
            yield skill_md
            return
        if depth >= MAX_SKILL_DEPTH:
            return
        try:
            children = sorted(directory.iterdir())
        except OSError:
            return
        for child in children:
            if not child.is_dir():
                continue
            if child.name in _SKIP_DIR_NAMES:
                continue
            yield from visit(child, depth + 1)

    try:
        top = sorted(root.iterdir())
    except OSError:
        return
    for child in top:
        if not child.is_dir():
            continue
        if child.name in _SKIP_DIR_NAMES:
            continue
        yield from visit(child, depth=1)


class SkillLoader:
    def __init__(
        self,
        skills_dir: Path | None = None,
        *,
        skills_dirs: Sequence[Path] | None = None,
    ) -> None:
        if skills_dirs is not None:
            self.skills_dirs: list[Path] = list(skills_dirs)
        elif skills_dir is not None:
            self.skills_dirs = [skills_dir]
        else:
            self.skills_dirs = []
        # Back-compat for callers/tests that read a single root.
        self.skills_dir = self.skills_dirs[0] if self.skills_dirs else None
        # Warnings from the most recent ``load`` (unreadable / invalid skill
        # files). Surfaced by the CLI's /skill command; reset on every load.
        self.errors: list[str] = []
        self._snapshot: list[dict[str, Any]] | None = None
        self._disabled_skill_ids: set[str] = set()

    @staticmethod
    def source_id(skill: dict[str, Any]) -> str:
        """Return the canonical identity for one skill source."""
        explicit = str(skill.get("source_id") or "").strip()
        if explicit:
            return explicit
        source = Path(str(skill.get("source") or ""))
        try:
            return str(source.expanduser().resolve())
        except OSError:
            return str(source.expanduser())

    @classmethod
    def skill_id(cls, skill: dict[str, Any]) -> str:
        """Return the logical identity shared by exact skill copies."""
        explicit = str(skill.get("skill_id") or "").strip()
        if explicit:
            return explicit
        return f"source:{cls.source_id(skill)}"

    @classmethod
    def sources(cls, skill: dict[str, Any]) -> tuple[dict[str, str], ...]:
        """Return every physical source represented by a logical skill."""
        raw_sources = skill.get("sources")
        sources: list[dict[str, str]] = []
        if isinstance(raw_sources, list):
            for raw in raw_sources:
                if not isinstance(raw, dict):
                    continue
                source = str(raw.get("source") or "")
                source_id = str(raw.get("source_id") or "")
                if not source_id and source:
                    source_id = cls.source_id({"source": source})
                if source_id:
                    sources.append({"source_id": source_id, "source": source})
        if sources:
            return tuple(sources)
        return (
            {
                "source_id": cls.source_id(skill),
                "source": str(skill.get("source") or ""),
            },
        )

    def fork_for_session(self) -> "SkillLoader":
        """Return a loader whose snapshot and access rules are session-local."""
        return SkillLoader(skills_dirs=self.skills_dirs)

    def discover(self) -> list[dict[str, Any]]:
        """Load skills and collapse byte-identical bundles across roots."""
        self.errors = []
        if not self.skills_dirs:
            return []

        skills: list[dict[str, Any]] = []
        by_skill_id: dict[str, dict[str, Any]] = {}
        seen_paths: set[Path] = set()

        for root in self.skills_dirs:
            if not root.exists():
                continue
            for path in _iter_skill_md_paths(root):
                try:
                    resolved = path.resolve()
                except OSError:
                    resolved = path
                if resolved in seen_paths:
                    continue
                seen_paths.add(resolved)
                skill = self._load_skill_md(path)
                if skill:
                    source_id = str(resolved)
                    skill_id = self._bundle_id(path.parent, source_id=source_id)
                    source = {
                        "source_id": source_id,
                        "source": str(path),
                    }
                    existing = by_skill_id.get(skill_id)
                    if existing is not None:
                        existing["sources"].append(source)
                        continue
                    skill["source_id"] = source_id
                    skill["skill_id"] = skill_id
                    skill["sources"] = [source]
                    by_skill_id[skill_id] = skill
                    skills.append(skill)

        return skills

    def set_snapshot(
        self,
        skills: Sequence[dict[str, Any]],
        *,
        disabled_skill_ids: Sequence[str] = (),
    ) -> None:
        """Freeze discovery records and access choices for this session."""
        self._snapshot = [dict(skill) for skill in skills]
        self._disabled_skill_ids = {
            str(skill_id) for skill_id in disabled_skill_ids
        }

    def set_disabled_skills(self, skill_ids: Sequence[str]) -> None:
        self._disabled_skill_ids = {str(skill_id) for skill_id in skill_ids}

    @property
    def disabled_skill_ids(self) -> frozenset[str]:
        return frozenset(self._disabled_skill_ids)

    def all_skills(self) -> list[dict[str, Any]]:
        """Return logical skills in precedence order, including disabled ones."""
        if self._snapshot is None:
            return self.discover()
        return [dict(skill) for skill in self._snapshot]

    def load(self) -> dict[str, dict[str, Any]]:
        skills: dict[str, dict[str, Any]] = {}
        for skill in self.all_skills():
            if self.skill_id(skill) in self._disabled_skill_ids:
                continue
            skills.setdefault(str(skill["name"]), skill)

        return skills

    def render_catalog(self, skills):
        """Render an always-on one-line catalog of every available skill for L1.
        Format:
        - Skill name: Skill description: Source file path
        Full skill content loads via the skill_view tool."""

        if not skills:
            return ""

        lines = ["Skill catalog (one-line summaries; load full instructions via the skill_view tool):"]
        for name in sorted(skills):
            desc = str(skills[name].get("description", "")).strip().splitlines()
            summary = desc[0].strip() if desc else ""
            source = str(skills[name].get("source", ""))
            line = f"- {name}: {summary}: {source}" if summary else f"- {name}: {source}"
            lines.append(line)
        return "\n".join(lines)

    def render_current_catalog(self) -> str:
        """Render a complete L3 catalog that supersedes earlier catalogs."""
        catalog = self.render_catalog(self.load())
        heading = (
            "Current effective skill catalog. Use this list instead of any "
            "earlier skill catalog in the conversation."
        )
        if not catalog:
            return heading + "\n- No skills are currently available."
        return heading + "\n" + catalog

    def _bundle_id(self, root: Path, *, source_id: str) -> str:
        """Hash all files in a skill bundle without including its location."""
        try:
            files = sorted(path for path in root.rglob("*") if path.is_file())
            digest = hashlib.sha256()
            for path in files:
                relative = path.relative_to(root).as_posix().encode("utf-8")
                content_digest = hashlib.sha256()
                with path.open("rb") as handle:
                    while chunk := handle.read(1024 * 1024):
                        content_digest.update(chunk)
                digest.update(len(relative).to_bytes(8, "big"))
                digest.update(relative)
                digest.update(content_digest.digest())
            return f"sha256:{digest.hexdigest()}"
        except OSError as exc:
            self.errors.append(f"{root}: could not fingerprint skill bundle: {exc}")
            return f"source:{source_id}"

    def _split_frontmatter(self, text: str) -> tuple[dict[str, Any], str]:
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) == 3:
                meta = yaml.safe_load(parts[1]) or {}
                if not isinstance(meta, dict):
                    meta = {}
                return meta, parts[2]
        return {}, text

    def _load_skill_md(self, path: Path) -> dict[str, Any] | None:
        try:
            text = path.read_text(encoding="utf-8")
            meta, body = self._split_frontmatter(text)

            raw_name = meta.get("name")
            if not raw_name or not str(raw_name).strip():
                self.errors.append(f"{path}: missing 'name' frontmatter")
                return None

            raw_description = meta.get("description")
            if not raw_description or not str(raw_description).strip():
                self.errors.append(f"{path}: missing 'description' frontmatter")
                return None

            return {
                "name": str(raw_name).strip(),
                "description": str(raw_description).strip(),
                "license": str(meta.get("license") or "").strip(),
                "compatibility": str(meta.get("compatibility") or "").strip(),
                "metadata": meta.get("metadata") or {},
                "body": body.strip(),
                "source": str(path),
            }
        except Exception as exc:
            self.errors.append(f"{path}: {exc}")
            return None
