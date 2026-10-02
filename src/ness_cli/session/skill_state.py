"""Global skill access defaults and thread snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ness_agent.skills import SkillLoader

from ness_cli.config.store import (
    atomic_write_json,
    locked_path,
    read_json_document,
)


@dataclass(frozen=True, slots=True)
class SkillSnapshot:
    skills: tuple[dict[str, Any], ...]
    disabled_skill_ids: frozenset[str]


def _skill_metadata(skill: dict[str, Any]) -> dict[str, Any]:
    sources = [dict(source) for source in SkillLoader.sources(skill)]
    primary = sources[0]
    return {
        "skill_id": SkillLoader.skill_id(skill),
        "source_id": primary["source_id"],
        "source": primary["source"],
        "sources": sources,
        "name": str(skill.get("name") or ""),
        "description": str(skill.get("description") or ""),
    }


def _bool_map(value: Any) -> dict[str, bool]:
    if not isinstance(value, dict):
        return {}
    return {str(key): bool(item) for key, item in value.items()}


def _text_map(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {str(key): str(item) for key, item in value.items()}


def _thread_map(document: dict[str, Any]) -> dict[str, Any]:
    value = document.get("threads")
    return dict(value) if isinstance(value, dict) else {}


def _stored_source(value: dict[str, Any]) -> dict[str, str] | None:
    """Validate a saved physical source before passing it to the skill loader."""
    fields = {}
    for key in ("source_id", "source"):
        raw = value.get(key)
        if raw is not None and (not isinstance(raw, str) or "\0" in raw):
            return None
        fields[key] = raw or ""
    if not any(item.strip() for item in fields.values()):
        return None
    try:
        fields["source_id"] = SkillLoader.source_id(fields)
    except (ValueError, RuntimeError):
        return None
    return fields


def _stored_skills(value: Any) -> list[dict[str, Any]]:
    """Read valid saved records without treating scalars as skill containers."""
    if not isinstance(value, list):
        return []
    records = []
    for raw in value:
        if not isinstance(raw, dict):
            continue
        skill_id = raw.get("skill_id")
        if skill_id is not None and not isinstance(skill_id, str):
            continue
        raw_sources = raw.get("sources")
        sources = []
        if isinstance(raw_sources, list):
            for item in raw_sources:
                if isinstance(item, dict):
                    source = _stored_source(item)
                    if source is not None:
                        sources.append(source)
        primary = _stored_source(raw)
        if primary is None:
            if not sources:
                continue
            primary = sources[0]
        records.append({**raw, **primary, "sources": sources})
    return records


def _merged_bool(values: Iterable[bool]) -> bool:
    """Availability wins when formerly separate copies disagree."""
    candidates = list(values)
    return any(candidates) if candidates else True


def _merged_writer(values: Iterable[str]) -> str:
    writers = {str(value) for value in values if value}
    return next(iter(writers)) if len(writers) == 1 else ""


class SkillStateStore:
    """Keep future defaults and isolated thread snapshots in one global file."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path).expanduser().resolve()

    def load_or_create(
        self,
        thread_id: str,
        discovered: Iterable[dict[str, Any]],
    ) -> SkillSnapshot:
        current_records = [dict(skill) for skill in discovered]
        current_by_skill = {
            SkillLoader.skill_id(skill): skill for skill in current_records
        }

        with locked_path(self.path):
            document = read_json_document(self.path)
            migrated = self._migrate_path_state(document, current_records)
            threads = _thread_map(document)
            raw_thread = threads.get(thread_id)

            if isinstance(raw_thread, dict) and isinstance(
                raw_thread.get("skills"), list
            ):
                if migrated:
                    atomic_write_json(self.path, document)
                return self._hydrate(raw_thread, current_by_skill)

            defaults = _bool_map(document.get("defaults"))
            writers = _text_map(document.get("writers"))
            available = {
                SkillLoader.skill_id(skill): defaults.get(
                    SkillLoader.skill_id(skill), True
                )
                for skill in current_records
            }
            thread = {
                "skills": [_skill_metadata(skill) for skill in current_records],
                "available": available,
                "baseline": dict(available),
                "baseline_writers": {
                    skill_id: writers.get(skill_id, "") for skill_id in available
                },
            }
            threads[thread_id] = thread
            document["threads"] = threads
            document["defaults"] = defaults
            document["writers"] = writers
            atomic_write_json(self.path, document)
            return self._hydrate(thread, current_by_skill)

    def apply(
        self,
        thread_id: str,
        disabled_skill_ids: Iterable[str],
    ) -> frozenset[str]:
        """Apply one thread's changed entries and merge future defaults."""
        disabled = {str(skill_id) for skill_id in disabled_skill_ids}
        with locked_path(self.path):
            document = read_json_document(self.path)
            threads = _thread_map(document)
            raw_thread = threads.get(thread_id)
            if not isinstance(raw_thread, dict):
                raise LookupError(f"missing skill snapshot for {thread_id}")

            skills = _stored_skills(raw_thread.get("skills"))
            skill_ids = {
                str(skill.get("skill_id") or "")
                for skill in skills
                if isinstance(skill, dict) and skill.get("skill_id")
            }
            previous = _bool_map(raw_thread.get("available"))
            baseline = _bool_map(raw_thread.get("baseline"))
            baseline_writers = _text_map(raw_thread.get("baseline_writers"))
            defaults = _bool_map(document.get("defaults"))
            writers = _text_map(document.get("writers"))

            for skill_id in skill_ids:
                proposed = skill_id not in disabled
                old_thread_value = previous.get(skill_id, True)
                if proposed == old_thread_value:
                    continue

                starting_default = baseline.get(skill_id, True)
                current_default = defaults.get(skill_id, True)
                starting_writer = baseline_writers.get(skill_id, "")
                current_writer = writers.get(skill_id, "")
                if (
                    current_default != starting_default
                    or current_writer != starting_writer
                ) and current_default != proposed:
                    resolved_default = True
                else:
                    resolved_default = proposed

                defaults[skill_id] = resolved_default
                writers[skill_id] = thread_id
                baseline[skill_id] = resolved_default
                baseline_writers[skill_id] = thread_id
                previous[skill_id] = proposed

            raw_thread["available"] = previous
            raw_thread["baseline"] = baseline
            raw_thread["baseline_writers"] = baseline_writers
            threads[thread_id] = raw_thread
            document["threads"] = threads
            document["defaults"] = defaults
            document["writers"] = writers
            atomic_write_json(self.path, document)

            return frozenset(
                skill_id for skill_id in skill_ids if not previous.get(skill_id, True)
            )

    def copy_thread(self, source_thread_id: str, target_thread_id: str) -> None:
        """Copy the source thread's effective snapshot for a fork."""
        with locked_path(self.path):
            document = read_json_document(self.path)
            threads = _thread_map(document)
            source = threads.get(source_thread_id)
            if not isinstance(source, dict):
                return
            defaults = _bool_map(document.get("defaults"))
            writers = _text_map(document.get("writers"))
            skills = _stored_skills(source.get("skills"))
            available = _bool_map(source.get("available"))
            threads[target_thread_id] = {
                "skills": skills,
                "available": available,
                "baseline": {
                    str(skill.get("skill_id")): defaults.get(
                        str(skill.get("skill_id")), True
                    )
                    for skill in skills
                    if skill.get("skill_id")
                },
                "baseline_writers": {
                    str(skill.get("skill_id")): writers.get(
                        str(skill.get("skill_id")), ""
                    )
                    for skill in skills
                    if skill.get("skill_id")
                },
            }
            document["threads"] = threads
            document["defaults"] = defaults
            document["writers"] = writers
            atomic_write_json(self.path, document)

    @staticmethod
    def _hydrate(
        thread: dict[str, Any],
        current_by_skill: dict[str, dict[str, Any]],
    ) -> SkillSnapshot:
        available = _bool_map(thread.get("available"))
        current_by_source = {
            source["source_id"]: skill
            for skill in current_by_skill.values()
            for source in SkillLoader.sources(skill)
        }
        hydrated: list[dict[str, Any]] = []
        for saved in _stored_skills(thread.get("skills")):
            skill_id = str(saved.get("skill_id") or "")
            if not skill_id:
                continue
            metadata = _skill_metadata(saved)
            live = current_by_skill.get(skill_id)
            if live is None:
                live = next(
                    (
                        current_by_source[source["source_id"]]
                        for source in metadata["sources"]
                        if source["source_id"] in current_by_source
                    ),
                    None,
                )
            skill = dict(live or {})
            skill.update(metadata)
            if live is not None:
                live_sources = {
                    source["source_id"]: source for source in SkillLoader.sources(live)
                }
                selected = next(
                    (
                        live_sources[source["source_id"]]
                        for source in metadata["sources"]
                        if source["source_id"] in live_sources
                    ),
                    next(iter(live_sources.values())),
                )
                skill["source_id"] = selected["source_id"]
                skill["source"] = selected["source"]
            skill.setdefault("body", "")
            hydrated.append(skill)
        skill_ids = {SkillLoader.skill_id(skill) for skill in hydrated}
        disabled = frozenset(
            skill_id for skill_id in skill_ids if not available.get(skill_id, True)
        )
        return SkillSnapshot(tuple(hydrated), disabled)

    @staticmethod
    def _migrate_path_state(
        document: dict[str, Any],
        current_records: list[dict[str, Any]],
    ) -> bool:
        """Collapse the earlier source-keyed state into logical skill IDs."""
        threads = _thread_map(document)
        records_by_thread = {
            thread_id: _stored_skills(thread.get("skills"))
            for thread_id, thread in threads.items()
            if isinstance(thread, dict)
        }
        needs_migration = any(
            not skill.get("skill_id")
            for records in records_by_thread.values()
            for skill in records
        )
        if not needs_migration:
            return False

        source_to_skill: dict[str, str] = {}
        group_sources: dict[str, set[str]] = {}
        for skill in current_records:
            skill_id = SkillLoader.skill_id(skill)
            for source in SkillLoader.sources(skill):
                source_id = source["source_id"]
                source_to_skill[source_id] = skill_id
                group_sources.setdefault(skill_id, set()).add(source_id)

        # Logical records identify every source copy, even when discovery no
        # longer finds them. Register these before resolving legacy records so
        # the order of records in a partially migrated file cannot split them.
        for records in records_by_thread.values():
            for skill in records:
                skill_id = str(skill.get("skill_id") or "")
                if not skill_id:
                    continue
                for source in SkillLoader.sources(skill):
                    source_id = source["source_id"]
                    source_to_skill.setdefault(source_id, skill_id)
                    group_sources.setdefault(skill_id, set()).add(source_id)

        for records in records_by_thread.values():
            for skill in records:
                if skill.get("skill_id"):
                    continue
                source_id = SkillLoader.source_id(skill)
                skill_id = source_to_skill.get(source_id, f"source:{source_id}")
                source_to_skill.setdefault(source_id, skill_id)
                group_sources.setdefault(skill_id, set()).add(source_id)

        for thread_id, thread in tuple(threads.items()):
            records = records_by_thread.get(thread_id, [])
            # Already logical snapshots and malformed containers are unrelated
            # to source-key migration. Keep their saved state intact.
            if not any(not skill.get("skill_id") for skill in records):
                continue
            old_available = _bool_map(thread.get("available"))
            old_baseline = _bool_map(thread.get("baseline"))
            old_writers = _text_map(thread.get("baseline_writers"))
            grouped: dict[str, dict[str, Any]] = {}
            value_keys: dict[str, list[str]] = {}

            for saved in records:
                was_logical = bool(saved.get("skill_id"))
                source_id = SkillLoader.source_id(saved)
                skill_id = str(saved.get("skill_id") or "") or source_to_skill.get(
                    source_id, f"source:{source_id}"
                )
                value_keys.setdefault(skill_id, []).append(
                    skill_id if was_logical else source_id
                )
                existing = grouped.get(skill_id)
                if existing is None:
                    metadata = _skill_metadata(
                        {
                            **saved,
                            "skill_id": skill_id,
                            "sources": [],
                        }
                    )
                    metadata["sources"] = []
                    grouped[skill_id] = metadata
                    existing = metadata
                known = {source["source_id"] for source in existing["sources"]}
                for source in SkillLoader.sources(saved):
                    if source["source_id"] not in known:
                        existing["sources"].append(dict(source))
                        known.add(source["source_id"])

            migrated_skills: list[dict[str, Any]] = []
            for skill in grouped.values():
                sources = skill["sources"]
                if not sources:
                    continue
                skill["source_id"] = sources[0]["source_id"]
                skill["source"] = sources[0]["source"]
                migrated_skills.append(skill)

            thread["skills"] = migrated_skills
            thread["available"] = {
                skill_id: _merged_bool(old_available.get(key, True) for key in keys)
                for skill_id, keys in value_keys.items()
            }
            thread["baseline"] = {
                skill_id: _merged_bool(old_baseline.get(key, True) for key in keys)
                for skill_id, keys in value_keys.items()
            }
            thread["baseline_writers"] = {
                skill_id: _merged_writer(old_writers.get(key, "") for key in keys)
                for skill_id, keys in value_keys.items()
            }
            threads[thread_id] = thread

        old_defaults = _bool_map(document.get("defaults"))
        old_default_writers = _text_map(document.get("writers"))
        logical_defaults: dict[str, bool] = {}
        logical_writers: dict[str, str] = {}
        mapped_keys: set[str] = set()
        for skill_id, source_ids in group_sources.items():
            keys = set(source_ids)
            mapped_keys.update((*source_ids, skill_id))
            if skill_id in old_defaults:
                # A logical default already accounts for its source copies.
                # Only actual remaining path defaults can conflict with it.
                keys.intersection_update(old_defaults)
                keys.add(skill_id)
            logical_defaults[skill_id] = _merged_bool(
                old_defaults.get(key, True) for key in keys
            )
            logical_writers[skill_id] = _merged_writer(
                old_default_writers.get(key, "") for key in keys
            )
        for key, value in old_defaults.items():
            if key in mapped_keys:
                continue
            skill_id = (
                key if key.startswith(("sha256:", "source:")) else f"source:{key}"
            )
            logical_defaults[skill_id] = value
            logical_writers[skill_id] = old_default_writers.get(key, "")

        document["threads"] = threads
        document["defaults"] = logical_defaults
        document["writers"] = logical_writers
        return True
