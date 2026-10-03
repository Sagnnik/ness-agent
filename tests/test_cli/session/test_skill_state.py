from __future__ import annotations

import json
from pathlib import Path

import pytest

from ness_agent.skills import SkillLoader
from ness_cli.session.skill_state import SkillStateStore


def _skill(path: Path, name: str = "skill") -> dict[str, str]:
    return {
        "name": name,
        "description": f"Description for {name}",
        "source": str(path),
        "source_id": str(path.resolve()),
        "body": f"Body for {name}",
    }


def test_new_threads_use_global_defaults_but_keep_isolated_snapshots(tmp_path: Path):
    store = SkillStateStore(tmp_path / "global" / "skill-state.json")
    record = _skill(tmp_path / "project" / ".agents/skills/a/SKILL.md")
    skill_id = SkillLoader.skill_id(record)

    first = store.load_or_create("session-a", [record])
    assert first.disabled_skill_ids == frozenset()
    assert store.apply("session-a", {skill_id}) == frozenset({skill_id})

    second = store.load_or_create("session-b", [record])
    assert second.disabled_skill_ids == frozenset({skill_id})
    assert store.load_or_create("session-a", [record]).disabled_skill_ids == frozenset(
        {skill_id}
    )

    store.apply("session-b", set())
    assert store.load_or_create("session-a", [record]).disabled_skill_ids == frozenset(
        {skill_id}
    )


def test_disjoint_changes_merge_for_future_threads(tmp_path: Path):
    store = SkillStateStore(tmp_path / "skill-state.json")
    first = _skill(tmp_path / "a/SKILL.md", "a")
    second = _skill(tmp_path / "b/SKILL.md", "b")
    first_id = SkillLoader.skill_id(first)
    second_id = SkillLoader.skill_id(second)

    store.load_or_create("session-a", [first, second])
    store.load_or_create("session-b", [first, second])
    store.apply("session-a", {first_id})
    store.apply("session-b", {second_id})

    future = store.load_or_create("session-c", [first, second])
    assert future.disabled_skill_ids == frozenset({first_id, second_id})


def test_conflicting_stale_changes_default_to_available(tmp_path: Path):
    store = SkillStateStore(tmp_path / "skill-state.json")
    record = _skill(tmp_path / "shared/SKILL.md")
    skill_id = SkillLoader.skill_id(record)

    store.load_or_create("session-a", [record])
    store.load_or_create("session-b", [record])
    store.apply("session-a", {skill_id})
    store.apply("session-b", {skill_id})
    store.apply("session-a", set())
    store.apply("session-b", set())

    # Session A changed the same default after B's last write. B then submits
    # the opposite stale choice; availability wins the disagreement.
    store.apply("session-a", {skill_id})
    store.apply("session-b", set())
    future = store.load_or_create("session-c", [record])
    assert future.disabled_skill_ids == frozenset()


def test_fork_copies_parent_snapshot(tmp_path: Path):
    store = SkillStateStore(tmp_path / "skill-state.json")
    record = _skill(tmp_path / "a/SKILL.md")
    skill_id = SkillLoader.skill_id(record)

    store.load_or_create("parent", [record])
    store.apply("parent", {skill_id})
    store.copy_thread("parent", "child")

    child = store.load_or_create("child", [])
    assert child.disabled_skill_ids == frozenset({skill_id})
    assert child.skills[0]["name"] == "skill"


def test_path_keyed_state_migrates_and_collapses_identical_sources(tmp_path: Path):
    state_file = tmp_path / "skill-state.json"
    first = _skill(tmp_path / ".codex/skills/browser-use/SKILL.md", "browser-use")
    second = _skill(tmp_path / ".cursor/skills/browser-use/SKILL.md", "browser-use")
    first_id = SkillLoader.source_id(first)
    second_id = SkillLoader.source_id(second)
    logical_id = "sha256:browser-use"
    current = {
        **first,
        "skill_id": logical_id,
        "sources": [
            {"source_id": first_id, "source": first["source"]},
            {"source_id": second_id, "source": second["source"]},
        ],
    }
    state_file.write_text(
        json.dumps(
            {
                "defaults": {first_id: False, second_id: False},
                "writers": {first_id: "session-a", second_id: "session-a"},
                "threads": {
                    "session-a": {
                        "skills": [first, second],
                        "available": {first_id: False, second_id: True},
                        "baseline": {first_id: True, second_id: True},
                        "baseline_writers": {first_id: "", second_id: ""},
                    }
                },
            }
        )
    )
    store = SkillStateStore(state_file)

    snapshot = store.load_or_create("session-a", [current])
    migrated = json.loads(state_file.read_text())

    assert len(snapshot.skills) == 1
    assert snapshot.disabled_skill_ids == frozenset()
    assert len(migrated["threads"]["session-a"]["skills"]) == 1
    assert migrated["defaults"] == {logical_id: False}


@pytest.mark.parametrize(
    ("skills", "legacy"),
    [
        (None, False),
        (None, True),
        (7, True),
        ("invalid", True),
        ({"bad": "container"}, True),
    ],
)
def test_bad_skill_container_cannot_block_another_thread(tmp_path, skills, legacy):
    path = tmp_path / "skill-state.json"
    record = _skill(tmp_path / "valid/SKILL.md")
    store = SkillStateStore(path)
    store.load_or_create("valid", [record])
    skill_id = SkillLoader.skill_id(record)
    store.apply("valid", {skill_id})
    document = json.loads(path.read_text())
    document["threads"]["broken"] = {"skills": skills, "available": None}
    if legacy:
        document["threads"]["legacy"] = {
            "skills": [_skill(tmp_path / "old/SKILL.md", "old")]
        }
    path.write_text(json.dumps(document))
    original = document["threads"]["valid"]

    snapshot = store.load_or_create("valid", [record])
    assert snapshot.disabled_skill_ids == frozenset({skill_id})
    created = store.load_or_create("new", [record])
    assert created.disabled_skill_ids == frozenset({skill_id})
    saved = json.loads(path.read_text())
    assert saved["threads"]["valid"] == original
    assert saved["threads"]["broken"] == document["threads"]["broken"]


@pytest.mark.parametrize("skills", [None, {"bad": "container"}])
def test_loading_bad_skill_container_creates_a_fresh_snapshot(tmp_path, skills):
    path = tmp_path / "skill-state.json"
    record = _skill(tmp_path / "valid/SKILL.md")
    path.write_text(json.dumps({"threads": {"broken": {"skills": skills}}}))
    snapshot = SkillStateStore(path).load_or_create("broken", [record])
    assert len(snapshot.skills) == 1
    assert snapshot.skills[0]["source_id"] == record["source_id"]


def test_malformed_list_items_are_skipped_during_migration_and_hydration(tmp_path):
    path = tmp_path / "skill-state.json"
    record = _skill(tmp_path / "valid/SKILL.md")
    invalid = [
        None,
        7,
        False,
        "invalid",
        [],
        {},
        {"skill_id": [], "source": record["source"]},
        {"source": []},
        {"source": "\u0000"},
    ]
    path.write_text(
        json.dumps(
            {
                "threads": {
                    "mixed": {
                        "skills": [*invalid, record],
                        "available": {record["source_id"]: False},
                    }
                }
            }
        )
    )
    store = SkillStateStore(path)
    snapshot = store.load_or_create("mixed", [record])
    assert len(snapshot.skills) == 1
    assert snapshot.disabled_skill_ids == frozenset({SkillLoader.skill_id(record)})
    assert len(json.loads(path.read_text())["threads"]["mixed"]["skills"]) == 1


def test_partially_migrated_state_preserves_existing_snapshot_and_default(tmp_path):
    path = tmp_path / "skill-state.json"
    record = _skill(tmp_path / "valid/SKILL.md")
    record["skill_id"] = "sha256:valid"
    store = SkillStateStore(path)
    store.load_or_create("valid", [record])
    store.apply("valid", {record["skill_id"]})
    document = json.loads(path.read_text())
    original = document["threads"]["valid"]
    document["threads"]["old"] = {"skills": [_skill(tmp_path / "old/SKILL.md", "old")]}
    path.write_text(json.dumps(document))

    created = store.load_or_create("new", [record])
    assert created.disabled_skill_ids == frozenset({record["skill_id"]})
    saved = json.loads(path.read_text())
    assert saved["threads"]["valid"] == original
    assert saved["defaults"][record["skill_id"]] is False
    assert saved["writers"][record["skill_id"]] == "valid"


@pytest.mark.parametrize("skills", [None, {"bad": "container"}])
def test_fork_handles_invalid_skill_containers(tmp_path, skills):
    path = tmp_path / "skill-state.json"
    path.write_text(json.dumps({"threads": {"parent": {"skills": skills}}}))
    store = SkillStateStore(path)
    store.copy_thread("parent", "child")
    child = store.load_or_create("child", [])
    assert child.skills == ()


@pytest.mark.parametrize("threads", [None, []])
def test_invalid_thread_container_does_not_block_creation(tmp_path, threads):
    path = tmp_path / "skill-state.json"
    path.write_text(json.dumps({"threads": threads}))
    record = _skill(tmp_path / "valid/SKILL.md")
    snapshot = SkillStateStore(path).load_or_create("new", [record])
    assert len(snapshot.skills) == 1


@pytest.mark.parametrize("bad_thread", [None, []])
def test_invalid_thread_record_is_preserved_during_migration(tmp_path, bad_thread):
    path = tmp_path / "skill-state.json"
    record = _skill(tmp_path / "valid/SKILL.md")
    path.write_text(
        json.dumps({"threads": {"broken": bad_thread, "old": {"skills": [record]}}})
    )
    snapshot = SkillStateStore(path).load_or_create("new", [record])
    assert len(snapshot.skills) == 1
    assert json.loads(path.read_text())["threads"]["broken"] == bad_thread


def test_logical_snapshot_skips_invalid_items_and_recovers_nested_sources(tmp_path):
    path = tmp_path / "skill-state.json"
    record = _skill(tmp_path / "valid/SKILL.md")
    skill_id = "sha256:valid"
    saved = {
        "skill_id": skill_id,
        "name": "skill",
        "sources": [
            None,
            [],
            {},
            {"source": ["bad"]},
            {"source": "\u0000"},
            {"source_id": record["source_id"], "source": record["source"]},
        ],
    }
    document = {
        "threads": {
            "valid": {
                "skills": [None, {}, {"skill_id": []}, saved],
                "available": {skill_id: False},
            }
        }
    }
    path.write_text(json.dumps(document))
    store = SkillStateStore(path)
    snapshot = store.load_or_create("valid", [{**record, "skill_id": skill_id}])
    assert len(snapshot.skills) == 1
    assert snapshot.skills[0]["source_id"] == record["source_id"]
    assert snapshot.disabled_skill_ids == frozenset({skill_id})
    assert json.loads(path.read_text()) == document
    store.copy_thread("valid", "child")
    child = store.load_or_create("child", [])
    assert child.disabled_skill_ids == snapshot.disabled_skill_ids
    assert len(child.skills) == 1
    assert store.apply("valid", []) == frozenset()


def test_mixed_thread_recognizes_all_logical_sources_without_discovery(tmp_path):
    path = tmp_path / "skill-state.json"
    first = _skill(tmp_path / "first/SKILL.md")
    second = _skill(tmp_path / "second/SKILL.md")
    skill_id = "sha256:shared"
    logical = {
        **first,
        "skill_id": skill_id,
        "sources": [
            {"source_id": record["source_id"], "source": record["source"]}
            for record in (first, second)
        ],
    }
    path.write_text(
        json.dumps(
            {
                "defaults": {skill_id: False, second["source_id"]: False},
                "writers": {skill_id: "old", second["source_id"]: "old"},
                "threads": {
                    "mixed": {
                        # A legacy copy can precede its logical record in a partially
                        # migrated snapshot. Discovery may no longer find these files.
                        "skills": [second, None, logical],
                        "available": {skill_id: False, second["source_id"]: False},
                        "baseline": {skill_id: False, second["source_id"]: False},
                        "baseline_writers": {
                            skill_id: "old",
                            second["source_id"]: "old",
                        },
                    }
                },
            }
        )
    )
    store = SkillStateStore(path)
    snapshot = store.load_or_create("mixed", [])
    assert len(snapshot.skills) == 1
    assert snapshot.disabled_skill_ids == frozenset({skill_id})
    saved = json.loads(path.read_text())
    assert saved["defaults"] == {skill_id: False}
    assert saved["writers"] == {skill_id: "old"}
    thread = saved["threads"]["mixed"]
    assert thread["baseline"] == {skill_id: False}
    assert thread["baseline_writers"] == {skill_id: "old"}
    assert {source["source_id"] for source in thread["skills"][0]["sources"]} == {
        first["source_id"],
        second["source_id"],
    }
    assert (
        store.load_or_create("mixed", []).disabled_skill_ids
        == snapshot.disabled_skill_ids
    )
    assert json.loads(path.read_text()) == saved


@pytest.mark.parametrize("value", [None, []])
def test_bad_setting_maps_do_not_block_migration_or_apply(tmp_path, value):
    path = tmp_path / "skill-state.json"
    record = _skill(tmp_path / "valid/SKILL.md")
    path.write_text(
        json.dumps(
            {
                "defaults": value,
                "writers": value,
                "threads": {
                    "old": {
                        "skills": [record],
                        "available": value,
                        "baseline": value,
                        "baseline_writers": value,
                    }
                },
            }
        )
    )
    store = SkillStateStore(path)
    snapshot = store.load_or_create("old", [record])
    assert len(snapshot.skills) == 1
    assert snapshot.disabled_skill_ids == frozenset()
    skill_id = SkillLoader.skill_id(record)
    assert store.apply("old", {skill_id}) == frozenset({skill_id})
    assert store.load_or_create("new", [record]).disabled_skill_ids == frozenset(
        {skill_id}
    )
