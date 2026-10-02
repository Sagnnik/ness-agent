from __future__ import annotations


from ness_agent.instructions import (
    ACT_MODE,
    COMPACTION,
    INIT_MEMORY,
    L0_HARNESS,
    PLAN_MODE,
    REFLECTION,
    SUBAGENT,
    THREAD_SUMMARY,
)
from ness_cli.instructions import (
    INSTRUCTION_FILES,
    clear_instruction_cache,
    default_instruction_files,
    load_instruction,
    packaged_instruction,
)
from ness_cli.prompts import (
    build_init_memory_prompt,
    default_aux_prompts,
    default_prompt_layers,
    plan_act_modes,
)


_SDK_PARITY = {
    "l0_harness.md": L0_HARNESS,
    "plan_mode.md": PLAN_MODE,
    "act_mode.md": ACT_MODE,
    "compaction.md": COMPACTION,
    "reflection.md": REFLECTION,
    "subagent.md": SUBAGENT,
    "thread_summary.md": THREAD_SUMMARY,
    "init_memory.md": INIT_MEMORY,
}


def test_every_legacy_instruction_has_a_packaged_replacement() -> None:
    files = default_instruction_files()

    assert set(files) == set(INSTRUCTION_FILES)
    assert all(content.strip() for content in files.values())
    assert set(_SDK_PARITY) < set(files)


def test_packaged_sdk_instructions_preserve_content() -> None:
    for name, expected in _SDK_PARITY.items():
        assert packaged_instruction(name) == expected.strip()


def test_project_override_wins_and_missing_files_fall_back(
    isolated_cli_env,
) -> None:
    clear_instruction_cache()
    directory = isolated_cli_env.root / "instructions"
    directory.mkdir()
    (directory / "l0_harness.md").write_text("PROJECT L0\n", encoding="utf-8")

    assert load_instruction("l0_harness.md", instructions_dir=directory) == (
        "PROJECT L0"
    )
    assert load_instruction("persona.md", instructions_dir=directory) == (
        packaged_instruction("persona.md")
    )


def test_prompt_layers_preserve_explicit_order(isolated_cli_env) -> None:
    directory = isolated_cli_env.root / "layers"
    directory.mkdir()
    (directory / "l0_harness.md").write_text("L0", encoding="utf-8")
    (directory / "persona.md").write_text("PERSONA", encoding="utf-8")

    layers = default_prompt_layers(instructions_dir=directory, l2_context="PROJECT")

    assert layers.build_l0() == "L0"
    assert layers.config.persona == "PERSONA"
    assert layers.config.l2_context == "PROJECT"


def test_auxiliary_mode_and_memory_prompts_use_explicit_directory(
    isolated_cli_env,
) -> None:
    directory = isolated_cli_env.root / "explicit"
    directory.mkdir()
    for name in (
        "compaction.md",
        "reflection.md",
        "subagent.md",
        "thread_summary.md",
        "init_memory.md",
        "plan_mode.md",
        "act_mode.md",
    ):
        (directory / name).write_text(f"BODY:{name}", encoding="utf-8")

    aux = default_aux_prompts(instructions_dir=directory)
    modes = plan_act_modes(
        plans_dir=isolated_cli_env.root / "plans",
        instructions_dir=directory,
    )

    assert aux.compaction == "BODY:compaction.md"
    assert aux.reflection == "BODY:reflection.md"
    assert aux.subagent == "BODY:subagent.md"
    assert aux.thread_summary == "BODY:thread_summary.md"
    assert modes.plan_mode_template == "BODY:plan_mode.md"
    assert modes.act_mode_template == "BODY:act_mode.md"
    assert build_init_memory_prompt("context", instructions_dir=directory) == (
        "BODY:init_memory.md"
    )


def test_goal_prompts_load_from_explicit_directory(isolated_cli_env) -> None:
    directory = isolated_cli_env.root / "goals"
    directory.mkdir()
    for name in ("goal_judge.md", "goal_repair.md", "goal_generic_repair.md"):
        (directory / name).write_text(f"BODY:{name}", encoding="utf-8")

    assert load_instruction("goal_judge.md", instructions_dir=directory) == (
        "BODY:goal_judge.md"
    )
    assert load_instruction("goal_repair.md", instructions_dir=directory) == (
        "BODY:goal_repair.md"
    )
    assert load_instruction("goal_generic_repair.md", instructions_dir=directory) == (
        "BODY:goal_generic_repair.md"
    )
