"""Build SDK prompt objects from CLI-owned instruction templates."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ness_agent.context.layers import AuxPrompts, PromptLayers, PromptLayersConfig
from ness_agent.options import ModeConfig

from ness_cli.instructions import load_instruction


def _instruction(name: str, *, instructions_dir: Path | None) -> str:
    return load_instruction(name, instructions_dir=instructions_dir)


def default_prompt_layers(
    *,
    instructions_dir: Path | None = None,
    l2_context: str | None = None,
    **overrides: Any,
) -> PromptLayers:
    """Build the normal prompt layers with optional explicit overrides."""
    overrides.setdefault("include_skill_catalog", False)
    config = PromptLayersConfig(
        l0=_instruction(
            "l0_harness.md",
            instructions_dir=instructions_dir,
        ),
        persona=_instruction(
            "persona.md",
            instructions_dir=instructions_dir,
        ),
        l2_context=l2_context,
        **overrides,
    )
    return PromptLayers(config)


def default_aux_prompts(
    *,
    instructions_dir: Path | None = None,
) -> AuxPrompts:
    """Build compaction, reflection, subagent, and memory prompts."""
    return AuxPrompts(
        compaction=_instruction(
            "compaction.md",
            instructions_dir=instructions_dir,
        ),
        reflection=_instruction(
            "reflection.md",
            instructions_dir=instructions_dir,
        ),
        subagent=_instruction(
            "subagent.md",
            instructions_dir=instructions_dir,
        ),
        thread_summary=_instruction(
            "thread_summary.md",
            instructions_dir=instructions_dir,
        ),
        init_memory=_instruction(
            "init_memory.md",
            instructions_dir=instructions_dir,
        ),
    )


def plan_act_modes(
    *,
    plans_dir: Path | None = None,
    instructions_dir: Path | None = None,
) -> ModeConfig:
    """Build plan and act mode configuration."""
    return ModeConfig(
        plans_dir=plans_dir,
        plan_mode_template=_instruction(
            "plan_mode.md",
            instructions_dir=instructions_dir,
        ),
        act_mode_template=_instruction(
            "act_mode.md",
            instructions_dir=instructions_dir,
        ),
    )


def build_init_memory_prompt(
    project_context: str,
    *,
    instructions_dir: Path | None = None,
) -> str:
    """Fill the project-context placeholder in the memory template."""
    template = _instruction(
        "init_memory.md",
        instructions_dir=instructions_dir,
    )
    return template.format(project_context=project_context)
