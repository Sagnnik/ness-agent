"""Load editable global instructions with packaged fallbacks."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_INSTRUCTIONS_DIR = Path(__file__).resolve().parent

INSTRUCTION_FILES: tuple[str, ...] = (
    "l0_harness.md",
    "persona.md",
    "plan_mode.md",
    "act_mode.md",
    "compaction.md",
    "reflection.md",
    "subagent.md",
    "thread_summary.md",
    "init_memory.md",
    "goal_judge.md",
    "goal_repair.md",
    "goal_generic_repair.md",
)


def _instruction_name(name: str) -> str:
    path = Path(name)
    if path.name != name or name not in INSTRUCTION_FILES:
        raise ValueError(f"unknown instruction: {name}")
    return name


def default_instruction_files() -> dict[str, str]:
    """Return all packaged templates keyed by filename."""
    return {
        name: (_INSTRUCTIONS_DIR / name).read_text(encoding="utf-8")
        for name in INSTRUCTION_FILES
    }


@lru_cache(maxsize=None)
def packaged_instruction(name: str) -> str:
    """Load a packaged instruction by filename."""
    filename = _instruction_name(name)
    return (_INSTRUCTIONS_DIR / filename).read_text(encoding="utf-8").strip()


@lru_cache(maxsize=None)
def _read_instruction_file(path: str) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


def load_instruction(
    name: str,
    *,
    instructions_dir: Path | None = None,
) -> str:
    """Prefer a non-empty editable template, then use the packaged default."""
    filename = _instruction_name(name)
    directory = instructions_dir

    if directory is None:
        from ness_cli.paths import config_dir_from_env

        directory = config_dir_from_env() / "instructions"

    path = directory / filename

    try:
        if path.is_file():
            content = _read_instruction_file(str(path.resolve()))
            if content:
                return content
    except OSError:
        pass

    return packaged_instruction(filename)


def clear_instruction_cache() -> None:
    """Clear process-local instruction caches."""
    _read_instruction_file.cache_clear()
    packaged_instruction.cache_clear()
