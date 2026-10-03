from __future__ import annotations

import json
from pathlib import Path

from langchain_core.tools import tool


@tool
def skill_view(name: str) -> str:
    """Load a specialized skill when the task at hand matches one of the skills listed in the system prompt. 
    Use this tool to inject the skill's instructions and resources into current conversation. 
    The output may contain detailed workflow guidance as well as references to scripts, files, etc in the same directory as the skill. 
    The skill name must match one of the skills listed in your system prompt.
    """
    from ness_agent.session_context import get_session_context

    rt = get_session_context()
    skills = rt.available_skills or {}
    skill = skills.get(name)
    if not skill:
        available = ", ".join(sorted(skills))
        return f"Error: unknown skill '{name}'. Available: {available}"

    body = skill.get("body") or ""
    skill_dir = Path(skill["source"]).resolve().parent

    linked_files: dict[str, list[str]] = {}
    for child in sorted(skill_dir.iterdir()):
        if not child.is_dir():
            continue
        files = sorted(
            str(p.resolve()) for p in child.rglob("*") if p.is_file()
        )
        if files:
            linked_files[child.name] = files

    payload = {
        "content": body,
        "linked_files": linked_files,
        "usage_hint": "To view linked files, call read(path=...) tool",
    }
    return json.dumps(payload, indent=2)
