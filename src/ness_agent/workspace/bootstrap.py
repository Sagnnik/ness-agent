from __future__ import annotations

import json
from pathlib import Path

from ness_agent.defaults import default_agent_profiles
from ness_agent.permissions import DEFAULT_RULES

# Project-local directories created under ``.ness/``.
# Global USER.md / plans live outside the project (see ness_cli.paths).
NESS_SUBDIRS = (
    "agents",
    "commands",
    "threads",
    "runtime/sessions",
    "runtime/shells",
)


def setup_ness_structure(ness_dir: Path) -> list[str]:
    """Create the standard project ``.ness/`` tree and default config files.

    Creates directories (agents/commands/threads/runtime/sessions|shells),
    default ``permissions.json``, ``hooks.json``, and ``mcp.json``, seeds
    built-in subagent profiles under ``agents/`` when missing, and creates an
    empty ``NESS.md`` when missing. Existing files are never overwritten.
    Returns a list of paths that were created.
    """
    created: list[str] = []
    root = Path(ness_dir)
    if not root.exists():
        root.mkdir(parents=True, exist_ok=True)
        created.append(str(root))

    for name in NESS_SUBDIRS:
        p = root / name
        if not p.exists():
            p.mkdir(parents=True, exist_ok=True)
            created.append(str(p))

    for path, content in {
        root / "permissions.json": json.dumps(DEFAULT_RULES, indent=2) + "\n",
        root / "hooks.json": "{}\n",
        root / "mcp.json": json.dumps({"mcpServers": {}}, indent=2) + "\n",
    }.items():
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            created.append(str(path))

    agents_dir = root / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)
    for filename, content in default_agent_profiles().items():
        path = agents_dir / filename
        if not path.exists():
            path.write_text(content, encoding="utf-8")
            created.append(str(path))

    ness_md = root / "NESS.md"
    if not ness_md.exists():
        ness_md.write_text("", encoding="utf-8")
        created.append(str(ness_md))

    return created
