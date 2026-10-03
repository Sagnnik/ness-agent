from __future__ import annotations

import os
import sys
from collections.abc import Sequence


def _worktree_name(argv: Sequence[str]) -> str | None:
    name: str | None = None

    for index, argument in enumerate(argv):
        if argument in {"--worktree", "-w"} and index + 1 < len(argv):
            name = argv[index + 1]
        elif argument.startswith("--worktree="):
            name = argument.split("=", 1)[1]

    return name


def bootstrap_worktree(argv: Sequence[str]) -> None:
    """Bootstrap a persistent worktree before loading config-dependent code."""
    name = _worktree_name(argv)
    if not name:
        return

    from ness_cli.workspace import WorktreeError, ensure_worktree

    try:
        path = ensure_worktree(name)
    except WorktreeError as exc:
        print(f"worktree error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    os.chdir(path)
    os.environ["NESS_AGENT_WORKTREE"] = name
    os.environ["NESS_AGENT_WORKTREE_PATH"] = str(path)


def main(argv: Sequence[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)

    bootstrap_worktree(args)

    if args and args[0] == "mcp":
        from ness_cli.mcp.cli import app

        app(args=args[1:], prog_name="ness mcp")
        return

    from ness_cli.cli import app

    app(args=args)


if __name__ == "__main__":
    main()
