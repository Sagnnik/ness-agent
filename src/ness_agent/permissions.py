from __future__ import annotations

import fnmatch
import json
import os
import tempfile
import threading
from pathlib import Path
from contextlib import contextmanager
from contextvars import ContextVar
from collections.abc import Iterator
from typing import Literal
from urllib.parse import urlparse, urlunparse

Decision = Literal["allow", "deny", "ask"]
SHELL_TOOL = "shell"
SHELL_COMMAND_ACTIONS = {"run", "start"}
FETCH_URL_TOOL = "fetch_url"
RuleScope = Literal["always", "session"]

DEFAULT_RULES = {
    "allow": [
        "read:*",
        "grep:*",
        "glob:*",
        "edit:*",
        "write:*",
        "delete:*",
        "web_search:*",
        "skill_view:*",
        "shell:jobs:*",
        "shell:read:*",
        "shell:run:pwd",
        "shell:run:ls*",
        "shell:run:git status*",
        "shell:run:git diff*",
        "shell:run:git log*",
        "shell:run:git show*",
    ],
    "deny": [
        "shell:run:rm*",
        "shell:run:sudo*",
        "shell:run:curl http*",
        "shell:run:wget *",
        "shell:start:rm*",
        "shell:start:sudo*",
        "shell:start:curl http*",
        "shell:start:wget *",
    ],
    "ask": ["*"],
}


class PermissionStore:
    def __init__(
        self,
        *,
        ness_dir: Path = Path(".ness"),
        project_root: Path | None = None,
        _persistent_lock: threading.RLock | None = None,
    ) -> None:
        """Store and evaluate tool-permission rules loaded from ``<ness_dir>/permissions.json``.

        Args:
            ness_dir: Directory containing the ``permissions.json`` file.
            project_root: Base for relative file paths and default native file access. Defaults to cwd.
        """
        self.ness_dir = Path(ness_dir)
        self.permissions_file = self.ness_dir / "permissions.json"
        self.project_root = (project_root or Path.cwd()).resolve()
        self._file_access: ContextVar[tuple[dict, ...]] = ContextVar(
            "ness_file_access", default=()
        )
        self._session_rules: dict[str, list[str]] = {"allow": [], "deny": []}
        self._persistent_lock = _persistent_lock or threading.RLock()

    def fork_for_session(self) -> "PermissionStore":
        """Create a project-backed view with independent temporary rules."""
        fork = PermissionStore(
            ness_dir=self.ness_dir,
            project_root=self.project_root,
            _persistent_lock=self._persistent_lock,
        )
        fork._session_rules = {
            "allow": list(self._session_rules["allow"]),
            "deny": list(self._session_rules["deny"]),
        }
        return fork

    def resolve_path(self, path: str) -> Path:
        """Resolve relative paths against the project without granting access."""
        try:
            candidate = Path(path)
            if not candidate.is_absolute():
                candidate = self.project_root / candidate
            return candidate.resolve()
        except Exception as exc:
            raise ValueError(f"Invalid path: {path} ({exc})") from exc

    def _protected_write(self, path: Path) -> bool:
        protected = (
            self.ness_dir.resolve(),
            *(self.project_root / name for name in (".git", ".ness")),
        )
        return any(path.is_relative_to(root.resolve()) for root in protected)

    @staticmethod
    def _file_rule(access: dict) -> str:
        return "filesystem:" + json.dumps(access, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _covers(access: dict, path: Path, write: bool, recursive: bool = False) -> bool:
        if not isinstance(access, dict) or not isinstance(access.get("path"), str):
            return False
        if not isinstance(access.get("write"), bool) or not isinstance(
            access.get("recursive"), bool
        ):
            return False
        if write and not access.get("write"):
            return False
        if recursive and not access["recursive"]:
            return False
        root = Path(access["path"])
        return root.is_absolute() and (
            path == root or (access["recursive"] and path.is_relative_to(root))
        )

    def _file_decision(
        self, path: Path, *, write: bool, recursive: bool = False
    ) -> Decision:
        rules = self._load()
        # Denials win over remembered grants and one-call approvals.
        for bucket in ("deny", "allow"):
            for rule in (*rules.get(bucket, []), *self._session_rules[bucket]):
                if not isinstance(rule, str) or not rule.startswith("filesystem:"):
                    continue
                try:
                    access = json.loads(rule.removeprefix("filesystem:"))
                    if (
                        bucket == "deny"
                        and isinstance(access, dict)
                        and access.get("write") != write
                    ):
                        continue
                    if self._covers(access, path, write, recursive):
                        return bucket
                except (TypeError, ValueError, KeyError):
                    continue
        if any(
            self._covers(access, path, write, recursive)
            for access in self._file_access.get()
        ):
            return "allow"
        return "ask"

    def file_access_requests(self, tool: str, args: dict) -> list[dict]:
        """Return canonical external paths whose access needs approval.

        Search directories request recursive read access; mutations request only
        their exact targets. Tool-wide allow rules do not grant filesystem access.
        """
        if tool not in {"read", "write", "edit", "delete", "grep", "glob"}:
            return []
        write = tool in {"write", "edit", "delete"}
        paths = args.get("paths", []) if tool == "delete" else [args.get("path", ".")]
        requests = []
        for raw in paths:
            try:
                path = self.resolve_path(raw)
            except (TypeError, ValueError):
                continue  # Invalid inputs are reported by the tool itself.
            if path.is_relative_to(self.project_root) or (
                write and self._protected_write(path)
            ):
                continue
            recursive = tool == "glob" or (tool == "grep" and path.is_dir())
            if self._file_decision(path, write=write, recursive=recursive) == "ask":
                requests.append(
                    {
                        "path": str(path),
                        "write": write,
                        "recursive": recursive,
                    }
                )
        return requests

    def remember_file_access(
        self, requests: list[dict], *, allow: bool, scope: RuleScope
    ) -> None:
        for access in requests:
            self.persist_rule(
                self._file_rule(access), "allow" if allow else "deny", scope=scope
            )

    @contextmanager
    def file_access(self, requests: list[dict]) -> Iterator[None]:
        """Install approvals only for this tool invocation and its worker context."""
        token = self._file_access.set(tuple(requests))
        try:
            yield
        finally:
            self._file_access.reset(token)

    def validate_path(
        self,
        path: str,
        *,
        write: bool = False,
        yolo_mode: bool = False,
        recursive: bool = False,
    ) -> str:
        """Resolve a path and enforce project scope or an approved external grant.

        YOLO bypasses scope and protected-write checks; OS permissions still apply.
        """
        resolved = self.resolve_path(path)
        if yolo_mode:
            return str(resolved)
        if write and self._protected_write(resolved):
            raise PermissionError(f"refusing to modify protected path {path}")
        if (
            not resolved.is_relative_to(self.project_root)
            and self._file_decision(resolved, write=write, recursive=recursive)
            != "allow"
        ):
            raise PermissionError(
                f"{resolved} is outside {self.project_root}; access requires approval"
            )
        return str(resolved)

    def relative_to_root(
        self, path: str, *, write: bool = False, yolo_mode: bool = False
    ) -> str:
        """Return a project-relative label, or an absolute external-path label."""
        resolved = Path(self.validate_path(path, write=write, yolo_mode=yolo_mode))
        if resolved.is_relative_to(self.project_root):
            return str(resolved.relative_to(self.project_root))
        return str(resolved)

    def check(self, tool: str, args: dict) -> Decision:
        """Return the effective decision (``"allow"``, ``"deny"``, or ``"ask"``)
        for a tool invocation without revealing which rule matched.

        See Also:
            :meth:`check_with_rule` — returns the matching rule as well.
        """
        decision, _ = self.check_with_rule(tool, args)
        return decision

    def check_with_rule(self, tool: str, args: dict) -> tuple[Decision, str | None]:
        decision, rule = self._check_tool_with_rule(tool, args)
        if decision == "deny":
            return decision, rule
        if tool in {"read", "write", "edit", "delete", "grep", "glob"}:
            write = tool in {"write", "edit", "delete"}
            paths = (
                args.get("paths", []) if tool == "delete" else [args.get("path", ".")]
            )
            for raw in paths:
                try:
                    path = self.resolve_path(raw)
                except (TypeError, ValueError):
                    continue
                if not path.is_relative_to(self.project_root):
                    if self._file_decision(path, write=write) == "deny":
                        return "deny", "filesystem access denied"
            if self.file_access_requests(tool, args):
                return "ask", "outside project access"
        return decision, rule
    def _check_tool_with_rule(self, tool: str, args: dict) -> tuple[Decision, str | None]:
        """Evaluate *tool* + *args* against persisted and session rules.

        Returns ``(decision, matched_rule)`` where *matched_rule* is the
        first rule that matched, or ``None`` if the catch-all ``"ask"``
        was used.

        Priority (first match wins): deny (persisted) → deny (session) →
        allow (persisted) → allow (session) → ask (persisted).
        """
        rules = self._load()
        key = self.pattern_key(tool, args)

        for r in rules.get("deny", []):
            if self._matches(r, key, tool): 
                return "deny", r

        for r in self._session_rules.get("deny", []):
            if self._matches(r, key, tool): 
                return "deny", r

        for r in rules.get("allow", []):
            if self._matches(r, key, tool): 
                return "allow", r

        for r in self._session_rules.get("allow", []):
            if self._matches(r, key, tool): 
                return "allow", r

        for r in rules.get("ask", []):
            if self._matches(r, key, tool): 
                return "ask", r

        return "ask", None

    def clear_session_rules(self) -> None:
        """Remove all session-scoped rules added via :meth:`persist_rule` during this session.

        Persisted (file-backed) rules are unaffected.
        """
        self._session_rules["allow"].clear()
        self._session_rules["deny"].clear()

    def _load(self) -> dict:
        with self._persistent_lock:
            if not self.permissions_file.exists():
                self.ness_dir.mkdir(parents=True, exist_ok=True)
                self._save(DEFAULT_RULES)
                return json.loads(json.dumps(DEFAULT_RULES))
            try:
                data = json.loads(self.permissions_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                data = {}
            for key in ("allow", "deny", "ask"):
                data.setdefault(key, [])
            return data


    def _save(self, rules: dict) -> None:
        with self._persistent_lock:
            self.ness_dir.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(
                dir=self.ness_dir,
                prefix=self.permissions_file.name,
                suffix=".tmp",
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(rules, handle, indent=2)
                    handle.write("\n")
                os.replace(tmp_name, self.permissions_file)
            except BaseException:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
                raise

    def remove_rule(self, bucket: Literal["allow", "deny"], index: int) -> str:
        """Remove a persisted rule at *index* from the given *bucket*.

        Returns the removed rule string.

        Raises:
            ValueError: If *index* is out of range for that bucket.
        """
        with self._persistent_lock:
            rules = self._load()
            try:
                removed = rules.get(bucket, []).pop(index)
            except IndexError as exc:
                raise ValueError(f"No {bucket} rule at index {index}") from exc
            self._save(rules)
            return removed

    def list_rules(self) -> str:
        """Return the full ruleset as a pretty-printed JSON string (read from disk)."""
        return json.dumps(self._load(), indent=2)

    def _shell_action(self, args: dict) -> str:
        return str(args.get("action") or "run").strip().lower()

    def _is_default_port(self, scheme: str, port: int) -> bool:
        return (scheme == "http" and port == 80) or (scheme == "https" and port == 443)

    def _normalize_permission_url(self, url: str) -> str:
        parsed = urlparse(url)
        hostname = parsed.hostname
        if not parsed.scheme or not hostname:
            return url.strip()

        scheme = parsed.scheme.lower()
        host = hostname.lower().rstrip(".")
        netloc = host
        if ":" in host and not host.startswith("["):
            netloc = f"[{host}]"
        try:
            port = parsed.port
        except ValueError:
            return url.strip()
        if port and not self._is_default_port(scheme, port):
            netloc = f"{netloc}:{port}"
        return urlunparse((scheme, netloc, parsed.path or "", parsed.params or "", parsed.query or "", ""))

    def pattern_key(self, tool: str, args: dict) -> str:
        """Build the pattern-matching key for a tool call.

        Format: ``tool:action:detail`` for shell, ``tool:url=<normalized>``
        for fetch_url, or ``tool:key1=val1,key2=val2`` for everything else.
        Used internally by :meth:`check_with_rule` and :meth:`_matches`.
        """
        if tool == SHELL_TOOL:
            action = self._shell_action(args)
            if action in SHELL_COMMAND_ACTIONS:
                return f"{SHELL_TOOL}:{action}:{args.get('command', '')}"
            parts = [f"{key}={args[key]}" for key in sorted(args) if key != "action"]
            detail = ",".join(parts) if parts else "*"
            return f"{SHELL_TOOL}:{action}:{detail}"
        if tool == FETCH_URL_TOOL:
            return f"{FETCH_URL_TOOL}:url={self._normalize_permission_url(str(args.get('url') or ''))}"
        if not args:
            return tool
        parts = [f"{key}={args[key]}" for key in sorted(args)]
        return f"{tool}:{','.join(parts)}"


    def default_rule_for(self, tool: str, args: dict) -> str:
        """Generate a sensible default permission-rule pattern for a tool call.

        For example, ``"shell:run:git *"`` for ``git`` commands,
        ``"shell:run:python -m pytest*"`` for ``python -m`` invocations,
        or a plain :meth:`pattern_key` for everything else.

        This is used by the permission-prompt flow to pre-fill a rule when
        the user chooses to remember their decision.
        """
        if tool == FETCH_URL_TOOL:
            return self.pattern_key(tool, args)
        if tool != SHELL_TOOL:
            return self.pattern_key(tool, args)
        action = self._shell_action(args)
        if action not in SHELL_COMMAND_ACTIONS:
            return self.pattern_key(tool, args)
        command = args.get("command", "").strip()
        parts = command.split()
        if not parts:
            return f"{SHELL_TOOL}:{action}:"
        if parts[0] == "git" and len(parts) >= 2:
            return f"{SHELL_TOOL}:{action}:git {parts[1]}*"
        if parts[0] == "python" and len(parts) >= 3 and parts[1] == "-m":
            return f"{SHELL_TOOL}:{action}:python -m {parts[2]}*"
        if parts[0] in {"npm", "npx", "pnpm", "yarn"} and len(parts) >= 3 and parts[1] == "run":
            return f"{SHELL_TOOL}:{action}:{parts[0]} run {parts[2]}*"
        return f"{SHELL_TOOL}:{action}:{parts[0]}*"

    def persist_rule(
        self,
        rule: str,
        bucket: Literal["allow", "deny"],
        scope: RuleScope = "always",
    ) -> None:
        """Save a rule to the *bucket* (``"allow"`` or ``"deny"``).

        When *scope* is ``"session"`` the rule is kept in memory for the
        duration of the session.  When *scope* is ``"always"`` (default)
        it is written to ``permissions.json`` on disk.

        Duplicate rules are silently ignored.
        """
        if scope == "session":
            if rule not in self._session_rules[bucket]:
                self._session_rules[bucket].append(rule)
            return
        with self._persistent_lock:
            rules = self._load()
            rules.setdefault(bucket, [])
            if rule not in rules[bucket]:
                rules[bucket].append(rule)
            self._save(rules)

    def _has_unquoted_shell_operators(self, command: str) -> bool:
        """Return True when command contains unquoted shell chaining or substitution."""
        in_single = False
        in_double = False
        i = 0
        length = len(command)
        while i < length:
            char = command[i]
            if in_single:
                if char == "'":
                    in_single = False
                i += 1
                continue
            if in_double:
                if char == "\\" and i + 1 < length:
                    i += 2
                    continue
                if char == '"':
                    in_double = False
                i += 1
                continue
            if char == "'":
                in_single = True
                i += 1
                continue
            if char == '"':
                in_double = True
                i += 1
                continue
            for operator in ("&&", "||", "$(", "\n"):
                if command.startswith(operator, i):
                    return True
            if char in ";|&`<>":
                return True
            i += 1
        return False

    def _shell_command_matches(self, rule: str, command: str) -> bool:
        if self._has_unquoted_shell_operators(command):
            return False
        return fnmatch.fnmatch(command.strip(), rule.strip())

    def _matches(self, rule: str, key: str, tool: str) -> bool:
        if rule == "*":
            return True
        if ":" not in rule:
            return fnmatch.fnmatch(tool, rule) or fnmatch.fnmatch(key, rule)
        rule_tool, rule_args = rule.split(":", 1)
        key_tool, _, key_args = key.partition(":")
        if rule_tool != key_tool:
            return False
        if rule_tool == SHELL_TOOL:
            if rule_args == "*":
                key_action, _, key_detail = key_args.partition(":")
                if key_action in SHELL_COMMAND_ACTIONS:
                    return self._shell_command_matches("*", key_detail)
                return True
            rule_action, _, rule_detail = rule_args.partition(":")
            key_action, _, key_detail = key_args.partition(":")
            if rule_action != "*" and rule_action != key_action:
                return False
            if key_action in SHELL_COMMAND_ACTIONS:
                return self._shell_command_matches(rule_detail, key_detail)
            return fnmatch.fnmatch(key_detail, rule_detail)
        if rule_tool == FETCH_URL_TOOL:
            return key_args == rule_args
        return fnmatch.fnmatch(key_args, rule_args)
