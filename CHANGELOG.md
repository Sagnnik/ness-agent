# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Per-session effective SDK configuration: `NessAgent.session()` now accepts main and reflection model overrides, `Session.config` / `Session.cost_tracker` expose the effective session view, and `Session.configure_models()` can rebind one live session without changing its siblings.
- `NessAgent.configure_default_models()` for updating the model defaults inherited by sessions created in the future.

### Changed

- A `NessAgent` now acts as a project-scoped owner of shared services and defaults while every `Session` receives isolated mutable runtime views for options, temporary permission rules, active MCP tools, and cost totals. Thread persistence, hooks, memory, tool definitions, persistent permission rules, tracing, and pricing remain shared.
- CLI model/provider/reasoning changes now update the selected thread and future sessions only; other live thread runtimes retain the model configuration with which they were created.
- Cost reporting is session-first: live usage accumulates in both the session tracker and the agent's current-process aggregate, while replayed durable usage restores only the session total.

### Fixed

- Concurrent `/new` and `/threads` runtimes no longer overwrite one another's effective models, temporary approvals, MCP activation, per-session callbacks, cost totals, or displayed model/provider metadata.
- Persistent permission updates are serialized and written atomically while `session` approval rules remain local to the session that created them.
- Resume restores historical thread cost without double-counting it as new agent spend, forks exclude inherited usage from child totals, and rollback preserves already-incurred session cost.
- Codex subscription conversations now use one stable UUID per thread for `prompt_cache_key` and the private backend's session-routing header without sending public-API-only cache controls; cache reads and writes are reported per call, append-only prefixes can be verified through privacy-safe debug fingerprints, and backend validation errors include their response detail and request ID.

## [0.2.2] - 2026-08-15 — Released

### Added

- `/export <path.html>` to write the current durable session as a self-contained, interactive HTML transcript with pre-compaction events, normalized JSONL download, image-byte omission, and overwrite protection.
- Concurrent CLI thread runtimes: `/threads` can switch between active turns without interrupting them, `/new` can start a fresh turn in the background, and live threads display animated working, waiting-for-input, and cancelling states.
- `/reflection` to immediately reflect on the unreflected conversation tail and update session memory.
- Public `ReflectionResult` export and `Session.run_reflection()` for on-demand reflection outside automatic thresholds.

### Changed

- The SDK `read` tool can now read absolute file paths outside the configured project root; relative paths still resolve from the project root, while write, edit, and delete paths remain project-bound.
- The TUI footer now labels Codex plan-backed usage as `subscription` instead of displaying the misleading `$0.0000`, while retaining dollar totals for usage-based providers.
- Codex subscription requests now retry transient SSE failures such as `server_is_overloaded`, service-unavailable, server, and rate-limit errors with bounded exponential backoff, jitter, and `Retry-After` support; non-transient stream errors still fail immediately.

### Fixed

- Thread runtime rekeying when forking a conversation under concurrent runtimes.

## [0.2.1] - 2026-08-12 — Released

### Added

- Codex subscription provider with browser and device-code authentication via `/login`, using the installed `codex` CLI app-server and isolated Ness global credentials (never reads or writes `~/.codex`).
- Pluggable model provider layer (`ness_cli/provider/`) with shared registry, profile selection, and separate Codex and OpenRouter adapters/chat models.
- Install scripts at `https://nessagent.dev/install.sh` (macOS/Linux) and `install.ps1` (Windows).
- Persistent session names through the SDK `Session.set_name()` / `ThreadStore.set_thread_name()` APIs and the interactive `/rename <name>` command.
- Local `YYYY-MM-DD HH:mm` update timestamps in the `/threads` picker.
- `/clear` to wipe the visible transcript without resetting the conversation.
- Cached Codex subscription status in `/status`, including account email/tier and every available usage-limit window (including weekly).
- Subscription-aware cost accounting in `CostTracker` / `TokenUsage`: provider `billing_mode` metadata keeps subscription-backed turns out of API cost estimation, and `cost_source` distinguishes provider-reported from estimated spend.

### Changed

- `/login` is the primary provider onboarding path: choose Codex subscription or OpenRouter API key, switch active providers mid-session, and reconnect or log out from a dedicated picker.
- `/status` labels subscription usage as `subscription` rather than estimating it as API spend.
- TUI pickers now expand responsively to show up to 12 choices while retaining transcript space, with improved approval and question layouts.
- **Breaking:** the SQLite `threads` table now requires a `name` column. Automatic database migrations are intentionally unsupported; back up or remove `.ness/threads/threads.db` before using this version. Removing it permanently discards saved threads unless they were backed up externally.

## [0.2.0] - 2026-08-07 — Released

### Added

- Public, adapter-neutral `MCPRuntime`, `MCPServerSpec`, and structured MCP state APIs for applications that want MCP connections without adopting Ness project configuration or UI policy.
- Native `.ness/mcp.json` loading in interactive and headless modes, with canonical `mcpServers` validation, custom `NESS_DIR` support, visible diagnostics, project-config fingerprints, and fail-closed headless trust.
- MCP stdio and Streamable HTTP clients with headers, `envFile`, safe child-process environments, Cursor and Claude interpolation syntax, and isolated per-server errors.
- Explicit `ness mcp status`, `ness mcp login`, and `ness mcp logout` OAuth management for Cursor `auth` and Claude `oauth` configurations, including loopback and manual callbacks, token refresh, project-scoped keyring storage, and an atomic `0600` file fallback.
- Transactional `ness mcp import` for Cursor- and Claude-compatible JSON files, with dry runs, server selection, explicit conflict replacement, redacted previews, canonical destination output, and import provenance.
- Multi-directory skill loading: `AgentSpec.skills_dirs` accepts an explicit, exhaustive list of skill roots, including nested category layouts. The Ness CLI loads `.ness/skills` plus the well-known project-local agent skill roots (`.agents/skills`, `.claude/skills`, `.codex/skills`, `.cursor/skills`) and only `~/.agents/skills` globally, passing them to the SDK explicitly; SDK helpers `merge_skill_dirs()` / `default_skill_search_dirs()` let other hosts opt into the same roots (`project_rels=` / `global_rels=` restrict the project-local and user-global sets).
- Cache-safe `ness_agent.summarize()` API and durable summary checkpoints for resume/rollback.
- Canonical model-facing history that preserves ordinary request prefixes while keeping L3 reminders out of durable transcripts.
- `ness --version` flag to print the installed version and exit.

### Changed

- Scans the directories provided (`skills_dir` or `skills_dirs`; both `None` disables skills). Pass `skills_dirs=merge_skill_dirs(project_root, your_dir)` to opt into the well-known roots. The Ness CLI end-user behavior is unchanged in kind: it passes its root list explicitly.
- Pinned the Python MCP SDK to `mcp>=1.27.1,<2` while Ness targets the v1 transport and OAuth APIs.
- TUI: the per-frame user-band width validation now rescans only newly appended transcript lines instead of the whole buffer, removing render-thread stalls (spinner stutter, laggy streaming echo) on very long transcripts; misfit detection and resize reflow behavior are unchanged.
- Fixed Ctrl+T thinking toggles corrupting active streamed answers, causing duplicate responses or preventing final Markdown rendering.
- Compaction now uses the main bound model, identical tools/session/system prefix, a human tail instruction, and a boundary-safe graph node.
- `/compact` retains the active user/tool turn verbatim and summarizes completed history only.
- Compaction checkpoints atomically retain the active semantic suffix, preventing SDK resume from dropping an unlogged user turn.
- Cache-safe forks retain the last successful model/tool binding; canonical image blocks are no longer stripped before compaction, and session pressure includes the stable system prefix.

### Removed

- Progressive tool-output compaction, the 40-message summary limit, `compaction_model`, `progressive_compact`, and `summarize_history`.
- `COMPACTION_INPUT_RESERVE` and `COMPACTION_OUTPUT_RESERVE`; use `COMPACTION_BUFFER_TOKENS` and `COMPACTION_SUMMARY_MAX_TOKENS`.
- **Breaking:** automatic first-start migration of project `.env` settings into global JSON configuration. Application settings now come only from the process environment or `configs.json` / `secrets.json`; MCP `envFile` remains supported explicitly.
- **Breaking:** `RunResult.usage`; use `RunResult.usage_total`, which aggregates every model call made during the turn.
- **Breaking:** the mixed `ness_agent.mcp.MCPManager`; SDK applications now use `MCPRuntime` with resolved server specs, while the Ness CLI owns project config, trust, OAuth persistence, and presentation through its adapter.

## [0.1.0] - 2026-07-31 — Released

### Added

- Initial public release of **Ness Agent** (SDK) and **Ness** (CLI).
- SDK: LangGraph agent loop, built-in tools, permissions, memory, skills, hooks, MCP, compaction, reflection, and tracing.
- CLI: interactive TUI (`ness`), headless print mode (`-p`), plan/act modes, git worktrees, global config, and `.ness/` project layout.

[Unreleased]: https://github.com/Sagnnik/ness-agent/compare/v0.2.2...HEAD
[0.2.2]: https://github.com/Sagnnik/ness-agent/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/Sagnnik/ness-agent/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/Sagnnik/ness-agent/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Sagnnik/ness-agent/releases/tag/v0.1.0
