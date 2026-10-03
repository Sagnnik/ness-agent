# Configuration

Ness splits **global** user data, **project** config, and **runtime** cache.

See also: [CLI guide](cli.md) · [Architecture](architecture.md)

---

## Directory layout

```text
# Global config (platformdirs user_config_dir("ness-agent"))
# Linux: ~/.config/ness-agent/
# macOS: ~/Library/Application Support/ness-agent/
# Windows: %APPDATA%\ness-agent\
USER.md                  Cross-repo user preferences
configs.json             Non-secret adapter settings (only values you changed)
secrets.json             API keys and other secrets (mode 0600)
mcp_oauth.json           OAuth fallback when no system keyring is available (mode 0600)
instructions/            Editable prompt templates (L0, persona, plan/act, aux, goal)
plans/<project-slug>/    Saved plan-mode output for this project

# Per-project cache (platformdirs user_cache_dir("ness-agent")/<hash>/)
cli_history              Prompt history for this project root
images/                  Normalized PNG scratch copies of pasted images

# Per-project .ness/ (NESS_DIR, default ".ness")
.ness/
├── NESS.md              Project conventions loaded into L1
├── permissions.json     Tool allow/deny/ask rules
├── hooks.json           Hook commands
├── mcp.json             Trusted stdio / Streamable HTTP MCP servers
├── agents/              Subagent definitions
├── commands/            User slash commands
├── threads/             Saved session trajectories (SQLite)
│   └── threads.db
└── runtime/
    ├── sessions/        Per-thread episodic memory (L3)
    │   └── mem_<thread_id>.md
    └── shells/          Background shell job metadata and logs

# Skills discovered when present (project before global, .agents first in each scope)
.agents/skills/  .claude/skills/  .codex/skills/  .cursor/skills/
~/.agents/skills/  ~/.claude/skills/  ~/.codex/skills/  ~/.cursor/skills/
```

Override roots with `NESS_AGENT_CONFIG_DIR`, `NESS_AGENT_CACHE_DIR`, and `NESS_DIR`. Skills may be nested under category folders (`category/skill/SKILL.md`); see [Skills in the CLI guide](cli.md#skills).

Pasted image scratch files live in `<cache-root>/<project-hash>/images/`.
`NESS_AGENT_CACHE_DIR` overrides the cache root; without it, Ness uses the
platform's `user_cache_dir("ness-agent")`. The TUI uses the paths resolved at
startup, even if the working directory or environment later changes.
Ness does not expire or delete these files automatically. You can delete the
`images/` directory manually without affecting staged images or saved
conversations, which keep the image data separately. Older shared scratch
files under `<platform-cache-root>/images/` are not moved or removed.

The CLI discovers the shared skill roots above and does not create skill directories during setup or `/init`. `.ness/skills` is no longer discovered and has no automatic migration.

Well-known-root discovery is Ness CLI policy: the CLI hands these directories to the SDK explicitly. SDK applications scan only the roots they configure (`skills_dir` / `skills_dirs`) and can opt into the same list via `default_skill_search_dirs()` or add a custom root with `merge_skill_dirs()`. See [SDK guide → Skills](sdk.md#skills).

---

## Settings resolution

Settings resolve in this order (highest wins):

1. CLI flags
2. Process environment variables
3. `secrets.json` / `configs.json`
4. Built-in defaults

`configs.json` is written lazily — it only contains values you changed via `/config` (defaults stay in code and evolve with upgrades).

Each save merges the requested changes into the latest documents under their
file locks and validates the result before writing either file. Changes to
other settings from another Ness process are preserved. Running settings and
selected-session updates come from the committed documents, with the same CLI
and environment precedence as startup.

Configuration patches are validated before saving. Each file is replaced
atomically, but a patch that changes both `configs.json` and `secrets.json` can
save one file before the other fails. Ness keeps successful saves. After a
write error, it reloads both files and reports which requested keys are saved
and which are unsaved, without printing secret values. A value already present
on disk counts as saved. You can retry the unsaved changes.

The configuration manager and selected session use the verified saved values,
with CLI and environment overrides still taking precedence. If the selected
session cannot apply those values, the error reports that too. If the files
cannot be read back or contain invalid settings, Ness reports an unknown save
status and retains its last valid runtime configuration.

MCP trust fingerprints and non-secret import provenance also live in `configs.json`. OAuth tokens and dynamic client registrations use the system keyring when available; `mcp_oauth.json` is an atomic project-scoped fallback and is never written when keyring storage succeeds.

Project `.env` files are not loaded or migrated for Ness application settings. Existing users should move those values to the process environment or enter them through `/config`; secret values are stored in `secrets.json` and other settings in `configs.json`. An MCP server may still opt into a dotenv file explicitly with its `envFile` field.

Use `/login` to authenticate, activate, reconnect, or log out of a model provider.
Selecting a connected provider activates it and opens its Reconnect/Log out menu;
selecting a disconnected provider starts authentication without an extra Connect step.
OpenRouter and OpenCode Go keys are stored separately in `secrets.json`. ChatGPT credentials for Codex
are managed by the system `codex` CLI app-server under
`<NESS_AGENT_CONFIG_DIR>/codex/` with file credential storage forced; Ness
does not reuse `~/.codex`.
Codex device-code login additionally requires **Device code authorization for
Codex** to be enabled in ChatGPT **Settings > Security**. Use browser login if
that account setting is unavailable.

Provider-specific model and reasoning choices are nested under
`provider_profiles` in `configs.json`, so switching providers restores each
provider's last selection. Legacy top-level OpenRouter settings remain valid.

`--reasoning-effort` checks the selected provider's available model metadata.
For example, OpenCode Go's `gpt-5.6-luna` accepts `none` and rejects `minimal`.
Provider catalog entries override fallback options and context windows. Codex
uses separate subscription defaults rather than API limits. Fallback matching
distinguishes model versions, so an unfamiliar version cannot inherit an older
version's limits. If selectable efforts are unavailable, Ness skips this CLI
check. If a context window is unavailable, Ness leaves it unknown.

Known API context windows, reasoning options, and vision support share one
fallback record per model family in `providers/model_metadata.py`. Provider
overrides keep OpenCode Go's supported effort choices and Codex subscription
limits separate. Protocol selection and billing remain provider-specific.
Eval bundle metadata stays separate from the runtime tables so a runtime
cleanup does not change a historical evaluation.

In the concurrent TUI, saved configuration is the default for new thread
runtimes. Changing the model, provider, or reasoning effort through `/config`
also rebuilds the currently selected thread, but it does not mutate sibling
threads that are already live. Selecting one of those threads later restores
its pinned runtime configuration.

Thread autosave follows the same session policy. Changing `auto_save_threads`
through `/config` updates the selected thread and the saved default for future
threads. Other open threads keep their own autosave setting, including any
turn already running. Their displayed settings match their persistence policy.
The autosave control shows the selected thread's setting and can apply a value
even when it already matches the saved default. Model reloads retain that
thread's behavior settings. The threads share one SQLite database, with a
separate autosave switch for each session. Turning autosave off skips new event
and checkpoint writes for that session without deleting its saved history.
Turning it back on resumes
saving subsequent writes; it does not backfill turns run while autosave was off.

---

## Environment variables

All except `NESS_DIR` are also editable via `/config` in the Ness TUI.

| Variable | Description |
|----------|-------------|
| `MODEL_PROVIDER` | Active provider (`openrouter` by default, `codex`, or `opencode`) |
| `MODEL_NAME` | Active provider model ID (`deepseek/deepseek-v4-flash` by default) |
| `REFLECTION_MODEL_NAME` | Model for background session-memory reflection (defaults to `MODEL_NAME`) |
| `ENABLE_APPROVAL` | Require approval for destructive tools |
| `AUTO_SAVE_THREADS` | Write thread events to `.ness/threads/` |
| `SESSION_END_REFLECTION` | Run a final reflection pass when a session ends (default off) |
| `REFLECTION_TOKEN_RATIO` | Fraction of usable context that must accumulate before reflection (default `0.4`; set `0` to disable) |
| `API_MAX_RETRIES` | Retries for chat API calls (default `3`) |
| `COMPACTION_BUFFER_TOKENS` | Context held back for cache-safe compaction input/output (default `16384`) |
| `COMPACTION_SUMMARY_MAX_TOKENS` | Maximum compaction summary output (default `4096`) |
| `COMPACTION_TOKEN_BUDGET` | Context-limit fallback when the model window is unknown (default `120000`) |
| `OPENROUTER_SESSION_ID` | Optional stable prompt-cache session id (defaults to active thread id) |
| `OPENROUTER_CACHE_TTL` | Anthropic prompt-cache lifetime (`5m` by default; `1h` supported) |
| `OPENROUTER_ANTHROPIC_MESSAGES` | Use OpenRouter Messages API for Anthropic models (default `true`) |
| `GOAL_JUDGE_MODEL` | Model for independent `/goal` judge (defaults to `REFLECTION_MODEL_NAME`) |
| `GOAL_MAX_ATTEMPTS` | Maximum worker/judge attempts for `/goal` (default `3`) |
| `OPENAI_BASE_URL` | Optional custom OpenAI-compatible base URL |
| `OPENAI_API_KEY` | Provider API key (also stored in `secrets.json` via `/config`) |
| `OPENCODE_GO_API_KEY` / `OPENCODE_API_KEY` | OpenCode Go subscription key (also stored separately in `secrets.json`) |
| `FORMAT_ON_WRITE` | Auto-format supported file types after writes (default `true`) |
| `NESS_DIR` | Project config directory (default `.ness`) |
| `NESS_AGENT_CONFIG_DIR` | Override global config root |
| `NESS_AGENT_CACHE_DIR` | Override cache root (OpenRouter catalog + per-project `cli_history`) |
| `EXA_API_KEY` | Optional Exa API key for higher-quality `web_search` and `fetch_url` ([exa.ai](https://exa.ai)) |

### CLI flags

Flags override env for a single run: `--model`, `--reflection-model`, `--api-key`, `--base-url`, `--openrouter-session-id`, `--reasoning-effort`, `--worktree` / `-w`, `--print` / `-p`, and `--yolo`.

`--yolo` is session-only and bypasses approval prompts, permission denials, native file-tool project scope, and protected-write checks in act mode. OS permissions, hook vetoes, and plan-mode read-only rules still apply. Normal-mode native file tools ask for approval before accessing outside-project paths. Once approval applies to that call; session and always decisions remember the approved path with separate read/write scope.

Use `/login` for provider authentication and switching. Use `/config` for the
active provider's model and reasoning settings, behavior, compaction, and
advanced options; provider-only fields are hidden when they do not apply.

The `/config` model picker reads the active provider's catalog. OpenRouter uses
its global disk cache or the packaged offline fallback without fetching data,
even when the cache is stale. Refresh is explicit. Run
`uv run python scripts/fetch_openrouter_models.py --refresh` to replace the
cache; provider callers can request `models(refresh=True)` as well. Both force
a refresh. A direct catalog `refresh()` without `force=True` uses the 24-hour
TTL to skip a fresh cache. Failed refreshes retain the previous catalog.

`codex app-server` handles ChatGPT authentication and credential management for the signed-in account. Ness performs model inference separately through its experimental ChatGPT-authenticated Codex Responses transport.
