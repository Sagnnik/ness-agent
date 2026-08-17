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

# Per-project .ness/ (NESS_DIR, default ".ness")
.ness/
├── NESS.md              Project conventions loaded into L1
├── permissions.json     Tool allow/deny/ask rules
├── hooks.json           Hook commands
├── mcp.json             Trusted stdio / Streamable HTTP MCP servers
├── agents/              Subagent definitions
├── commands/            User slash commands
├── skills/              Project-local SKILL.md skills (highest precedence)
├── threads/             Saved session trajectories (SQLite)
│   └── threads.db
└── runtime/
    ├── sessions/        Per-thread episodic memory (L3)
    │   └── mem_<thread_id>.md
    └── shells/          Background shell job metadata and logs

# Also discovered by the Ness CLI when present (after .ness/skills; project before global)
.agents/skills/  .claude/skills/  .codex/skills/  .cursor/skills/
~/.agents/skills/
```

Override roots with `NESS_AGENT_CONFIG_DIR`, `NESS_AGENT_CACHE_DIR`, and `NESS_DIR`. Skills may be nested under category folders (`category/skill/SKILL.md`); see [Skills in the CLI guide](cli.md#skills).

Well-known-root discovery is Ness CLI policy: the CLI hands these directories to the SDK explicitly. SDK applications scan only the roots they configure (`skills_dir` / `skills_dirs`) and can opt into the same list via `merge_skill_dirs()` — see [SDK guide → Skills](sdk.md#skills).

---

## Settings resolution

Settings resolve in this order (highest wins):

1. CLI flags
2. Process environment variables
3. `secrets.json` / `configs.json`
4. Built-in defaults

`configs.json` is written lazily — it only contains values you changed via `/config` (defaults stay in code and evolve with upgrades).

MCP trust fingerprints and non-secret import provenance also live in `configs.json`. OAuth tokens and dynamic client registrations use the system keyring when available; `mcp_oauth.json` is an atomic project-scoped fallback and is never written when keyring storage succeeds.

Project `.env` files are not loaded or migrated for Ness application settings. Existing users should move those values to the process environment or enter them through `/config`; secret values are stored in `secrets.json` and other settings in `configs.json`. An MCP server may still opt into a dotenv file explicitly with its `envFile` field.

Use `/login` to authenticate, activate, reconnect, or log out of a model provider.
Selecting a connected provider activates it and opens its Reconnect/Log out menu;
selecting a disconnected provider starts authentication without an extra Connect step.
OpenRouter keys are stored in `secrets.json`. Codex subscription credentials
are managed by the system `codex` CLI under `<NESS_AGENT_CONFIG_DIR>/codex/`
with file credential storage forced; Ness does not reuse `~/.codex`.
Codex device-code login additionally requires **Device code authorization for
Codex** to be enabled in ChatGPT **Settings > Security**. Use browser login if
that account setting is unavailable.

Provider-specific model and reasoning choices are nested under
`provider_profiles` in `configs.json`, so switching providers restores each
provider's last selection. Legacy top-level OpenRouter settings remain valid.

In the concurrent TUI, saved configuration is the default for new thread
runtimes. Changing the model, provider, or reasoning effort through `/config`
also rebuilds the currently selected thread, but it does not mutate sibling
threads that are already live. Selecting one of those threads later restores
its pinned runtime configuration.

---

## Environment variables

All except `NESS_DIR` are also editable via `/config` in the Ness TUI.

| Variable | Description |
|----------|-------------|
| `MODEL_PROVIDER` | Active provider (`openrouter` by default, or `codex`) |
| `MODEL_NAME` | Model passed to `ChatOpenRouter` (`deepseek/deepseek-v4-flash` by default) |
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
| `FORMAT_ON_WRITE` | Auto-format supported file types after writes (default `true`) |
| `NESS_DIR` | Project config directory (default `.ness`) |
| `NESS_AGENT_CONFIG_DIR` | Override global config root |
| `NESS_AGENT_CACHE_DIR` | Override cache root (OpenRouter catalog + per-project `cli_history`) |
| `EXA_API_KEY` | Optional Exa API key for higher-quality `web_search` and `fetch_url` ([exa.ai](https://exa.ai)) |

### CLI flags

Flags override env for a single run: `--model`, `--reflection-model`, `--api-key`, `--base-url`, `--openrouter-session-id`, `--reasoning-effort`, `--worktree` / `-w`, `--print` / `-p`, and `--yolo`.

`--yolo` is session-only and bypasses approval prompts and persisted permission denials in act mode; hook vetoes and plan-mode read-only rules still apply.

Use `/login` for provider authentication and switching. Use `/config` for the
active provider's model and reasoning settings, behavior, compaction, and
advanced options; provider-only fields are hidden when they do not apply.

The `/config` model picker reads the active provider's catalog. OpenRouter's
global disk cache is reused for 24 hours with a packaged offline fallback;
Codex models are supplied by `codex app-server` for the signed-in account.

Codex access tokens are refreshed through `account/read` shortly before JWT
expiry and once after an HTTP 401. Network, rate-limit, and server failures do
not trigger credential refresh. The subscription Responses transport is kept
inside `src/ness_cli/provider/codex/` because that endpoint is experimental.
