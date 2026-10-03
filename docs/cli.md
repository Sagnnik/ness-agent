# Ness CLI

**Ness** is the interactive coding-agent CLI shipped with Ness Agent. It supports plan/act modes, filesystem-driven extension points under `.ness/`, and a full TUI for approvals, thread history, and configuration.

See also: [Configuration](configuration.md) · [Architecture](architecture.md) · [SDK](sdk.md)

---

## Getting started

```bash
pip install ness-agent
ness --version               # verify the install
export OPENAI_API_KEY=...    # optional; or connect via /login
ness
```

Initialize a project and global config:

```bash
ness
/init
```

`/init` creates project `.ness/` directories for agents, commands, threads, and runtime data, plus permissions, hooks, MCP configuration, default agent profiles, and an empty `NESS.md`. It also ensures global config (`USER.md`, `instructions/`, `plans/<slug>/`). Skill directories are discovered when present and are not created by `/init`.

Or skip the env var and run `/login`. Choose Codex to sign in with ChatGPT
using managed browser/device authentication, or OpenRouter/OpenCode Go for a masked API-key prompt. Device-code
login must first be enabled in ChatGPT **Settings > Security** by turning on
**Device code authorization for Codex**; browser login does not require that setting.
Selecting a connected provider makes it active and exposes only **Reconnect** and
**Log out**. A disconnected provider starts its login flow immediately—there is
no separate Connect step.
On WSL, Ness prints the authentication URL without launching a browser; copy it
into the browser of your choice. While any Codex login is pending, use the visible
**Cancel pending login** action (or press Esc/Ctrl+C) to return to the session.

---

## Headless one-shot queries (`-p` / `--print`)

Run a single query without opening the TUI; the final response goes to stdout and the process exits:

```bash
ness -p "what does the auth module do?"
cat build-error.txt | ness -p "explain the root cause" > diagnosis.txt
ness -p --yolo "run the test suite and fix any failures"
```

Approvals are deny-by-default in print mode: tools already allowed by `.ness/permissions.json` run normally, anything that would prompt is auto-denied and the denial is fed back to the model, and `--yolo` bypasses the gate. `-p` composes with the other flags — `--worktree`, `--resume`, `--model`, etc.

- **stdout** — final response only
- **stderr** — diagnostics and the `ness --resume <thread_id>` hint
- **Exit codes:** `0` success, `1` turn error, `2` usage error, `130` interrupted

---

## Parallel sessions (git worktrees)

Run a second agent in an isolated checkout and branch without touching your main working tree:

```bash
# Terminal 1 — main checkout
ness

# Terminal 2 — isolated agent (creates .ness/worktrees/auth on first launch)
ness --worktree auth
```

Each worktree gets its own branch (`worktree-<name>`), file edits, and runtime data (`.ness/threads/`, `.ness/runtime/sessions`, shell jobs). Tracked `.ness` files (agents, permissions, NESS.md) and project skills under `.agents/skills/` inherit from git. Config and secrets are global (see [Configuration](configuration.md)), so worktrees need no per-checkout setup. Re-launching with the same `--worktree` name reuses the existing checkout. Merge back with normal git when done (`git merge worktree-auth`, etc.).

---

## Skills

Primary project skills live under `.agents/skills/<name>/SKILL.md`. Ness discovers these shared and compatibility locations when present:

**Project-local:** `.agents/skills/`, `.claude/skills/`, `.codex/skills/`, `.cursor/skills/`  
**User-global:** `~/.agents/skills/`, `~/.claude/skills/`, `~/.codex/skills/`, `~/.cursor/skills/`

This discovery is a Ness CLI policy. The CLI passes these roots to the SDK explicitly. The SDK scans only the directories it is given; embedders opt into these roots via `default_skill_search_dirs()` or add a custom root with `merge_skill_dirs()` (see [SDK guide → Skills](sdk.md#skills)).

Project roots take precedence over user-global roots. Within each scope, `.agents/skills` wins on name collisions, followed by `.claude/skills`, `.codex/skills`, and `.cursor/skills`. Nested category layouts (`category/skill/SKILL.md`) are supported. A directory with `SKILL.md` is a skill (resources like `scripts/` are not scanned as skills).

The CLI does not discover `.ness/skills` or migrate its contents. Create skill directories yourself or use a skill installer. Directory symlinks are supported, and links to the same skill are deduplicated; Ness does not create or manage those links.

```text
.agents/skills/react_component/SKILL.md
.agents/skills/product-a/skill-one/SKILL.md
```

Each `SKILL.md` may include YAML frontmatter:

```markdown
---
name: react_component
description: Create React components matching project conventions.
---
# React Component

Skill instructions go here.
```

Skill loading is two-stage. A one-line catalog of available skills (`name: description`, plus path) is sent through L3 on the first turn, after availability changes, and after compaction. Full `SKILL.md` bodies load when the model calls the `skill_view` tool (or `read`s the catalog path); that content stays in the conversation as a tool message. There is no separate loaded-skill summary. After compaction, a general reminder tells the model to reload instructions for any skill procedure it needs.

Type `$` at the start of a word to open the skill list in the input chrome. Keep typing to filter, use Up/Down to highlight a skill, and press Tab to insert `$<name>` and request it for that prompt. Enter sends the prompt; Esc closes the list. Without a selection, `$` stays ordinary text, even if you type a matching skill name. Removing a selected marker removes its request. Selections stay with their prompt when queued.

Skill availability defaults and thread snapshots are saved in the global
configuration directory's `skill-state.json`. Ness skips malformed saved skill
entries and rebuilds a requested thread's snapshot if its skill-list container
is invalid. Damaged records in other threads cannot block opening your thread.
Older snapshots migrate automatically while valid thread selections stay intact.

---

## Permissions

`.ness/permissions.json` uses glob-style rules:

```json
{
  "allow": ["read:*", "grep:*", "shell:run:git status*"],
  "deny": ["shell:run:rm -rf*", "shell:run:sudo*"],
  "ask": ["*"]
}
```

Deny rules win over allow rules. Rules are evaluated in order: persistent deny, session deny, persistent allow, session allow, then ask. Shell command allow/deny rules reject commands with unquoted shell operators (`;`, `&&`, `|`, `>`, `<`, newlines, etc.) so chained or redirect commands fall through to ask instead of matching a prefix rule.

`web_search:*` is allowed by default. `fetch_url` asks for approval per normalized URL; approving one URL does not approve a different path or query, and changing `max_characters` does not require a new approval.

Approval choices are `y` yes, `S` session (allow for this CLI run), `a` always, `n` no, `N` never, `d` diff, and `s` show args.

### Web search providers

`web_search` and `fetch_url` pick a provider automatically:

| Provider | When used | Notes |
|----------|-----------|-------|
| **Exa** | `EXA_API_KEY` is set | Semantic search, content highlights, reliable fetch |
| **DuckDuckGo fallback** | No Exa key | Keyword search via DuckDuckGo HTML; direct HTTP fetch with `trafilatura` / BeautifulSoup extraction |

The fallback requires no API key but is less capable: no neural search, weaker snippets, no JavaScript rendering on fetch, and occasional DuckDuckGo rate limits or CAPTCHAs. Set `EXA_API_KEY` when you need more reliable web access.

---

## MCP

Ness Agent connects to local **stdio** and remote **Streamable HTTP** MCP servers at CLI startup, discovers their tools, and exposes them to the agent.

**Security:** stdio servers run arbitrary commands with your user permissions, and remote servers receive the configured headers. Before starting a changed non-empty configuration, interactive Ness shows a redacted server summary and asks you to trust that exact configuration. The approval is stored by project and config fingerprint in global `configs.json`; changing a command, endpoint, credential template, or server set requires approval again. Headless mode never prompts and skips untrusted MCP servers, even with `--yolo`; run interactive Ness once to approve them.

Configure MCP servers in `.ness/mcp.json` with the `mcpServers` key (same shape as Cursor):

```json
{
  "mcpServers": {
    "filesystem": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "."],
      "envFile": ".env",
      "env": {"API_KEY": "${env:FILES_API_KEY}"},
      "cwd": "${workspaceFolder}",
      "startup_timeout": 20
    },
    "hosted": {
      "type": "http",
      "url": "https://example.com/mcp",
      "oauth": {
        "clientId": "${MCP_CLIENT_ID}",
        "callbackPort": 8787,
        "scopes": "read write"
      }
    }
  }
}
```

Per-server fields:

- `type`: optional `stdio`, `http`, or `streamable-http`. When omitted, `command` implies stdio and `url` implies HTTP.
- `command` / `args`: stdio process to spawn. `command` may also be an array like `["npx", "-y", "..."]`; its remaining elements are prepended to `args`.
- `env`: optional stdio environment overrides. Ness inherits only the MCP SDK's safe environment allowlist, then applies `envFile`, then `env`.
- `envFile`: optional dotenv file for stdio servers, resolved from the project root.
- `cwd`: working directory for the server process (defaults to the project root).
- `url` / `headers`: Streamable HTTP endpoint and optional request headers.
- `auth`: Cursor static OAuth shape (`CLIENT_ID`, optional `CLIENT_SECRET`, and optional scope list).
- `oauth`: Claude OAuth shape (`clientId`, optional `clientSecret`, `callbackPort`, scopes, and token endpoint authentication method). Omit the client ID to use dynamic registration.
- `startup_timeout`: seconds to wait for connect + tool discovery (default `20`).

String fields support Cursor interpolation (`${env:NAME}`, `${userHome}`, `${workspaceFolder}`, `${workspaceFolderBasename}`, `${pathSeparator}`, `${/}`) and Claude interpolation (`${NAME}`, `${NAME:-default}`). Expansion uses the Ness process environment; `envFile` values are only passed to the stdio child. Missing required variables invalidate that server without blocking others.

OAuth never launches a browser during normal interactive or headless startup. Authenticate explicitly with `ness mcp login <server>`; use `--no-open` to open the printed URL yourself or `--manual-callback` to paste a callback URL over SSH. Stored tokens refresh automatically. `ness mcp logout <server>` removes local credentials but does not revoke the provider-side token; it exits with an error instead of claiming success when keyring deletion cannot be verified.

Management commands:

```text
ness mcp status [server]
ness mcp login <server> [--callback-port PORT] [--no-open] [--manual-callback]
ness mcp logout <server>
ness mcp import <cursor-or-claude.json> --dry-run
ness mcp import <cursor-or-claude.json> [--server NAME] [--replace NAME] [--yes]
```

Imports are explicit and transactional. Differing name conflicts abort the entire import unless each is named with `--replace`, and a destination edit made after confirmation aborts rather than being overwritten. Imports are written to `.ness/mcp.json` but never grant execution trust; the next interactive startup still shows the redacted trust prompt. Import provenance is kept in global `configs.json`.

SSE, WebSocket, `headersHelper`, OAuth metadata URL overrides, automatic config discovery, and native registry/add commands are not supported yet. Explicit `sse` and `ws` entries are reported as unsupported rather than guessed.

Tools are exposed as `mcp__<server>__<tool>`. The startup header shows connected MCP servers in Add-ons; use `/mcp` for the full server and tool list. Connection failures are shown as a startup warning. Startup failures do not stop the CLI.

**Approval and permissions:** all `mcp__*` tools require approval when `ENABLE_APPROVAL=true`. You can add explicit rules in `.ness/permissions.json`:

```json
{
  "allow": ["mcp__filesystem__read_file", "mcp__filesystem__list_directory"],
  "deny": ["mcp__filesystem__write_file"],
  "ask": ["*"]
}
```

**Subagents:** subagents are read-only. MCP tools, write tools, shell execution, nested subagents, and `todo` are rejected even if listed in frontmatter.

**Prompt size:** tool descriptions include the full MCP input schema so the model can handle complex arguments. Servers with many tools may increase token usage.

---

## Subagents

Subagents live in `.ness/agents/<name>.md`:

```markdown
---
tools: [read, grep, glob, web_search, fetch_url]
---
You are a read-only explorer. Return concise findings with file references.
```

The `spawn_subagent` tool runs one or more filtered, isolated read-only graphs. Only the parent agent can spawn subagents; nested spawning is blocked by the read-only tool filter. Always pass `tasks` — a non-empty list of `{name, prompt, label?}` — plus optional `max_concurrency` and `timeout`. Never call with bare top-level `name`/`prompt`. A single investigation still uses a one-item list:

```python
spawn_subagent(tasks=[{"name": "explore", "prompt": "Find route handlers"}])
```

The parent agent waits until every subagent completes, fails, or times out.

Batch mode validates every task before starting any of them and returns one structured result with each task's status, duration, thread id, label, and output.

---

## Slash commands

Shift+Tab toggles plan/act mode without rebuilding the graph or invalidating the prompt cache. Current mode appears in the prompt prefix and footer. Type `/` for the command picker or `/help` for the full list.

An error stops the current prompt queue and leaves unstarted prompts queued for review. An interruption also stops queue processing; explicitly cancelling clears the queue. Warnings do not stop the queue. A later final response does not clear an earlier error; headless mode uses the same rule when choosing its exit code.

**General**

- `/help`: show the command reference.
- `/login`: authenticate or manage model providers in a dedicated picker. Selecting a connected provider activates it and offers Reconnect/Log out; selecting a disconnected provider starts authentication immediately. Provider changes rebuild the model while preserving the thread.
- `/config`: edit provider keys/endpoints, model/reasoning, behavior toggles, compaction budgets, and advanced options (persisted to global `configs.json` / `secrets.json`). Model/provider/reasoning changes rebuild the selected thread and become defaults for future threads; other live thread runtimes keep their existing model configuration.
- `/exit` or `/quit`: end the session.

**Session**

- `/status`: show provider authentication, account email/tier, every available usage-limit window (including OpenCode Go's rolling 5-hour, weekly, and monthly windows), reset times/credits, and the selected session's token/cache/cost stats. Subscription calls are labeled `subscription` rather than estimated as API spend.
- `/threads`: open a scrollable saved-thread picker, ordered by recent updates and prefixed with local `YYYY-MM-DD HH:mm` timestamps. Threads with active turns show an animated working indicator; switching away does not interrupt them.
- `/rename <name>`: set or update the current session's persistent display name (1–80 characters; requires thread autosave).
- `/fork`: choose a human message, copy the conversation state before it into a child thread, and prefill that message for editing. Forking copies session memory/checkpoints but leaves current working-tree files unchanged.
- `/goal <objective>`: run up to `GOAL_MAX_ATTEMPTS` worker attempts, defaulting to three. Only completed attempts reach validation and the isolated read-only judge. Unmet validation or verdicts produce repair instructions for the next attempt. A worker error or interruption stops the goal immediately, records the reason in goal history, and skips judging and automatic retries.
- `/save`: archive the current thread with a headline summary.
- `/new`: archive and start a fresh thread. During an active turn, the running thread stays in the background and the new thread starts independently.
- `/compact`: request a cache-safe summary at the next model boundary; the active user/tool turn remains verbatim unless it exceeds the retained-suffix budget, in which case older in-turn work is summarized and a coherent recent suffix is kept.
- `/reflection`: immediately reflect on conversation messages added since the last successful reflection and update session memory.
- `/export <path.html>`: write the current durable session as a self-contained, interactive HTML transcript. The export retains events from before compactions, includes an in-page normalized JSONL download, omits pasted image bytes, and refuses to overwrite an existing file. Quote paths that contain spaces.

When an API response omits its cost, Ness estimates it from the provider's
model metadata. Rates refresh when main, reflection, or goal-judge models are
created and when the model catalog is loaded or refreshed. Existing sessions
share those rates; previously recorded costs and token totals stay unchanged.
Provider-reported costs take precedence, and subscription calls do not receive
API cost estimates.

**Context & memory**

- `/skill`: inspect skills and manage availability. Select a skill for a prompt by typing `$` and pressing Tab in the skill list. The model loads its instructions through `skill_view`.
- `/init`: initialize project `.ness/` and ensure global config.
- `/memory` or `/memory add <note>`: read or append project memory.
- `/memory create [force]`: opt-in LLM draft of `NESS.md` from project context (`force` overwrites non-empty content).
- `/user` or `/user add <note>`: read or append user preferences.

**Tools & policy**

- `/permissions`: list/edit permission rules.
- `/hooks`: list hooks.
- `/mcp`: list MCP server status and tools.

**Input**

- `/clear`: clear the visible transcript without resetting the conversation.
- `/copy`, `/copy code`, `/copy <n>`: copy assistant output.
- `Ctrl+G`: paste an image from the clipboard into the prompt as `[Image #N]`. The image is resized (max 2000px long edge, max 5 MB) and sent to vision-capable models. If the system clipboard supplies copied filenames, Ness uses the first readable image file and skips non-image or unreadable files.
- `@path/to/file`: attach a file's contents to the next prompt — its current contents are inlined as a `<document>` block above your text. Type `@` to see tracked paths and untracked, non-ignored project files; ↑/↓ to pick, Enter or Tab to complete, Esc to dismiss. Suggestions use a 30-second cached file index. Mention tokens persist on resume/rollback and re-expand from disk.

Pasted images also create PNG scratch files under the project's configured
cache directory. These files remain until you delete them manually. See
[cache locations and cleanup](configuration.md#directory-layout).

Markdown files under `.ness/commands/*.md` become project-local slash commands. Their body is used as a prompt template with `{{args}}` substitution.

---

## Thread events

Autosave is per open thread. A `/config` change applies to the selected thread
and becomes the default for new threads. Other open threads keep their setting.
Turning autosave off preserves existing history but skips new saves; enabling
it again does not backfill unsaved turns.

When autosave is on, Ness Agent stores events in `.ness/threads/threads.db`:

- **`threads`**: user `session-*` metadata (explicit names, cost, turns, summaries, archive state)
- **`events`**: append-only JSON payloads for user sessions only
- **`subagents`**: subagent run metadata (status, output, duration) linked to a parent `session-*` thread

Event kinds stored in `events.payload` (session threads only):

```json
{"kind": "user", "content": "...", "t": "..."}
{"kind": "assistant", "content": "...", "tool_calls": [], "t": "..."}
{"kind": "tool", "tool": "read", "args": {}, "result": "...", "call_id": "...", "duration_ms": 10, "exit": "ok", "t": "..."}
{"kind": "approval", "tool": "edit", "decision": "yes", "t": "..."}
{"kind": "usage", "model": "deepseek/deepseek-v4-flash", "input_tokens": 100, "cached_input_tokens": 40, "output_tokens": 20, "cost_usd": 0.0001, "cost_source": "provider", "t": "..."}
{"kind": "reflection", "prompt": "...", "response": {"new_bullet_points": []}, "message_index": 12, "memory_updated": true, "error": "", "t": "..."}
{"kind": "compaction_llm", "instruction": "...", "response": "...", "source_event_seq": 12, "active_user_seq": 13, "trigger": "automatic", "before_tokens": 101000, "after_tokens": 9000, "active_suffix_messages": 1, "t": "..."}
{"kind": "compact", "content": "manual compaction requested", "t": "..."}
```

`/threads` lists user `session-*` threads only. Each row starts with its locally converted update datetime and uses an explicit `/rename` name when present, then falls back to the archived summary or first user message. The original conversation shows `×N` when it has forks; each fork shows `fork #k` in creation order. Fork lineage is stored explicitly on the thread row; inherited usage remains in the copied event history but is excluded from the child thread's cost totals. Subagent trajectories are not stored in `events`; subagent LLM usage rolls up into the parent session's `threads` aggregates. Subagent outputs are stored in the `subagents` table.

Selecting an idle saved thread rebuilds user messages, assistant tool-call turns, and tool results from saved events. A thread that is already live keeps its own model, temporary approvals, MCP activation, graph, cancellation state, and cost total when selected again. The startup `--resume <thread_id>` flag remains available for automation. `spawn_subagent` tool output is supplemented from linked subagent outputs when available.

When a thread contains a successful new-format `compaction_llm` checkpoint, resume starts from its summary and replays only raw events after `source_event_seq`. Raw conversation events remain available for audit, rollback, and forks; L3 reminder messages are never written to the event log.

Idle threads are archived on `/save`, `/new`, thread switching/forking, and session exit. Switching or using `/new` during an active turn leaves that live thread unarchived until it finishes. Archived threads get a headline summary from the first user message.

`/rollback [seq]` returns to the checkpoint before the selected user turn. It restores recorded file changes, removes files created by the discarded work, and preserves your Git staging choices and unrelated files. Without a sequence number, it opens the turn picker. Automatic rollback requires autosave and a Git checkpoint from a repository with an initial commit.

Rollback checks the recorded files before restoring them. Later changes to those files, overlapping tool operations, incomplete mutation records, background shell starts, and older mutating checkpoints without ownership records cause a failure instead of an automatic restore. Git-ignored files are outside shell snapshot coverage. Creating, changing, or deleting an ignored file during a tool window causes rollback to fail and preserve history, rather than report an incomplete restore as successful. Ignored-file checks use filesystem metadata without saving contents to Git; the agent's thread, shell-log, and session-memory directories are excluded. File changes observed during a foreground tool's execution window are attributed to that tool; external writers during that window cannot be distinguished from the command itself. Managed mutations are blocked during restoration, so use automatic rollback without concurrent external writers.

History is discarded only after file restoration, memory restoration, and conversation replay succeed. On failure, history and checkpoints remain available for retry. An I/O or replay failure can leave files, memory, or live graph state partially restored; fix the reported cause and retry, or resume the thread from its retained history.

> **Breaking database change:** the current release does not migrate older thread databases. If `.ness/threads/threads.db` predates persistent session names, Ness stops with an incompatibility error. Back up or remove that file so Ness can create the current schema; removing it discards saved threads.
