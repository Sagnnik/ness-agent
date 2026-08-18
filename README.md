<p align="center">
  <img src="https://raw.githubusercontent.com/Sagnnik/ness-agent/main/assets/banner-light-geo.svg" alt="Ness Agent — hackable coding-agent harness" width="100%">
</p>

# Ness Agent

[![CI](https://github.com/Sagnnik/ness-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/Sagnnik/ness-agent/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/ness-agent)](https://pypi.org/project/ness-agent/)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/github/license/Sagnnik/ness-agent)](LICENSE)

Ness Agent is an experimental, hackable coding-agent harness for engineers who want to own the loop. It ships as a **Python SDK** you can embed in your own tools and **Ness**, an interactive CLI for day-to-day coding sessions.

<p align="center">
  <img src="assets/ness_agent_sc.png" alt="NessAgent terminal UI showing project context, tool execution, and Act mode" width="100%">
</p>
<p align="center"><a href="https://raw.githubusercontent.com/Sagnnik/ness-agent/main/assets/ness-demo6.mp4">Open the demo video</a></p>
<p align="center"><em>NessAgent TUI in Act mode.</em></p>

> **0.x experimental** — APIs may change until 1.0. See [CHANGELOG](https://github.com/Sagnnik/ness-agent/blob/main/CHANGELOG.md).

## Table of contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick start — CLI (Ness)](#quick-start--cli-ness)
- [Quick start — SDK](#quick-start--sdk)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## Overview

| Component | What it is |
|-----------|------------|
| **Ness Agent SDK** | LangGraph agent loop, built-in tools, permissions, memory, skills, hooks, MCP, compaction, reflection, tracing |
| **Ness CLI** | Terminal UI (`ness`), plan/act modes, git worktrees, global config, `.ness/` project layout |


## Installation

Requires **Python 3.12+**.

### CLI — global install (recommended)

On Linux/Mac Os:
```bash
curl -fsSL https://nessagent.dev/install.sh | sh
```

On Windows:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://nessagent.dev/install.ps1 | iex"
```

Or install it directly with [uv](https://docs.astral.sh/uv/) or [pipx](https://pipx.pypa.io/):

```bash
uv tool install --upgrade ness-agent
# or: pipx install ness-agent
```

Launch `ness` and run `/login` to sign in to Codex with ChatGPT or configure an
OpenRouter API key. Codex authentication uses the installed `codex` CLI
app-server's managed browser/device flow and stores credentials in Ness's
isolated global config directory; it never reads or changes `~/.codex`.

> [!NOTE]
> Ness's Codex model transport is experimental. Authentication and credential
> management are handled by Codex app-server, while Ness sends inference
> requests directly to the ChatGPT Codex Responses endpoint used by Codex CLI.

For environment-based OpenRouter setup:

```bash
export OPENAI_API_KEY=your-openrouter-api-key
export OPENAI_BASE_URL=https://openrouter.ai/api/v1
```

Add those lines to `~/.bashrc` or `~/.zshrc` to persist across sessions. Alternatively, `/login` accepts and masks the key (stored in global `secrets.json`).

Ness does not automatically read or migrate a project `.env` file. You may
still keep local values in `.env`, but load them into the process explicitly
(for example, `uv run --env-file .env ness`) or use `/config`.

Optional tracing support:

```bash
uv tool install 'ness-agent[tracing]'
```

Ensure `~/.local/bin` (uv) or pipx's bin directory is on your `PATH`.

### SDK — library install

For embedding the agent harness in your own Python project (use a venv):

```bash
pip install ness-agent
pip install 'ness-agent[tracing]'   # optional

# In a uv-managed project (adds to pyproject.toml):
uv add ness-agent
uv add ness-agent --extra tracing   # optional
```

**From source (contributors):**

```bash
git clone https://github.com/Sagnnik/ness-agent.git
cd ness-agent
uv sync
uv run ness
```

Editable global install from a clone:

```bash
git clone https://github.com/Sagnnik/ness-agent.git
cd ness-agent
uv tool install -e .
ness
```

## Quick start — CLI (Ness)

```bash
ness
/init                       # create .ness/ and global config
```

Headless one-shot:

```bash
ness -p "what does the auth module do?"
```

Parallel isolated session:

```bash
ness --worktree feature-x
```

Full CLI reference: [docs/cli.md](https://github.com/Sagnnik/ness-agent/blob/main/docs/cli.md) · Configuration: [docs/configuration.md](https://github.com/Sagnnik/ness-agent/blob/main/docs/configuration.md)

## Quick start — SDK

```python
import asyncio
from langchain_core.tools import tool
from ness_agent import NessAgent, PromptLayers, PromptLayersConfig

@tool
def ping() -> str:
    """Return pong."""
    return "pong"

async def main() -> None:
    agent = NessAgent(
        model=your_chat_model,
        tools=[ping],
        prompt=PromptLayers(PromptLayersConfig(l0="You are a helpful agent.", persona="Be concise.")),
    )
    session = agent.session(thread_id="demo-1")
    result = await session.run("say hello")
    print(result.assistant_message)

asyncio.run(main())
```

Full SDK guide: [docs/sdk.md](https://github.com/Sagnnik/ness-agent/blob/main/docs/sdk.md) · Architecture: [docs/architecture.md](https://github.com/Sagnnik/ness-agent/blob/main/docs/architecture.md)

## Documentation

| Guide | Description |
|-------|-------------|
| [docs/README.md](https://github.com/Sagnnik/ness-agent/blob/main/docs/README.md) | Documentation index |
| [docs/sdk.md](https://github.com/Sagnnik/ness-agent/blob/main/docs/sdk.md) | SDK usage, public API, tracing |
| [docs/sdk-api.md](https://github.com/Sagnnik/ness-agent/blob/main/docs/sdk-api.md) | SDK API reference — signatures and contracts |
| [docs/cli.md](https://github.com/Sagnnik/ness-agent/blob/main/docs/cli.md) | Ness TUI, slash commands, MCP, permissions |
| [docs/configuration.md](https://github.com/Sagnnik/ness-agent/blob/main/docs/configuration.md) | Global config, `.ness/` layout, env vars |
| [docs/architecture.md](https://github.com/Sagnnik/ness-agent/blob/main/docs/architecture.md) | Prompt layers, modes, memory, compaction |

## Contributing

Contributions welcome. See [CONTRIBUTING.md](https://github.com/Sagnnik/ness-agent/blob/main/CONTRIBUTING.md) for dev setup and PR guidelines.

## License

Licensed under the Apache License 2.0. See [LICENSE](https://github.com/Sagnnik/ness-agent/blob/main/LICENSE).
