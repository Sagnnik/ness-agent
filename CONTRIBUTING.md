# Contributing to Ness Agent

Thanks for your interest in Ness Agent. This project is experimental (0.x): public APIs may change until 1.0.

## Development setup

Requirements: **Python 3.12+** and [uv](https://docs.astral.sh/uv/) (recommended).

```bash
git clone https://github.com/Sagnnik/ness-agent.git
cd ness-agent
uv sync
cp .env.example .env   # optional; tests use dummy keys by default
```

Install optional extras when needed:

```bash
uv sync --extra tracing
```

## Project layout

- `src/ness_agent/` — SDK (agent loop, tools, permissions, memory, etc.)
- `src/ness_cli/` — Ness CLI adapter and TUI (`ness` entry point)
- `tests/` — pytest suite (`test_sdk_*`, `test_cli/`, etc.)

## Running tests

```bash
OPENAI_API_KEY=test uv run python -m compileall -q .
OPENAI_API_KEY=test uv run pytest -q
```

Opt-in suites:

```bash
# Paid provider smoke tests (needs a real API key)
OPENROUTER_LIVE_TEST=1 OPENAI_API_KEY=... uv run pytest -q -m live

# Wheel build + clean install smoke test
PACKAGING_SMOKE=1 uv run pytest -q -m packaging
```

## Pull requests

1. Small PRs only — keep changes focused and minimal. Large PRs will not be accepted.
2. Large feature requests — open an issue first to discuss the design before writing any code.
2. Branch from `main` (e.g. `feat/cli-cron`, `fix/session-resume`).
3. Keep changes focused; match existing code style in touched files.
4. Ensure tests pass before opening a PR.

Link issues with `Closes #123` in the PR description when applicable.

## License

By contributing, you agree that your contributions are licensed under the
[Apache License 2.0](LICENSE), the same license as the project.
