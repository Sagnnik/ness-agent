# Harbor evals

The default adapter runs Ness through OpenRouter. The `evals/codex` adapter runs
the same Ness SDK through the ChatGPT-authenticated Codex Responses transport.

Install the eval dependencies:

```bash
uv sync --group evals
```

Run one selected Terminal-Bench 2.1 task on Modal:

```bash
PYTHONPATH=. uv run --group evals harbor run \
  -d terminal-bench/terminal-bench-2-1 \
  --include-task-name terminal-bench/headless-terminal \
  --agent evals.ness_harbor_agent:NessAgent \
  --model openrouter/deepseek/deepseek-v4-flash \
  --env modal \
  --env-file .env \
  --jobs-dir evals/jobs \
  --n-concurrent 3 \
  --n-attempts 3
```

Use another `--include-task-name` value to select a different task. Repeat the
flag to run several named tasks in one job. `--n-concurrent` is the number of concurrent sandboxes while
`--n-attempts` is the number of attemps on the same task. Rewards are averaged by the number of attempts

Or run it with the config file:
```bash
PYTHONPATH=. uv run --group evals harbor run \
    -c evals/configs/tb-single-task.yaml \
    --env-file .env
```

To run the Ness SDK through the ChatGPT-authenticated Codex model, use the
separate Codex config. `CODEX_FORCE_AUTH_JSON=true` makes the adapter read
`~/.codex/auth.json` on the host, upload it only for the agent phase, and remove
the temporary copy when the trial ends:

```bash
CODEX_AUTH_JSON=true \
PYTHONPATH=. uv run --group evals harbor run \
    -c evals/configs/tb-single-task-codex.yaml \
    --env-file .env
```
The auth file is not committed and must not be placed under `evals/jobs`.

Codex subscription runs attach an API-equivalent cost estimate to each usage
event. The standard short-context rates (USD per 1M tokens) are:

| Model | Input | Cached input | Output | Cache write |
| --- | ---: | ---: | ---: | ---: |
| `gpt-5.6-sol` | $4.00 | $0.40 | $20.00 | $5.00 |
| `gpt-5.6-terra` | $2.00 | $0.20 | $12.00 | $2.50 |
| `gpt-5.6-luna` | $0.20 | $0.02 | $1.20 | $0.25 |

These mirror the [OpenAI API pricing schedule](https://developers.openai.com/api/docs/pricing)
for estimation only; they are not ChatGPT subscription charges. Prompts over
272K input tokens use the long-context multipliers from that schedule.

If you want to view the jobs info:
```bash
harbor view jobs
```
and visit `localhost:8080`