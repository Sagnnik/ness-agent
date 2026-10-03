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
`--n-attempts` is the number of attempts on the same task. Rewards are averaged by the number of attempts.

Or run it with the config file:
```bash
PYTHONPATH=. uv run --group evals harbor run \
    -c evals/configs/tb-single-task.yaml \
    --env-file .env
```

To run the Ness SDK through the ChatGPT-authenticated Codex model, use the
separate Codex config. `CODEX_AUTH_JSON=true` makes the adapter read
`~/.codex/auth.json` on the host, upload it only for the agent phase, and remove
the temporary copy when the trial ends:

```bash
CODEX_AUTH_JSON=true \
PYTHONPATH=. uv run --group evals harbor run \
    -c evals/configs/tb-single-task-codex.yaml \
    --env-file .env
```
The auth file is not committed and must not be placed under `evals/jobs`.

The Codex adapter is the frozen `ness-agent-0.2.4-codex-eval-v1` snapshot. Its
Harbor installer reads the package pin from `evals/codex/codex_chat_model.py`
and installs the released `ness-agent==0.2.4` in the sandbox. The runner checks
that version and its legacy `ness_cli.provider.codex` layout before using the
SDK. A current checkout with the ported `ness_cli.providers` layout is not a
compatible replacement, even if its development version is still `0.2.4`.

The conversion code, pricing estimates, and context windows in this eval copy
stay independent of CLI changes. To upgrade an eval, create a new reviewed
snapshot with a new revision/package pin and update its conversion, usage,
cost, and sandbox compatibility tests together. Preserve the old snapshot and
job configuration when reproducing historical runs.

The release wheel verified for this snapshot is
`ness_agent-0.2.4-py3-none-any.whl`, SHA-256
`feb1fe05cdcd765f040cfd5ee5d4f358597d93ef8d4af8e0c7546e6d6196c121`.

[The failed-or-errored rerun config](configs/tb-failed-or-errored-codex.yaml)
selects 44 unique tasks from the recorded September 28 nine-job sweep. It
includes every task with a reward-zero attempt, an unscored attempt, or a trial
exception, including exceptions on reward-one attempts. The selection covers
82 historical attempts: 63 reward-zero, 13 unscored, and six reward-one with
errors. It schedules three fresh attempts per task, 132 trials total, with the
original model, concurrency, and immutable dataset revision.

This config selects tasks; it does not upgrade the installed Ness package.
Before using it to measure branch fixes, prepare the compatible adapter snapshot
and updated package pin described above. Its current agent path still installs
the historical `0.2.4` release.

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
