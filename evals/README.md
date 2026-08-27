# Harbor evals

The current adapter runs Ness through OpenRouter. It does not enable image input yet.

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

If you want to view the jobs info:
```bash
harbor view jobs
```
and visit `localhost:8080`