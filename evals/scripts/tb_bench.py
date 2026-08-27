from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path


JOB_DIR = Path("evals/jobs/tb-text-only-20-codex")
OUTPUT_FILE = Path("evals/TB_BENCH.md")
ATTEMPTS_PER_TASK = 5
PLANNED_TRIALS = 100
TASKS = [
    "adaptive-rejection-sampler",
    "cancel-async-tasks",
    "circuit-fibsqrt",
    "cobol-modernization",
    "compile-compcert",
    "configure-git-webserver",
    "constraints-scheduling",
    "count-dataset-tokens",
    "custom-memory-heap-crash",
    "db-wal-recovery",
    "distribution-search",
    "extract-elf",
    "feal-differential-cryptanalysis",
    "feal-linear-cryptanalysis",
    "filter-js-from-html",
    "fix-code-vulnerability",
    "fix-git",
    "fix-ocaml-gc",
    "kv-store-grpc",
    "largest-eigenval",
]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def money(value: float | None) -> str:
    return "not available" if value is None else f"${value:,.6f}"


def number(value: float | int | None, digits: int = 2) -> str:
    return "not available" if value is None else f"{value:,.{digits}f}"


def percent(value: float | None) -> str:
    return "not available" if value is None else f"{value * 100:.2f}%"


def duration(value: float | None) -> str:
    return "not available" if value is None else f"{value:,.2f} s"


def pass_at_k(rewards_by_task: dict[str, list[int]], k: int) -> float | None:
    if not rewards_by_task or min(len(values) for values in rewards_by_task.values()) < k:
        return None

    scores = []
    for values in rewards_by_task.values():
        n = len(values)
        successes = sum(values)
        if n - successes < k:
            scores.append(1.0)
            continue
        probability_all_fail = 1.0
        for index in range(k):
            probability_all_fail *= (n - successes - index) / (n - index)
        scores.append(1.0 - probability_all_fail)
    return statistics.mean(scores)


def main() -> None:
    job_result = read_json(JOB_DIR / "result.json")
    stats = job_result.get("stats") or {}
    trial_files = sorted(JOB_DIR.glob("*/result.json"))
    trials = [read_json(path) for path in trial_files]

    rewards: list[float] = []
    rewards_by_task: dict[str, list[int]] = defaultdict(list)
    durations: list[float] = []
    input_tokens: list[int] = []
    cache_tokens: list[int] = []
    output_tokens: list[int] = []
    costs: list[float] = []

    for trial in trials:
        verifier = trial.get("verifier_result") or {}
        reward_values = (verifier.get("rewards") or {}).values()
        reward = next((value for value in reward_values if isinstance(value, (int, float))), None)
        if reward is not None:
            rewards.append(float(reward))
            task_name = trial.get("task_name", "unknown")
            rewards_by_task[task_name].append(int(reward == 1.0))

        if trial.get("started_at") and trial.get("finished_at"):
            durations.append(
                max(0.0, (timestamp(trial["finished_at"]) - timestamp(trial["started_at"])).total_seconds())
            )

        usage = trial.get("agent_result") or {}
        if isinstance(usage.get("n_input_tokens"), (int, float)):
            input_tokens.append(int(usage["n_input_tokens"]))
        if isinstance(usage.get("n_cache_tokens"), (int, float)):
            cache_tokens.append(int(usage["n_cache_tokens"]))
        if isinstance(usage.get("n_output_tokens"), (int, float)):
            output_tokens.append(int(usage["n_output_tokens"]))
        if isinstance(usage.get("cost_usd"), (int, float)):
            costs.append(float(usage["cost_usd"]))

    total_input = sum(input_tokens) if input_tokens else None
    total_cache = sum(cache_tokens) if cache_tokens else None
    total_output = sum(output_tokens) if output_tokens else None
    total_cost = sum(costs) if costs else None
    pass_count = sum(reward == 1.0 for reward in rewards)
    observed = len(trials)
    completed = stats.get("n_completed_trials", observed)
    errored = stats.get("n_errored_trials", 0)
    cancelled = stats.get("n_cancelled_trials", 0)
    retries = stats.get("n_retries", 0)

    lines = [
        "# Terminal-Bench 2.1 — Ness Agent",
        "",
        "Run: `tb-text-only-20-codex` · Model: `gpt-5.6-luna`",
        "",
        f"Tasks: {len(TASKS)} · Attempts per task: {ATTEMPTS_PER_TASK} · Total planned trials: {PLANNED_TRIALS}",
        "",
        "Task names:",
        "",
        *[f"- `{task}`" for task in TASKS],
        "",
        "## Run summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Planned trials | {PLANNED_TRIALS:,} |",
        f"| Completed trials | {completed:,} |",
        f"| Errored trials | {errored:,} |",
        f"| Cancelled trials | {cancelled:,} |",
        f"| Retry count | {retries:,} |",
        f"| Overall reward mean | {number(statistics.mean(rewards) if rewards else None, 4)} |",
        f"| Overall pass count | {pass_count:,} / {observed:,} |",
        f"| Overall pass rate | {percent(pass_count / observed if observed else None)} |",
        f"| Harbor pass@2 | {percent(pass_at_k(rewards_by_task, 2))} |",
        f"| Harbor pass@5 | {percent(pass_at_k(rewards_by_task, 5))} |",
        "",
        "## Durations",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Trials with duration | {len(durations):,} |",
        f"| Mean | {duration(statistics.mean(durations) if durations else None)} |",
        f"| Median | {duration(percentile(durations, 0.50))} |",
        f"| P95 | {duration(percentile(durations, 0.95))} |",
        f"| Maximum | {duration(max(durations) if durations else None)} |",
        "",
        "## Tokens and costs",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Total input tokens | {number(total_input, 0)} |",
        f"| Average input tokens | {number(total_input / len(input_tokens) if input_tokens else None, 2)} |",
        f"| Total output tokens | {number(total_output, 0)} |",
        f"| Average output tokens | {number(total_output / len(output_tokens) if output_tokens else None, 2)} |",
        f"| Total cached input tokens | {number(total_cache, 0)} |",
        f"| Cache hit rate | {percent(total_cache / total_input if total_input else None)} |",
        f"| Total cost | {money(total_cost)} |",
        f"| Average cost | {money(total_cost / len(costs) if costs else None)} |",
        f"| Cost per successful trial | {money(total_cost / pass_count if total_cost is not None and pass_count else None)} |",
        "",
    ]
    report = "\n".join(lines)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
