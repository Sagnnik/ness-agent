# Terminal-Bench 2.1 — Ness Agent (full sweep, updated Luna)

Run: `tb-full-01..09-of-09-codex` · Model: `codex/gpt-5.6-luna` (unpinned alias) · Agent: `ness-agent==0.2.4` · Date: 2026-09-28 · Attempts per task: 3

Tasks: 89 across 9 configs (10 each, last = 9) · Trials: 267 completed · Scored: 254 (13 setup-phase infra flakes unscored)

## Full-sweep summary

| Metric | Value |
| --- | ---: |
| Completed trials | 267 |
| Errored trials | 33 |
| Scored trials | 254 |
| Pass count | 191 / 254 |
| Trial pass rate | 75.20% |
| Task-mean pass@1 | 75.67% |
| Harbor pass@2 | 85.06% |
| Harbor pass@3 | 87.50% |

## Per-config results

| Config | Trials | Errored | Pass | Mean |
| --- | ---: | ---: | ---: | ---: |
| `tb-full-01-of-09-codex` | 30 | 3 | 23 / 30 | 0.767 |
| `tb-full-02-of-09-codex` | 30 | 0 | 23 / 30 | 0.767 |
| `tb-full-03-of-09-codex` | 30 | 3 | 17 / 30 | 0.567 |
| `tb-full-04-of-09-codex` | 30 | 4 | 23 / 30 | 0.767 |
| `tb-full-05-of-09-codex` | 30 | 7 | 23 / 26 | 0.767* |
| `tb-full-06-of-09-codex` | 30 | 6 | 25 / 27 | 0.833* |
| `tb-full-07-of-09-codex` | 30 | 6 | 19 / 24 | 0.633* |
| `tb-full-08-of-09-codex` | 30 | 0 | 23 / 30 | 0.767 |
| `tb-full-09-of-09-codex` | 27 | 4 | 15 / 27 | 0.556 |

`*` scored denominator excludes setup-phase infra flakes with no verifier reward (05: 4, 06: 3, 07: 6 PyPI/apt failures).

## Durations (267 trials)

| Metric | Value |
| --- | ---: |
| Mean | 622.10 s |
| Median | 396.50 s |
| P95 | 1,891.70 s |
| Maximum | 3,718.90 s |

## Tokens and costs

| Metric | Value |
| --- | ---: |
| Total input tokens | 312,386,373 |
| Average input tokens | 1,229,867.60 |
| Total output tokens | 4,076,809 |
| Average output tokens | 16,054.76 |
| Total cached input tokens | 291,913,984 |
| Cache hit rate | 93.45% |
| Total cost | $14.824900 |
| Average cost | $0.05837 |
| Cost per successful trial | $0.07762 |

## Failure taxonomy (76 failed entries: 63 reward-0 + 13 unscored setup flakes)

- Luna capability: ~57 (vision/FEN errors, XSS over-refusal + fixation, asyncio semantics, workspace hygiene, slow-origin download strategy, SQL plans, ISA fidelity, glyph/axis misreads, formatting vs verifier regex, off-by-one tuning, grad-comm/shard bugs, search blowups)
- Harness/infra flakes (retryable, unscored): 13 — PyPI `sse-starlette`/`click` fetch timeouts (05: 4, 06: 3), Debian-security `bullseye` apt 404 on qemu images (07: 6)
- Harness false negatives (scored 0, agent correct): 2 — Valgrind SIGSEGV in its own loader under gVisor (`custom-memory-heap-crash`, identical fix as passer)
- Ness-agent bugs: 2 — compaction crash `400 Unsupported parameter: max_output_tokens` (`extract-moves-from-video__Jf6HwCK`, `mteb-leaderboard__egWL2hm`; long-context trials ~80 steps / ~10M input tokens, died before writing artifacts)
- Strict/brittle verifiers (agent functionally close): 3 — `build-pmars` glob `[0]` dir ordering, `install-windows-3.11` image-path string match (VM booted), `mcmc-sampling-stan` regex drops `e+01`
- Timeout-as-mechanism: 4 timeouts on trials that still passed (shared-env artifacts final before kill); all other timeouts reflect luna search inefficiency, not harness faults
- Minor collection artifact: `ness.txt` 0-byte on timeout kills (transcript survives in `trajectory.json`)

Per-task fail details and evidence: see `evals/TB_FULL_FAILED.html` (aggregated failed-run artifact, self-contained).

---

# Terminal-Bench 2.1 — Ness Agent (previous run, kept for reference)

Run: `tb-text-only-20-codex` · Model: `gpt-5.6-luna`

Tasks: 20 · Attempts per task: 5 · Total planned trials: 100

Task names:

- `adaptive-rejection-sampler`
- `cancel-async-tasks`
- `circuit-fibsqrt`
- `cobol-modernization`
- `compile-compcert`
- `configure-git-webserver`
- `constraints-scheduling`
- `count-dataset-tokens`
- `custom-memory-heap-crash`
- `db-wal-recovery`
- `distribution-search`
- `extract-elf`
- `feal-differential-cryptanalysis`
- `feal-linear-cryptanalysis`
- `filter-js-from-html`
- `fix-code-vulnerability`
- `fix-git`
- `fix-ocaml-gc`
- `kv-store-grpc`
- `largest-eigenval`

## Run summary

| Metric | Value |
| --- | ---: |
| Planned trials | 100 |
| Completed trials | 100 |
| Errored trials | 11 |
| Cancelled trials | 0 |
| Retry count | 0 |
| Overall reward mean | 0.7100 |
| Overall pass count | 71 / 100 |
| Overall pass rate | 71.00% |
| Harbor pass@2 | 80.00% |
| Harbor pass@5 | 85.00% |

## Durations

| Metric | Value |
| --- | ---: |
| Trials with duration | 100 |
| Mean | 533.60 s |
| Median | 358.79 s |
| P95 | 1,482.73 s |
| Maximum | 1,808.42 s |

## Tokens and costs

| Metric | Value |
| --- | ---: |
| Total input tokens | 67,266,295 |
| Average input tokens | 672,662.95 |
| Total output tokens | 1,489,589 |
| Average output tokens | 14,895.89 |
| Total cached input tokens | 62,163,456 |
| Cache hit rate | 92.41% |
| Total cost | $4.051344 |
| Average cost | $0.040513 |
| Cost per successful trial | $0.057061 |
