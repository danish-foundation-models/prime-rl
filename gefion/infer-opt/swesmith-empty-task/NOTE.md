# swesmith rollouts with no task: root cause and fix (2026-10-31)

## Root cause: dataset rows with no issue text

The upstream SWE-smith datasets have no `problem_statement` for many rows. Our task list does not remove them.

- `SWESmithTaskset.load()` sets `prompt=row["problem_statement"]` without checking it (`deps/prime-envs/environments/swe/swesmith_env/swesmith_env/taskset.py`, the `yield SWESmithTask(...)` block). An empty statement becomes an empty prompt.
- The oracle manifest `manifests/ucloud-task-ids-20261024-oracle/swe-smith-train.task-ids.json` keeps a task when the no-edit run scores 0 and the gold-patch run scores 1. It never looks at the prompt, so empty-statement tasks pass.
- No fetch fails and no filter misfires. The "Network protocol blocked…" text is not an error. `append_user_notice()` (`deps/verifiers/verifiers/v1/dialects/base.py`, called from `chat.py:500`, `anthropic.py:542` and `responses.py:527`) adds it to the first user message of every request whenever the runtime is network-restricted (`block = ["*"]`). With a real task the notice is a trailing paragraph. With an empty task it is the whole message. In `stream/00001.jsonl`, all 6,717 swesmith model calls have `policy = null`, so nothing was actually blocked.

Rows with no statement in the pinned snapshots (`count_dataset.py`):

| lang | rows | empty | in our 6,805 ids | empty in ids |
| --- | ---: | ---: | ---: | ---: |
| py | 50,908 | 11,437 (22.5%) | 4,735 | 990 (20.9%) |
| go | 8,212 | 6,583 (80.2%) | 795 | 631 (79.4%) |
| java | 7,470 | 766 (10.3%) | 64 | 15 |
| js, ts, rs, cpp | | 0 | 1,211 | 0 |
| **total** | | | **6,805** | **1,636 (24.0%)** |

In job 534489's traces so far (`scan_traces.py`, `rewards.py`, 1,226 swesmith train rollouts):

- 338 rollouts (28%) have an empty task. They cover 14 distinct tasks: 10 py and 4 go, about 24 rollouts each (3 harnesses × 8). All 14 are in the empty list above.
- The bash and opencode harnesses show only the notice. The pi harness puts its own system text in the first user message, so its rollouts get the harness preamble with no task.
- Empty-task rollouts are solved 1 time in 172 scored. Rollouts with a statement are solved 168 times in 460.
- The degenerate-turn stop rate is about the same with and without a task (49% vs 48%). Empty tasks do not cause the degeneracy studied in `fp8-quality/REPORT.md`.

**What it costs:** nearly every empty-task group scores 0 for all rollouts. The `zero_advantage` gate (`advantage_range`, rejecting [0, 0]) drops those groups, so they rarely reach the trainer. The real cost is wasted rollout compute: about a quarter of swesmith sandbox and inference time. The rare group with one lucky solve does train, on a prompt with no task.

## Fix

**1. Config, needed now.** Point the 3 swesmith sources at a task list without the empty rows. I built it here: `swe-smith-train.nonempty.task-ids.json` (5,169 ids = 6,805 − 1,636; sha256 `3d38800a…c466`). Copy it next to the original manifest and change the following in `rl.toml` (`task_ids_file` at lines 653, 747 and 841, one per `swesmith-env-{opencode,pi,bash}` source):

```toml
[orchestrator.train.source.env.taskset]
id = "swesmith-env"
task_ids_file = "/dcai/projects/iu_0145/rasmus/manifests/ucloud-task-ids-20261024-oracle/swe-smith-train.nonempty.task-ids.json"
```

`filter_fn = "lambda r: bool((r['problem_statement'] or '').strip())"` alone would **not** work with `task_ids_file`. `load()` raises `Unknown, filtered, or unavailable task names` for every listed id that a filter drops. The task list itself has to change, or the code in fix 2 does.

Apply this at the next restart, not by editing the live config. Changing the list changes the task indices. Before resuming, check that the `difficulty_pool` sampler checkpoint keys tasks by name rather than by index. If it keys by index, start the curriculum state fresh.

**2. Code, so the problem cannot come back.** `swesmith-skip-empty-statement.diff` (prime-envs, not applied) makes `load()` skip rows with an empty `problem_statement`, log how many it skipped, and leave those ids out of the unknown-ids check. With it, the current manifest works unchanged and future manifests are safe. The oracle builder (`manifests/.../runs/finalize.py`) could also report `empty_problem_statement` as its own outcome.

**3. Optional follow-up in verifiers.** `append_user_notice()` adds "Network protocol blocked fetching a resource…" to every restricted rollout, even when `mediate_external_capabilities` removed nothing. It tells the model something false on every swesmith, openswe and other restricted task. Adding it only when `capabilities` is non-empty would remove this. It changes the prompt distribution for all envs, so it is a separate decision.

## Files

- `scan_traces.py <traces_dir> <out.json>`: per-language split of empty vs non-empty tasks and where the notice appears.
- `rewards.py <traces_dir>`: solve and stop rates, empty vs non-empty.
- `count_dataset.py <task_ids.json> <out.json>`: empty statements in the pinned HF snapshots (local parquet cache).
- `empty_task_ids.json`: the 1,636 empty-statement ids in the current manifest.
- `empty_tasks_in_traces.json`: the 14 that job 534489 has sampled so far.
- `swe-smith-train.nonempty.task-ids.json`: the filtered manifest.
- `swesmith-skip-empty-statement.diff`: the taskset patch.
