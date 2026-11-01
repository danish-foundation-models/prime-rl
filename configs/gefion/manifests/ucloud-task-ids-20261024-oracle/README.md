# Oracle-validated task ids (openswe, swesmith-env), 2026-10-25

These lists keep only tasks that grade correctly with the sanitized tasksets in prime-envs `679dfd3d7`.

| File | Tasks | From |
| --- | --- | --- |
| `swe-smith-train.task-ids.json` | 6,805 | 10,000 sampled from `../ucloud-task-ids-20261010b/swe-smith-train.task-ids.json` |
| `openswe.task-ids.json` | 877 | 1,466 checked of 10,000 sampled from `../ucloud-task-ids-20261010b/openswe.task-ids.json` (run stopped early) |

Use a file as the taskset's `env.taskset.task_ids_file`. Both load with the pinned tasksets and their default revisions.

## Rule

A task is kept only when both runs succeeded without errors:

- A no-edit run scored 0.
- A gold-patch run scored 1.

Both runs used fresh, relay-only UCloud sandboxes, the same runtime settings as r21, and a 1,200 s scoring limit.

Every other sampled task is listed in `<family>.report.json` under `tasks`, with an outcome and a reason:

| Outcome | Meaning |
| --- | --- |
| `free_reward` | Scores 1 with no edits |
| `gold_fails` | The gold patch scores 0 |
| `gold_error` / `noedit_error` | The task's own grading failed: gold patch does not apply, scoring timeout, checkout failure |
| `image_build_failed` | The image recipe does not build |
| `unchecked` | An infrastructure error, or never run (most of OpenSWE) |

`by_group` breaks these counts down by SWE-smith language and by OpenSWE eval-script flag.

## Reproduce or extend

- `runs/launch.sh` resumes both runs. It skips cells that already have a result.
- `runs/finalize.py` rewrites the lists and reports from `runs/*.jsonl`. A `.redo.jsonl` file overrides the first pass.
- The probe is `tmp/git-leak-probe/probe.py`.
- The samples are `*.sample.task-ids.json`, drawn with `random.Random(20261024)`.

To grow OpenSWE, rerun `runs/launch.sh` and then `runs/finalize.py`. Before resuming, check that the gateway's image index answers. The oracle stopped on 2026-10-25 while the gateway returned `503 routing state unavailable`.

## Context

The full write-up of the issues and fixes is the Claude doc "SWE RL task integrity: git leaks and broken graders" (https://claude.ai/artifact/PPH9oZqykMUPAjKbmvLJPv).

## swe-smith-train.nonempty-stable.task-ids.json (2026-11-01)
`swe-smith-train.nonempty.task-ids.json` minus 4 tasks whose scorer gave both 1 and 0 to identical exact
fixes in run gefion-super-r6s2638-8t4p4d-tp8-r2 (exact-fix rollouts scored 0 in 2/5 to 5/11 cases):
Byron__dua-cli.8570c154.func_pm_op_change__ore7h0ad, benoitc__gunicorn.bacbf8aa.func_pm_remove_assign__d8mlhfip,
facelessuser__soupsieve.a8080d97.func_basic__e262md05, Netflix__falcor.39d64776.func_pm_remove_assign__rcvwwkoy.
