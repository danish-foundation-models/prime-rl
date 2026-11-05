# Task lists for the r5 restart from step 110 (2026-11-05)

These lists are derived from the lists r5 trained on up to step 110.

| File | Source list | Removed |
|---|---|---|
| `openswe.task-ids.json` (830) | `ucloud-task-ids-20261024-oracle/openswe.task-ids.json` (877) | 47 broken or suspect |
| `scaleswe-train.task-ids.json` (2,433) | `ucloud-task-ids-20261010b/scaleswe-train.task-ids.json` (2,468) | 35 broken or suspect |
| `swe-smith-train.no-java.task-ids.json` (5,090) | `ucloud-task-ids-20261024-oracle/swe-smith-train.nonempty-stable.task-ids.json` (5,165) | all `java:` tasks (49); 25 broken or suspect; `cpp:Tencent__ncnn.a64aa7ff.func_pm_aug_assign_swap__l5h15lyi` (scored 1 with no fix in place) |
| `tmax.task-ids.json` (14,597) | `ucloud-task-ids-20261010b/tmax.task-ids.json` (14,600) | 3 broken or suspect |
| `terminal-lego.medium-hard.task-ids.json` (4,483) | `ucloud-task-ids-20261010b/terminal-lego.task-ids.json` (13,765) | keeps only the medium and hard labels |

- **Why Java is out of swesmith and Rust/C++ are back:** these tasks are graded in the agent's own sandbox, on whatever build state the agent left behind. In training, exact gold fixes scored 0 in 42% of Rust, 19% of C++ and 18% of Java rollouts. The swesmith scoring audit (`/dcai/projects/iu_0145/rasmus/reports/swesmith-scoring-20261105/`) found and fixed the causes for two of the three:
  - **Rust:** cargo's automatic cache cleanup, triggered by the agent's partial build, deletes test-only crates from the offline cache. The fix turns cleanup off at setup.
  - **C++:** `ctest --rerun-failed`, plus agent reconfiguration of the build dir. The fix strips the flag and restores the image's CMake cache.
  - **Java:** the cause is unknown, so it stays out.
- **Source of the broken and suspect lists:** `excluded-broken-and-retired.from-step95.csv`, where `kind` starts with `broken`.
- **How the run uses these lists:** the restart does not load the `*.task-ids.json` lists above. It keeps the step-110 task lists. The sampler skips the dropped tasks by key, listed in `<taskset>.exclude-keys.json`.
  - **Why:** a task key is a content hash that includes the task's row index. A shorter list therefore re-keys every task after the first removed one, and the step-110 sampler history no longer matches. Only 1 of 864 openswe history keys matched the shorter list.
  - **The failure it caused:** job 536582 failed at startup on this. The terminal-lego and tmax task caches record the task list they were built from.
  - **How the keys were made:** `exclude-keys.json` holds the keys of tasks in the old list that are not in the list above. Tasks were compared on content without the row index. Script: `exclude_keys.py`. terminal-lego is matched by task name instead (`exclude_keys_terminal_lego.py`), because its cached rows and a fresh load differ in paths.
- **Retired tasks (`retire-*`) are not removed from these lists.** The sampler now retires them itself (`retire_solved_after` / `retire_unsolved_after`), using the same rules.
