# LUMI agentic environment portfolio

This is the environment split for Nemotron 3 Super RL on LUMI. It follows the
large-corpus training and held-out benchmark split described in Prime
Intellect's [Scaling Agentic RL](https://www.primeintellect.ai/blog/scaling-agentic-rl)
inventory. An environment is marked ready only after at least one real task
image completes its setup hook through the UCloud runtime.

## Initial training mix

The checked-in smoke config uses equal source sampling so a short validation
run exercises source diversity rather than reproducing corpus-size frequency.
It monitors but does not enforce the post-batch zero-advantage filter: a plumbing
run must still exercise the trainer and weight broadcast when an early policy
checkpoint happens to score zero on every member of a tiny sampled group. Restore
enforcement for production training so zero-gradient groups do not consume steps.
The supported smoke caps trainer sequences at 8K and uses context parallelism
two across 64 trainer ranks. Job `21541900` completed three variable-length
trainer steps across all four training sources and all three held-out
environments with no sandbox setup or authentication errors and 0% error in
the three required rollout batches. Rank-zero throughput was 1,175, 2,290, and
3,366 tokens/s; peak rank memory was 59.4 GiB. All three 224.77 GiB broadcasts
reloaded into the optimized vLLM server, and the final 225 GiB, 80-shard
checkpoint was marked stable. Two surplus Multi-SWE rollouts logged transient
relay transport exceptions after the required batches: one poll timeout
recovered and completed, while one response disconnect was drained at shutdown.

That smoke is a systems validation, not evidence of learning quality: its three
tiny rollout batches happened to contain 0/8 trainable groups and therefore
produced zero loss. A production run must restore the zero-advantage filter,
use enough concurrent groups to obtain reliable reward density, and verify a
non-zero update early before consuming the full allocation.
Production mixture weights should be selected from observed reward density,
error rate, rollout length, and duplicate rate after the smoke run.

| Environment | Rows | UCloud check | Role |
|---|---:|---|---|
| `scaleswe-v1` | 17,202 | 4/4 sampled images set up | train |
| `r2e-gym-v1` | 4,522 | 4/4 sampled images set up | train |
| `swelego-v1` | 4,323 | 4/4 sampled images set up | train |
| `multiswe-v1` | 2,232 | 4/4 sampled images set up | train |

The initial mix therefore covers 28,279 validated SWE tasks across Python, C,
C++, Go, Java, JavaScript, Rust, and TypeScript. The operational smoke config is
`configs/lumi/rl_nemotron3_super_swe_ucloud_smoke.toml`.

These tasksets select their container image and resource limits per task. Keep
`UCLOUD_SANDBOX_PREWARM=0` for this portfolio: prewarming the generic harness
runtime would attach capacity for the wrong image. The smoke config also uses
the shared, pre-populated `HARBOR_CACHE_DIR` so startup does not make concurrent
eval loaders initialize the same Harbor credentials and exports. Set the
mini-SWE harness `TMPDIR=/var/tmp`: UCloud mounts `/tmp` as a 64 MiB tmpfs, which
is too small to unpack `uv`, while `/var/tmp` uses the task's root filesystem.
The Harbor adapter likewise stages verifier test archives under `/var/tmp`, so
large benchmark test bundles do not exhaust that fixed tmpfs during scoring.
UCloud creation pacing is shared through
`~/.cache/verifiers/limiter/ucloud-sandbox.bucket`. The limiter must update that
cursor with a locked positional write; append-mode truncate/rewrite can leave a
NUL-filled gap on the shared filesystem when many env workers start together.
Parkable scale runs can also receive retryable `node_active_exec_deferred` 503s
after creation while the direct node makes active CPU or memory headroom. Set
`exec_attempts` and `retry_interval_seconds` on the UCloud runtime so these
responses backpressure setup instead of failing the rollout; the current
mid-scale config allows 120 attempts at five-second intervals.

## Held-out eval mix

These smaller tasksets remain eval-only and have representative UCloud setup
checks:

| Environment | Rows | UCloud check |
|---|---:|---|
| `swebench-verified-v1` | 500 | ready |
| `swebench-multilingual-v1` | 300 | ready |
| `terminal-bench-2-v1` | 89 | ready |

The smoke config evaluates one row from each at startup. Scale runs should use a
small fixed interim subset and run the complete suites separately at milestone
checkpoints; full benchmark rollouts should not block every training step.

## Deferred coverage

| Environment | Intended role | Current blocker |
|---|---|---|
| `swesmith-v1` | train | `swesmith` and `swebench` loader dependencies are not in the operational SIF metadata layer; task-image access remains unprobed. |
| `openswe-v1` | train | The configured Hugging Face identity does not have access to the gated `GAIR/OpenSWE` dataset. |
| `swerebench-v2-v1` | train | Task images use private `prime/primeintellect/...` references that UCloud workers cannot pull. |
| `terminal-lego-v1` | train | Dataset loading works without `git-lfs`, but the private `terminal-lego/...` image namespace needs a bulk UCloud mirror. |
| `tmax-v1` | train | The taskset is integrated and bounded managed images work, but full-corpus training needs bulk mirroring of its per-task Harbor images. |
| `openthoughts-tblite-v1` | eval | Its private task-image namespace needs a UCloud mirror. |
| `senior-swe-bench-v1` | eval | Its private task-image namespace needs a UCloud mirror. |
| Search training and BrowseComp eval | train/eval | The current bench has no configured search-provider credential; judge endpoints and costs also need an explicit choice. |

Use `scripts/lumi_validate_ucloud_taskset.sh` for representative setup checks.
Harbor-backed probes must run serially because Harbor 0.14 writes its shared
anonymous credentials cache non-atomically.
