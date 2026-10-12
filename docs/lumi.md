# LUMI Guide

PRIME-RL owns training, inference, orchestration, and the Slurm runtime on
LUMI. The repository remains a live checkout; generated outputs live outside
the checkout under the bench artifact tree.

See [LUMI downstream maintenance](lumi-downstream.md) for the integration
branch policy, refresh procedure, and dependency-fork requirements.

## Runtime layout

- Project: `project_465002751`
- Checkout: `/scratch/project_465002751/rasmus/prime-rl`
- Runtime image:
  `/scratch/project_465002751/shared/sif/lumi-posttrain-u24r71-t211-vllm024.sif`
- Training artifacts: `/scratch/project_465002751/rasmus/artifacts/training`
- Hugging Face cache: `/scratch/project_465002751/.cache/huggingface`
- Other caches: `/scratch/project_465002751/rasmus/.cache`
- Temporary files: `/scratch/project_465002751/rasmus/tmp`
- Launcher uv: `/scratch/project_465002751/rasmus/bin/uv`

The SIF contains the compiled ROCm, Torch, Triton, vLLM, and Python dependency
stack. Run it directly with `singularity exec`; do not add `--rocm` and do not
activate a writable overlay.

PRIME-RL and its in-repo workspace packages are mounted from the checkout and
placed first on `PYTHONPATH`. vLLM discovers PRIME-RL patches through the
`vllm.general_plugins` distribution entry point, so
`scripts/lumi_prepare_runtime.sh` maintains a small commit-keyed metadata target
under `.cache/prime-rl/metadata`. The metadata target never replaces live source.
Its key also includes `pyproject.toml`, because entry points are packaging
metadata. Ordinary dirty source edits intentionally reuse the target and are
picked up immediately through `PYTHONPATH`.

RL jobs also add every local package beneath
`deps/verifiers/environments/` and `deps/research-environments/environments/`
to the live-source path. This is generated centrally by
`scripts/lumi_runtime_env.sh`; repository-bundled environments such as
`reverse-text` therefore remain outside the SIF and track checkout edits.

## Launch surfaces

Use the repository-owned wrappers from the checkout root:

```bash
./scripts/lumi_submit_sft.sh --dry-run
./scripts/lumi_submit_rl.sh --dry-run
```

Remove `--dry-run` to submit. Select an experiment with `CONFIG`:

```bash
CONFIG=$PWD/configs/lumi/rl_single_node.toml ./scripts/lumi_submit_rl.sh
CONFIG=$PWD/configs/lumi/sft_single_node.toml ./scripts/lumi_submit_sft.sh
```

RL dry-runs render configs and Slurm scripts without cleaning checkpoints,
rollouts, broadcasts, or an external checkpoint directory. If a queued run already
contains health-monitor artifacts, validate changed settings with a distinct
`--run.name <validation-name>`; do not use `--clean` to bypass the artifact guard.
Queued Slurm scripts retain their attempt-specific config paths and submitted
environment. Editing the source TOML alone does not update those inputs.
The batch script reloads the bench and worktree `.env` files at startup; a
worktree `PRIME_RL_METADATA_DIR` pin can select a verified immutable SDK layer
without resubmitting the job. Keep credentials in the bench `.env`.

The wrappers invoke the native entry points with `uv run` inside the SIF and
forward remaining arguments. The container always renders in dry-run mode; if
the caller did not pass `--dry-run`, the wrapper submits the resulting script
with the host `sbatch`. Slurm does not need to be installed in the image. The
wrappers do not create or synchronize a host `.venv`, so a fresh checkout cannot
trigger a full GPU dependency install on the login node. A bench-level workflow
should call these wrappers with an absolute `CONFIG` path rather than reproducing
their Slurm or container logic.

The active LUMI templates are:

- `src/prime_rl/templates/lumi/single_node_rl_lumi.sbatch.j2`
- `src/prime_rl/templates/multi_node_rl_lumi.sbatch.j2`
- `src/prime_rl/templates/lumi/single_node_sft_lumi.sbatch.j2`
- `src/prime_rl/templates/lumi/multi_node_sft_lumi.sbatch.j2`

The generic single-node configs write beneath
`/scratch/project_465002751/rasmus/artifacts/training`. Experiment configs must
also use that artifact root; generated files do not belong in the repository.

## Environment overrides

The runtime has stable defaults, with these supported overrides:

- `CONFIG`: input TOML for a submit wrapper
- `SIF`: alternate immutable dependency image
- `LUMI_ROOT`: bench root containing caches, temporary files, and siblings
- `LUMI_UV`: uv executable bound into the SIF for the launcher
- `PRIME_RL_HF_HOME`: Hugging Face cache root
- `PRIME_RL_UV_CACHE_DIR`, `PRIME_RL_XDG_CACHE_HOME`,
  `PRIME_RL_WANDB_CACHE_DIR`, `PRIME_RL_WANDB_CONFIG_DIR`,
  `PRIME_RL_MPLCONFIGDIR`, `PRIME_RL_TMPDIR`: non-home runtime state
- `PRIME_RL_METADATA_DIR`: prepared distribution-metadata target inside a job
- `NEVER_CLEAN_OUTPUT_DIR` and the `PRIME_RL_SFT_DEBUG_*` diagnostics are
  forwarded through the clean launcher container when set by a workflow
- `SBATCH_BIN`: host submission executable, useful for submission-path tests

The supported Hugging Face cache knob is `HF_HOME`. Do not split Hub or datasets
caches into separate paths.

## Source and plugin preflight

The runtime helper can be exercised without allocating a GPU:

```bash
SIF=/scratch/project_465002751/shared/sif/lumi-posttrain-u24r71-t211-vllm024.sif \
  ./scripts/lumi_prepare_runtime.sh
```

The printed directory must contain `prime_rl-*.dist-info/entry_points.txt` and
the PRIME-RL command wrappers. At job startup the Slurm templates prepend live
source directories to `PYTHONPATH` and add the metadata target's `bin` directory
to `PATH`.

## Monitoring

```bash
squeue -u "$USER" -o "%.18i %.9P %.30j %.8T %.10M %.6D %R"
sacct -j <jobid> --format=JobID,JobName%30,Partition,State,ExitCode,Elapsed,NNodes,NodeList -P
```

For a run output directory:

- Slurm output: `<output_dir>/job_<jobid>.log`
- Trainer: `<output_dir>/logs/trainer.log`
- Orchestrator: `<output_dir>/logs/orchestrator.log`
- Inference: `<output_dir>/logs/inference.log`
- Per-node streams: `<output_dir>/logs/{trainer,inference}/node_<rank>.log`

## Troubleshooting

`FATAL: SIF not found`

- Verify the shared image path or set `SIF` explicitly.

`expected one prime_rl vLLM plugin`

- Run `scripts/lumi_prepare_runtime.sh` and ensure its returned directory is on
  `PYTHONPATH` after the live source paths.
- Do not install PRIME-RL dependencies over the immutable image at job startup.

Imports resolve from a stale source copy

- Ensure `/workdir/src` and the in-repo workspace package paths precede
  `PRIME_RL_METADATA_DIR` on `PYTHONPATH`.
- The metadata target is intentionally last; it supplies distribution metadata
  and console scripts, not authoritative source.

Unexpected writes under `$HOME`

- Inspect the cache variables above. LUMI launchers set all of them beneath the
  project scratch trees and set `PYTHONNOUSERSITE=1` in the container.

See [Nemotron Super RL](lumi-rl-super.md) for the main/LUMI runtime, optimized
serving, private checkpoint conversion, and pilot readiness checks.

### Agentic rollout capacity

Agent episodes alternate model calls with setup, tools, and grading. Episode
concurrency is not the same as active decode requests. For eight inference
replicas, the scale config starts at 256 episodes and permits adaptive control
within 128–512, with environment multiplex ceilings of 512. Batch size remains
64 and group size eight. Confirm sandbox CPU admission capacity as well as KV
headroom before raising load; KV feedback alone does not measure sandbox load.
Use full publisher pools, including MultiSWE and TMax, with recipe-backed image
preparation. Representative environment checks do not imply task allowlists; see
[expanded sources](lumi-rl-expanded-sources.md).

### RL startup and task pools

The orchestrator loads task pools concurrently with inference model startup and
startup weight synchronization. Taskset construction and finite decoding run off
the event loop; sources are still decoded sequentially to avoid GIL contention
and datasets/tqdm global-state races. Identical read-only training task pools are
shared across harness variants. Curriculum/runtime state remains separate.

Bounded evals materialize only the requested prefix. A full eval (`num_examples=-1`)
and every finite training source retain their full pools. Cache identity includes
the materialization limit, so an eval prefix can never replace a full train pool.

Before diagnosing dataset delay, distinguish inference wait, startup broadcast,
training task load, and eval load from the orchestrator log. A failed env-server
can end the allocation before any dataset load: inspect per-source logs as well
as the Slurm log. Host MCP tools use the serving environment's interception
transport. With `interception.type="ucloud"`, shared and per-task host tools use
our UCloud HTTP relay; no Prime tunnel or Prime account is required. Host state
callbacks retain the public relay URL. `UCloudRuntime.host_url` translates only
restricted guests' framework URLs to the configured private relay origin,
preserving capability paths and signed state query parameters. Validate both a
real MCP request and its state callback before launching a new remote-tool env.

Pilot monitoring must check explicit fatal orchestrator/trainer records even before
any episodes exist. `lumi_watch_rl_pilot.py --stop-on-infra-failure` cancels its
specified job immediately on these records as well as on sustained non-image
execution failures or loss of useful progress. Confirmed image-build failures
quarantine affected tasks across training harnesses; they do not stop the job.
See `docs/lumi-rl-expanded-sources.md` for thresholds and resume-safe monitoring. Slurm RUNNING and responsive inference health endpoints do not establish
training health: a failed orchestrator can remain alive while its dataset-loader
thread unwinds. The monitor retains fatal records across subsequent polls.

Inference admin HTTP clients use fresh connections (`max_keepalive_connections=0`).
Concurrent Arrow/Python task decoding may hold the GIL long enough to race the
engine's short keep-alive timeout. Reusing health-check sockets for a one-shot
NCCL initialization can fail after only some workers entered the collective.
Do not retry ambiguous NCCL initialization requests without server-side
idempotency. The generation clients and their connection pools are separate.

## Throughput-aware RL scheduling

See [throughput scheduling](throughput-scheduling.md) for independent rollout
inventory, balanced consumption, bounds, and monitoring. Enable it explicitly with
`orchestrator.concurrency.throughput_balancing = true`; dry-run the selected config
before submission.

For worktree-based LUMI runs, provision the validated `third_party/llmd/bin`
executables (`epp`, `envoy`, and `pd-sidecar` for disaggregated inference).
They are external artifacts, so Git worktree creation does not copy them. A shared
symlink must target a container-visible `/scratch/...` path. The native RL submit
wrapper now checks the rendered router backend and runs `lumi_check_router.sh`
inside the operational SIF during dry-run and submission. Do not accept a render-only
check as executable availability validation. Missing router binaries cause exit 127
on inference node zero and tear down the whole allocation.

The wrapper also refuses, at dry-run and submission, a config whose inference request
slots cannot serve every rollout in flight: `num_infer_replicas` x
`[inference.vllm] max_num_seqs` must be at least `[orchestrator.concurrency]
max_inflight` (`scripts/lumi_check_inference_slots.py`). Below that, rollouts queue in
vLLM while their sandboxes idle; the r21 smoke r4 ran 4 x 64 slots for 512 rollouts.
Check vLLM's KV cache usage after raising `max_num_seqs`.

The orchestrator's concurrency controller sizes the in-flight pool from engine KV use. A
rollout's KV keeps growing long after admission (SWE contexts reach 128K, math answers
32K), so KV use lags admissions by a rollout's lifetime. It therefore grows the pool by
only 10% per pipeline turnover, KV trims (above 80% mean use across engines) lower the cap
and let rollouts drain, and each trim sets a ceiling at 90% of the pool that overflowed,
which rises 2% per turnover while the engines stay clear. vLLM preemptions and a persistent
queue cut the cap by 20% and 10%, once per drain. Nothing cancels rollouts. Trims read
the mean because one engine holding a few long SWE contexts says nothing about the
fleet's headroom.

On 2026-10-13 (r21 13a) cancelling still made things worse. The orchestrator's
event loop stalled for 25-59 s at a time while it shipped three buffered steps. Packing
took 66-127 s in a thread, competing for the GIL. Rollouts stopped sending requests,
so the engines drained to ~100 running. When the loop resumed, ~900 rollouts sent their
next turn at once. One engine preempted, and two escalating cuts cancelled ~700
rollouts. Inference then ran at a third of its throughput.

Packing and per-rank encoding now run in a spawned worker process
(`BatchPacker.pack_encoded`). A watchdog thread logs `Event loop stalled for Ns; thread
stacks:` with every thread's stack when the loop goes quiet for 10 s. Check
`orchestrator.log` for that line first when inference throughput drops to zero.

Slurm commands such as `squeue` are host tools, not included in the SIF. For exact-run
sandbox cleanup, perform the ended-job check on the host and pass a fresh, job-bound
proof to the container-side SDK cleanup; recheck each sandbox's exact run label.

### Sandbox SDK release updates

Submit through `CONFIG=... ./scripts/lumi_submit_rl.sh`. The generated Slurm
script requires `PRIME_RL_METADATA_DIR` in the submission environment; it does
not embed that path. Direct `sbatch` after a separate dry-run loses the wrapper
export and can fail before startup.

The UCloud client is three released packages pinned in
`scripts/lumi_prepare_runtime.sh`: the `rlrs/verifiers` fork, `rlrs/verifiers-ucloud`
and the sandbox SDK wheel that verifiers-ucloud was released with. The preparer
installs them into an immutable metadata layer; jobs import them from that layer,
never from `deps/` checkouts. To update, change the pins to the new release tags,
re-run the preparer, and verify the imported versions and paths inside the job
container. Do not edit an existing metadata layer or the shared SIF in place.
Recheck launch dry-runs after updating dependencies.

#### ECHO and restricted-network prompt stability

The Super r8 configuration keeps score centering and adds tool-observation CE at
0.02 for SWE, terminal, and SQL sources. Admission retains independently supervised
CE-only groups; category targets, inventory limits, and policy-lag checks still
apply. The ECHO per-source override explicitly preserves behavior shaping.

The verifiers dialect helper anchors a synthetic network-policy user notice after
leading system/developer messages when a harness supplies no user message. It must
not append that notice after the growing assistant/tool history: doing so breaks
prefix reuse and fragments the exact-token training graph. Insertion is idempotent
and provider-capability restrictions still apply. Keep this fix in the live
verifiers source used by the interception server; it does not require rebuilding
sandbox harness bundles.

Before resubmission, run the algorithm/curriculum regressions, replay stored native
traces through ECHO, and check combined score-centering/CE gradients on a GPU.
The bounded GPU probe and native-token replay are recorded under
`artifacts/training/echo-investigation-20260923`. They do not constitute a distributed
128K throughput benchmark; monitor the first real trainer steps separately.
