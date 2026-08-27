# LUMI Guide

PRIME-RL owns training, inference, orchestration, and the Slurm runtime on
LUMI. The repository remains a live checkout; generated outputs live outside
the checkout under the bench artifact tree.

See [LUMI agentic environment portfolio](lumi-agentic-environments.md) for the
large-corpus training mix, held-out eval tasksets, and current UCloud coverage.

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
Its key also includes `pyproject.toml` and the runtime-preparation script,
because entry points and the explicitly layered packages are packaging
metadata. Ordinary dirty source edits intentionally reuse the target and are
picked up immediately through `PYTHONPATH`.

The metadata target also supplies the exact locked Harbor CLI, Multi-SWE scorer
and its missing transitive packages, plus the `modelexpress`, `torchao`,
`mamba-ssm`, `prime-sandboxes`, and `vllm-router` runtime companions.
`mamba-ssm` is compiled for `gfx90a` and is required by context-parallel
NemotronH training. The router wheel supplies the default local inference
front end when the base image predates PRIME-RL's router integration. The
`prime-sandboxes` pin must satisfy the version required by the live Verifiers
checkout; otherwise importing `verifiers.v1` can fail before configuration is
parsed. Its matching `connectrpc` and `protobuf-py` distributions are layered
too, so they take precedence over the legacy `connect-python` module path in
the base image. Container launchers also mask that legacy distribution's
metadata, because current `prime-sandboxes` rejects a partial in-place upgrade.
Packages are listed explicitly and installed without dependency resolution so
this thin layer does not replace the SIF's tested Torch, Pydantic, FastAPI, or
other core runtime stack. This keeps taskset downloads, scoring, and model
loading aligned with the checkout when the immutable SIF contains an older
compatible dependency stack.

RL jobs also add every local package beneath `deps/prime-envs/environments/`,
`deps/verifiers/environments/`, and `deps/research-environments/environments/`
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
rollouts, broadcasts, or an external checkpoint directory.

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

Multi-replica RL deployments use one global router on inference node 0. The
rendered launcher must expose one `INFER_URLS` value for the policy client and
one space-separated `ADMIN_URLS` entry per vLLM server. Admin traffic bypasses
the router for weight updates and health checks. Do not create one policy URL
per replica: `model.client.base_url` is a single router URL, and extra values
will be rejected by the orchestrator CLI.

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
  `PRIME_RL_HARBOR_CACHE_DIR`,
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

Run CPU-only unit tests against live source and the base SIF, without appending
the prepared metadata target. The layered `mamba-ssm` package is GPU-runtime
state: its upstream ROCm autotuner queries the device architecture during
import, so importing it on a login node with no visible GPU is unsupported.
Validate that layer with a GPU probe or training smoke.

Single-node launchers normalize the Slurm allocation to logical
`CUDA_VISIBLE_DEVICES` IDs and unset the node-wide `HIP_VISIBLE_DEVICES` and
`ROCR_VISIBLE_DEVICES` values before PRIME-RL partitions GPUs among child
processes. PyTorch ROCm supports this compatibility variable; keeping the
node-wide ROCm variables would defeat the per-child inference/trainer split.

TMax task records may refer to registry images that UCloud workers cannot pull.
For a bounded run, export the Harbor taskset, build each selected task's
`environment/Dockerfile` under a stable gateway image id, and map the task name
to that id with `taskset.image_overrides`:

```bash
uv run scripts/build_ucloud_harbor_image.py \
  --task-dir /path/to/export/tmax/task_000001_example \
  --image-id prime-rl-tmax-task-000001-example
```

Keep `taskset.tasks` aligned with the prepared image set. Do not assume the
registry reference in a Harbor `task.toml` is public or available on every
sandbox node. UCloud's direct runtime requires numeric OCI identities; the
Verifiers adapter maps `runtime.user = "root"` to UID `0` when constructing the
sandbox request. Use the current mini-SWE harness version; its sandbox program
pins the LiteLLM release used by Verifiers so Python 3.10 task images resolve a
compatible environment.

Probe a representative task image and its setup hook through the operational
SIF before adding a taskset to a UCloud training mix:

```bash
./scripts/lumi_validate_ucloud_taskset.sh r2e-gym-v1
./scripts/lumi_validate_ucloud_taskset.sh scaleswe-v1 \
  --num-tasks 3 --shuffle --max-concurrent 3
```

Run Harbor-backed taskset probes serially. Harbor 0.14 stores anonymous session
state in one user-level credentials file and its writes are not atomic, so
parallel dataset exports can observe a partially written JSON document. This
does not affect already-exported tasksets or ordinary non-Harbor environment
loads.

## Monitoring

Start the local run dashboard on a named user access node and keep it in a
detached `tmux` session. Bind it to loopback; use an SSH tunnel rather than
exposing the service on the shared login network:

```bash
hostname  # record the exact UAN, for example uan01
tmux new-session -d -s prime-rl-dashboard-7799 -c "$PWD" \
  './scripts/lumi_run_in_container.sh dashboard \
  /scratch/project_465002751/rasmus/artifacts/training \
  --host 127.0.0.1 --port 7799'
curl -fsS http://127.0.0.1:7799/api/runs >/dev/null
```

From a workstation, tunnel through LUMI's public login endpoint to that exact
UAN, then open `http://localhost:7799`:

```bash
ssh -N -J "$USER@lumi.csc.fi" \
  -L 7799:127.0.0.1:7799 "$USER@uan01"
```

The dashboard includes metrics, resolved configs, component logs, and rollout
traces. Inspect or stop the server with `tmux attach -t
prime-rl-dashboard-7799` or `tmux kill-session -t prime-rl-dashboard-7799` on
the recorded UAN.

```bash
squeue -u "$USER" -o "%.18i %.9P %.30j %.8T %.10M %.6D %R"
sacct -j <jobid> --format=JobID,JobName%30,Partition,State,ExitCode,Elapsed,NNodes,NodeList -P
```

For a run output directory:

- Slurm output: `<output_dir>/job_<jobid>.log`
- Trainer rank 0: `<output_dir>/logs/trainer/node_0.log`
- Orchestrator: `<output_dir>/logs/orchestrator.log`
- Inference rank 0: `<output_dir>/logs/inference/node_0.log`
- Per-node streams: `<output_dir>/logs/{trainer,inference}/node_<rank>.log`

Long quiet intervals are normal during the first compiled trainer step and
while vLLM reloads a 120B checkpoint. Do not classify them as CP or RCCL hangs
from log silence alone. Check that the rank processes are runnable and sample
the node's kernel GPU-busy counters from an overlapping, CPU-only step:

```bash
srun --jobid=<jobid> --overlap -w <trainer-node> -N1 -n1 \
  --cpus-per-task=1 --gpus=0 bash -lc \
  'ps -C PRIME-RL::Train -o pid,stat,pcpu,etime,wchan:24,comm; \
   for f in /sys/class/drm/card*/device/gpu_busy_percent; do \
     printf "%s=" "$f"; cat "$f"; done'
```

During a filesystem weight broadcast, the inference log reports checkpoint
prefetch progress and ends with a successful `POST /update_weights`. A real
trainer failure has a rank traceback; search every node log before resubmitting:

```bash
rg -n 'Fatal|ERROR|Traceback|InductorError|OutOfMemory' \
  <output_dir>/logs/trainer/node_*.log
```

Set `slurm.cleanup_grace_period` explicitly for scale jobs. It preserves a
short window for an in-flight checkpoint after a component failure; the
Nemotron smoke uses 600 seconds instead of the generic one-hour default so a
failed allocation does not look indefinitely hung.

The RL launcher also deletes run-labelled UCloud sandboxes on exit. Cleanup
uses concurrent, idempotent deletion so cancellation can remove hundreds of
sandboxes within Slurm's termination window; HTTP 404 means the sandbox is
already gone, while other deletion errors are reported.

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
- Harbor tasksets use a job-scoped node-local cache under `/tmp` by default.
  Set `PRIME_RL_HARBOR_CACHE_DIR` only when that data must persist; do not
  redirect a large task export to project scratch or the login-node home.
