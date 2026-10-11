---
name: start-run
description: How to launch prime-rl training runs — the `rl`, `sft`, `inference`, and `evals` entrypoints, their config classes, and single-node/SLURM/dry-run modes. Use when starting a run or picking the right entrypoint.
---

# Start a run

All entrypoints run via `uv run <command>` and accept TOML configs via `@ path/to.toml` plus CLI overrides.

SLURM launches write generated scripts and coordination files under `<run_dir>/launcher/`, with batch logs under `launcher/logs/`. Local launches do not create this directory. Every launch writes configs and `command.txt` under `configs/attempt_<n>/`. `configs/latest` points to the current attempt. The command uses shell-safe quoting.

## Run directories

`output_dir` (default `outputs`) groups related runs; each run writes all its artifacts (logs, configs, checkpoints, broadcasts, rollouts) to its own run directory `<output_dir>/<run_name>`. `run.name` auto-generates as `<envs>--<model>--<short-id>` (SFT: `<dataset>--<model>--<short-id>`), so every launch gets a fresh, readable run directory; `run.dir` overrides the directory leaf when it should differ from the name. Pass `--run.name <name>` to make the run directory predictable — required to resume the run later (`--resume`, or `--resume.step N`, reuses the named run directory; without `[ckpt]` it loads but saves no new checkpoints). Launching into a run directory that already contains artifacts fails unless resuming or `--clean` is set (which wipes only that run directory).

## Config system at a glance

[`pydantic-config`](https://github.com/PrimeIntellect-ai/pydantic-config) — Pydantic-based TOML + CLI loader. Highlights (see the `configs` skill for full mechanics):

- Config files via `@ path` (TOML / YAML / JSON); CLI args layer on top, deep-merged with class defaults.
- Nested groups via dotted CLI paths — kebab-case on the CLI, snake_case in TOML.
- Bool toggles: bare `--flag` enables, `--no-flag` disables (nested too).
- Lists: space-separated or JSON literal. Dicts: JSON literal, deep-merged with file values.
- Optional sub-configs (`WandbMonitorConfig | None`): bare `--monitors.wandb` enables defaults; `--monitors.wandb @ wandb.toml` enables from a file; `--no-monitors.wandb` disables.
- Discriminated unions are switched by the `type` tag (e.g. `--optimizer.type muon`).
- Validation aliases let renamed fields keep working; legacy keys can be remapped in a `model_validator(mode="before")`.
- Auto-generated `--help` panels from `Field(description=...)` or PEP 224 docstrings.
- Friendly errors: required-field boxes, validator errors point at the offending flag, unknown flags get a "did you mean" hint.
- State-only optimizer offload remains enabled by default with `model.optim_cpu_offload = true`.
- For gradients, FP32 masters, optimizer state, and optimizer-in-backward CPU execution, set
  `model.optim_cpu_offload = false` and `model.full_offload = true`. This mode uses the native
  CPU optimizer kernel, only supports AdamW and SignSGD (SignSGD is stateless and
  halves the host RAM footprint), and disables gradient clipping. Use a
  `[model.full_offload]` table only to select the Torch debugging backend or disable NUMA binding.

## `rl` — RL training

Launches inference server, orchestrator, and trainer as subprocesses.

```bash
uv run rl @ examples/basic/reverse-text/rl.toml
uv run rl @ examples/basic/reverse-text/rl.toml --dry-run                                # write scripts, don't run
```

- Config: `RLConfig` (`packages/prime-rl-configs/src/prime_rl/configs/rl.py`)
- Entrypoint: `src/prime_rl/entrypoints/rl.py`
- SLURM: single- and multi-node
- Multi-node SLURM stops after `.trainer.done` for trainer-only fake-data runs. Runs with inference stop after both `.trainer.done` and `.orchestrator.done`.
- NIXL on SLURM: install NIXL and ModelExpress with the provided scripts. The job starts ModelExpress and Redis unless `slurm.launch_modelexpress = false`.
- Environment packages: before launching a config with a non-core verifier env id,
  verify the package imports under `uv run` (for example
  `uv run python -c "import importlib.util; print(importlib.util.find_spec('r2e_gym'))"`).
  If a local env exists under `deps/prime-envs/environments/` or
  `deps/verifiers/environments/` but does not import, install the env workspace
  members with `uv sync --all-extras --all-packages` (all) or `uv sync --all-extras
  --package prime-rl --package <env>` (one) — they're auto-discovered, no
  `pyproject.toml` edit needed. Keep `--all-extras` for training so a targeted
  package sync does not prune accelerator dependencies from the environment.

## `sft` — SFT training

Launches torchrun internally — never call torchrun directly.

```bash
uv run sft @ examples/basic/reverse-text/sft.toml
uv run sft @ examples/basic/reverse-text/sft.toml --slurm
uv run sft @ examples/basic/reverse-text/sft.toml --dry-run
```

- Config: `SFTConfig` (`packages/prime-rl-configs/src/prime_rl/configs/sft.py`)
- Entrypoint: `src/prime_rl/entrypoints/sft.py`
- SLURM: single- and multi-node
- Multi-node online evals use one SLURM job with `num_train_nodes + num_infer_nodes` nodes. The generated `launcher/sft.sbatch` assigns inference nodes first, then trainer nodes.

## `inference` — vLLM server

OpenAI-compatible API plus prime-rl custom endpoints (`/update_weights`, `/load_lora_adapter`, `/init_broadcaster`). Always use this entrypoint — never `vllm serve` directly. It starts a `vllm-router` on `server.port` (default 8000, the client-facing URL) fronting the engine on `backend_port` (default 8100); admin endpoints must target the engine port directly.

```bash
uv run inference --vllm.model Qwen/Qwen3-0.6B
uv run inference --vllm.model Qwen/Qwen3-0.6B --vllm.enforce-eager
```

Smoke checks:

```bash
curl http://<host>:<port>/health
curl http://<host>:<port>/v1/models
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3-0.6B", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 50}'
```

On LUMI, launch through `scripts/lumi_submit_rl.sh` or
`scripts/lumi_submit_sft.sh`. They use the immutable dependency SIF without
`--rocm` or an overlay, while the checkout remains authoritative on
`PYTHONPATH`. `scripts/lumi_prepare_runtime.sh` supplies the distribution
metadata required for vLLM plugin discovery. Set `CONFIG` to select an
experiment and keep its `output_dir` beneath
`/scratch/project_465002751/rasmus/artifacts/training`.

Nemotron 3 Super runs on the current LUMI image use Triton 3.7's compiler fix
for the ROCm SSD chunk-scan kernel. Do not set
`PRIME_RL_PATCH_VLLM_SSD_CHUNK_SCAN` or enable the old branch-local kernel
workaround. When the operational image changes, verify its Triton version and
run a normal TP=8 Nemotron inference smoke before a scale run.

For multi-node LUMI RL, a clean co-located orchestrator exit can precede the
trainer's final optimizer step. Keep the launcher supervision that waits for
the trainer in this case; do not treat orchestrator completion as a reason to
terminate the remaining trainer process. Trainer rank 0 must then publish the
job-scoped completion sentinel so separate inference nodes tear down cleanly.

- Config: `InferenceConfig` (`packages/prime-rl-configs/src/prime_rl/configs/inference.py`)
- Entrypoint: `src/prime_rl/entrypoints/inference.py`
- SLURM: single-node, multi-node, and disaggregated deployments

## `evals` — multi-env evals

Runs the configured eval sources against a live inference server. Standalone (no `[online]` block): one epoch of every source against the served weights, then exit. Add `[ckpt]` (`interval` counts completed task groups in the eval-source order) to make the run interruptible, then use `--resume`, `--resume.step N`, or `--resume.dir path/to/checkpoints/step_N`; for standalone evals, checkpoint step N means resume from task cursor N. Checkpoints store only the cursor, not generated episode records; partially completed groups and groups beyond the durable cursor are retried. Resume loads without `[ckpt]` but does not save new checkpoints. Checkpoint/resume is rejected with `[online]` because that process is coupled to the trainer's live broadcast handshake. With `[online]` (`broadcasts_dir`, `max_steps`, `resume_step`): watch the broadcasts dir for stable `step_{n}` weight broadcasts and evaluate each — the `sft` launcher writes this config for online evals. By default a newer checkpoint cancels unfinished episodes from the prior eval. Set `eval.cancel_on_new_checkpoint = false` to drain every epoch. The trainer can idle while it waits for slow evals. Launcher-managed SFT evals use NCCL weight broadcast by default, including multi-node SLURM deployments. LoRA and external inference use filesystem broadcast.

```bash
uv run inference --vllm.model Qwen/Qwen3-4B   # start inference separately
uv run evals @ eval.toml
```

Minimal standalone `eval.toml`:

```toml
model = "Qwen/Qwen3-4B"

[eval.client]
base_url = "http://localhost:8000/v1"

[eval.concurrency]  # adaptive; same controller as [orchestrator.concurrency]
min_inflight = 8
max_inflight = 128

[[eval.source]]
num_examples = 32   # always cap eval size for smokes
group_size = 4
env.taskset.id = "aime25"
env.agent.harness.id = "null"
env.agent.runtime.type = "subprocess"

[ckpt]               # optional: make a standalone eval resumable
interval = 10        # completed task groups between cursor saves
```

- Env servers: spawned by the evals process, one per source without an explicit `serve.address`, at `tcp://127.0.0.1:<eval.env_server_base_port + index>`; logs at `{output_dir}/logs/latest/envs/eval/{name}.log`.
- External inference APIs (no vLLM `/metrics`, e.g. Prime Inference) have no load signal for adaptive concurrency: the startup `/metrics` probe fails fast unless the band is pinned (`min_inflight = max_inflight`). Full example: `configs/evals/swe.toml` (SWE-bench Verified + Terminal-Bench 2 on Prime Inference, `agent.timeout.rollout = 3600`).
- Config: `EvalsConfig` (`packages/prime-rl-configs/src/prime_rl/configs/evals.py`)
- Entrypoint: `src/prime_rl/entrypoints/evals.py` (implementation: `src/prime_rl/evals/evals.py`)

## Exporting checkpoints

Trainer checkpoints are DCP-sharded (`<run_dir>/checkpoints/step_{n}/trainer`). Convert to HF safetensors with `uv run python tools/convert_dcp_to_bf16.py <run_dir>/checkpoints/step_{n}` (writes `<ckpt_dir>/weights`, serveable via `uv run inference --vllm.model <dir>`; model config auto-read from the run’s `configs/latest/resolved/trainer.json`/`sft.json`; multi-rank via `torchrun --nproc-per-node N`; full fine-tunes only, LoRA rejected). Quantize a bf16 HF dir to blockwise FP8 with `tools/convert_bf16_to_fp8.py <dir>` (vLLM-native format), or straight from a checkpoint with `tools/convert_dcp_to_fp8.py <ckpt_dir>` (rank-parallel, writes only `<ckpt_dir>/weights-FP8`, no bf16 on disk); dequantize fp8-only releases with `tools/convert_fp8_to_bf16.py <dir>`. Caveat: on SM120 GPUs (RTX PRO 6000) vLLM 0.26 picks `CutlassFp8BlockScaledMMKernel` for blockwise-fp8 checkpoints and it silently degrades outputs — serve with `VLLM_DISABLED_KERNELS=CutlassFp8BlockScaledMMKernel,MarlinFP8ScaledMMLinearKernel` to fall back to the Triton kernel.

## Summary

| Command | Purpose | Typical use |
|---------|---------|-------------|
| `rl` | Full RL pipeline | Production RL training |
| `sft` | Supervised fine-tuning | SFT and hard-distill |
| `inference` | vLLM server | Standalone serving / debugging |
| `evals` | Multi-env evals | Standalone evals / SFT online evals |

## Key paths

- `src/prime_rl/entrypoints/` — `rl`, `sft`, `inference` (+ `trainer`, `orchestrator` for direct launches)
- `packages/prime-rl-configs/src/prime_rl/configs/` — all config classes
- `configs/debug/` — minimal debug configs
- `examples/` — full example configs (e.g. `reverse-text/`)

## Dashboard

Interactive launches auto-start one shared dashboard daemon per user (process title
`PRL::Dashboard`) and end startup with a `Dashboard · <url>` banner. Relay
that URL to the researcher. Discovery: `~/.cache/prime-rl/dashboard/daemon.json` holds
the live `url` (the port can differ from 7788 when it was taken). `--no-dashboard`
opts a run out.

## LUMI Super RL

Follow `docs/lumi-rl-super.md` for the shared SIF and external UCloud provider.
Prepare and pin runtime metadata before submitting GPUs; do not change active
job source. Use the native LUMI dry-run and inspect attempt-specific JSON configs.

For the Super ROCm stack, test router capture with
`scripts/lumi_super_router_probe.py`, then compare saved samples using
`scripts/lumi_super_replay_parity.py` under the target CP/EP trainer topology.
The latter performs frozen-weight forwards with replay off/on and a replay
backward check without optimizer updates. Do not infer replay correctness from
successful config validation. Test NCCL/RCCL startup and changed-weight transfer
in a separate bounded two-update run before scheduling a longer pilot.

LUMI weight transfer uses the dedicated `PRIME_RL_WEIGHT_NCCL_ALGO`,
`PRIME_RL_WEIGHT_NCCL_PROTO`, and `PRIME_RL_WEIGHT_HIGH_PRIORITY` settings from
`docs/lumi-rl-super.md`. Keep overrides scoped to the weight communicator;
forcing Ring/LL across the whole trainer changes its collective performance.
The complete OFI host-staging, DMA-BUF, high-priority, and hardware-queue bundle
still applies. Use Super Mamba prefix caching in `align` mode with FP32 SSM
states. Validate cache hits and cold/warm parity serially at7K/32K/130K using
`scripts/lumi_super_prefix_verify.py`; do not run a competing stress client.

When disabling Super prefix caching, also omit `vllm.mamba_block_size`.
The optimized vLLM stack rejects a prefix-cache block size with caching off,
before loading the model. Keep `mamba_ssm_cache_dtype=float32`.

Super NCCL reload needs GPU headroom beyond the inference cache budget.
Use an explicit `vllm.kv_cache_memory_bytes` budget validated by
`scripts/lumi_super_capacity.sbatch`, including occupied-cache reloads. A
utilization fraction alone can leave almost no usable cache after profiling.
Set `PRIME_RL_WEIGHT_BUCKET_BYTES=268435456` to bound transient receive buffers;
deploy the matching versioned-wire receiver first. Whole tensors are not split.
The private nonlocal-expert reload overlay is built with
`scripts/lumi_prepare_super_reload_overlay.py`; select it through
`PRIME_RL_SUPER_RELOAD_OVERLAY` before sourcing the Super bootstrap. NCCL update RPCs are single-attempt, including timeouts: a
partly consumed receive stream cannot safely be replayed. On failure leave the
engines paused and restart the failed run; never serve partially loaded weights.

NCCL receive tensors must own storage for only that weight. Deferred vLLM
layer processing can retain small gate weights until finalization; returning
views into a flat dtype bucket pins a whole multi-gigabyte MoE layer per gate.
`receive_state_dict` clones slices of multi-weight buckets before yielding them.
The wire probe retains small BF16 weights across repeated transfers and checks
exact storage size and values; headroom alone does not fix bucket retention.

For multiple inference replicas, combine llm-d KV-utilization, queue and active-
request scorers with prefix affinity. Validate both HTTP routes with
`scripts/lumi_kv_router_probe.py --prefix --output <artifact-directory>`.
Prefix affinity is approximate and must yield to heavy KV occupancy.

For LUMI Super NCCL startup, enable `PRIME_RL_PAUSE_NCCL_INIT=1` alongside
`HSA_ENABLE_SDMA=0`. Engines pause before communicator creation and resume only
on success. Keep the bounded weight buckets and private reload overlay when
using the validated 14 GiB/device cache profile; validate multi-replica transfer
and trainer 128K memory before submitting the full pilot.

For Nemotron Super parallelism preflight, validate `ParallelDims` on CPU and
inspect the model's Mamba group count. EP must be divisible by CP, and this
Mamba implementation requires its state groups to be divisible by CP. Super
has eight groups: CP16/EP16 passes the generic mesh check but is not supported
by the current model implementation. Keep CP8 when using the existing kernels.

For the SFT-matched Super 128K RL pilot, use the 16t8i config and retain CP8/EP8.
Eight training nodes double persistent shard sizes relative to sixteen; batch
one is already in use. Select the consolidated `PRIME_RL_SUPER_PROFILE` before
sourcing the inference bootstrap; its manifest preserves the validated kernels,
helper modules, and reload fix. Keep the eight inference replicas and validate
host-cache restoration plus weight reload before releasing dependent training.

Before a LUMI trainer-only diagnostic, check the rendered script includes the
same OFI bundle as SFT/pilot: DMA-BUF disabled, high-priority RCCL streams and
two GPU hardware queues. These are template defaults, not just pilot config
settings. A topology comparison without them is not a valid memory test.

Scope Super's SDMA-off workaround to inference. For the SFT-matched trainer,
use SDMA=1 and omit forced RCCL channel counts; validate at least three updates
with the full OFI/high-priority/queue bundle. A Python stack snapshot inside
GEMM or a collective is not proof of deadlock: compare successive snapshots,
GPU activity and completed updates. `py-spy` can collect Python stacks from the
allocated node without modifying the runtime image.

For trainer-to-inference RCCL, use identical channel policy from process startup.
The RL inference bootstrap must clear the evaluation profile's fixed16-channel
settings; both roles use automatic selection. A late context-manager env change
can be ineffective because RCCL caches parameters. Validate heterogeneous sender
and receiver roles; sourcing the inference bootstrap on both probe nodes masks
this mismatch. Keep the CPU-store channel-settings agreement check enabled.

For agentic capacity changes, check both orchestrator concurrency and per-source
`serve.pool.multiplex`. Measure active decoding separately from episode count.
Use full publisher task pools with prepared images or recipe-backed image caching.
Do not require individual oracle validation for task admission; representative
environment checks are sufficient. TMax includes its own hidden verifiers. Keep
image build failures separate from rewards. See docs/lumi-rl-expanded-sources.md.
The sandbox
harness Python must use isolated imports to avoid task checkout packages
shadowing its bundled dependencies.

For full-pool LUMI agentic RL, startup inference synchronization must precede
materializing tasksets; otherwise trainer ranks can expire their collective
watchdog while the orchestrator is still loading data. Use bounded batch decoding
and sequential off-loop source loading. Verify full task counts without restoring
small image-validation allowlists. See `docs/lumi-rl-expanded-sources.md`.

For Terminal-Lego, use a fresh recipe index after verifier/preparation changes.
Validate candidate execution and reward exit-status behavior, then syntax-check all
rendered verifiers. Restore only declared empty workspace directories; never fabricate
missing task input assets. Keep unknown runtime logic out of dependency bootstrap.
Do not call an offline syntax/context check a live sandbox validation. Inspect UCloud
capacity/DNS failures separately from task packaging and verifier failures.

For the UCloud agentic pilot, validate parking via the managed agent and real
harness probes documented in `docs/lumi-rl-expanded-sources.md` before submission.
`parkable=true` alone is insufficient: the adapter must use `start_agent()` and
bind relay registration to the gateway-confirmed sandbox handle/generation.
Keep explicit parking in both train/eval runtime configs and capacity preparation.

When diagnosing parked-rollout failures, distinguish generation latency from
response-commit/wake latency. Verify bounded idempotent response retries and
concurrent park/resume before restarting. Stale image-build receipts require
backend validation; see the recovery section in docs/lumi-rl-expanded-sources.md.

Live image-cache validation must exercise an existing receipt using the real SDK
client. Tiny parking probes do not exercise long-running lease-renewal races;
cover the completed-request renewal race through the SDK worker lifecycle.

For UCloud gateway runs, follow "UCloud gateway runtime" in `docs/lumi-rl-super.md`:
export the gateway's allowlists, load every taskset with the final config, regenerate
Harbor task caches after allowlist or revision changes, and enable
`[orchestrator.image_prefetch]` so upcoming task images build ahead of sampling.

Start pilot observation with `lumi_watch_rl_pilot.py --stop-on-infra-failure`
and explicit job/run paths. Use `--max-failed-episodes 15 --poll-seconds 15`
for these runs. Confirmed failed images are quarantined across training
harnesses and excluded from the global failure window; backend failures still
count. Enable the progress guard (default 600 seconds after useful work starts,
3600 seconds for initial startup). See `docs/lumi-rl-expanded-sources.md` for
resume filtering and exact stop semantics. Zero reward is never a stop signal.

## Score-centering RL loss

Use `[trainer.loss] type = "score_centering"` with `top_k = 32` to opt into
standalone SC (`importance_sampling = "none"`); `"tis"` composes truncated IS.
The combined RL config enables generation-time sampler head capture and the
inference logprob capacity. Follow `docs/score-centering.md` for separate process
configs, supported full-vocabulary sampling, transport requirements, and checks.
SC requires additional sampler evidence; existing sampled-token-only traces are
not sufficient. Keep the default loss unless SC is explicitly selected.

When mixing harnesses over the same deterministic finite tasksets, enable
`share_taskset = true` on the train sources. The orchestrator caches by resolved
taskset class and full config within the collection; source cursors and curriculum
state remain independent. Keep sharing disabled for stochastic loaders. Infinite
streams are never shared. Confirm `Reusing finite task pool` in startup logs and
check host memory alongside inference CPU cache; duplicating large coding datasets
per harness can consume tens of GiB and repeat minutes of Arrow decoding.

When node loss is explicitly tolerated, use native whole-episode retries restricted
to `SandboxNodeLost` and bounded attempts. Monitor with `--tolerate-node-lost` so
those exhausted attempts are counted but excluded from the automatic-stop window.
Other errors and mixed failures retain the stop rule. Never resume a lost sandbox.

For mixed-duration agent pilots, use the optional taskset-family lookahead and eval
concurrency budget described in `docs/lumi-rl-harnesses.md`. Validate streaming
harness requests through TrainClient with exact token/logprob/top-k capture before
launch; EvalClient-only harness acceptance does not prove train compatibility.

Offline coding-harness bundles must include executable rg/fd, with their directory
on the agent PATH. Before a pilot, run the real tool-chain probe described in
`docs/lumi-rl-harnesses.md`; a single shell echo does not validate glob/grep or edit.

### LUMI sandbox SDK updates

Pin the released wheel in `scripts/lumi_prepare_runtime.sh` and the local
verifiers-ucloud package dependency, and refresh that package's lockfile. Rebuild
the immutable metadata layer and check the actual SDK import from the container
used by the job; standalone probe/cleanup helpers must resolve that layer too.
Run adapter compatibility tests and the bench image check before the native
launcher dry-run. Keep existing metadata layers available for rollback.

For ECHO-enabled runs, source `algo` overrides replace global settings: copy any
required behavior shaping explicitly. Verify that the advantage gate admits
nonzero auxiliary CE supervision, that rewritten assistant histories retain
observation supervision without resampling action tokens, and that initial tool
demonstrations remain masked. On restricted-network harnesses, synthetic policy
notices must occupy a stable prefix position rather than moving with each turn.

For shared task-difficulty replay across harnesses, use `difficulty_pool` with
`strategy = "smoothed_replay"` and the same explicit `shared_group` for the same
taskset. Keep different source seeds; outcomes share by task identity while
sampling and gates remain separate. Verify history/RNG checkpoint restore and
image quarantine after sampler changes. See `docs/throughput-scheduling.md`.
