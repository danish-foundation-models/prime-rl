# Nemotron Super RL on LUMI

Use `configs/lumi/rl_nemotron_super_r5_pilot.toml` through
`scripts/lumi_submit_rl.sh --dry-run`. The LUMI template retains main's
attempt-specific JSON configs, global router, environment servers, and successful
trainer/orchestrator completion gates. The shared PyTorch 2.13 / ROCm 7.2 / vLLM 0.28
SIF (`lumi-posttrain-u24r72-t213-vllm028.sif`) remains immutable. Live source, the
prepared package layers and the gfx90a vLLM overlay supply the rest. Do not run the
CUDA GPU extras installer on this bench.

## Inference stack (r72)

Super inference runs the SIF's vLLM 0.28 with the gfx90a source overlay from
`rlrs/vllm` branch `perf/rocm-gfx90a-nemotron-v028-super`. It carries the kernels the
r71 runs took from the frozen Super profile and the `vllm-569869cdc` layer: AITER
paged attention and custom all-reduce on gfx90a, shuffled KV cache without
materialization, fused large ReLU-squared MoE with local-row expert assignment and a
local-expert sum, the single-group top-k fast path, the BF16-to-FP32 router GEMM, FA2
varlen prefill, MI250X MoE/SSU tuning, the expert-parallel layerwise-reload fixes, and
private per-rank pinned buffers for host KV offload. vLLM 0.28 otherwise has every rank
`cudaHostRegister` the whole shared offload region; with 192 GiB on a node, ranks 2-7
fail and die at their next kernel launch with `hipErrorInvalidValue`.

Validated on one TP8/EP8 replica with the stock HF model against stock vLLM 0.28: 93K-token
teacher-forced NLL 1.8047 vs 1.8086 (mean |Δ| 0.085 nats/token, the run-to-run floor),
GSM8K 99% vs 98%, and 9,933 vs 6,260 tok/s at 128K, concurrency 8, prefix caching off.

1. Materialize the overlay once per revision on the login node:
   `scripts/lumi_prepare_vllm_overlay.sh <rev>`. It exports the files the revision
   changes against the SIF's vLLM commit into `.runtime/vllm-overlays/<rev>` and
   refuses deletions and non-Python files.
2. Set in `[inference.env_vars]`:
   `PRIME_RL_INFERENCE_PROFILE = "nemotron-super-gfx90a"` and
   `PRIME_RL_VLLM_OVERLAY = "<printed directory>"`. The template exports the overlay
   before `srun`; `lumi_job_container.sh` bind-mounts its files read-only over the SIF's
   vLLM (directories for comma-named tuning files). `lumi_super_inference_env.sh`
   sets the profile's AITER switches and per-image vLLM/Triton/Inductor caches.
3. Set `inference.vllm.worker_cls =
   "prime_rl.inference.vllm.worker.nemotron_super.VerifiedSuperWorker"`; it audits
   representative checkpoint tensors on every rank after the initial load.

`PRIME_RL_SUPER_PROFILE` and `PRIME_RL_SUPER_RELOAD_OVERLAY` belong to the retired r71
image; the inference env refuses them. The sections below that mention the frozen
profile, `dfm_evals` helpers or the reload overlay describe how r1-r21 ran.
`configs/lumi/rl_nemotron_super_r5_128k_16t4i_sc_r72_smoke_r1.toml` is r21 on this
stack at smoke scale.

Prepare runtime metadata on the submit host and pin its returned path before
submitting GPU jobs. Never edit launcher scripts or imported source while a job
using them is active. Use a new versioned checkout for subsequent changes.
LUMI jobs must not run broad process-name cleanup or wildcard filesystem removal.
Slurm owns process teardown. Per-job temporary state stays under bench `tmp`.

The four training tasksets are `scaleswe`, `r2e-gym`, `swelego`, and `multiswe`.
The external `verifiers-ucloud` provider supplies runtime/interception type
`ucloud` with SDK 0.4.18. Provider runtime labels are a dictionary. Preserve
runtime-specific fields through config serialization and env-server transport.
The bash harness forwards `tool_timeout` to local shell commands as well as MCP
calls; a timed-out shell command's process group is terminated before returning
a tool error to the agent. The generated standalone script must import `os` and
`signal` itself. Verify a real timed-out child in both `run_bash` and the bundled
`CHAT_PROGRAM_SOURCE`; a `--help` check does not exercise timeout cleanup. Refresh
the offline bundle script SHA256 allowlist whenever bundled source changes.
Infrastructure errors remain invalid episodes.

Use a private conversion directory for checkpoint 1250. MTP parameters are
excluded from policy optimization. Start with optimized non-speculative serving;
MTP rollout inference requires a separate target-logprob and weight-reload check.
The pilot explicitly selects GRPO advantages and main's IPO loss, with no length
penalty, FP32 optimization/reduction defaults, and fresh AdamW at 1e-6.

The pilot config is provisional until GPU compatibility, optimized serving,
rollout/action-logprob integrity, and memory checks pass. Successful config
rendering alone does not authorize claiming an RL update worked.

The training renderer clamps each completion to the engine context remaining
after rendering the complete tool history. A full rendered context ends the
rollout with `max_context_tokens`; it remains scoreable instead of being marked
as a provider outage. Explicit prompt-only caps remain separate.

The LUMI submission wrapper validates existing run artifacts during rendering.
Dry runs never delete run directories; use a unique run name for a fresh run.

On ROCm, distributed setup pins the local GPU but omits eager `device_id`
binding when initializing the default group. This avoids RCCL communicator
splitting deadlocks with overlapping CP/EP meshes. Verify subgroup collectives
across nodes before the full-model benchmark. OFI remains enabled.

Optimized inference must expose both the frozen profile root (sitecustomize)
and its `source` directory (the frozen `dfm_evals` kernel namespace). Imports
alone do not exercise that namespace; the full model's MoE forward does.

Freeze and expose `nemotron_moe`, `nemotron_prefill`, and `nemotron_worker`
helpers with the optimized profile. Run the EP workspace poison checks before
loading checkpoint weights; initial tensor audits use `VerifiedSuperWorker`.

Preconvert the full Super checkpoint as a separate CPU workload with 16 CPU
threads using `uv run python scripts/lumi_convert_nemotron.py SOURCE CACHE/prime`.
Dry-run that command with `--dry-run` first. The converter uses native operations,
checks representative first/last experts, and publishes a unique staged directory
by rename after writing `.prime-v1`. It refuses incomplete existing caches and
never deletes them. Set trainer.model.conversion_dir to CACHE and depend on the
conversion job before allocating trainers. This avoids 63 ranks idling behind
master-rank conversion with training's one-thread CPU setting.

This account currently has MaxSubmitJobs=0 on CPU partitions. Use a short
`dev-g` allocation requesting one GPU for preconversion until CPU submission
is available; no model GPU computation is performed during conversion. The native model
registry does initialize GPU-dependent imports, so execution needs that visible
GPU; the dry run validates only source metadata and shard presence on CPU.

Conversion reads complete shards and clones only one layer's tensors into
independent CPU storage. It applies the native conversion chain, checks the full
first/last expert tensors for every MoE layer, writes that layer, and releases it
before loading the next layer. Do not keep the complete source and converted
model in memory. An index and marker are published only after all layers pass.

Load shared credentials from both bench `post/.env` and `prime-rl/.env`, then
checkout-local `.env` overrides. The submission wrapper scans resolved JSON
configs and checks all four UCloud endpoint/token variables by presence before
GPU allocation; it never prints their values.

Native SWE tasksets use framework-only networking. SDK 0.4.18 / gateway 0.5.33+
enforce the named relay-only policy from sandbox creation. Configure both the
runtime and interception `guest_relay_url` as
`http://gateway-live-ucloud-20260824a:8092`; external workers keep the public
`https://app-sandboxes-relay-v2.cloud.sdu.dk` endpoint. Preserve the authenticated
SDK tunnel path so both model calls and task-state requests reach interception.
Stage the pinned portable Python bundle through `offline_python_bundle`; the
restricted guest cannot download Python, uv or harness dependencies. Check
its sidecar archive/script hashes before use. Unknown scripts fail explicitly.
Native eval can exit zero while every episode is invalid; inspect `ok` and
`errors` in traces, not just its exit status.

The initial 32K pilot uses CP4/EP8 on eight trainer nodes. Keep FP32 optimizer
and reduction defaults. Full-length synthetic updates validate memory and
compute throughput; they do not establish real rollout validity, policy parity,
reward quality or learning. Do not report the fallback A100 MFU as MI250X MFU.

RL trainer Triton caches use `DurableFileCacheManager`: write to a private file,
flush/fsync it, then atomically rename it into the shared cache. This retains
cross-rank reuse while preventing readers from observing unflushed metadata.
A missing child file, malformed manifest, or malformed child JSON makes the
entire group a cache miss; never return a partial group without its GPU binary. Use the durable-v1 namespace;
leave previous cache namespaces intact. Interrupted private files are retained,
with no automatic sweep. The optimized inference profile owns its separate cache.

Before a new weight-transport or router-replay pilot, run a bounded integration
check. `scripts/lumi_super_router_probe.py --model CHECKPOINT --output DIRECTORY`
checks routed-expert capture on short, multi-turn, and long prompts, including a
repeated prefix-cache request. Serve the optimized model with
`vllm.enable_return_routed_experts=true`. Require distinct in-range choices at
every MoE layer, full token alignment, and stable repeated routing; a non-null
array alone is insufficient. Capture validation does not replace a trainer
forward/backward replay check.

`weight_broadcast.type = "nccl"` uses RCCL on ROCm and transfers layerwise model
weights through the configured OFI network. It still uses small filesystem
handshake markers; durable training checkpoints also remain on disk. Test both
the startup broadcast and an updated policy broadcast before a longer pilot.
Keep the optimized inference profile and FP32 optimizer/reduction defaults.

`scripts/lumi_super_replay_parity.py --config RESOLVED_TRAINER_JSON --samples
CAPTURE_DIRECTORY --output DIRECTORY` runs under torchrun with the intended
trainer topology. It compares completion logprobs against those exact captured
samples with replay disabled/enabled, and checks finite replay gradients without
an optimizer update. Inspect the numerical report before enabling replay in RL.

The validated SFT OFI settings are a complete bundle: the absolute OFI plugin
path, `NCCL_NET=Libfabric`, `NCCL_NET_GDR_LEVEL=0`,
`NCCL_DMABUF_ENABLE=0`, `TORCH_NCCL_HIGH_PRIORITY=1`, and
`GPU_MAX_HW_QUEUES=2`. Retain the CXI/HMEM settings. Do not carry over host
staging alone. PyTorch's high-priority setting does not select the stream for
vLLM's separate `PyNcclCommunicator`; its initialization and weight collectives
need an explicitly selected high-priority stream as well. Exercise the native
wire helpers with `scripts/lumi_super_nccl_probe.py` before full-model transfers.

Use `vllm.enable_prefix_caching=true` with `vllm.mamba_cache_mode="align"`
for Super. Keep FP32 SSM states and the optimized inference kernels. Validate
real cache hits and cold/warm parity serially at multiple context lengths when
changing the runtime; do not substitute the unvalidated `all` cache mode. Greedy token identity alone is a sensitive check,
not proof of a broken cache: inspect token logprobs and numerical differences
before attributing the failure. Do not enable router replay solely from one
short prompt; compare multiple lengths and check its backward path.

Use `PRIME_RL_WEIGHT_NCCL_ALGO=Ring`, `PRIME_RL_WEIGHT_NCCL_PROTO=LL`,
and `PRIME_RL_WEIGHT_HIGH_PRIORITY=1` for the LUMI asymmetric weight group.
`WeightBroadcastCommunicator` applies algorithm/protocol overrides only during
its construction and restores the process environment. Its dedicated stream
waits for the caller before reading tensors and makes the caller wait before
consuming received tensors; allocation lifetime is recorded on that stream.
Do not apply these algorithm/protocol overrides to trainer collectives.
Re-test the native wrapper, including default-stream allocation and repeated
payload verification, when changing this transport. The RCCL 7.1.1 tuning table
also has a known sub-64-byte indexing defect fixed upstream in commit
`dea50b5e1121ca638f495bd16c8d69243997fbaf`; do not assume generic Simple/Ring
forcing cures tiny-message initialization failures.

Submit RL using `CONFIG=/absolute/path/to/config.toml
./scripts/lumi_submit_rl.sh` after its dry-run. The wrapper exports the prepared
`PRIME_RL_METADATA_DIR` into the Slurm job. A rendered `rl.sbatch` is not
self-contained: submitting it from a fresh shell without that variable exits
before starting any model process. Do not bypass the wrapper after inspection.

When disabling Super prefix caching, also omit `vllm.mamba_block_size`.
The optimized vLLM stack rejects a prefix-cache block size with caching off,
before loading the model. Keep `mamba_ssm_cache_dtype=float32`.

Super NCCL reload needs GPU headroom beyond the inference cache budget.
Reserve space for a full unsharded MoE layer (about 5.35 GiB), overlap with the
previous receive buffer, and layer processing. Use an explicit `vllm.kv_cache_memory_bytes` budget validated by the
capacity probe; a utilization fraction alone can leave almost no cache after
model profiling. Verify full startup and updated-policy loads before raising it. NCCL update RPCs retry connection-establishment failures up to three attempts.
Read/write timeouts and HTTP error responses are never retried: a
partly consumed receive stream cannot safely be replayed. On failure leave the
engines paused and restart the failed run; never serve partially loaded weights.

NCCL receive tensors must own storage for only that weight. Deferred vLLM
layer processing can retain small gate weights until finalization; returning
views into a flat dtype bucket pins a whole multi-gigabyte MoE layer per gate.
`receive_state_dict` clones slices of multi-weight buckets before yielding them.
The wire probe retains small BF16 weights across repeated transfers and checks
exact storage size and values; headroom alone does not fix bucket retention.


### Pilot failure checks

NCCL broadcaster initialization must propagate every non-404 HTTP error. Do
not continue into startup synchronization after a failed initialization.
For persistent RL inference, set `PRIME_RL_FAIL_ON_INFERENCE_EXIT=1` in
`[env_vars]`: the rank wrapper then treats even a zero-status server exit as a
failure, allowing Slurm to tear down a stalled allocation. Normal run completion
is handled by the shared completion monitor.

An `Unexpected exception logging compilation metrics` error for a function-valued
Dynamo config is caught logging code; it also occurs in successful runs. Check
subsequent server readiness and worker exits before attributing a failure to it.
For native worker deaths, preserve the core dump and inspect it with libraries
from the exact SIF. Enable `NCCL_DEBUG=INFO`, `NCCL_DEBUG_SUBSYS=INIT,NET`, and
`PYTHONFAULTHANDLER=1` on a diagnostic retry. A Slurm RUNNING state alone does
not establish component health.

### 128K cache capacity and routing

Use `scripts/lumi_super_capacity.sbatch` for the isolated two-node cache sweep.
Prepare `PRIME_RL_METADATA_DIR` with `scripts/lumi_prepare_runtime.sh`, inspect the
capacity JSON configs, and run `sbatch --test-only` before submission. The sender
replays policy weights from SFT1250 with the production NCCL wire protocol,
excluding MTP. It checks a full128K request before and after identical reloads,
and reloads with >=80% occupied cache. Results are under
`artifacts/training/nemotron-super-r5-rl-main-20260916/capacity/job-<id>`.
The sweep stops on the first failure; a native crash is not a cache-capacity
measurement. `PRIME_RL_WEIGHT_MEMORY_PROBE=1` records receiver allocator peaks
and free memory around reloads. Keep headroom above the measured peak; successful
startup alone does not validate a cache budget. This test creates no sandboxes.

For multiple replicas, use llm-d scorers for KV cache utilization, queue depth,
active requests, and prefix affinity. Enable Mamba prefix caching with `align`
mode and FP32 SSM state after the serial parity checks below. Prefix affinity
and consistent hashing alone do not establish occupancy balancing. Install pinned
router binaries with `scripts/install_llmd.sh`; keep Go caches under bench
`.cache`. Validate routing through both `/v1/chat/completions` and
`/inference/v1/generate` with deliberately uneven worker loads before RL.

The capacity sweep defaults to 8 and 10 GiB/device; override with
`CAPACITY_BUDGETS="8 9 10"` at submission. The router diagnostic
`scripts/lumi_kv_router_probe.py --output <artifact-directory>` uses the native
EPP/Envoy binaries and both request routes, then swaps worker occupancy. Its
Prometheus fixtures must declare `# TYPE ... gauge`, as the production extractor
reads gauge/counter values rather than untyped metrics.

For lower NCCL reload peaks, set `PRIME_RL_WEIGHT_BUCKET_BYTES=268435456` on
the sender. This emits the versioned bounded-bucket wire format; deploy the
matching receiver before enabling it. Zero retains the legacy dtype-group
wire format, which the new receiver also accepts. Buckets never split a tensor,
so one tensor can exceed the configured limit. Small received weights still
own their storage, preventing deferred loaders from retaining unrelated buffers.

When running the standalone wire probe, set `PRIME_RL_PROBE_NATIVE_WRAPPER=1`
to exercise the production communicator and its scoped Ring/LL settings.
Without it, the probe uses raw PyNcclCommunicator and does not apply the
`PRIME_RL_WEIGHT_NCCL_*` overrides.

Use `lumi_kv_router_probe.py --prefix --output <artifact-directory>` to check
prefix affinity together with occupancy balancing through both API routes.
The prefix scorer is approximate: it tracks routed prefixes rather than
confirming every resident KV block. Keep its weight below the occupancy scorer
so affinity yields to a heavily loaded worker. Policy-version cache salts are
included in the router's prefix hashes.

For the nonlocal-expert retention fix, build a private overlay with
`scripts/lumi_prepare_super_reload_overlay.py --source <installed-vllm>/model_executor/model_loader/reload/layerwise.py --output <artifact-directory>`.
Set `PRIME_RL_SUPER_RELOAD_OVERLAY` to its printed directory. The Super bootstrap
composes this with the frozen optimized profile; it does not replace the SIF or
edit installed packages. The builder verifies the expected source fragment and
records source/output SHA256 values. Existing overlay files are immutable.
The patched loader releases weights only when the native loader reports both
zero elements loaded and explicit `False` (nonlocal expert); other loading and
finalization behavior remains native. Check every worker's startup log for
`Reload overlay enabled` before interpreting memory measurements.

Run prefix parity serially before capacity stress: use `CAPACITY_PREFIX_STRICT=1`
for the built-in 7K/32K/130K token-request check, including per-response cached-token
counts. Do not send a separate parity client during the capacity run. Cold/cold
controls can diverge under different batch compositions. The capacity driver saves
initial, first-reload, and occupied-reload responses, uses distinct cache salts
for those comparisons, and waits for aborted stress requests to drain before
the final comparison. An OOM-free reload alone does not establish output parity.

The LUMI template exports `PRIME_RL_SUPER_RELOAD_OVERLAY` before sourcing the
Super bootstrap, so it can compose the import hook before Python starts. Keep
this ordering; exporting the selector afterward silently omits the overlay.

For the validated LUMI weight-communicator startup, set `PRIME_RL_PAUSE_NCCL_INIT=1`
in the job environment alongside `HSA_ENABLE_SDMA=0`. This pauses all engines
before collective initialization, resumes after success, and leaves them paused
on initialization failure. The HSA workaround is empirical; the underlying runtime
fault is not yet isolated.

### Host-memory prefix cache

Configure local GPU-to-host prefix caching through PRIME's native tier:

```toml
[inference.kv_cache_offload]
type = "native"
[inference.kv_cache_offload.cpu]
num_bytes = 206158430208 # 192 GiB aggregate per TP8 inference node
```

Keep Mamba `align`, FP32 SSM state, and the explicit GPU KV budget. The native
OffloadingConnector supports hybrid cache groups and divides this host budget
across TP workers. This is a local prefix-reuse tier: replicas do not share a
host cache, and host capacity does not directly increase simultaneous GPU decode
slots. There is no filesystem tier. Policy-version cache salts also apply to
offloaded block hashes.

Validate host restoration with `CAPACITY_HOST_CACHE=1 CAPACITY_PREFIX_STRICT=1`
and the host-enabled capacity config. The diagnostic alone sets `PRIME_RL_CACHE_DIAGNOSTICS=1` to mount the cache-reset
routes so it can reset GPU prefixes without clearing the external host cache.
Do not set `VLLM_SERVER_DEV_MODE=1`: its RLHF `/update_weights` route shadows
PRIME's endpoint and rejects the native weight-reload body.
It requires positive external-cache hit counters and matching generated tokens
after reset, then exercises empty and occupied weight reloads. Do not enable
dev routes on the production pilot. Preserve the host-enabled test report before
claiming this tier is validated on the deployed runtime.

Initialize the weight communicator before diagnostic generation/host-cache
traffic, matching orchestrator startup. Synchronize the worker device before
communicator creation. This ordering is under validation for the HSA startup
crash; SDMA-off alone has not eliminated it. For 128K CP8 training, validate
activation offloading (`max_inflight_activations=2`) before pilot submission:
the configuration without activation offloading can OOM during backward.

Before increasing context parallelism, validate both the device mesh and model
head/group dimensions on CPU. `ParallelDims(dp_replicate=1, dp_shard=-1,
cp=..., pp=1, ep=..., world_size=64)` validates the mesh: EP must be a multiple
of CP. The current Super Mamba implementation also requires CP to divide its
eight state groups and grouped normalization boundaries. CP16 is therefore not
supported merely by selecting EP16. The RL loader already uses one packed
sequence per microbatch and accumulates gradients to the global batch size;
reducing global batch size alone does not reduce a full128K microbatch's peak.
Native `optim_cpu_offload=true` keeps Adam state on host between GPU updates;
it does not change optimization/reduction dtypes or move model weights.

## SFT-matched 128K pilot

Use `configs/lumi/rl_nemotron_super_r5_128k_16t8i_pilot.toml` for the
16-trainer/8-inference-node pilot. Keep CP8/EP8, full activation checkpointing,
microbatch one and GPU-resident optimizer state. The eight-node trainer has
larger persistent shards than the working sixteen-node SFT; its memory results
are not interchangeable. The synthetic full-length diagnostic also uses a
narrow token distribution and can induce severe expert imbalance; it is a
stress check, not a throughput prediction for real rollouts.

`scripts/lumi_prepare_super_profile.py` consolidates the validated evaluation
source, frozen helper modules and nonlocal-expert reload fix into one
content-addressed profile with source hashes. Set `PRIME_RL_SUPER_PROFILE` in
`[inference.env_vars]`; the launcher exports it before loading the serving
environment. This retains the compiled vLLM/AITER dependency layer and immutable
SIF. It does not upgrade kernels. The same source profile can be selected by
evaluation launchers through `DFM_EVALS_VLLM_SOURCE` and `VLLM_SRC_OVERRIDE`;
do not change profiles for already-submitted evaluations. Import checks that
load the ROCm platform require an allocated GPU. Validate prefix/host restoration
and occupied-cache reload with the consolidated profile before launching RL.

The LUMI RL template sets the OFI host-staging communication bundle for every
role, including trainer-only diagnostics: `NCCL_DMABUF_ENABLE=0`,
`TORCH_NCCL_HIGH_PRIORITY=1`, and `GPU_MAX_HW_QUEUES=2`. Do not rely on these
being present only in a full pilot's `[env_vars]`. When comparing trainer
configurations, inspect the rendered environment as well as the model mesh.
Use a bounded collective timeout for diagnostics; extending it does not repair
a stalled all-gather.

Keep inference-only communication workarounds scoped to inference.
The SFT-matched trainer profile uses `HSA_ENABLE_SDMA=1` and automatic RCCL
channel selection (omit `NCCL_MIN_NCHANNELS`/`NCCL_MAX_NCHANNELS`). The RL inference bootstrap uses the same automatic channel policy and retains
its SDMA-off setting. RCCL channel counts must match the trainer from process
startup; they cannot be corrected reliably by changing environment variables
only when creating the weight communicator.
Use `configs/lumi/rl_nemotron_super_r5_128k_16t8i_pilot_r3.toml` to express this
separation; do not export `HSA_ENABLE_SDMA=0` globally for all pilot roles.
Check multiple completed optimizer updates, not just GPU allocation or the
first forward, before declaring a trainer memory/communication test successful.

The launcher and RL inference bootstrap clear inherited fixed channel counts.
The weight communicator exchanges channel settings through its CPU store before
RCCL initialization and fails explicitly if ranks disagree. Standalone sender
probes must use the trainer environment, not the inference environment on both
nodes: identical test roles can hide production bootstrap mismatches.

## Full 200-step SC run

The configuration `configs/lumi/rl_nemotron_super_r5_128k_16t8i_sc_full200_r3.toml`
starts from final SFT step 2619 with 16 trainer and eight inference nodes. It uses
128K context, 200 optimizer steps, constant 1e-6 LR, GRPO groups of eight, and the
throughput scheduler's 72/40/8/8 SWE/terminal/SQL/search target per batch of 128.
Agent turns are capped at 200; coding trajectories retain the 200K total-token
budget with harness compaction. The adaptive episode cap starts at 256 (128–512).

Checkpoints include optimizer and orchestrator/scheduler state every ten steps and
at completion. Retention keeps the latest two and steps divisible by 50. Full Super
checkpoints are approximately 1.4 TiB each; allow about 8.4 TiB for retained saves
and another checkpoint of transient headroom. Retention refuses symlinked or
redirected checkpoint roots/steps and incomplete trainer checkpoints without DCP
metadata. Never point checkpoint output at another run or at shared model exports.

Online evals run every 25 steps (held-out diagnostics, TBLite, BIRD dev and
BrowseComp-Plus); SWE-bench Verified/Multilingual, Terminal-Bench 2 and Senior
SWE-Bench run every 50. This is sampled monitoring, not full benchmark reporting.
The existing four-task held-out diagnostics remain excluded from training.
Initial eval is skipped; final policy evaluation still runs at step 200.

Use the normal native dry-run and submission workflow. The allocation is 48 hours,
the partition maximum, not a promise that 200 agentic updates finish in 48 hours.
If it times out, first confirm the old Slurm job has ended and clean up only its
exact run-labelled sandboxes/reservations. Resume with an explicit complete step:

```bash
CONFIG=$PWD/configs/lumi/rl_nemotron_super_r5_128k_16t8i_sc_full200_r3.toml \
  ./scripts/lumi_submit_rl.sh --resume.step <N> --dry-run
CONFIG=$PWD/configs/lumi/rl_nemotron_super_r5_128k_16t8i_sc_full200_r3.toml \
  ./scripts/lumi_submit_rl.sh --resume.step <N>
```

Require both `checkpoints/step_N/trainer/.metadata` and
`checkpoints/step_N/orchestrator/progress.pt`; do not use bare `--resume` after
an interrupted save, because latest-step discovery includes incomplete directories.
Do not alter source names on resume. A changed category mix or source ratio refuses to
resume unless `[orchestrator.concurrency] rebalance_on_resume = true`: the scheduler then keeps
its learned rollout durations and yields and restarts its served counts and credits under the
new targets. To continue in a new run directory, fork the checkpoint with
`--resume.dir <old run>/checkpoints/step_N`; sampler state, the quarantine list and progress
carry over. A changed `[trainer.optim] lr` does not take effect on resume: the checkpointed
optimizer state and LR schedule carry the old rate (the step log's `LR` shows it). Set `[orchestrator.eval] retrigger_on_resume = true` so a resume after a crash reruns
the evals that were in flight at the checkpoint step; by default they are skipped.
Use `scripts/lumi_watch_rl_pilot.py` for ongoing capacity renewal and fatal-error
monitoring also on full runs; it does not automatically resubmit a timed-out job.

### Bounded behavior shaping

The full-run configuration enables `[orchestrator.algo.behavior_penalty]` with
`penalty = 0.1`. Before the GRPO group baseline, a flagged trajectory receives
`max(0, task_reward - 0.1)`, including otherwise successful trajectories. Multiple
defects receive only one deduction. Raw task rewards remain unchanged;
`behavior/task_reward` and `behavior/training_reward` expose the distinction.
All-zero reward groups still have zero advantages; ordinary below-mean GRPO
advantages can be negative even though these shaped rewards cannot.

Detection scans sampled assistant messages for unparsed function/parameter bodies
regardless of their opening token, including missing openings. It excludes parsed
tool arguments, fenced/inline/blockquote examples, observations and errored traces.
Reasoning without content or a tool call is penalized only when a normal model
stop ends the task, not when context/turn/token limits or infrastructure errors
terminate it. Complete unparsed action bodies can still be flagged on a truncated
trajectory. Per-trace `info.behavior_shaping` records the affected message indices.

OpenSeeker uses `configs/search/openseeker_judge_ab.txt` so both its prompt and
parser request A/B labels. Keep the rubric unchanged when adjusting answer format.

The 256-rollout configuration is
`configs/lumi/rl_nemotron_super_r5_128k_16t12i_sc_full200_r11.toml`: 16 trainer
nodes, 12 TP8 inference replicas, group size 8, and OpenCode/Pi/Bash for the
SWE and terminal sources. Source-family weights are preserved when excluding
Prime Agent. Category targets are SWE 112, terminal 64, SQL 16, search 16,
tool use 32, instruction 16. The adaptive rollout cap remains 512; raising
inference capacity does not by itself authorize more concurrent sandboxes.

The r11 configuration uses shared smoothed replay with 64 recent valid outcomes,
a 20% maximum replay share, and a 16-observed-task warmup. The LUMI runtime pins
sandbox SDK 0.4.27, whose exec implementation drains output concurrently with
stdin writes. Use the immutable metadata layer from `lumi_prepare_runtime.sh`;
do not replace SDK files in an existing running layer.


### Image-build recovery and immutable recipe repair

Use SDK 0.4.28+ for read-only image-build polling retries. The UCloud adapter also
uses this waiter for cached receipt validation, so a transient status read does
not discard the receipt or create a duplicate build. A confirmed missing COPY
input follows image quarantine. Exhausted polls retain an infrastructure type
and a shared incident identity for the monitor; ordinary rollout failures retain
individual accounting. Fatal, no-progress, and sustained-low-throughput checks
remain active.

The prepared next-run config is
`configs/lumi/rl_nemotron_super_r5_128k_16t12i_sc_full200_r12.toml` and uses
`artifacts/training/nemotron-super-r5-rl-scale-20260917/all-tasks/images.v8.sqlite`.
Build a new index with the owning environment tool:

```bash
uv run python deps/prime-envs/tools/ucloud_recipe_repairs.py SOURCE.sqlite DESTINATION.sqlite
```

The tool refuses an existing destination. It retains all unrelated recipes and
build cache identities. Terminal-Lego verifier preparation uses temporary CA
trust when the task intentionally breaks certificates; that trust must not be
present in the final runtime. Monitor this path with the maintained health
watcher and retain its exact-job cancellation and exact-run cleanup guards.

## UCloud gateway runtime (verifiers-ucloud 0.3.x)

Runs use the UCloud production gateway `https://app-sandboxes-gw.cloud.sdu.dk` and
relay `https://app-sandboxes-relay-v3.cloud.sdu.dk` from the bench `.env` files.
Every sandbox reaches the relay by the gateway VM's name,
`guest_relay_url = "http://sandbox-gateway-gateway-1:8092"` (gateway 0.9.70+), set on both
the runtime and the interception. Do not use the gateway's private address, which changes
when UCloud restarts its VM, or the public relay URL: model calls through it work, but the
node never pauses the waiting sandbox. The checkout's `.env` pins the package
layer built by `lumi_prepare_runtime.sh` (verifiers-ucloud, the verifiers fork and the SDK,
all from tags); repin it after every rebuild, because jobs source it at startup.
The other live sources are submodules pinned to pushed fork commits: `deps/prime-envs`
(rlrs/research-environments), `deps/renderers` (rlrs/renderers, with the sampler top-k
logprobs that score centering needs) and `deps/text2sql-env`. In a fresh worktree, run
`git submodule update --init` for them; a worktree's submodules inherit URLs already set in
the shared repository config, so check each `git -C deps/<name> remote get-url origin`.
research-environments uses Git LFS, which LUMI lacks: check it out with
`-c filter.lfs.process= -c filter.lfs.smudge=cat -c filter.lfs.required=false`.

Runtime settings: `managed_agent` is the default; set `park_interactive = true` so
OpenCode and Pi park too, and `repair_loopback_hosts = true`. Unknown runtime keys are
rejected, so configs that still use `parkable`, `image_recipe_db` or `image_build_cache`
fail to load. Task images are gateway image names (`image_reference_type = "name"`): the
gateway creates sandboxes only for names in its image index and builds recipe images on
demand.

OpenCode and Pi take Node and their packages from the gateway's read-only `vf-agents`
toolkit: `toolkits = ["vf-agents:latest"]` and `harness_toolkit = "vf-agents"` link its
trees to `/var/tmp/vf-*`, where the harnesses find themselves installed. Do not use
`offline_harness_bundle` for them: unpacking 0.5-0.7 GB into every sandbox at once ran
sandboxes out of disk (`No space left on device`) in the r21 smoke r2. The backend builds
the toolkit from `harnesses-20260920/search-tools-v1/{opencode,pi}-assets.tar.gz`.

Allowlists come from the gateway, read with the SDK key:

```bash
verifiers-ucloud summary
verifiers-ucloud task-ids artifacts/manifests/ucloud-task-ids-<date>
```

Each training taskset takes its environment's file as `task_ids_file`. Loading fails when a
listed task is unknown or removed by the source's own filters, so derive a list without
held-out tasks instead of filtering them. Remove tasks whose images are broken the same way, and record why next to the list (`swe-smith-train.removed.json` drops the charset_normalizer image, which lacks ripgrep, and one kakoune task whose branch is not in its image). SWE-smith grading queries the GitHub API from
the env server (`swesmith` profile helpers) and reads `GITHUB_TOKEN`; without it LUMI's
shared outbound IP hits the anonymous limit (60/h) and grading fails with `403: rate limit
exceeded` or `Mirror clone repo must be created first`. The bench `prime-rl/.env` holds the
key as `GITHUB_API_KEY`; the checkout `.env` exports `GITHUB_TOKEN` from it. Load every taskset with the final config before
submitting. Harbor task caches (`task_data_file`) record the full taskset config, so
regenerate them with `deps/prime-envs/tools/ucloud_cache_harbor.py` whenever
`task_ids_file` or a pinned `revision` changes. Terminal-Lego's pinned revision must match
its git checkout under `HF_HOME`. Harbor datasets are cached under the container home
`$LUMI_ROOT/.home/.cache/harbor`; check loads with `--home` set to it.

Before a run, check the gateway with `scripts/lumi_probe_ucloud_runtime.py group agent acp
--output DIR` inside the operational SIF and the run's Python path. It uses r21's runtime
settings and deletes only its own sandboxes. A waiting sandbox is paused in place and its
status stays `running`; it is parked (and its relay request reattachable) only under node
memory pressure, so the probes require answered calls with intact memory, not `parked`.

`[orchestrator.image_prefetch]` keeps upcoming images built: every `interval_seconds` it
asks the gateway to ensure the images of each name-based env's next `lookahead_tasks`
unseen tasks. Smoothed replay explores unseen tasks in a seeded order, so those tasks are
known before they are sampled. Sources with the same `shared_group` share that order,
so one build serves every harness of a family. The gateway starts at most 32 builds per call, and a build
can take several minutes. Failed and unknown images are quarantined. The step metrics
`image_prefetch/<state>` show how far ahead builds are.

Math sources need no sandbox. `i3-math` (INTELLECT-3-RL math) trains on the subprocess
runtime with the `null` harness; `aime25` and `aime26` evaluate. Runs set
`HF_DATASETS_OFFLINE=1`, so download their datasets first by loading the tasksets on a login
node with `HF_DATASETS_OFFLINE=0 HF_HUB_OFFLINE=0`. `i3-math` grades with a uv script
(`verify.py`, math-verify); prepare it once against the shared `UV_CACHE_DIR` so rollouts do
not resolve it from PyPI. Its `filter` keeps problems by Qwen3-4B's solve rate; r21 keeps
those solved at most 4 of 8 times. The math tasksets default to no network
(`network_allow = []`), which the subprocess runtime cannot enforce: verifiers then fails
every episode before it starts ("requires a network policy"), the orchestrator logs
`Episode failed before any trace`, and a category quota on that source blocks every batch.
Set `env.taskset.network_allow = ["*"]` on them. Per-call token caps follow one rule, in training and eval alike:
multi-turn agent sources use the run-wide `max_completion_tokens` (8192 in r21) and
single-turn sources (`max_turns = 1`), where one call is the whole answer, override it to
32768. A 4096 cap cut every math answer before its boxed result, 18-22% of the single-turn
NeMo answers, and 8-14% of Bash turns; every cut-off single-turn answer scores 0, and a
category quota on such a source blocks every batch. Before adding a source, run a few real
rollouts with `eval` on its exact harness and runtime; loading the taskset is not enough.

`[orchestrator.degenerate_turns]` keeps derailed turns out of training: a train trace with a
sampled turn of at least `min_tokens` whose mean negative sampling logprob exceeds
`max_mean_nll` is marked failed (`DegenerateTurn`). At temperature 1 with no tail cut-off the
Bash harness occasionally spirals into thousands of near-random tokens (mean NLL ~8 against
~0.65); in r21 these grew to 1% of Bash turns and 17% of Bash rollouts and drove its entropy up.

`general-agent` (multi-turn tool use, 4,417 tasks with their own tools and databases) also
runs locally: each task's tools are served host-side over MCP to the `null` harness and
graded by replaying its gold tool chain. Its corpus comes through Harbor into the container
home cache (`$LUMI_ROOT/.home/.cache/harbor`); load it once online before a run. r21 keeps
only hard tasks, those GPT-5-mini solves at most half the time (`max_pass_rate = 0.5`, 1,385
tasks; GLM-5.3-flash solved 88% of the 0.95 band), and gives it one group (8) per batch in its own `toolbench` category. ProgramBench and NL2RepoBench are benchmarks (200 and 104 tasks)
that need their images on the gateway; use them as evals, not training.
