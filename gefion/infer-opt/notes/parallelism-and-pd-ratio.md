# Parallelism layouts, the prefill stall, and the P:D ratio (Nemotron-3 Super, vLLM 0.28, Gefion H100)

Date: 2026-10-29. Read-only study. No jobs were submitted and nothing outside this file was edited.

Sources:
- vLLM 0.28.0 (`g2cf0a6915`), at `phase0/prime-rl/.venv/lib/python3.12/site-packages/vllm`. Below, `vllm/` means that directory.
- prime-rl at `phase0/prime-rl` (commit `138d1b5c`).
- vllm-router 0.2.0. The wheel is from `PrimeIntellect-ai/router`, tag v0.2.0. Its Rust source was fetched read-only from GitHub. The files cited are `src/routers/http/vllm_pd_router.rs`, `pd_router.rs`, `server.rs` and `routers/router_manager.rs`.
- Measurements: job 533117 (`results/mtppd-s-533117.out`, FP8 DP8+EP8 decode-only and prefill-only), the nsys profile 533102, and the TP8 RL run log `runs/gefion-super-tp8-4t1i-r1/logs/attempt_1/inference.log`.

## Summary

- **The DP8 stall comes from where the work runs, not from the communication.** Under DEP, all of a prefill chunk's non-expert work runs on one GPU: Mamba, attention, the shared expert and the latent projections. For an 8704-token chunk that is about 119 TFLOP, or about 200 ms at 600 TFLOP/s, plus long-context attention. The other 7 ranks wait inside the MoE `AllGatherV`. Only splitting that work fixes it (TP, or a TP×DP hybrid), or moving it elsewhere (PD), or making the ranks wait at the same time.
- **vLLM 0.28 has a knob for the last option.** `prefill_schedule_interval = N` (scheduler config, DP only) admits prefills only on every N-th step, and that step is the same on all DP ranks. Their stalls then overlap instead of adding up. It needs no code, and prime-rl passes it through `[inference.vllm]`. It is the cheapest thing to test first.
- **Prefix-cache granularity depends on the layout.** It is a side effect of the hybrid page-size rule: the attention block size is 4160 tokens under DP8 (FP8 run) but 1040 under TP8 (both from the logs). Each turn re-prefills, on average, about half a block beyond the last block-aligned Mamba checkpoint. That is probably a large part of why prompt tokens per generated token are about 3 under DP8 and only 1.4 in the TP8 run.
- **PD roles are per request.** prime-rl starts every engine with `kv_role = "kv_both"`, and the NIXL behaviour is chosen per request by `do_remote_prefill` / `do_remote_decode`. vllm-router always runs both stages (it has no bypass threshold) and cannot move a worker between pools at runtime in PD mode. prime-rl's **llm-d** router backend already has the threshold (`[inference.router] type = "llm-d"`, `non_cached_tokens`). The binaries are not installed in this checkout.
- **Several features are blocked or unsuitable for this model.** DCP and PCP are blocked for Mamba. DBO needs DeepEP or nixl_ep and the V1 runner, and with Mamba it is untested. EPLB is supported but does nothing for the stall. Wide EP across nodes makes the stall worse.
- **The 150 ms TP8 chunk is not one bug.** The roofline at realistic efficiencies is about 95–135 ms. The named excess costs are:
  - about 7 GB of TP all-reduce per step, which falls back from symmetric memory to NCCL because 8704 tokens is just over the 64 MiB limit;
  - long-context attention: 13–37 ms per rank at 32k–100k positions;
  - latent projections and the router gate, replicated on all 8 ranks: 7.3 TFLOP per rank, of which 7/8 is wasted.

---

## 0. Model numbers used below (from `models/nemotron-super-r6-step2638/config.json`)

| item | value |
|---|---|
| Mamba2 (40 layers, d_inner 8192, 128 heads × 64, 8 groups, d_state 128) | 4.38 B params |
| GQA attention (8 layers, 32 q heads, **2 KV heads**, head_dim 128) | 0.29 B |
| Routed experts: 512 × (1024→2688→1024, relu²) × 40, latent MoE (`moe_latent_size` 1024) | 112.7 B total, **4.84 B active** (top-22) |
| Shared expert (4096→5376→4096) × 40 | 1.76 B |
| Latent projections fc1/fc2 (4096↔1024) × 40 | 0.34 B |
| Router gate (4096×512, bf16 → fp32 out) × 40 | 0.08 B |
| Embedding + LM head | 2 × 0.54 B |
| **Total / active (without LM head)** | **120.7 B / 11.7 B**. That is 23.4 GFLOP per prefill token in linear layers |
| KV cache per token (8 layers × 2 heads × 128 × K,V × bf16) | **8 KiB** |
| Mamba state per sequence (fp32 SSM 128×64×128 × 40 layers, plus conv) | **168 MB + 2.5 MB** |

The non-expert part (Mamba + attention + shared + latent + gate) is **6.8 B params, 58% of the active FLOPs.** Under DP attention, this is the part that runs on a single GPU for that GPU's prefill.

---

## 1. Prefill vs decode node ratio

### 1.1 Sizing method

Measure the following:

| symbol | what | how to measure |
|---|---|---|
| `G` | target generated tok/s, cluster-wide | orchestrator, or the sum over vLLM "Avg generation throughput" |
| `r` | **computed** prompt tokens per generated token in the colocated layout | vLLM "Avg prompt throughput" counts computed tokens only (`v1/metrics/loggers.py:146`, `stats.computed`). The TP8 log 08:17–09:00 (running > 100) gives 5.3k / 3.8k = **1.39**. The DP8 runs give ~3, see 1.4 |
| `r_PD` | the same in PD mode | see below. It is **larger than `r`** with vllm-router |
| `Π_P(c)` | prefill throughput of one P node at the context positions you actually prefill at | a P node at saturation ("Avg prompt throughput" with waiting > 0). Synthetic reference: **93.7k tok/s/node**, FP8 DP8+EP8, 4–8k prompts, no prefix cache (533117). Long positions add `16384·c·8` FLOP of attention per token, about +22% at c = 40k and +37% at 64k, so expect ~60–75k at RL positions |
| `g_D(s)` | decode tok/s of one D node at the target per-sequence speed `s` | decode-only step time `T(n)` per rank with n seqs/rank, s = tokens_per_step / T, g_D = 8·n·s |
| `u_P` | target P utilisation, 0.7–0.8 | keeps the P queue, and so each turn's TTFT, bounded |

Then, with `u_P` keeping some headroom on P:

```
n_D  = ceil( G / g_D(s) )                         # also check memory: n·(168 MB·(1+k_mtp) + c_avg·8 KiB) per GPU
n_P  = ceil( G · r_PD / (u_P · Π_P) )
n_P / n_D = g_D(s) · r_PD / (u_P · Π_P)
```

**Decode-side numbers (533117, FP8 DP8+EP8, allgather, prefix caching off, short contexts):**
- **Step-time fits.** No MTP: T(n) ≈ 14 + 0.20·n ms (17.4 / 21.2 / 27.4 / 33.9 ms at 16 / 32 / 64 / 96 seqs per rank). MTP k=1: T ≈ 16 + 0.34·n ms at 1.79 accepted tokens per step on real loads.
- **Long contexts** add the KV read: about `n · c_avg · 8 KiB / 2.7 TB/s`, which is 0.12 ms per sequence at c_avg = 40k. The Mamba state traffic is already in the fit.
- **Example: no MTP, s = 40 tok/s, c_avg = 40k.** T = 25 ms gives n ≈ 34/rank, so g_D ≈ 11k tok/s per node. Memory per GPU: 34 × (168 MB + 330 MB) ≈ 17 GB, which fits.
- **Example: MTP, s = 40 tok/s.** n ≈ 60–70/rank, so g_D ≈ 19–22k per node. This is **memory-bound**: MTP doubles the Mamba blocks, giving 0.66 GB per sequence at 40k.

**`r_PD` with vllm-router.**
- Every turn goes P → D (`vllm_pd_router.rs` `process_vllm_two_stage_request_discovered`: stage 1 sets `max_tokens = 1`, stage 2 sends `kv_transfer_params`).
- P never sees the tokens D generated in the previous turn, so P **re-prefills them** in the next turn's prompt.
- Hence `r_PD ≈ r + (1 − 1/turns_per_rollout)`, about r + 1 for long agent episodes.
- With a bypass threshold (1.2), short continuations stay on D, which already holds that KV. Then `r_PD` approaches `r`, and most of it is first turns and long tool outputs.

**Worked ratios** (Π_P = 65k, u_P = 0.8):

| decode config | g_D | r_PD = 2.4 (TP8-like r + 1) | r_PD = 4 (DP8-like r + 1) |
|---|---|---|---|
| DP8 no MTP, 40 tok/s/seq | 11k | n_P/n_D = 0.51 | 0.85 |
| DP8 MTP, 40 tok/s/seq | 20k | 0.92 | 1.54 |

So on two nodes, **1P:1D is about right** without MTP, with P somewhat idle. With MTP on D, P becomes the bottleneck unless the threshold bypass reduces `r_PD`. On 3 nodes, use 1P:2D without MTP and 2P:1D with MTP. **Measure `r_PD` and `Π_P` in the running PD test first: the ratio is linear in both.**

**KV and Mamba transfer cost per turn.**
- **What moves.** D pulls the KV for the prompt tokens it doesn't already hold in its prefix cache, at 8 KiB per token, plus the full Mamba state. The state is 168 MB in fp32, or 84 MB with `mamba_ssm_cache_dtype = float16`. With Mamba, P computes N−1 tokens and D recomputes the last one (`nixl/base_scheduler.py` `_truncate_mamba_request_for_prefill`, `_get_remote_prefill_token_count`).
- **Bandwidth is not the issue.** At ~11 turns/s per D node (11k tok/s, ~1k tokens per turn), that is ~1.9 GB/s of Mamba state plus ~0.4 GB/s of KV, against ~50 GB/s per NDR port and 8 ports per node. That holds only on nodes with `iommu=pt`; the report (§9) has the IOMMU caveat.
- **Latency.** Each turn adds roughly 5–10 ms for 168 MB over one rank pair, plus the P queue and P compute time. RL throughput does not care as long as there is enough concurrency, but rollouts get longer.
- **Requirements.** `VLLM_SSM_CONV_STATE_LAYOUT=DS` (asserted in `nixl/base_worker.py:322`). Keep **equal TP on P and D**, for two reasons:
  - The hybrid attention block size depends on TP: 4160 at TP1/TP2, 2080 at TP4, 1040 at TP8.
  - `nixl/base_worker.py:202` asserts against "head-sharded attention reads with P_TP > D_TP and heterogeneous block sizes". With 2 KV heads, P = TP8 / D = TP1 is exactly that case.

### 1.2 Can it adapt on the fly?

**The engine role is per request; `kv_role` is mostly cosmetic.**
- prime-rl builds `kv_role = "kv_both"` for every engine (`packages/prime-rl-configs/src/prime_rl/configs/inference.py:575–580`). vLLM warns that `kv_both` is deprecated (`nixl/connector.py:126`).
- **What each request does is set by its `kv_transfer_params`** (`nixl/pull_scheduler.py`):
  - `do_remote_decode = true`: act as P. The prompt is truncated by one token for Mamba, and the blocks are held until D reads them.
  - `do_remote_prefill = true`: act as D. Pull the remote blocks beyond the local prefix-cache hit.
  - neither: an ordinary local prefill + decode.
- `kv_role` only matters in two places: deferred block freeing for consumers when there is more than one in-flight batch (`v1/core/sched/scheduler.py:155`), and partial-tail offload for producers (`:1237`).
- **So any engine can serve as P, D or colocated at the same time.** The roles are fixed only by the router's pool membership and by per-role engine settings fixed at startup: all2all backend, cudagraph mode, max_num_seqs and budget.

**Built into vLLM, but not usable through vllm-router today.**
- `kv_recompute_threshold` (default 64, `kv_connector_extra_config`): skip a remote pull below N tokens.
- `bidirectional_kv_xfer` (default off): D keeps its blocks for `decoder_kv_blocks_ttl = 480` s and returns `remote_block_ids` in its response. P can then **pull the previous turn's KV back from D** instead of re-prefilling it (`nixl/pull_scheduler.py:67–101, 205–281`).
- Bidirectional transfer would remove the `r_PD − r` overhead, but the client or router must carry D's returned `kv_transfer_params` into the next turn's P request. vllm-router overwrites the prefill params (`build_prefill_kv_transfer_params` always emits nulls), so this does not happen today.

**vllm-router 0.2.0 PD mode.**
- **No bypass.** Every request is two-stage, and there is no token threshold. The CLI threshold flags are cache-aware load-balancing knobs only.
- **Policies.** `--prefill-policy` and `--decode-policy` are separate, and prime-rl passes one `--policy`. With `consistent_hash` on `x-session-id`, a session sticks to the same P and the same D.
- **Pool membership is fixed after startup.** `/add_worker` on the PD router returns "PD router requires specific add_prefill_server or add_decode_server methods" (`pd_router.rs:889`). `add_prefill_server` / `add_decode_server` exist in Rust but have no HTTP route in non-IGW mode (`server.rs:1245–1256`); `POST /workers` with `worker_type` only works with `--enable-igw` (`router_manager.rs:193`). `/remove_worker` works.

**What prime-rl exposes.**
- `deployment.type = "disaggregated"` with fixed counts: `prefill_nodes_per_replica`, `decode_nodes_per_replica`, `num_prefill_replicas`, `num_decode_replicas`.
- Per-role `*_env_vars` and `*_vllm_overrides`.
- Template defaults: P uses `deepep_high_throughput`; D uses `deepep_low_latency` and `FULL_DECODE_ONLY` (`templates/multi_node_rl.sbatch.j2:356–415`). The PD test overrides both to allgather.
- **The threshold already exists via the llm-d backend**:
  - Config: `[inference.router] type = "llm-d"`, `non_cached_tokens = N` (default 16; `configs/inference.py:330`).
  - Template: `templates/llmd/epp_pd.yaml.j2`. The EPP's `prefix-based-pd-decider` runs decode-only when fewer than N prompt tokens are uncached on the chosen D. A `pd-sidecar` on each decode node orchestrates the remote prefill.
  - Install: `scripts/install_llmd.sh`. `third_party/llmd` is not present in this checkout.
  - Caveats: llm-d is rejected with `enable_return_routed_experts`; its prefix index is approximate; and it is untested with `/inference/v1/generate` token-id requests at our scale.
- There is no dynamic role assignment anywhere.

### 1.3 What a dynamic scheme would cost

| option | what | effort | value |
|---|---|---|---|
| A. llm-d threshold | `router.type = "llm-d"`, `non_cached_tokens` ≈ 1–4k; install the binaries | config plus 1 install step; then validation (routing correctness, prefix-scorer accuracy on token-id requests) | Short tool-output turns stay on D, which holds the KV, so `r_PD` → `r`. Each D DP rank then does small local prefills, so combine with `prefill_schedule_interval` on D |
| B. Threshold in vllm-router | In `vllm_pd_router.rs`, before stage 1: count prompt tokens (prime-rl sends token ids), estimate D's cached prefix from per-session `prompt_len + completion_len` of the previous turn (exact under consistent-hash affinity), and send directly to D without `kv_transfer_params` below T | ~150–250 lines of Rust plus tests, 2–4 days | Same as A, and keeps vllm-router (routed experts, logprob merge) |
| C. Bidirectional multi-turn | Also carry D's returned `kv_transfer_params` into the next P request (`bidirectional_kv_xfer = true`) | B plus 1–2 days; D must keep blocks ≥ the turn gap (`decoder_kv_blocks_ttl`) | P stops re-prefilling the generated tokens. Mostly redundant if B or A already keeps continuations on D |
| D. Role shifting | Admin endpoint to move a URL between pools, plus a controller on P queue depth / D KV usage | ~1 day router plus 2–3 days controller and drain logic | Coarse: one 8-GPU DP group per move, nodes must drain first (no request migration), and consistent-hash remaps lose prefix affinity. Per-role configs differ. **Not worth it** compared with A/B plus a static ratio |

---

## 2. Other layouts and the prefill-stall problem

**How DEP lockstep works in 0.28.**
- Every DP rank runs a forward pass each step, and idle ranks run a dummy batch (`v1/engine/core.py:927`, `DPEngineCoreProc`). The MoE allgather therefore makes every rank wait at every MoE layer for the slowest rank.
- `v1/worker/dp_utils.py` takes the **minimum** cudagraph mode across ranks and pads to the max token count. In a step where any rank prefills more than the capture size, all ranks run eager or piecewise.
- The 533102 profile shows the waiting as `AllGatherV_RING_LL` time: 2.6–3.6 s of ~6 s per rank, average 1.3 ms and max 17.6 ms per call.

| layout / feature | supported for Nemotron-H in 0.28? | decode step | prefill stall | config | risks |
|---|---|---|---|---|---|
| **DEP (DP8+EP8), current** | yes | ~21.5 ms (17–34 ms at 16–96 seqs/rank) | **worst**: the chunk's dense work (~119 TFLOP per 8704 tokens) runs on 1 GPU while 7 wait | `data_parallel_size = 8`, `enable_expert_parallel`, `all2all_backend = "allgather_reducescatter"` | none new |
| **DEP + `prefill_schedule_interval = N`** | **yes**. `config/scheduler.py:143`; engine side `v1/engine/core.py:2137`; scheduler side `v1/core/sched/scheduler.py:519, 552, 922, 1166`. DP only | unchanged on non-prefill steps | Prefills are admitted only when `step_counter % N == 0`, which is the same step on every rank, so the stalls overlap. Ideal: up to ~8× less stalled time if ranks have similar prefill load | `[inference.vllm] prefill_schedule_interval = 4..16`. Size `max_num_batched_tokens` for N steps of prefill demand. Per rank: ~3 × 21 tokens/step, so 8704 is plenty | Throttling switches off on a rank whose last release step left requests waiting (`prefill_capacity_bound = bool(self.waiting)`), so a saturated rank breaks the alignment. TTFT grows by ≤ N steps (~0.1–0.4 s, irrelevant for RL) |
| **TP2×DP4 + EP8** (or TP4×DP2) | yes, and the prime-rl template supports `tensor_parallel_size` with DP per node. Sequence-parallel MoE turns on automatically when TP > 1 and DP > 1 with allgather or DeepEP (`config/parallel.py:673`). Nemotron-H then runs the gate, latent projections and shared expert on 1/TP of the tokens (`models/nemotron_h.py:147–247`) | slightly above DP8. Phase-0 bf16 TP4×DP2 was ~14.5k engine peak vs ~17.9k for DP8 (before the infer-opt patches) | Per-chunk stall shrinks ~TP× (dense work on TP GPUs), and the number of independent lockstep groups drops to 8/TP. Combines with `prefill_schedule_interval`. Attention block size: 4160 at TP2, **2080 at TP4**, so TP4 also halves the per-turn re-prefill | `tensor_parallel_size = 2` (or 4), `data_parallel_size = 4` (or 2), `enable_expert_parallel` | TP4 replicates KV (2 heads), so KV memory ×2. Uneven replica load (seen in §11) |
| **TP8+EP8, one engine** | yes | 10–25 ms | Chunk spread over 8 GPUs, but the whole engine stalls 100–300 ms per chunk (~45% of the time). See §3 for why it is that slow | as now | Without DP there is no sequence-parallel MoE, so the latent projections and gate are replicated on 8 ranks and each MoE layer does 2 all-reduces (latent 1024-dim **and** shared 4096-dim, `fused_moe/runner/moe_runner.py:434–451`) |
| **Chunked-prefill budget** | yes | – | Changes only how stalls are distributed, not the total: total stall ≈ Σ prefill tokens × per-token cost either way. Smaller chunks lower each step's spike, at slightly worse GEMM efficiency | `max_num_batched_tokens`, `long_prefill_token_threshold` | **Keep ≥ 4176 if MTP + prefix caching** (the 0.28 corruption bug). Under TP8, **8192 instead of 8704** keeps the 64 MiB all-reduce on the symmetric-memory path (`all_reduce_utils.py:58`, `size <= 64 MiB` for world size 8 on sm90; 8704 × 4096 × 2 B = 68 MiB goes to NCCL) |
| **Wide EP (EP16/32, DP attention across nodes)** | DeepEP 1.2.1 and NVSHMEM are installed. vLLM backends `deepep_high_throughput` / `deepep_low_latency` (`config/parallel.py:188`); `VLLM_DEEPEP_BUFFER_SIZE_MB`; internode HT uses NVSHMEM RDMA | Expert-weight reads per rank halve (14 GB → 7 GB FP8 at EP16), but each decode step adds 2 × 40 IB dispatch/combine round trips. For this model's small latent experts that is a net loss or a wash | **Worse**: one rank's prefill now stalls 16–32 GPUs | `all2all_backend = "deepep_low_latency"` (decode) / `"deepep_high_throughput"`, DP across nodes | Needs IBGDA working (NVSHMEM IBGDA; usually the driver options `NVreg_EnableStreamMemOPs=1` and `PeerMappingOverride=1`; not checked, login node has no GPU) and `iommu=pt` nodes (§9). Phase 0: DeepEP-LL asserted `num_rdma_bytes < INT_MAX` at 4K/16K tokens, because the LL buffer scales with `moe.max_num_tokens = max_num_batched_tokens` × 512 experts (`fused_moe/layer.py:352`) and so only works with a decode-sized budget. DeepEP has an "Unsupported hidden" switch, and 1024-dim latent / top-22 are untested |
| **DBO (dual-batch overlap)** | partly. `enable_dbo` requires an all2all backend in {deepep_low_latency, deepep_high_throughput, nixl_ep} (`config/vllm.py:1681`); yield points exist only in the DeepEP/nixl_ep prepare/finalize and the modular kernel. Not in the V2 runner (`config/vllm.py:2452`). prime-rl forces V2 when `enable_return_sampling_mask` is set (`src/prime_rl/inference/server.py:17`); Nemotron-H is V1 by default | Overlaps the EP all2all with compute (~3–8 ms/step of comm here) | Splits each batch in two at the token midpoint (`v1/worker/ubatch_utils.py:63`), so it can overlap one ubatch's comm with the other's compute. **It does not remove the dense prefill work** | `enable_dbo = true`, `dbo_decode_token_threshold`, `dbo_prefill_token_threshold`, DeepEP backend | **No Mamba guard.** A request split across the two ubatches must hand its SSM state from ubatch 0 to ubatch 1 inside the same layer, and that path is unverified (silent-corruption risk). Run parity tests before any use |
| **EPLB** | yes. `NemotronHForCausalLM` implements `MixtureOfExperts`, and online FP8 MoE has `supports_eplb = True` (`quantization/online/moe_base.py:133`) | small: per-rank expert load is already fairly even at 512 experts × top-22 | **none**: the imbalance is in DP attention, not in the experts | `enable_eplb`, `eplb_config.num_redundant_experts`, `window_size`, `step_interval` | Physical-expert remapping vs prime-rl's per-step weight reload (`PRIME_RL_RELOAD_FILTER_NEMOTRON_EXPERTS`) and the `ep_compact_moe` patch; prime-rl turns it off for PD by default. Skip |
| **Decode context parallel (DCP)** | **no**. `MambaManager` asserts "DCP not support mamba now" (`v1/core/single_type_kv_cache_manager.py:1310`). Also GQA DCP needs TP > 2 KV heads (`config/model.py:1387`) | – | – | `decode_context_parallel_size` | blocked |
| **Prefill context parallel (PCP)** | **no**. Same file asserts "PCP not support mamba now"; PCP runtime is V2-only and MLA-only (`config/vllm.py:621, 2375`) | – | – | `prefill_context_parallel_size` | blocked |
| **Sequence parallelism pass (`enable_sp`, async TP `fuse_gemm_comms`)** | settable with TP > 1, but off by default for MoE (`config/vllm.py:274`, `IS_DENSE`). It forces `splitting_ops = []` and FULL cudagraphs (`config/compilation.py:1216`), which conflicts with Mamba's piecewise splitting. The MoE all-reduces are inside the MoE custom op, out of the pass's reach | – | small at best (attention/Mamba all-reduce → RS+AG with overlap) | `compilation_config.pass_config.enable_sp / fuse_gemm_comms` | high (graph mode); not recommended |
| **PD (NIXL)** | yes, including Mamba (3-read conv transfer, DS layout) and hetero-TP for Mamba. See §1 | D steps are pure decode (~21 ms at ~22/rank DP8) | **gone on D** (except the 1-token Mamba recompute); P is throughput-bound | §1 | `r_PD > r`; equal TP on both sides; one extra hop per turn |
| Elastic EP, MoRI, nixl_ep, flashinfer_nvlink_* | exist in `All2AllBackend` | – | – | – | not relevant on H100/IB for this problem |

---

## 3. Why a TP8 prefill chunk of 8704 tokens takes ~150 ms

**Per-rank work for TP8+EP8 with DP1.**
- Mamba, attention and the shared expert are split 8 ways.
- **The latent projections and gate are replicated.** `ReplicatedLinear` / `GateLinear` (`models/nemotron_h.py:149, 186–200`); `disable_tp` only applies under sequence-parallel MoE.
- The experts are split by EP.

| component | per-rank work | estimate | notes |
|---|---|---|---|
| Split GEMMs (Mamba in/out, attn qkv/o, shared expert), FP8 `CutlassFp8BlockScaledMMKernel` (log) | 14.0 TFLOP | 14–20 ms at 700–1000 TFLOP/s | FP8 dense peak is 1979; vLLM's CUTLASS blockwise sm90 kernel typically reaches only 35–55% of it. `use_deep_gemm = false` sets **both** `VLLM_USE_DEEP_GEMM` and `VLLM_MOE_USE_DEEP_GEMM` to 0 (`server.py:28–30`), so the DeepGEMM linear kernel (first in the priority list, `kernels/linear/__init__.py:430`) is never tried |
| **Replicated** latent proj (FP8) + gate (bf16 cuBLAS → fp32) | 7.3 TFLOP | ~10 ms | 7/8 of this is redundant. It disappears with TP×DP (sequence-parallel MoE) |
| Routed experts, FlashInfer CUTLASS FP8 (TRT-LLM DeepGEMM-style grouped kernel, `fp8_gemm_kernel<2688,1024,…>` in 533102) | 10.5 TFLOP (64 local experts × ~374 rows each) | ~15–25 ms | Small K = 1024 / N = 1024 gives a short mainloop, so efficiency is ~30–50%. Rows go in 3 chunks via `fi_moe_chunk` (4096) |
| Attention core (FA3 bf16, 4 q heads per rank, causal) | 0.6 / 2.9 / 5.2 / 9.7 / 14.9 TFLOP at chunk start 0 / 16k / 32k / 64k / 100k | 2 / 7 / 13 / 24 / 37 ms at 400 TFLOP/s | **At ≥ 64k, attention exceeds all the linear layers.** Profile 533102 shows single-GPU FA3 prefill calls of up to 3.6 ms per layer |
| TP all-reduce: 88 × 71 MB (Mamba out, attn o, shared) + 40 × 17.8 MB (latent) ≈ 7.0 GB | – | **20–35 ms** at 350–450 GB/s busbw | Not overlapped. 71 MB is just above the 64 MiB symmetric-memory limit, so it runs on NCCL ring (backend order CUSTOM → SYMM_MEM → PYNCCL, from the log). `fuse_allreduce_rms` only covers ≤ 1 MB at TP8 |
| Mamba SSD/conv (Triton, 16 heads per rank), norms, FP8 activation quantization, elementwise | – | ~8–12 ms | |
| **Total** | | **~95–135 ms** (16 ms if all linear FLOPs ran at FP8 peak) | measured ~150 ms |

**Conclusion.** The chunk is ~7× slower than pure FP8 peak, but only ~1.1–1.6× slower than a realistic estimate. The excess is spread across several costs, in rough order:

1. TP all-reduce (~25% of the step).
2. Attention at long positions.
3. Replicated latent and gate work (~7%).
4. GEMM efficiency: CUTLASS blockwise on large M, and small-K grouped MoE.

To confirm, run one nsys capture of a TP8 mixed step with `analysis/nsys_breakdown.py`; the existing 533102 profile is DP8.

**Cheap TP8 checks.**
- `max_num_batched_tokens = 8192`, so the all-reduce stays on symmetric memory.
- Microbenchmark DeepGEMM against CUTLASS blockwise on the in_proj shape `(8704 × 4096) · (4096 × 18560)/8`. If DeepGEMM wins, set `VLLM_USE_DEEP_GEMM=1` while keeping `VLLM_MOE_USE_DEEP_GEMM=0`. That needs a small prime-rl change, because `server.py` overwrites both.
- None of these fixes the structural issue: with DP = 1, TP8 cannot use sequence-parallel MoE.

---

## 4. Recommendation: what to test next on 1–2 inference nodes

1. **DEP + `prefill_schedule_interval` (1 node, config only).** DP8+EP8 FP8 as now, plus `[inference.vllm] prefill_schedule_interval = 4`, then 8 and 16.
   - Measure: stalled-step share (steps > 50 ms), per-sequence tok/s, gen and prompt tok/s, and how often `prefill_capacity_bound` fires (waiting > 0).
   - Cheapest test, and it targets the lockstep stall directly.
2. **TP2×DP4 + EP8 (sequence-parallel MoE) with the best interval from step 1 (1 node).** Then **TP4×DP2**, which also halves the hybrid block size to 2080 and so the per-turn re-prefill.
   - Measure computed-prompt/generated (`r`) alongside throughput. The layout changes `r` through the block size.
3. **PD 1P:1D on 2 nodes, both DP8+EP8 (equal TP), as the running test does. Then 1P:1D with the llm-d router and `non_cached_tokens` ≈ 2048** (install `scripts/install_llmd.sh`; set `prefill_schedule_interval` on D).
   - Measure `r_PD`, P saturation throughput `Π_P` and D `g_D`, then rescale the ratio with §1.1.
   - Add MTP on D only after measuring `Π_P`, because MTP roughly doubles the P demand per D node.
4. **TP8 hygiene (1 node, if TP8 stays the baseline):** `max_num_batched_tokens = 8192`, a DeepGEMM-for-linears microbenchmark, and an nsys capture of a mixed step to confirm the §3 breakdown.
5. **Defer:**
   - vllm-router bypass threshold (§1.3 B): only if llm-d can't be used.
   - DBO with DeepEP: V1 runner only, Mamba correctness unverified, DeepEP untested at hidden 1024 / top-22.
   - Wide EP across nodes: needs IBGDA and IOMMU-fixed nodes, and makes the stall worse.
   - EPLB.
   - DCP/PCP/SP: blocked or unsuitable for Mamba.

---

## 5. PD results: 1P:1D, both DP8+EP8 FP8, allgather, vllm-router (job 533326, 2026-10-29)

**Setup.** `phase0/full/gefion_pd_r1.toml` (made by `make_pd_config.py`). P = dgx047 and D = dgx048, both with iommu=pt. The trainer is on 4 nodes. Both roles use `allgather_reducescatter`, and D keeps `FULL_AND_PIECEWISE`. No CPU offload. `VLLM_SSM_CONV_STATE_LAYOUT=DS`, and UCX on all 8 IB ports for both roles. Sampling uses /metrics A/B snapshots 60 s apart, summed over the 8 per-rank ports, plus `pd_metrics.py` on the stats log lines.

**What broke (job 533137).** The template sets `LOCAL_IP=$(hostname -I | awk '{print $1}')`. On Gefion the first address is `bmc_redfish0`, 169.254.0.18, which is the same on every node. It becomes `VLLM_NIXL_SIDE_CHANNEL_HOST`, so D handshook with its own engines. The result was "Remote NIXL agent engine ID mismatch", every transfer failed, and D produced garbage. `render_pd.sh` now patches LOCAL_IP to skip 169.254.x. The non-PD launches are unaffected, because they only use LOCAL_IP within a node.

**Decode node (D):**

| window | running | gen tok/s | per-seq tok/s | mean ITL | ITL distribution | KV usage |
|---|---|---|---|---|---|---|
| 11:12 (60 s) | 127 | 6.38k | 50 | 19.7 ms | 100% in 10–25 ms | 9% |
| 11:32 (60 s) | 145 | 6.79k | 47 | 22.2 ms | 100% in 10–25 ms | 17% |
| 11:10–11:35 (log) | 133 | 6.35k | 47.6 | | | 17% |
| 11:35–11:59 (log) | 162 | 6.81k | 42.0 | | | 24% |
| 12:01 (60 s, after the weight update) | 187 | 7.63k | 40.8 | 24.2 ms | 91.5% ≤ 25 ms, 8.5% 25–50 ms, none > 50 ms | 21–25% |

- **The lockstep stall is gone.** No token takes more than 50 ms, against 100–450 ms stalls on DP8 colocated.
- **Per-sequence speed** is 41–51 tok/s, against ~21 for DP8 and ~30 for TP8 (mean ITL 33 ms).
- **D's own local compute is ~6 tok/s.** That is just the 1 recomputed token per turn for Mamba.
- **D's prefix cache holds the conversation.** Of D's prompt tokens per turn, ~93% are local cache hits, and only the new tokens (tool output) arrive over NIXL: 11–22k tok/s.

**Prefill node (P):**

| window | computed tok/s | cached tok/s | hit rate | running / waiting | mean computed per request | prefill time per request | TTFT on P |
|---|---|---|---|---|---|---|---|
| 11:12 | 20.8k | 105.7k | 84% | 5 / 1 | 3.4k | 0.55 s | 0.75 s |
| 11:32 | 15.3k | 129.9k | 89% | 2 / 0 | 3.9k | 0.51 s | 0.68 s |
| 11:10–11:35 (log) | 17.4k | | | 1.8 / 0.2 | | | |
| 11:35–11:59 (log) | 19.4k | | | 2.1 / 0.1 | | | |
| 12:01 | 26.2k | 129.9k | 83% | 1 / 0 | 4.2k | 0.89 s | |

- **P was never saturated.** Its waiting queue was ≈ 0.
- **Estimate of Π_P.** Prefill iterations ≈ requests: 226 iterations for 233 requests, so each request is a single ~4k-token step on one rank, taking ~0.5 s under DP lockstep at an average prompt length of ~37k. Eight ranks doing that concurrently gives **Π_P ≈ 8 × 4.06k / 0.51 s ≈ 64k tok/s**. This matches the §1.1 estimate of 60–75k. It was not measured at saturation.

**r_PD** (P computed per D generated token):
- 2.75 over 11:10–11:35, 2.86 over 11:35–11:59, and 3.4 in the minute right after the weight update (fresh rollouts).
- **Use ~2.85.**
- The share of it that D actually needs, external KV tokens per generated token, is 1.7–2.8. P's re-prefill of D's previous-turn output (about r_PD − r) is ≈ 0.6–1.0.

**KV transfer (NIXL, decode side).**
- 6,901 transfers over 11:10–11:35 and 7,227 over 11:35–11:59, i.e. 4.6–5 per second.
- Average transfer **21–24 ms**, mean p90 22–25 ms, **~203 MB each** (168 MB Mamba state plus KV), ~8.5–10 GB/s per transfer, ~1 GB/s per node pair in total.
- **0 transfer failures** after the fix.
- D's TTFT is ~0.14 s. P's TTFT plus transfer plus D's TTFT adds ≈ 0.8–1.0 s per turn.

**Training.**
- **Step 1 took 45m55s**, against 1h16m for TP8 and 1h47m for DP8 under the same shakedown. Trainable 97/128, Turns 17.0.
- **The weight update (NCCL, inference_world_size = 16) reached all 8 P and all 8 D engines.** Every engine logged "Checkpoint reload completed" in ~10.5 s and returned `/update_weights` 200. Generation and KV transfers continued normally afterwards.
- **DegenerateTurn count before step 1: 94**, the same as TP8 (94) and below DP8 (131). There is no sign that the transferred Mamba/KV state is corrupt.

**P:D ratio for 8 inference nodes** (u_P = 0.8, Π_P ≈ 64k, r_PD ≈ 2.85):

| D operating point | g_D per D node | n_P/n_D | 8-node split | P utilisation | cluster gen tok/s |
|---|---|---|---|---|---|
| s ≈ 40 tok/s, 23 seqs/rank (as measured at 187 running) | 7.6k (measured) | 0.42 | **3P:5D** (2P:6D would put P at ~100%) | 56% | **~38k** |
| s ≈ 30 tok/s (TP8 parity), ~40–65 seqs/rank (extrapolated, T ≈ 33 ms) | ~10–14k | 0.55–0.78 | **4P:4D** | 4P:4D at 45–62%; 3P:5D would be at 74–104% and P-bound at ~54k | 40–56k |

For comparison, 8 colocated nodes give ~22k (DP8) and ~32k (TP8).

**Recommendation: 3P:5D** at the measured operating point (~40 tok/s per sequence, ~7.6k gen tok/s per D node), with P at ~56% utilisation. **Move to 4P:4D** if D is pushed to ~30 tok/s per sequence for more throughput per D node.
- The orchestrator must keep ~190+ rollouts in flight per D node, ~950+ in total. The shakedown only fills one D node.
- If the llm-d threshold (§1.3 A) brings r_PD down to ~1.8, **2P:6D** fits at ~65% P utilisation.
- Re-measure Π_P under saturation by pushing more load through one P before going below 3 P nodes.
