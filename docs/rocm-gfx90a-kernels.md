# ROCm gfx90a inference kernels

PRIME-RL includes opt-in inference kernels for GLM-5.2 and Nemotron 3 Super
on AMD CDNA2 (`gfx90a`). They are deliberately capability-gated.
Narrow wrapper paths retain and call the stock vLLM operation outside their
measured contract. The representation-changing INT8 path cannot retain fallback
FP8 weights, so it
validates its complete contract and fails before mutating a layer.

These kernels are specialized integration code, not general ROCm defaults.
Keep the stock vLLM path for other models or deployment shapes.

## Support matrix

| Path | Supported contract | Opt-in |
|---|---|---|
| Dense INT8 | GLM-5.2 OCP E4M3FN block-FP8 checkpoint; BF16 activations; the seven `(N, K)` projection shapes and `M <= 2048` listed in the tuning metadata | `PRIME_ROCM_CDNA2_INT8=1` and `quantization = "prime_cdna2_int8"` |
| Routed-expert INT8 MoE | GLM-5.2 routed experts only: 256 global / 16 local experts, top-8, hidden size 6144, intermediate size 2048, gated SiLU, BF16, no expert bias, and global-EP `M <= 4096` | Same custom quantization opt-in |
| FP32-output router gate | Exact GLM-5.2 BF16 `[M, 6144]` activation and BF16 `[256, 6144]` weight, `1 <= M <= 2048`; deterministic FP32 split-K reduction | `PRIME_ROCM_GFX90A_GATE_LINEAR=1` with `inference.enable_fp32_router_logits = true` |
| Sparse-indexer decode | One-token OCP E4M3FN paged MQA, 32 heads, head dimension 128, and cache block size 1, 16, 32, or 64 | `PRIME_ROCM_SPARSE_INDEXER_FALLBACK=1` and `PRIME_ROCM_SPARSE_INDEXER_TRITON_DECODE=1` |
| Sparse MLA | GLM BF16 sparse MLA with 512 NoPE + 64 RoPE dimensions, a 512-wide value, a static configured top-k of 2048, and no valid ragged row longer than 2048 | `PRIME_ROCM_SPARSE_MLA_TRITON=1` |
| Hybrid long-context attention | Nemotron 3 Super BF16 TP8: four local query heads, one local KV head, head size 128, physical page 2048, and sequence length 4,096 through 262,144 | `PRIME_ROCM_HYBRID_PAGED_ATTN=1` and `mamba_block_size = 1024` |
| Dynamic INT8 KV attention | Nemotron 3 Super BF16 TP8 with vLLM NHD `int8_per_token_head` cache, inline FP32 scales, the same head/page contract, decode through 262,144, and cached-prefix prefill chunks through 32,768 | `kv_cache_dtype = "int8_per_token_head"`, `PRIME_ROCM_NEMOTRON_INT8_PAGED_ATTN=1`, and the pinned prefix source hook |
| One-group biased-sigmoid router | Nemotron 3 Super FP32 512-expert sigmoid router, one group, top-22 experts, normalized weights, and at most 32,768 scheduler tokens | `PRIME_ROCM_NEMOTRON_GROUPED_TOPK=1` plus both pinned router source hooks |
| Routed-expert ReLU-squared | Nemotron 3 Super BF16 contiguous `[M, 2688]`, `5,632 <= M <= 720,896` | `PRIME_ROCM_NEMOTRON_RELU2=1` |
| Large-prefill MoE tiles | Nemotron 3 Super TP8/EP8 expert shapes `(64, 2688, 1024)` and `(64, 1024, 2688)`, top-22, and `4,096 <= M <= 32,768` | `PRIME_ROCM_NEMOTRON_MOE_CONFIG=1` |

All paths require a ROCm PyTorch build, a `gfx90a` GPU, and the exact
vLLM distribution described below. The INT8 path is currently specific to
`zai-org/GLM-5.2-FP8`'s model contract; a similar architecture or another
GLM release is not implicitly supported.

## Exact runtime and topology

The private vLLM symbols used by these integrations are pinned to distribution
`0.24.0+lumi_aif_gfx90a_ee0da84`, corresponding to vLLM commit
`ee0da84ab9e04ac7610e28580af62c365e898389`. Startup validates the
distribution and every imported private-function signature before binding the
symbols. The measured software stack is PyTorch `2.11.0+rocm7.1` and Triton
`3.7.1`.

For GLM MLA, validate the canonical `qk_head_dim`, `qk_nope_head_dim`, and
`qk_rope_head_dim` fields rather than the generic `head_dim` alias.
Transformers v5.4 maps that alias to `qk_rope_head_dim`, and vLLM corrects the
loaded value to 64 even though the NoPE dimension is 192.

The routed MoE path requires this exact deployment:

- server tensor parallelism 8 and data parallelism 2;
- expert parallelism 16, yielding 16 local experts per rank;
- the vLLM MoE-local factors `tp_size=1`, `dp_size=2`, `pcp_size=1`, and
  `sp_size=8`;
- `all2all_backend = "allgather_reducescatter"`;
- EPLB disabled; and
- at most 2048 scheduler tokens per DP replica.

The DP2 all-gather makes the largest production MoE cell `M=4096`, even though
each local scheduler is capped at 2048. DP1, DP4, a different EP factor, PCP,
or SP factor is a different kernel workload and is rejected.

A minimal standalone PRIME-RL configuration contains:

```toml
enable_expert_parallel = true
all2all_backend = "allgather_reducescatter"
enable_eplb = false
enable_dbo = false
use_deep_gemm = false

[model]
name = "zai-org/GLM-5.2-FP8"
dtype = "bfloat16"

[parallel]
tp = 8
dp = 2

[vllm_extra]
quantization = "prime_cdna2_int8"
moe_backend = "triton"
max_num_batched_tokens = 2048

[env_vars]
PRIME_ROCM_CDNA2_INT8 = "1"
PRIME_ROCM_CDNA2_INT8_AUDIT = "1"
```

For an RL config, nest these tables under `inference`. This INT8 path is
static-only, so it is suitable for a frozen rollout teacher or evaluation
server, not a server receiving base-weight updates during training.

## Nemotron 3 Super long-context deployment

The final measured Nemotron deployment is one MI250X node with TP8,
expert parallelism disabled (TP-MoE), BF16 model weights,
`max_num_batched_tokens = 32768`, and a 2,048-token hybrid attention page.
Set `mamba_block_size = 1024`; vLLM aligns the attention page to 2,048 while
preserving the hybrid cache capacity. A minimal kernel selection is:

```toml
enable_expert_parallel = false
all2all_backend = "allgather_reducescatter"
enable_eplb = false
enable_dbo = false
use_deep_gemm = false
enable_fp32_router_logits = true

[model]
name = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"
dtype = "bfloat16"
max_model_len = 131072 # or 262144

[parallel]
tp = 8
dp = 1

[vllm_extra]
max_num_batched_tokens = 32768
max_num_seqs = 64
mamba_block_size = 1024

[env_vars]
PRIME_ROCM_HYBRID_PAGED_ATTN = "1"
PRIME_ROCM_NEMOTRON_GROUPED_TOPK = "1"
PRIME_ROCM_NEMOTRON_RELU2 = "1"
PRIME_ROCM_NEMOTRON_MOE_CONFIG = "1"
```

Decode attention dispatches only for sequence lengths from 4,096 through
262,144. Cached-prefix prefill dispatches for chunks of at most 32,768 tokens
and total sequence lengths through 262,144; an uncached full prefill remains
on the stock vLLM path. The router, activation, and MoE selectors likewise fall
back outside the shapes in the support matrix.

For the capacity-oriented dynamic INT8 cache path, select vLLM's Triton
attention backend and replace the BF16 attention opt-in:

```toml
[vllm_extra]
kv_cache_dtype = "int8_per_token_head"

[env_vars]
VLLM_ATTENTION_BACKEND = "TRITON_ATTN"
VLLM_TUNED_CONFIG_FOLDER = "/workdir/src/prime_rl/inference/vllm/tuned_configs/gfx90a"
PRIME_ROCM_HYBRID_PAGED_ATTN = "0"
PRIME_ROCM_NEMOTRON_INT8_PAGED_ATTN = "1"
```

This path uses vLLM's per-token, per-head dynamic quantization and inline FP32
scale representation. PRIME-RL replaces the generic decode kernel and adds
per-token scale loads to the pinned prefix-prefill kernel. The selected decode
launch uses 1,024-token partitions, 32-token tiles, four warps, and one stage;
the cached-prefix prefill launch uses `BLOCK_M=256`, `BLOCK_N=64`, and eight
warps. Apply
[`vllm-ee0da84-nemotron-int8-prefix-hook.patch`](vllm-ee0da84-nemotron-int8-prefix-hook.patch)
to the pinned vLLM source. The installer rejects a missing or changed hook.
The BF16 and INT8 attention opt-ins are alternatives, not a combined mode.

The Nemotron checkpoint declares one router group. The vLLM router factory
therefore reduces the grouped router to `FusedTopKBiasRouter`; optimizing only
`GroupedTopKRouter._compute_routing` does not reach production. Both the factory
choice and fused biased-sigmoid implementation are inside the compiled source
closure. Apply
[`vllm-ee0da84-nemotron-grouped-topk-hook.patch`](vllm-ee0da84-nemotron-grouped-topk-hook.patch)
and
[`vllm-ee0da84-nemotron-fused-topk-bias-hook.patch`](vllm-ee0da84-nemotron-fused-topk-bias-hook.patch)
to the pinned vLLM source. The grouped hook selects the fused router only for
the exact one-group contract. The fused hook then calls the pinned
`vllm_topk_sigmoid` primitive directly and retains stock behavior for every
other routing contract.

A changed hook or kernel source needs a fresh `VLLM_CACHE_ROOT`; the AOT cache
can otherwise reuse a graph compiled before the source seam existed. For an
experiment, bind the patched module over the matching file in the pinned image.
For a production runtime, rebuild and verify the image with the patch applied;
the current shared SIF does not contain this source change.

The final 128k concurrency-8 run completed at 25.298 output tokens/s, 2.56x
the 9.863 tokens/s stock baseline, with zero failures. Mean TTFT was 75.27
seconds and mean TPOT was 168.74 ms. The 256k boundary run completed a
261,632-token prompt plus 512 decode tokens in 50.76 seconds at 10.086 output
tokens/s, also with zero failures. The complete profiles and stopping analysis
are in
`/scratch/project_465002751/rasmus/artifacts/profiling/nemotron3-super-128k-20260725/analysis.md`.

The final dynamic-INT8 runtime stores about 1.995 million KV tokens per rank:
15.22 nominal complete 128k contexts or 7.61 complete 256k contexts. This is a
lossy cache representation: kernel-level numerical checks and throughput runs
do not replace downstream task evaluation before making it a deployment
default.

The repository also carries the measured MI250X selective-state-update table
under `src/prime_rl/inference/vllm/tuned_configs/gfx90a`. Point
`VLLM_TUNED_CONFIG_FOLDER` at that directory. At the c8 production cell
(effective batch 128), the selected `(BLOCK_SIZE_M, num_warps) = (8, 2)`
launch is 1.43x faster than the fallback heuristic in isolation. In the final
full-model trace it reduced selective-state-update time from 6.566 to 6.059 ms
over 16 decode steps (7.7%) and step time from 37.038 to 36.868 ms (0.46%).

## Runtime representation and numerical contract

The checkpoint remains serialized OCP E4M3FN with `[128, 128]` block scales.
After the initial load, eligible dense and routed-expert weights are
dequantized and requantized once to signed INT8 with `[1, 32]` groups and FP16
scales. BF16 activations are dynamically quantized to matching 32-element
groups. Matmuls then use native gfx90a INT8 MFMA instructions.

This conversion is intentionally lossy. Kernel acceptance uses independent
dequantized checkpoint references and the following envelopes:

- dense output: relative L2 at most `0.025` and cosine similarity at least
  `0.999`;
- routed MoE output relative to the accepted stock-FP8 path: relative L2 at
  most `0.06` and cosine similarity at least `0.999`; and
- the production MoE API must agree with its preallocated kernel path within
  relative L2 `0.002` and cosine similarity `0.999`.

These are kernel gates, not a model-quality guarantee. A release still needs
paired logits or token-ID checks and task evaluation against the same
checkpoint on the stock runtime.

The conversion is one-way for the lifetime of a worker. Reload mode must be
`none`; checkpoint reload, kernel reload, `/update_weights`, and PRIME-RL
base-weight broadcast are unsupported. Restart workers from the source
checkpoint to install new base weights. The custom quantization config also
keeps `self_attn.kv_b_proj` on stock FP8 because vLLM's MLA post-load
processing requires the original FP8 method.

## Routed and shared experts

The fused INT8 MoE kernel computes routed experts only. On the supported
GLM-5.2 integration, the single shared expert remains a separate set of dense
projections and may use the dense INT8 method. Because the quantization method
reports `supports_internal_mk=False`, vLLM's MoE runner executes the shared
expert externally, using its no-overlap or multi-stream schedule, and adds that
result separately. The routed method accepts either both shared-expert
arguments or neither and rejects mismatched presence; it never silently
absorbs shared-expert work into the routed kernel.

## Environment compatibility

The supported INT8 environment uses the custom quantization name,
`moe_backend = "triton"`, BF16 activations, legacy
all-gather/reduce-scatter dispatch, no EPLB, no DBO, and no DeepGEMM. Other
quantization layouts, static activation scales, FP16 activations, biased
experts, alternative all-to-all backends, or shapes absent from the metadata
are outside the measured contract.

Sparse selector rules are independent of the INT8 selector:

- `PRIME_ROCM_SPARSE_MLA_TRITON` and
  `PRIME_ROCM_SPARSE_MLA_REFERENCE_FALLBACK` are mutually exclusive.
- `PRIME_ROCM_SPARSE_MLA_PREFILL_Q_CHUNK` applies only to the reference
  fallback, not the Triton kernel.
- `PRIME_ROCM_SPARSE_INDEXER_TRITON_DECODE` is incompatible with the
  bulk-host-context mode. The Triton decode path keeps context lengths and
  block tables on the GPU and is graph compatible; the bulk-host-context
  fallback performs host copies and is not.
- AITER sparse MLA targets newer AMD hardware. These opt-ins replace its
  gfx90a-incompatible backend while retaining vLLM's paged and ragged metadata
  contracts.
- Missing AITER tuning metadata on gfx90a is a capability miss, not a table
  entry to guess. Continue through the validated stock or reference fallback
  unless the user selected an exact-contract opt-in that must fail closed.

## Absorbed MLA BMM promotion

The standalone BF16 absorbed-MLA kernels cover GLM-5.2's exact TP8 Q-absorb
and V-up views. Production dispatch is limited to the two measured token
counts, `M=16` and `M=32`; every other token count or tensor layout continues
through vLLM's stock `torch.bmm` path. LUMI job `20191714` selected:

| Operation | M | `(BLOCK_M, BLOCK_N, BLOCK_K, warps, stages, waves/EU)` | Speedup |
|---|---:|---|---:|
| Q absorb | 16 | `(16, 64, 64, 4, 2, 1)` | 1.125x |
| Q absorb | 32 | `(16, 64, 64, 4, 1, 2)` | 1.159x |
| V up | 16 | `(16, 64, 64, 4, 2, 2)` | 1.071x |
| V up | 32 | `(16, 64, 64, 4, 2, 1)` | 1.100x |

The pinned vLLM already exposes `_v_up_proj`, but keeps Q absorption inline in
`MLAAttention.forward_impl`. The PRIME-RL installer therefore requires a
narrow `_q_absorb(self, q_nope, out)` seam by default and fails before
mutating vLLM when it is absent. The minimal behavior-preserving upstream
change is recorded in
[`vllm-ee0da84-mla-q-absorb-hook.patch`](vllm-ee0da84-mla-q-absorb-hook.patch).
It only extracts the existing `torch.bmm` call into a method; PRIME-RL does
not intercept global `torch.bmm` dispatch or duplicate `forward_impl`.
V-up-only installation remains available explicitly with
`install_glm52_gfx90a_mla_bmm(require_q_absorb=False)`.

Use `PRIME_ROCM_CDNA2_INT8_AUDIT=1` during validation. Each worker emits JSON
records for every completed dense or expert transcode and every intentional
stock-FP8 exemption. Treat missing, duplicate, or unexpected records as a
failed model-load census.

## Release evidence and source closure

Microbenchmarks select candidates; they do not release a kernel. The final
performance gate must load the complete model, pass paired semantic and
quality checks, and run fixed prefix-friendly and random workloads through a
global-concurrency sweep to saturation. Report completed requests, requested
and emitted output tokens, latency, and the single-router service-aggregate
output tokens/s.

Always report the resource denominator beside throughput. If a service
allocates `N` GCDs, cost-normalized throughput is
`output_tokens_per_second_per_GCD = service_output_tokens_per_second / N`.
Do not multiply a router aggregate by its replica count or confuse combined
prompt and output throughput with output throughput. Before accepting a
target, compare
the intended decode batch with a roofline estimate covering bytes and FLOPs
per step, active-expert HBM traffic, and collective traffic. A target above
that bound is a denominator or model-arithmetic problem, not a kernel-tuning
goal.

Audit the installed model, not merely the installer. Require one successful
post-`process_weights_after_loading` record per expected layer per worker and
reject missing, duplicate, or unexpected layer ids. Do not emit an installed
marker when a capability check or hook installation is rejected.
Treat shared redirected worker stdout as a byte stream rather than an atomic
line protocol: concurrent worker prefixes and plain-text markers can
interleave. Use structured JSON records carrying rank, PID, layer, and event
identity for the authoritative per-worker census. Plain-text marker counts
may be checked in aggregate, but a missing prefix on one interleaved line is
not by itself evidence that installation failed.

Freeze the complete transitive worker source closure for every release job.
Include package `__init__.py` files, worker bootstrap and patch modules,
`padded_input_scrub.py`, kernel and integration modules, tuning metadata, and
upstream source patches. Hash before submission and verify in the allocation;
place submit scripts and result manifests outside any reviewed root whose
digest they embed, avoiding self-referential manifests. Hashes do not cover
mode bits, so separately require every submitted or invoked `.sh` and
`.sbatch` file to be executable, run `bash -n` on them, and dry-run the
rendered launch before `sbatch`.

## Reproducing or extending tuning

The machine-readable tuning record is
[`gfx90a_int8_tuning.json`](../src/prime_rl/inference/vllm/kernels/gfx90a_int8_tuning.json).
It contains the exact runtime, model and topology contract, candidate spaces,
timing protocol, correctness gates, and selected dense/MoE launch
configurations. The Python dispatch tables must resolve to exactly the same
launch dataclasses as its selected mappings.

To retune a shape:

1. Use one otherwise idle gfx90a GCD, the pinned software stack, a fresh
   Triton cache, and production weight/activation layouts.
2. Build input values from a fixed seed. For MoE, use unique top-8 routes per
   token; for the DP2 `M=4096` case, balance 32,768 routes across 256 global
   experts so exactly 2048 routes reach the 16 local experts.
3. Sweep every candidate in the metadata with `triton.testing.do_bench`.
   Record median and p95, then remeasure the eight fastest candidates three
   times with 50 ms warmup and 250 ms measurement windows. Select by the
   median of those medians.
4. Compare the independent source reference, stock FP8, the selected raw
   kernel, and the production API. Apply the numerical envelopes above and
   require the production API to outperform stock FP8.
5. Capture and replay the warmed production call in a GPU graph. Inspect the
   generated AMDGCN assembly for native INT8 MFMA and the absence of FP8
   decode in the matmul loop.
6. Update the JSON and Python table together, rerun the complete current-source
   kernel gate, then run a full-model load census, paired-quality gate, and
   end-to-end throughput profile.

Nearest-bucket dispatch is permitted only within the listed maximum token
bound. A new projection shape, topology, dtype, or larger token count requires
its own measured entries and release gates.

The router-gate candidate has a separate unsubmitted correctness, graph, and
tuning harness under
`artifacts/training/glm52-gate-linear-gfx90a-20260724-v1`. Its seed dispatch
is not a measured production configuration: run that harness, replace the
seeds with recorded winners, then include the gate in the same full-model
quality and throughput release sequence.
