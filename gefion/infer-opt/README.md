# infer-opt: Nemotron-3 Super inference optimizations for vLLM 0.28 on DGX H100

Kernels, patches and tooling for the PRIME-RL inference nodes (DP8+EP8 bf16 per node). Results
and the reasoning behind each change are in section 12 of `../gefion-phase0-report.md`.
Nothing here edits vLLM or prime-rl: patches load from `sitecustomize.py` when this directory
is on `PYTHONPATH` and `INFER_OPT` names them. That covers every process vLLM spawns (API server,
engine cores, workers), because they inherit the environment.

## Use

```bash
export PYTHONPATH=/dcai/projects/iu_0145/rasmus/infer-opt${PYTHONPATH:+:$PYTHONPATH}
export INFER_OPT=ep_compact_moe,fast_generate
export VLLM_TUNED_CONFIG_FOLDER=/dcai/projects/iu_0145/rasmus/infer-opt/moe_configs
```

In a PRIME-RL config, the same three variables go in `[inference.env_vars]`, or in the sbatch
template. Later, the patches can move into `prime_rl.inference.patches` as ordinary
`monkey_patch_*` functions: each `install()` is self-contained.

## Patches (`INFER_OPT=...`)

| name | where it runs | what it does |
|---|---|---|
| `ep_compact_moe` | workers | Triton MoE under DP+EP for non-gated (relu²) experts: aligns only the local experts, applies relu² only to the rows those experts own, and sums only the local slots. It keeps vLLM's GEMM kernel and is bitwise-identical to stock at equal tile configs. Falls back to stock for anything else (LoRA, quantized, gated, no EP). |
| `fast_generate` | API server | `/inference/v1/generate` with top-k logprobs: sets `detokenize=False` and `flat_logprobs`, stores each request's logprob rows in one numpy buffer (stock keeps ~10^8 Python objects at RL shapes, and GC stalls the event loop for minutes), and serialises with orjson. The JSON has the same keys and values as stock. Skipped for requests with stop strings or streaming, and when routed experts are returned. |
| `fi_moe_chunk` | workers | Calls FlashInfer's CUTLASS MoE in token chunks (`INFER_OPT_FI_MOE_CHUNK`, default 4096). Under DP+EP allgather, online FP8 otherwise allocates workspace for the whole gathered prefill batch (14 GiB peak activation per GPU). Results are per-token independent and unchanged. Only affects the FlashInfer CUTLASS backend (FP8 on H100). |
| `nvtx` | workers | NVTX ranges around the eager step phases, LM head and sampler, for `nsys`. Profiling only. |

`moe_configs/` holds Triton tile configs for `E=64,N=2688` (Super's per-rank expert shape)
tuned on H100 for the compact path, which vLLM 0.28 does not ship. They are picked up through
`VLLM_TUNED_CONFIG_FOLDER`.

## Status (from section 12)

Single node, DP8+EP8 bf16, RL-shaped load (1–8K prompts, 8–32K outputs, 33 logprobs): stock does
~4.1k output tok/s, and `ep_compact_moe` + tuned configs + `fast_generate` + ReplaySSM + `gpu_memory_utilization=0.93`
do 12.4k. The config is `../phase0/smoke/inference_recommended.toml`.

| item | status |
|---|---|
| `ep_compact_moe` + `moe_configs/` | ready: bitwise-equal to stock GEMMs at equal tiles; decode within run-to-run noise |
| `fast_generate` | ready: JSON checked equal to stock `_create_tokens_logprobs` on CPU, and run under the RL load |
| `use_replayssm = true` (vLLM option) | ready, monitor: mismatch k3 +9% over 8K tokens; no KV connectors or spec decoding |
| `mamba_ssm_cache_dtype = "float16"` | experimental: +44% at RL shapes (memory), but 8K–32K parity not measured yet |
| MTP k=1 + prefix caching on + `max_num_batched_tokens = 4176` | ready for correctness: logprobs as exact as no-MTP; 1.88 tokens/step; memory-bound in bf16 (7.8k on the math load). FP8 is the intended way to pay for its Mamba state |
| MTP with prefix caching on and `max_num_batched_tokens` < 4176 | **do not use**: vLLM 0.28 corrupts decoding at positions [budget, 4176) |
| DP8 online FP8 + `fi_moe_chunk` | experimental: 23.0k on the math load, but logprob mismatch 4.2x bf16; FP8+MTP hits a FlashInfer DeepGEMM JIT failure |
| `nvtx` | profiling only |

## Tools

- `bench/rl_load.py`: closed-loop, RL-shaped load against `/inference/v1/generate`. It sends what the
  orchestrator sends (token ids, 33 logprobs, GRPO groups of 8 that share a prompt), with
  log-uniform prompt and output lengths, and measures throughput from `/metrics` over a fixed window.
- `bench/parity.py`: logprob parity between servers (seeded decode, teacher-forced prefill, and
  decode-vs-prefill mismatch, the k3 KL estimate).
- `bench/make_config.py`: the PRIME-RL inference config for a variant (the DP8+EP8 baseline plus overrides).
- `tools/moe_tune.py`: stock vs compact MoE microbenchmark and the Triton config tuner (`bench` and `tune` modes).
- `analysis/nsys_breakdown.py`: per-decode-step GPU time by component, from an `nsys` sqlite export.
- `analysis/engine_log_tput.py`: node throughput integrated from vLLM's per-engine stats lines (works when `/metrics` is unresponsive).
- `bench/make_i3math_prompts.py`, `bench/data/i3math_prompts.json`: tokenized i3-math prompts for realistic loads (`rl_load.py --prompts-file`) and parity (`parity.py --prompts-file`).
- `analysis/mismatch_bursts.py`: finds runs of tokens decoded from a corrupted state (sampler logprob far above teacher-forced).
- `analysis/pyspy_summary.py`: summary of a py-spy collapsed-stack file. py-spy cannot attach to the server on Gefion (`ptrace_scope=1`).
- `jobs/*.sbatch`: the single-node jobs behind section 12 (`lib.sh` has the shared helpers).
