# FP8 + MTP: where the "1.3k tok/s" came from, and MTP on PD decode nodes (vLLM 0.28)

Date: 2026-10-29. Jobs run for this note: 533117 (`jobs/mtppd_s.sbatch`), 533209 (`jobs/mtppd_t.sbatch`).

## 1. The 1.3k figure

Source: `../phase0/smoke/inference_recommended.toml` line 12 ("FP8 + MTP reached only 1.3k tok/s on the
same load"). The numbers behind it are the multi-turn load (`rl_load.py --turns 8`, 1408 closed-loop
clients, 360 s warmup, 240 s window) in:

| job | variant | gen tok/s | prompt tok/s (incl. cached) | local prefix hits | max_num_seqs/rank |
|---|---|---|---|---|---|
| 531400 | FP8+MTP | 1056 | 49.6k | 0.5% | 96 |
| 531400 | FP8 | 5310 | 13.9k | 36% | 176 |
| 531424 | FP8+MTP | 1277 | 46.4k | 0.3% | 96 |
| 531424 | FP8 | 5396 | 7.8k | 47% | 176 |
| 531430 | FP8+MTP + `mtp_keep_last_block` | 1321 | 63.5k | 30% (+~50% host tier) | 96 |

`prompt_tokens_total` counts every prompt token, cached or not (`v1/metrics/stats.py` PromptTokenStats.total).

### What the engine logs show (`analysis`: per-10 s sums over the 8 "Engine NNN:" lines)

`results/fp8mtp-q-531424/fp8_mtp1.log`:
- 22:57:50–22:59:40: decode phase, 768 running (96/rank), **~24k gen tok/s**, prompt ~0.2k/s.
- from 23:02: prefill-bound, ~650–700 running + ~700 waiting, **prompt ~47k tok/s at 0.3% hits, gen ~1.2k**.
  The measured window (23:03:30–23:07:30) lies entirely in this phase.

`results/fp8mtp-q-531424/fp8_nomtp.log`: 1330–1369 running (176/rank) throughout, gen 6–14k, prompt
~9k/s at 47% hits. It never reaches a prefill-bound phase within its window.

Whole-run totals (all 10 s buckets with 8 engines): MTP 4.0M generated tokens in ~550 s
(~7.3k/s) vs no-MTP 5.6M in ~650 s (~8.6k/s); MTP processed 17.0M prompt tokens vs 5.4M. Job 531430
(MTP + keep-last-block): 4.8M in ~500 s (~9.6k/s). Client side, the MTP runs completed more turns
(3719 vs 3168 in 531424) in less time.

### Root cause

Not a slower MTP decode step. The 240 s window compared two different phases of a non-stationary load:

1. **MTP lost the prefix cache on re-sent turns.** vLLM 0.28 drops the last matching block of every
   prefix hit for EAGLE/MTP and backs the Mamba align checkpoint off one block (see
   `infer_opt/mtp_keep_last_block.py`). With 4176-token blocks a turn reused nothing below ~8.3k
   tokens of context: 0.3–0.5% hits vs 36–47% without MTP. So every turn re-prefilled its whole history.
2. **The MTP server reached the long-context turns sooner.** At 96 seqs/rank it decoded each
   sequence ~1.75x faster (31 vs 18 tok/s/seq), so its 1408 clients finished turn 1 during warmup and
   the window landed on turns 2+ (contexts of 10–65k tokens). The no-MTP server ran 1369 sequences
   at once at 18 tok/s each, and its window was still mostly turn-1 decode.
3. **Prefill steps stall all ranks (the 533102 finding).** Once the engine is prefill-bound, each
   step is an ~8.7k-token chunk taking ~0.7–1.3 s, and through the MoE allgather every DP rank's
   decode waits for it. Decode then gets ~1.79 tokens per ~1.3 s per sequence, which gives the ~1.2k tok/s.
   The lockstep amplifies (1) and (2) but is not MTP-specific.
4. Lower capacity: MTP needs 1 + k Mamba state blocks per request (`MambaSpec.num_speculative_blocks`),
   so it was capped at 96 seqs/rank vs 176. Measured below: at equal concurrency MTP uses 2x the cache
   (kv_usage 0.98 vs 0.49 at 96/rank, prefix caching off).

In job 531430 the windowed counters also cover periods where the engine was nearly idle (e.g.
23:47:00, 13 running, 0 waiting) followed by bursts of ~1000 arrivals. That pattern suggests the
single-process client or API server was busy with ~1000 large logprob responses at once. This is a
likely harness artifact that this note does not verify.

### What is *not* the cause (measured in job 533117)

- **MTP prefill cost:** prefill-only (4–8k prompts, max_tokens 1, prefix caching off) runs at 93.7k
  prompt tok/s without MTP and 91.4k with MTP, so the drafter adds ~2.4% to prefill.
- **Drafter cost per decode step:** ~0. An MTP step with 16 seqs/rank (32 verified tokens) takes
  21.4 ms, against 21.2 ms for a no-MTP step with 32 seqs/rank. An MTP step costs what its token count costs.
- **CUDA graphs:** spec-decode shapes are captured. The MTP logs show `cudagraph_mode FULL_AND_PIECEWISE`,
  "Capturing CUDA graphs (decode, FULL)" with 26 sizes, and max_cudagraph_capture_size 384
  (= 96 × 2 × 2). Under DP the graph mode is synced to the minimum across ranks
  (`v1/worker/dp_utils.py`), so in a step where any rank prefills more than the graph size, every
  rank runs eager. That happens with and without MTP.

## 2. Decode-only step time (job 533117)

FP8 DP8+EP8, allgather_reducescatter, prefix caching off, synthetic 2–4k prompts, ignore_eos, 32k
outputs. Nothing finishes in the 90 s window, so the window has no prefill. Step time per rank =
tokens_per_step × concurrency / gen_tok_s.

| seqs/rank | no-MTP gen tok/s | no-MTP step | MTP gen tok/s | MTP tokens/step | MTP step | gain at 1.79 tok/step |
|---|---|---|---|---|---|---|
| 16 | 7,344 | 17.4 ms | 11,756 | 1.962 | 21.4 ms | **1.46x** |
| 32 | 12,098 | 21.2 ms | 18,495 | 1.954 | 27.1 ms | 1.40x |
| 64 | 18,690 | 27.4 ms | 26,463 | 1.952 | 37.8 ms | 1.30x |
| 96 | 22,638 | 33.9 ms | 30,261 | 1.938 | 48.9 ms | 1.24x (MTP cache 98% full) |

Acceptance on synthetic prompts is inflated (95–96%), so the gain column uses the 1.79 tokens/step
measured on the real multi-turn load (job 531430). Math loads gave 1.86–1.88. The gain column is
`1.79 × t_noMTP / t_MTP`.

EQUAL_MEMORY_PLACEHOLDER

## 3. MTP on PD decode nodes in vLLM 0.28

Supported in vLLM 0.28, but **both roles must run the same `speculative_config`**:

- `nixl/metadata.py`: NIXL_CONNECTOR_VERSION 7 ("Validate EAGLE/MTP speculative configuration
  compatibility") puts `speculative_config` (method, draft model, kv dtype, parallel_drafting) in the
  handshake compatibility hash. A prefill node without MTP and a decode node with MTP fail the handshake.
- The draft layer's KV is transferred like any other layer. `gpu_model_runner.initialize_kv_cache`
  registers every layer in the kv_cache_config, the MTP layer included, with the connector. The
  Nemotron-H MTP layer is `mtp_hybrid_override_pattern = "*E"` (one attention + one MoE, no Mamba), so
  it adds one attention layer to the transferred KV. The prefill side computes it, because the
  EAGLE/MTP proposer runs after every step, prefill included.
- Spec decode with kv_transfer on the decode side is explicitly handled:
  - the scheduler defers lookahead slots during async KV load ("Allocate speculative lookahead slots
    later to avoid mismatching local and remote block counts", `v1/core/sched/scheduler.py` ~1004);
  - the NIXL scheduler clips the Mamba groups' speculative scratch slots from transfers
    (`nixl/base_scheduler.py` `get_exchange_clipped_blocks`, `_ssm_spec_blocks`);
  - for Mamba, P computes N-1 tokens and D recomputes the last token (`_truncate_mamba_request_for_prefill`,
    `_get_remote_prefill_token_count`).
- Known wart: on P, the drafter's KV at position N-2 is written with P's sampled token, not prompt
  token N-1, because P's prompt is truncated. Only the drafter reads that entry, so outputs and
  logprobs are unchanged and acceptance drops slightly at the start of a request.

prime-rl wiring (`src/prime_rl/templates/multi_node_rl.sbatch.j2` PD branch; `_launch_rank.sh.j2`):
`deployment.prefill_vllm_overrides` / `decode_vllm_overrides` are JSON-merged into `[inference.vllm]`
per rank (`--vllm @ vllm-overrides.json`), and the vLLM config allows extra keys. Because both sides
need it, put `speculative_config` in the shared `[inference.vllm]` (or in both override dicts):

```toml
[inference.vllm]
speculative_config = { method = "mtp", num_speculative_tokens = 1 }

[deployment]   # decode side
decode_vllm_overrides = { enable_prefix_caching = false, mamba_cache_mode = "none", max_num_seqs = 96 }
```

Further points for this setup:
- Patches: `dg_jit_per_proc` (FP8 FlashInfer DeepGEMM JIT race) is needed. `mtp_keep_last_block`
  only matters where prefix caching is on, i.e. on the prefill side for multi-turn reuse.
- Keep `max_num_batched_tokens >= 4176` wherever MTP and prefix caching are both on (the 0.28
  corruption bug). With prefix caching off on D the bug does not apply.
- The PD template's decode role uses `all2all_backend = deepep_low_latency` and `FULL_DECODE_ONLY`
  graphs. With MTP, each rank sends 2 tokens per sequence per step, so the DeepEP-LL dispatch buffer
  (moe.max_num_tokens) must cover 2 × max_num_seqs. Spec-decode FULL graphs require the attention
  backends to report UNIFORM_BATCH. Both worked under allgather (job 531430); they are not tested under
  DeepEP-LL + FP8 CUTLASS.
- Memory: MTP doubles the Mamba state blocks per request on D. Size max_num_seqs to ~half of the no-MTP value.
