# Does FP8 / PD inference cause the degenerate turns? (job 534489, 2026-10-31)

Scope: is the online `fp8_per_block` + TP8/EP8 + prefill/decode (NIXL) inference stack degrading
sampling enough to explain 41–48% of SWE/terminal rollouts ending in a degenerate turn?
This was done without GPUs: vLLM 0.28 source reading, an analysis of the traces of all 12 Gefion runs
with traces, and a literature search. A GPU A/B is ready to run but has **not** been submitted.

## Verdict

| suspect | likelihood it is a main cause | basis |
|---|---|---|
| PD / NIXL KV and Mamba-state transfer | **very unlikely (<5%)** | Non-PD FP8 runs of the same checkpoint degenerate at the same rate, step 1, policy 0 (table 2). No turn-start signature. |
| Online FP8 (`fp8_per_block`) | **unlikely as the main cause (~15%). Plausible as a minor contributor (~30% that it adds ≥20% relative).** Needs the A/B. | The degeneration comes from a fat, smooth tail in the T=1 distribution. That tail is already there at step 1 on the start weights (1.3–1.9% of sampled tokens in *healthy* turns have p < 4.5e-5). FP8's measured logit perturbation (~0.03 nats mean, k3 ≈ 5e-3) is one to two orders of magnitude too small to create it. FP8 does quantize layers that NVIDIA keeps in bf16 (§1), and the mismatch is not corrected in the loss (`importance_sampling = "none"`). |
| What the traces point to instead | – | Sampling at T=1 with no tail cut-off, on long agentic contexts with long turns. The degenerate rate rises 6× from <4k to >32k context, and 20× from <1k to 4–16k-token turns. The per-turn cap is 16384 here against 8192 on LUMI, and the checkpoint is r6 rather than r5. The LUMI comparison is confounded by all of this (§2.5). |

What would change this verdict: in the A/B, bf16 rescoring gives the sampled tail tokens markedly
lower probability than FP8 does (mean B−F < −0.3 nats for sampler logprob < −5, or FP8/bf16 tail mass
> 1.2), or bf16 regeneration has a lower degenerate rate with a paired CI that excludes 0.

## 1. What `fp8_per_block` quantizes in vLLM 0.28 for Nemotron-3 Super

Sources: `vllm/config/quantization.py` (`_ONLINE_SHORTHANDS`), `layers/quantization/online/{base,fp8}.py`,
`models/nemotron_h.py`, `layers/mamba/mamba_mixer2.py`, `layers/vocab_parallel_embedding.py`.

- **Scheme.** W8A8 FP8 e4m3:
  - Weights use static 128×128 block scales. They are computed at load time from the bf16 weights, and again after every trainer broadcast (`monkey_patch_online_fp8_parameter_cast`).
  - Activations use dynamic per-token, per-128-channel scales (`GroupShape(1,128)`). Accumulation is fp32.
  - The shorthand quantizes **every `LinearBase` and the routed experts**, with an empty ignore list. There is no calibration.
- **Quantized in Super** (88 layers: 40 Mamba, 40 MoE, 8 attention):
  - routed experts (512, top-22, relu²): FlashInfer CUTLASS block-FP8 on H100, plus `fi_moe_chunk`
  - shared expert `up_proj`/`down_proj`
  - **MoE latent projections `fc1_latent_proj`/`fc2_latent_proj`** (4096↔1024). Every routed contribution passes through these, and `routed_scaling_factor = 5.0` scales that branch.
  - **Mamba `in_proj`** (z, x, B, C, dt) and `out_proj`. The 16 dt rows per TP rank get their own 128-row block, so no scale is shared with the larger z/x rows.
  - **attention `qkv_proj`/`o_proj`** on all 8 attention layers, which are the model's only exact long-range retrieval path.
- **Not quantized:**
  - `lm_head`: `ParallelLMHead` gets `None` from the online config, so it falls back to `UnquantizedEmbeddingMethod`, and prime-rl runs it in fp32.
  - embeddings
  - router `GateLinear` (fp32 compute, fp32 logits)
  - Mamba `conv1d` (hard-coded `quant_config=None`)
  - SSM state (fp32 cache), A/D/dt-bias, norms
  - KV cache: bf16, since the online config returns no attention/KV method
- **Compared with NVIDIA's official FP8 Super checkpoint** (tech report arXiv 2604.12374, Table 6): NVIDIA keeps **attention QKV/O and the latent projections in bf16**, along with embeddings, output and conv1d. Only the experts and the Mamba GEMMs are FP8, with calibrated PTQ. Our online scheme also quantizes the two groups NVIDIA kept. The Nano report (2512.20848) likewise keeps attention and the Mamba layers next to it in bf16 because other choices lost accuracy.
- **Effect on logits and tails.** The output head is exact. FP8 error reaches the logits only through the final hidden state, as a roughly uniform additive logit perturbation. That perturbation changes tail probabilities multiplicatively by e^ε. With mean |ε| ≈ 0.03–0.1 nats, it moves the expected tail mass by well under 1% relative. This is why FP8 noise alone is a poor fit for the fat tail in §2.4. Errors that build up with context (the dt/decay path, attention over 64k+ tokens) are the plausible way FP8 *could* matter. The A/B splits Δ by context length for this reason.
- **Measured before this run (report §12.7, one node, math load):**
  - FP8's decode-vs-own-teacher-forcing mismatch was k3 5.5e-3, against 1.3e-3 for bf16 (4.2×). No corrupted sequences.
  - "Score FP8 rollouts on a bf16 server before using FP8 for RL" was recommended and never done. This A/B is that check.
  - In the live run, the trainer's bf16-vs-FP8 `Mismatch KL` is 0.0062–0.0068, with `[trainer.loss] importance_sampling = "none"`. The same loss setting was used on LUMI, but there the sampler was bf16.

## 2. What the traces show

Scripts: `extract_turns.py` (one row per sampled turn, all runs, `data/*.turns.jsonl`), `summarize_runs.py`, and
`characterise.py` (outputs in `data/char_*.txt`). "Degenerate" means the run's rule: a sampled turn of at least 100 tokens with mean
NLL > 3, recomputed from the logprobs, so runs without the detector are counted the same way.
Agentic envs only (swesmith, scaleswe, openswe, terminal-lego, tmax).

### 2.1 PD is not the cause: natural experiment, same checkpoint, all FP8, policy-0 rollouts only

| run | inference | turns ≥100 tok | degenerate turns | at 16–64k context | rollouts with one |
|---|---|---|---|---|---|
| **8t3p5d-tp8 (live, 534489)** | TP8, PD 3P:5D | 26,186 | **2.66%** | 3.49% | 37.2% |
| 8t2p6d-tp8 | TP8, PD 2P:6D | 32,633 | 2.60% | 3.44% | 32.6% |
| pdtp8-4t1p1d | TP8, PD 1P:1D | 1,210 | 3.22% | 4.69% | 31.5% |
| routetest-4t2p2d | TP8, PD 2P:2D | 11,039 | 3.07% | 4.37% | 39.8% |
| pd-4t1p1d (+ run-533326) | DP8, PD | 3,627 / 6,169 | 3.89% / 2.79% | 6.31% / 3.88% | 43.9% / 28.5% |
| **tp8-4t1i** | **TP8, no PD** (otherwise the live config, plus CPU KV offload) | 4,868 | **3.64%** | 5.43% | 31.3% |
| shakedown-4t1i (3 attempts) | DP8, no PD | 6,306 / 1,820 / 2,963 | 3.82% / 3.46% / 3.41% | 5.4% / 5.0% / 5.1% | 36.2% / 27.8% / 28.2% |
| tp8mtp-4t1i | TP8 + MTP, no PD | 1,400 | 2.71% | 4.27% | 20.8% |
| pdmtp-4t1p1d | DP8 + MTP + PD | 802 | **16.1%** | – | 21.4% |

- **Non-PD is no better than PD**, so PD is not the cause.
  - The non-PD runs had no live detector, so they kept generating after a degenerate turn. That biases their per-turn rate slightly upward.
  - The rollout-level rates are equivalent.
  - `tp8-4t1i` differs from the live run's inference config only in `use_pd_kv_transfer`, the conv-state layout and the KV offload.
- **No PD signature inside turns.**
  - The first sampled token of a turn is produced from the prefill node's transferred KV and state. Its mean NLL is 1.65–1.75 in PD runs and 1.61–1.83 in non-PD runs: the same.
  - The share of degenerate turns that are garbage from their first 64 tokens is 18% in the live run and 24% non-PD.
  - Tool-output size before the turn has no effect except above 16k tokens, and that is the same with and without PD.
- **Aside: MTP + PD is broken.** First-token NLL is 8.7 and 16% of turns are degenerate. It is already dropped from the config, but don't revive it without a fix.

### 2.2 The problem is present before any RL update

Policy-0 rollouts were sampled from the unmodified r6 step-2638 weights loaded from disk, with no weight reload. They already show
28–44% degenerate rollouts. This rules out the FP8 re-quantization on weight broadcast and RL drift as the *origin*.
They can still be amplifiers: entropy rises over steps 1–4.

### 2.3 Where it happens (live run, step 1, 38,851 turns ≥100 tokens; `data/char_live_step1.txt`)

| context (prompt tokens) | <4k | 4–8k | 8–16k | 16–32k | 32–64k | 64–96k |
|---|---|---|---|---|---|---|
| degenerate | 0.60% | 0.70% | 1.84% | 2.76% | 3.32% | 3.39% |
| median NLL, healthy turns | 0.53 | 0.75 | 0.79 | 0.97 | 1.07 | 1.08 |

| turn length (tokens) | <256 | 256–1k | 1k–4k | 4k–8k | 8k–16k | ≥16k (cap) |
|---|---|---|---|---|---|---|
| degenerate | 0.18% | 0.79% | 3.8% | 15.6% | 18.6% | 17.4% |

- **Harness:** bash 1.98%, opencode 2.51%, pi 2.75%. **Env:** scaleswe 1.9%, openswe 2.0%, swesmith 2.6%, terminal-lego 2.9%, tmax 4.8%. The same pattern holds in the non-PD run.
- **Drift within a turn is smooth.**
  - Degenerate turns of ≥4k tokens: mean NLL is 1.37 at tokens 1–15, then 1.56 at 16–63, 1.92 at 64–255, 2.56 at 256–1023, and 3.9 beyond.
  - Healthy turns of ≥4k tokens also drift, from 1.1 to 1.8.
  - The median derailment onset (first 64-token window above 3 nats) is 147 tokens into the turn. 0.1% start beyond 8192.
- This is the profile of a self-reinforcing slide. It is not a state-transfer glitch (a jump at turn start) and not a kernel bug (spikes at fixed positions or batch sizes).

### 2.4 The mechanism: a fat semantic tail at T=1 with no cut-off

- **Rate.** In *healthy* (non-degenerate) turns, 7–10% of sampled tokens have p < e⁻⁵ and **1.3–1.9% have p < e⁻¹⁰**. That is roughly one 1-in-20,000 token every 50–80 tokens. The rate grows with context (1.30% at <8k, 1.91% at 32–64k), with the same values PD or not.
- **Shape.** The NLL histogram decays smoothly, about 0.7× per nat out to 17 nats, with no spike at a uniform-floor value (ln 131072 ≈ 11.8).
- **The tokens are ordinary words in slightly wrong places:** "the integration is ‹like›", "global ‹made›-available plugins", "child ‹concussion› field". Each one moves the context off-distribution a little, and long turns compound it into word salad.
- **FP8 does not fit.** The per-token FP8 error needed to produce such a tail is about 2 nats, against about 0.03 measured. The base checkpoint's T=1 distribution is the likely source.
- **Fixes consistent with this mechanism:**
  - The main session's T=0.8 mitigation (in progress).
  - A real tail cut-off: `top_k` with `enable_return_sampling_mask` so the trainer masks consistently. vLLM requires `top_k > 0` for that.
  - A shorter per-turn cap.

### 2.5 Why the LUMI comparison is not like-for-like

- LUMI's "1% of Bash turns / 17% of Bash rollouts" was r21's *worst*, after many RL steps from r5 step 2619 (`docs/lumi-rl-super.md`). Gefion's Bash at step 1 is 1.6% of all turns and 31% of rollouts.
- The per-turn cap was 8192 on LUMI against 16384 here.
  - 29% of degenerate turns here are longer than 8192 tokens.
  - Turns of 4k–16k tokens degenerate at 16–19%.
  - A cap also stops a derailing turn early, which keeps its mean NLL under 3 more often.
- The "0.65" LUMI NLL matches Gefion's short-context value (0.53–0.75 below 8k), while Gefion's agentic contexts mostly sit at 16–64k.

### 2.6 Unrelated issue found in passing (follow-up)

134 of 763 swesmith rollouts (18%) in the live run have **no task**:
- The problem statement is empty.
- The only user message is "Network protocol blocked fetching a resource. Continue without those capabilities…".
- The model then explores aimlessly ("I'm not sure where I am").

Their degenerate rate is about the same (43% vs 45%), so this does not explain the degeneration, but these rollouts are wasted and are trained on. Script: `first_user.py`.

## 3. Literature (web, Oct 2026; details in the session log)

- **No known vLLM bug matches this.** I found nothing for online `fp8_per_block` word salad on NemotronH, or for NIXL hybrid-SSM transfer corrupting outputs.
  - Hybrid-SSM P/D (vLLM blog 2026-04-21, PRs #36687/#37416/#37635) supports homogeneous TP with DS layout. It was benchmarked on Super FP8 for throughput only, with no published accuracy check.
  - A confirmed older bug (stale SSM state, PR #32118, fixed in January) produced exactly this kind of occasional gibberish. That is why §2.1 matters.
- **FP8 rollouts with a bf16 trainer:**
  - Jet-RL (2601.14243): bf16-train/FP8-rollout collapses on long-horizon tasks. TIS-based fixes are "fragile".
  - verl: an "obvious accuracy drop" without TIS; token-TIS with C=2 matches bf16.
  - NeMo-RL: FP8 generation-only loses ~3 points on Llama-8B, and IS narrows but does not close the gap.
  - vLLM's bitwise-consistency blog (2026-10-14): a hybrid Mamba/MoE (Nemotron-3.5) baseline collapsed with entropy rising (0.05 → 2.6) while mismatch-KL *fell*. On hybrids, mean mismatch-KL is not a reliable alarm; watch entropy and tails.
  - None of these gives a hard healthy/unhealthy KL threshold.
- **Cheap training-side mitigation if FP8 stays:** `[trainer.loss] importance_sampling = "tis"` (`tis_cap = 2.0` is already set). It is independent of the root cause.

## 4. GPU A/B (ready, not submitted)

### Set (built on CPU): `data/abset.jsonl`, 465 turns, 108 MB, from `build_set.py`

- Policy-0 rollouts only, so the bf16 checkpoint on disk *is* the sampling policy.
- Contexts are reconstructed token-exact from the trace nodes and checked against `usage.prompt_tokens`. 35 compacted or branched rollouts were skipped.
- **Scoring:** 189 `normal` turns (about 50 per context bucket: <8k, 8–32k, 32–64k, ≥64k) and 84 `degen` turns. Total 9.4M prompt and 0.73M completion tokens.
- **Generation:** 132 `gen_degen` contexts (the next turn degenerated) and 144 `gen_normal` contexts (16–64k).

### Arms (`ab.sbatch <arm> [stage]`, one node each, TP8+EP8, no PD, otherwise the live config from `make_cfg.py`)

| arm | what |
|---|---|
| `bf16` | `quantization` unset (bf16 weights) |
| `fp8` | `fp8_per_block`, exactly as in the live run |
| `fp8keep` (optional) | `fp8_per_block` with NVIDIA's keep-list (attention qkv/o and latent projections in bf16) via `quantization_config.ignore`. Matching was checked with `should_ignore_layer`. It is the candidate fix if FP8 turns out guilty. For production, its weight-reload path needs a check. |

### Stages

1. **`score`.** Teacher-forced prompt logprobs (`/inference/v1/generate`, `prompt_logprobs`): k=1 for all 273 turns, and k=20 for turns ≤32k tokens. `max_num_batched_tokens 4096` and `gmu 0.85` in every arm, because prompt logits are fp32 and full-vocab.
2. **`gen`.** 8 resamples per context with the live sampling parameters: T=1, top_p 1, top_k −1, max_tokens 16384, repetition_detection 16/64/8. The seeds are fixed per (context, i), so arms are paired on common random numbers.
3. **`variants`** (optional, on the `fp8` arm). The same generation with T=0.8 (the main session's mitigation) and with min_p=0.02. These test the tail hypothesis directly. Their logprobs are processed (truncated/tempered), so compare the degenerate *rates*.

All stages resume: rerunning skips finished turns and samples. Outputs go to `results/<arm>/`.

### Commands

```bash
cd /dcai/projects/iu_0145/rasmus/infer-opt/fp8-quality
```
```bash
sbatch ab.sbatch bf16
```
```bash
sbatch ab.sbatch fp8
```
Optional, after looking at the first two:
```bash
sbatch ab.sbatch fp8keep
```
```bash
sbatch ab.sbatch fp8 variants
```
Analysis (CPU, any node):
```bash
python3 analyze_scores.py --set data/abset.jsonl --bf16 results/bf16/score_k1.jsonl --fp8 results/fp8/score_k1.jsonl --bf16-topk results/bf16/score_k20.jsonl --fp8-topk results/fp8/score_k20.jsonl
```
```bash
python3 analyze_gen.py --a bf16=results/bf16/gen.jsonl --b fp8=results/fp8/gen.jsonl
```
```bash
python3 analyze_gen.py --a fp8=results/fp8/gen.jsonl --b t08=results/fp8/gen_t08.jsonl
```

### Reading the output

- **`B-F` (pure quantization):**
  - overall k3 and mean |Δ|
  - by context length: does it grow with context?
  - the **tail table**: mean(B−F) for sampled tokens binned by sampler logprob. Negative values at lp < −5 mean FP8 inflates the tail.
  - onset windows of degenerate turns: does bf16 make the derailing tokens less likely?
  - top-k tail mass F/B
- **`F-T`.** The live PD decode logprobs against non-PD FP8 prefill on the same tokens. This is the PD/decode-path check: anything above about 5e-3 k3, or a spike at tokens 0–16, would be a PD finding.
- **`B-T`.** It should land near the trainer's 0.0065 mismatch KL. That validates the setup.
- **Generation.** Degenerate rate per arm, the paired discordant counts, and a bootstrap CI over contexts.
  - Power: about 1,050 samples per role. A 10% → 5% change on `gen_degen` or 3.5% → 1.75% on `gen_normal` is detectable.
  - Effects of 25% relative are borderline.

### Cost (the QOS is full while 534489 runs; submit only when nodes are free and you agree)

| what | expected | worst case (time limit) |
|---|---|---|
| `bf16` + `fp8`, all stages (2 nodes in parallel, ~1 h wall) | **~2 node-hours**. Per arm: ~7 min startup, ~10–15 min scoring, ~25–35 min generation. | 5 node-hours |
| score stage only, both arms | ~0.8 node-hours | 1.5 |
| `fp8keep` | +1 | +2.5 |
| `fp8 variants` | +1.2 | +2.5 |

The timings extrapolate from earlier single-node runs: TP8 startup 380–440 s, ~10k tok/s engine decode. Long agentic
contexts will be slower. The scripts were dry-run against a mock server and on synthetic score files only. The first real run
may need small fixes. The most likely ones are a prompt-logprob memory spike (lower `max_num_batched_tokens`) and
`repetition_detection` being rejected by the generate endpoint (rerun `gen_ab.py` with `--no-repetition-detection` for both arms).

## 5. Checkpoint comparison: did our SFT cause the fat tail? (r6 vs NVIDIA, ready, not submitted)

**Question.** Is the fat T=1 tail from §2.4 something our SFT introduced, or does NVIDIA's post-trained model have it too?
- r6 was SFT'd from the *base* model: LUMI `nemotron-super-base-r6-16n-cp8-ep8-bs32-lr1e5-r1`, CP8, packed.
- The comparison model is NVIDIA's post-trained `NVIDIA-Nemotron-3-Super-120B-A12B-BF16`.

### Can the comparison be like-for-like? Mostly yes (checked on CPU, `template_check.py`)

- **Same tokenizer.** `tokenizer.json` is byte-identical. Same architecture and config apart from metadata (`max_position_embeddings` 262144 vs 1048576).
- **Same chat format.** The two `chat_template.jinja` files differ in one line: the default of `truncate_history_thinking` is True for NVIDIA and False for r6.
  - That flag only drops reasoning from assistant turns *before the last user message*.
  - Our agent trajectories have one user message followed by assistant/tool turns. Rendering with True or False gives identical tokens for all 227 sampled turns checked, across bash, opencode, pi, nemo_local and null harnesses.
  - So NVIDIA's own default template produces exactly our contexts. No separate renderer is needed. Both arms get the trace's token ids unchanged.
- **Template vs trace tokens.** Re-rendering the trace messages with the HF template reproduced the trace tokens exactly in 175 of 227 turns. The rest differ only through trimmed trailing whitespace in tool arguments or non-canonical BPE splits of the same text: the trace keeps the tokens as sampled.
- **One real caveat: whose history it is.** Mid-trajectory contexts are r6's own rollouts. For NVIDIA they are off-policy history in r6's style, which could inflate NVIDIA's NLL on later strata.
  - The `ck_turn0` stratum (system prompt, tools and task only, no earlier model output) is fully fair to both models. Treat it as the primary comparison and the deeper strata as supporting.
  - A fully on-policy test, with NVIDIA running whole episodes in the envs, needs the RL stack with sandboxes and is not scripted here.
- **NVIDIA's own sampling defaults.** NVIDIA's `generation_config.json` sets `temperature 1.0, top_p 0.95`, so NVIDIA itself recommends cutting the tail. r6's `generation_config` sets neither, and the RL run uses top_p 1.0.

### Set: `data/ckptset.jsonl` (387 contexts, `build_set.py --ckpt`)

Random policy-0 agentic turns, regardless of how the next turn went, so rates are representative.

| stratum | contexts | median context | median original turn length | originally degenerate |
|---|---|---|---|---|
| `ck_turn0` (first turn: system + task) | 100 | 2.2k | 148 | 1 |
| `ck_lt8k` | 99 | 4.4k | 190 | 0 |
| `ck_8_32k` | 94 | 17.3k | 350 | 5 |
| `ck_ge32k` | 94 | 41.8k | 456 | 3 |

13 compacted or branched rollouts were skipped.

### Arms and stages (same `ab.sbatch`)

- **Arms:**
  - `bf16`: r6, bf16
  - `nvidia`: NVIDIA's model, bf16, otherwise identical (TP8+EP8, no PD, fp32 lm_head and router, fp32 Mamba state, `max_num_batched_tokens 4096`, gmu 0.85)
- **`ckpt` stage.** 8 seeded resamples per context at the live T=1 settings (top_p 1, top_k −1, max_tokens 16384, repetition_detection). The seeds are the same in both arms. 3,096 samples per arm, with token ids kept.
- **`xscore` stage.** Teacher-forced scoring of the *other* arm's samples (2 per context, 774) on this arm's server, so each model rates the other's generations. It waits up to 45 min for the other arm's `gen_ckpt.done`, so submit the two jobs together.

### Commands

```bash
cd /dcai/projects/iu_0145/rasmus/infer-opt/fp8-quality
```
```bash
sbatch ab.sbatch bf16 ckpt,xscore
```
```bash
sbatch ab.sbatch nvidia ckpt,xscore
```
Analysis (CPU):
```bash
python3 analyze_ckpt.py --a r6=results/bf16/gen_ckpt.jsonl --b nvidia=results/nvidia/gen_ckpt.jsonl --a-on-b results/nvidia/xscore.jsonl --b-on-a results/bf16/xscore.jsonl
```

The r6 arm can share one server start with the FP8 A/B's bf16 arm: `sbatch ab.sbatch bf16 score,gen,ckpt,xscore`. That saves about 0.15 node-hours, but the NVIDIA job then waits longer for r6's samples before it can cross-score. Separate jobs are simpler.

### Output (per stratum and arm)

- degenerate rate (≥100 tokens, mean NLL > 3)
- share of sampled tokens with p < e⁻¹⁰ and with p < e⁻⁵ (own T=1 logprobs)
- median per-sample NLL, length and finish reasons
- NVIDIA − r6 differences with 95% bootstrap CIs over contexts
- cross-scores: each arm's mean NLL under the other model, and for the tokens one model sampled at p < e⁻¹⁰, how the other model rates them

### How to read it

| outcome | meaning | implication |
|---|---|---|
| NVIDIA's p<e⁻¹⁰ share is far lower than r6's (r6 ≈1.3–1.9% in the traces), especially on `ck_turn0`, and NVIDIA degenerates less | our SFT fattened the tail | Look at the SFT recipe and data (from base, packed CP8, lr 1e-5). Consider starting RL from a better-calibrated checkpoint. Truncated sampling still helps in the meantime. |
| similar tail and degenerate rate | the tail is intrinsic to the model family at T=1, consistent with NVIDIA's own top_p 0.95 default | Fix it at sampling time: `top_k`/`top_p` with `enable_return_sampling_mask` so the trainer stays consistent, or keep T<1. |
| r6's tail tokens also get p < e⁻¹⁰ from NVIDIA | they are genuine errors that r6 over-weights | supports the SFT explanation |
| NVIDIA rates r6's tail tokens as plausible | r6 is just higher-entropy, not making errors | weakens the SFT explanation |
| NVIDIA looks fine on `ck_turn0` but worse on deep strata | likely the off-policy-history caveat, not a real difference | lean on the `ck_turn0` result |

**Power.** About 775 samples per stratum and 3,100 per arm. The tail share is computed over millions of tokens, so its CIs will be tight. The degenerate rate is about 2–3% for r6, roughly 70 events per arm, so a halving is detectable overall but only roughly per stratum.

### Cost (nodes are busy; submit only when you decide)

| what | expected | worst case (time limit) |
|---|---|---|
| `bf16` + `nvidia` with `ckpt,xscore` (2 nodes in parallel, ~1 h wall) | **~2 node-hours**. Per arm: ~7 min startup, ~20–30 min generation (about 2.5–4M output tokens), ~15–20 min cross-scoring (~13M prompt tokens), plus waiting for the other arm. | 5 node-hours |

Everything was dry-run against a mock server: generation with token ids, `score.py --gen`, `analyze_ckpt.py`, and retries at concurrency 256. Both clients now retry transport errors and 5xx responses with backoff, so one failed request no longer kills a run. That fix also applies to the FP8 A/B.

## Files

`/dcai/projects/iu_0145/rasmus/infer-opt/fp8-quality/`:

- this report
- trace analysis: `extract_turns.py`, `summarize_runs.py`, `characterise.py`, `first_user.py`
- set building: `build_set.py` (`--ckpt` for the checkpoint set)
- template check: `template_check.py` (r6 contexts vs NVIDIA's chat template)
- GPU scripts: `make_cfg.py`, `score.py`, `gen_ab.py`, `ab.sbatch` (arms `bf16 | fp8 | fp8keep | nvidia`)
- analysis: `analyze_scores.py`, `analyze_gen.py`, `analyze_ckpt.py`
- `data/`: turn tables for 12 runs, characterisation outputs, `abset.jsonl`, `ckptset.jsonl`

Nothing in the live run, its configs, `phase0/full` or the prime-rl tree was modified. No jobs were submitted.
