# Claim check: "cache only the final hidden state, then KL is a single dot product"

Date: 2026-10-26. Context: PRIME-RL score-centering RL of Nemotron-3 Super (d=4096, V=131072, bf16 body, fp32 LM head).
PRIME-RL paths below are relative to `/dcai/projects/iu_0145/rasmus/phase0/prime-rl` (read only, nothing edited).

## Verdict

**Partly correct, but the claim mixes up two things.** DeepSeek did say to cache only the last-layer hidden
state. They said it about *teacher* states in **on-policy distillation** (DeepSeek-V4), and they then rebuild
the **full logits** through the prediction head at training time. That is a full `d x V` matmul per token, not
"a single dot product". It saves storage. It does not save compute, and it does not avoid materializing logits:
the logits are built on the fly in chunks and never stored.

For our score-centering run it is **not worth pursuing as a throughput fix**:

- It moves about 32x more bytes per token than the binary top-k (8 KiB against 256 B).
- It needs the *sampler's* LM head (which lags the trainer by up to `max_off_policy_steps` versions) for the result to be exact.
- It needs non-trivial vLLM and PRIME-RL changes.

It becomes interesting only if we wanted **full-vocabulary** score centering or KL instead of the top-32 plus tail approximation.

## 1. What DeepSeek actually said

**DeepSeek-V4 tech report, §5.2.2 "Efficient Teacher Scheduling for Full-Vocabulary OPD".**
Source: arXiv 2606.19348, https://arxiv.org/html/2606.19348. Verbatim:

> "Furthermore, naively materializing logits for a vocabulary size |V|>100k across all teachers is prohibitive,
> even when spooled to disk. We address this by caching only the last-layer teacher hidden states in a
> centralized buffer during the forward pass. At training time, these cached states are retrieved and passed
> through the corresponding prediction head module to reconstruct the full logits on the fly. This design incurs
> negligible recomputation overhead while completely circumventing the memory burden associated with explicit
> logits materialization. To mitigate the GPU memory footprint of the teacher prediction head, we order training
> samples by teacher index during data dispatching. [...] Finally, the exact KL divergences between teacher and
> student logits are computed using a specialized TileLang kernel [...]"

The motivation is in §5.1.2 "On-Policy Distillation". Prior OPD work "simplif[ies] the full-vocabulary KL loss
into a token-level KL estimate at each token position". The report says this "leads to high variance in gradient
estimation and often causes training instability. Therefore, we adopt full-vocabulary logit distillation in our OPD."

Points to note about this source:

- **Whose states.** The cached states belong to the **teacher**: a separate frozen model, run in a teacher forward
  pass, with more than ten teachers. They are not the rollout sampler's states. A frozen teacher means its
  prediction head is known and unchanging, so "pass it through the head" is exact.
- **What happens to the logits.** They are rebuilt *in full*, and the KL is the *exact full-vocabulary* KL. The goal
  is to compute **more** (full vocab instead of a sampled-token estimate), not less.
- **What "materialization" means here.** It means storing or spooling the logits. The logits are still computed.
- **The word "compressed".** It does not appear in this passage. In DeepSeek papers it refers to the compressed KV
  latents of MLA and of V4's CSA/HCA attention, which are unrelated. The colleague's wording probably blends the two.

**DeepSeek-V3.2, §3.1 "Unbiased KL Estimate".** Source: arXiv 2512.02556, https://arxiv.org/html/2512.02556.
This is the RL KL-to-reference used in GRPO. It is the k3 estimator on the **sampled token**, multiplied by the
importance ratio pi_theta/pi_old. It needs only sampled-token logprobs. No hidden states are involved.

The same section's "Keep Sampling Mask" (keep the top-p/top-k truncation mask from sampling and apply it to
pi_theta) is what PRIME-RL's `sampling_mask` already implements: `src/prime_rl/trainer/models/layers/lm_head.py:107`
and lines 186-216.

I found no DeepSeek source, in the R1, GRPO/DeepSeekMath, V3.2 or V4 reports, that says KL is "a single dot product".

## 2. The math

With logits `z_v = w_v^T h / T` and normalizer `logZ = logsumexp_v z_v`:

- Sampled-token logprob: `log pi(y) = w_y^T h / T - logZ`.
  - The numerator **is** one dot product.
  - `logZ` needs all V = 131072 dot products, i.e. a `[d x V]` matvec, 2dV = 1.07 GFLOP per token.
  - There is no identity that gives `logZ` from `h` without touching every row of `W`.
- Per-token KL estimators on the sampled token, with `r = log pi_q(y) - log pi_p(y)`: `k1 = r`, `k2 = r^2/2`,
  `k3 = e^{-r} - 1 + r`. Each needs **both** normalizers, `logZ_q` and `logZ_p`.
- Full KL: `KL(q||p) = sum_v q_v (log q_v - log p_v)`. This needs all V logits under both heads.
- When "one dot product" is true: if the sampler shipped `(h, logZ_q)` and the trainer held the **exact sampler
  head `W_q`**, then `log q(y) = w_{q,y}^T h - logZ_q` is one dot product. But the sampler already returns
  `log q(y)` itself in 4 bytes, so this buys nothing. And the trainer side, `log p(y)`, needs the trainer's own
  `logZ_p`, which it **already computes anyway**: the fused LM head does an online logsumexp over vocab chunks,
  `lm_head.py:94-105` and `:193-214`.

What each approach saves:

| | Storage / transfer | Compute |
|---|---|---|
| Sampler top-k (current) | 33 x (id, logprob) per token | free: vLLM already computes full logits for sampling |
| Cached hidden state | 1 x h per token (avoids storing V logits) | **adds** a full `W_q h` recompute per token on the trainer (chunked; logits never stored) |
| "Single dot product" | n/a | only valid for the numerator; `logZ` still costs a full matvec |

## 3. Numbers for Nemotron-3 Super

Per generated token:

| payload | bytes/token | at 4.1k tok/s | 128k-token sample |
|---|---|---|---|
| top-32 binary as stored (int32 id + fp32 logprob) | 256 B | 1.0 MB/s | 32 MiB |
| top-33 JSON on the wire (about 60-80 B/entry) | about 2-2.6 KB | about 10 MB/s | about 300 MiB |
| hidden state bf16 (4096 x 2) | 8 KiB | 33.6 MB/s | 1 GiB |
| hidden state fp32 | 16 KiB | 67 MB/s | 2 GiB |
| hidden state bf16, base64 in JSON | about 10.9 KB | about 45 MB/s | about 1.3 GiB |

The hidden state is **32x** larger than the stored top-k and about **4x** larger than the JSON top-k.

Our measured bottleneck was the API server's per-entry Python work (33 dicts per token, "token_id:" strings,
JSON encoding). It was not raw bytes. A single contiguous base64 tensor per request would be CPU-cheap per token,
but the same bottleneck can be fixed more cheaply by returning top-k as packed binary arrays. That is roughly what
our patch does, and it keeps a 32x smaller payload.

On the trainer side, buffering hidden states for a 64-sample x 128k batch (`configs/lumi/*.toml`: `seq_len = 131072`,
`batch_size = 64`) is up to about 64 GiB per step at the worst-case length, against about 2 GiB for top-k.

Recompute cost on the trainer, with `W_q` of 4096 x 131072 = 537M params (1 GiB bf16, 2 GiB fp32):

- **Compute.** Forward only, no backward through q: 2dV = 1.07 GFLOP/token. The trainer already does this
  forward for p, plus about 2x more in backward. So it adds about +33% to LM-head cost, and about 1-2% of total
  per-token training FLOPs for a model with about 12B active params (about 70+ GFLOP/token for forward plus backward).
- **Memory.** It can reuse the chunked online-logsumexp kernel, so no `[N, V]` buffer: `lm_head.py:124-224`, vocab
  chunk 8192. A second head must be resident: +1-2 GiB per rank, or sharded.

So "negligible recompute" (DeepSeek's wording) holds for us too. The cost is in bytes and plumbing, not FLOPs.

## 4. Does it fit score centering as PRIME-RL implements it?

`ScoreCenteringLoss.loss`, `src/prime_rl/trainer/rl/loss.py:193-237`, uses:

- `sampler_topk_ids` (sampler's top-32 set) and `sampler_topk_logprobs` = `log q_i` on that set
  (`loss.py:206-208`). `q_tail = 1 - sum q_i` (`:223`) relies on these being **unnormalized full-distribution**
  logprobs (`deps/renderers/renderers/client.py:134-171`, "never normalize the head").
- `trainer_topk_logprobs` = `log p_i` gathered at the **sampler's** ids, computed in the fused head
  (`lm_head.py:200-202, 220-221`; `train.py:462`).
- `inference_logprobs`, the sampled-token `log q(y)`, used for the TIS weight (`loss.py:230`).

  `p_tail = 1 - sum p_i`, `rho = q_tail/p_tail`, `alpha = rho * w(-log rho)`,
  `residual_i = q_i w(log p_i - log q_i) - alpha p_i` (detached), `correction = sum_i residual_i log p_i`,
  `loss = -A * (w_y log p(y) - correction)`.

Could the trainer rebuild `{ids, log q_i, log q(y)}` from a sampler `h`? Only if three conditions hold.

1. **Same head weights.** q is the inference policy at version `v - s`, with `s <= max_off_policy_steps`
   (default 8: `packages/prime-rl-configs/src/prime_rl/configs/orchestrator.py:681`; 5 in
   `configs/lumi/rl_nemotron_super_r5_128k_16t8i_scale.toml:68`). Using the trainer's current `W_theta` on `h_old`
   gives `softmax(W_theta h_old)`. That is a hybrid that is **neither q nor p**, so the loss would be biased in an
   uncontrolled way. Fixes:
   - (a) keep a ring of LM-head snapshots per policy version, 6-9 x 1 GiB bf16 on host, and pick the one matching
     the sample's version; or
   - (b) freeze the LM head during RL. Then `W_q = W_theta` up to dtype.
2. **Same numerics.** vLLM samples from bf16 logits computed with its own kernels, TP layout and fp32 log_softmax
   (`logprobs_mode = "processed_logprobs"`, `configs/inference.py:623-625`, `configs/rl.py:658`). The trainer would
   recompute in fp32 from a bf16 `h`. The result is close but not bit-identical to the distribution actually sampled.
   That is fine for `q_i`, but `log q(y)` should still come from vLLM, which returns it anyway (4 B).
3. **Same processing.** Temperature is known per token. Any top-p/top-k sampling mask or logit processors must be
   replayed identically. `sampling_mask` already exists for this.

If all three hold, it works and allows **more** than today: the trainer can compute q over the full vocabulary,
so it can use exact full-vocab score centering or KL with no top-32 plus tail approximation. That is the DeepSeek-V4
OPD use case.

## 5. Bottom line and what it would take

- **The claim as stated ("single dot product, no logits") is a misunderstanding.**
  - A log-normalizer always needs the full `W h` matvec.
  - DeepSeek's actual point is storage: cache `h`, not `[T, V]` logits, then recompute full logits on the fly through a *known, fixed* head.
- **For our current top-32 score centering it is a worse design.**
  - It is 32x the bytes of binary top-k.
  - It needs versioned sampler heads, and it adds trainer compute and memory.
  - The vLLM API-server bottleneck is better solved by binary top-k transport, which we already did.
- **Worth pursuing only if** we decide we want full-vocabulary score centering, full KL, or OPD with frozen teachers.
  For OPD the teacher head is fixed, so the scheme is clean, and an OPD teacher pass under our control could keep `h` locally.
- **What it would take:**
  - vLLM: capture the final-norm output per generated token in the model runner. There is no stable public
    `/generate` field for last-layer hidden states of sampled tokens, so this means a custom patch or side channel
    shipping `[T, 4096]` bf16 as binary (not JSON), tagged with the policy version.
  - PRIME-RL transport: a new `TrainingSample` field through `orchestrator/trajectories.py`,
    `trainer/batch.py:300-320, 455-463, 849`, plus about 8 KiB/token buffer and disk budgets (about 1 GiB per 128k sample).
  - Trainer: hold `W_q` for each in-flight version (or freeze the head). Extend `_SequenceChunkedLogProbEntropyFn`
    to run a second chunked pass with `W_q` that yields `log q` over the vocab (or its top-k). Then either feed the
    existing loss or write a full-vocab variant.
  - Validation: compare reconstructed `log q(y)` against vLLM's returned `inference_logprobs` per token as a mismatch metric.
