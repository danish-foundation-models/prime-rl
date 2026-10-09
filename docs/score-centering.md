# Optional score centering

Score centering corrects prefix-wise drift between the rollout sampler and the
trainer. It is an optional RL loss; IPO remains the default. The implementation
follows [Score Centering Stabilizes Off-policy Reinforcement Learning](https://arxiv.org/abs/2609.20807),
including its top-k head plus rescaled trainer-tail approximation.

Enable standalone SC in a combined RL config:

```toml
[trainer.loss]
type = "score_centering"
top_k = 32
importance_sampling = "none"
```

For SC composed with truncated importance sampling, set
`importance_sampling = "tis"` and optionally `tis_cap = 2.0`. The sampled-token
importance weight and centering coefficients are detached. Group advantages and
RL component weights/normalization remain unchanged. SC does not add IPO's KL
regularizer. `tail_epsilon = 1e-6` floors tail masses for numerical stability.

The combined RL config sets `score_centering_top_k` in each train source's
`sampling.extra_body`, raises the inference `max_logprobs` to at least `top_k + 1`,
and selects `processed_logprobs`. Separate orchestrator/inference processes must
set those fields explicitly. Use positive temperature with full-vocabulary
sampling (`top_p = 1`, no sampling `top_k`). Truncated sampling is rejected for
this option. The loss head size `top_k` does not truncate generation.

The renderer requests one extra logprob because ServingTokens can put the sampled
token first even when it is outside the highest-probability head. It selects the
highest-probability k entries without renormalizing them. The sampled token's own
logprob is retained independently. Missing, malformed, or misaligned head data
fails rather than silently falling back to ordinary policy gradient.

Sampler IDs (int32) and logprobs (float32) travel as binary arrays on assistant
trace nodes and the training transport. They are excluded from disk trace JSON
and sandbox-facing model responses. Samples retain token alignment across turns,
shared prefixes, truncation, packing, padding, label shifts, and context-parallel
shards. Only RL member tokens contribute to SC. Prompt/tool/padding rows carry
invalid IDs and never contribute. Turning SC off requests no additional sampler
probabilities and uses the existing loss path.

The fused LM head gathers selected logits during its existing sequence/vocabulary
chunks and scatters their gradients during recomputation. It does not materialize
full sequence-by-vocabulary logits. Native model forwards and the HF wrapper pass
selected IDs through; the vanilla-head fallback also supports SC.

Monitor `sc_sampler_head_mass`, `sc_trainer_head_mass`, `sc_correction`, and
`sc_importance_weight`. Compare k=32 against k=128 if retained mass is low. The
head adds 8*k binary bytes per token before padding; inference JSON serialization
adds more. A head-only benchmark cannot establish full rollout/training overhead.

Validation covers dense-reference SC/TIS+SC gradients, constant-reward expected
gradient cancellation, selected-token gradients across vocabulary chunks on CPU
and GPU, missing-data rejection, binary trace round-trips, multi-turn alignment,
packing/padding/truncation, and combined config setup. Before a production switch,
measure end-to-end vLLM throughput and run the short RL pilot. No sandbox or
training job is started by enabling or testing the config.

On LUMI, the loss/head unit tests need a GPU allocation because the trainer imports
GPU kernels during collection. Run renderer client regression checks with the
optimized Super vLLM source profile, which provides the native token-in/token-out
API. The isolated command is `python -m pytest deps/renderers/tests/test_client.py
--noconftest --import-mode=importlib -q`, invoked through the container's `uv run`;
it avoids loading the renderer parity suite's unrelated conftest dependencies.

## Final-checkpoint pilot on LUMI

`configs/lumi/rl_nemotron_super_r5_128k_16t8i_sc_pilot_r14.toml` selects
final SFT export step 2619, SC alone with k=32, and an independent model
conversion directory. Do not reuse a conversion directory from another export.
The combined RL config enables sampler head capture and processed logprobs
(max_logprobs=33) automatically. The adapter includes verifiers-ucloud main
964708a's descriptor-limit preparation and SDK 0.4.22 connection pools while
retaining the local managed/parkable runtime and private guest relay origin.
Before submission, run the native launcher with `--dry-run` and verify real
park/resume/tool execution using the current SDK.
