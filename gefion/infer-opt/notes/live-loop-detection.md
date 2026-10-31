# Live loop and degeneration detection during rollouts

Date: 2026-10-29. Read-only investigation. No jobs submitted and no code edited.
Analysis scripts are in the session scratchpad (`loopsim.py`, `nllsim.py`, `combo.py`, `bench.py`).

## TL;DR

* Repetition loops are **not filtered or penalized today**. `orchestrator/loop_annotations.py` only
  records metrics. The only filter is `DegenerateTurn` (mean sampled NLL > 3 over a turn of at least 100
  tokens). It catches high-entropy gibberish, not loops: a loop like `of of of …` has an NLL near 0.
* Both failure modes are expensive. They are detected only after the rollout ends.
  * **DegenerateTurn traces** use about 55% of all train decode tokens in the shakedown run and 32% in
    tp8. Of that, **30% (shakedown) and 10% (tp8) of all train decode tokens are generated in turns
    *after* the degenerate turn**, in traces that are then thrown away.
  * **Turns that hit the 16k per-turn cap** (`finish_reason=length`) hold **35% (shakedown) and 31%
    (tp8) of all train decode tokens**. About 60-70% of them are token loops that an online detector
    flags long before 16k tokens.
* Recommended design, in order of value and cost:
  1. **Between-turn DegenerateTurn abort** in the verifiers interception server. It gives exactly the
     same training outcome as today (same traces dropped) and saves **~30% / ~10%** of train decode
     tokens. It also removes the worst straggler rollouts. Effort: about 0.5-1 day.
  2. **In-engine repetition stop** using vLLM 0.28's built-in `SamplingParams.repetition_detection`,
     passed through `orchestrator.train.sampling.extra_body`, plus finish-reason plumbing in
     verifiers. It saves another **~8% / ~6%** of decode tokens on top of (1). Effort: about 1 day.
     Engine overhead is about 1 ms/step at 150 sequences, about 2-3% of a 40 ms step, and probably
     hidden by async scheduling. This needs measuring.
  3. Combined estimate: **~40% (shakedown) / ~18% (tp8) fewer train decode tokens**. The real
     wall-clock gain should be larger, because the saved tokens are at long contexts and in straggler
     rollouts.
* Do **not** add an in-turn sliding-window NLL abort. It saves only about 3% more tokens. It also
  changes which traces are dropped: 8-20% of traces would be killed that the post-hoc rule keeps,
  including 10-35 traces with reward > 0.
* Cross-turn tool-call cycles (option c) barely occur: 2 recovered events in tp8 and none in the
  shakedown run. They are not worth building now.

## 1. What exists today

### 1a. `DegenerateTurn` filter (the only active filter)

* Code: `src/prime_rl/orchestrator/degenerate.py::find_degenerate_turn`. It finds the first sampled
  node with at least `min_tokens` logprobs whose mean `-logprob` is greater than `max_mean_nll`.
* Runs in: `orchestrator/dispatcher.py` (around line 639), when the episode task completes, for train
  traces only, after the whole rollout has finished (including scoring).
* Effect: it appends `vf.Error(type="DegenerateTurn")` and sets `trace.ok = episode.ok = False`.
  `algo/base.py::iter_trainable_traces` skips `has_error` traces, so the **whole trace is dropped**.
  It gets no reward and no mask, and it does not count in the group baseline. The other members of the
  group still train against a baseline computed over the survivors only (`train_sink._process_group`).
  The env retry list only contains `SandboxNodeLost`, so the trace is not retried.
* Config: `packages/prime-rl-configs/.../orchestrator.py::DegenerateTurnConfig`, set through
  `[orchestrator.degenerate_turns]`. Our config (`gefion_full_r1.toml`) has `enabled = true`,
  `min_tokens = 100`, `max_mean_nll = 3.0`.
* Frequency in the logs:
  * shakedown (`runs/gefion-super-shakedown-4t1i-r1/logs/latest/orchestrator.log`): 231 events in
    about 3 h, and 231 of 947 train traces (24%). The degenerate turn averages 6,630 tokens, ranging
    from 105 to 16,384. 38 of them hit the 16k cap. These turns hold 1.53M tokens in total.
  * tp8 (`runs/gefion-super-tp8-4t1i-r1/logs/latest`): 45 events so far (about 52/h). They average
    5,148 tokens, ranging from 178 to 16,384, and 5 hit the 16k cap.

### 1b. `loop_annotations` (observation only)

* Code: `src/prime_rl/orchestrator/loop_annotations.py::annotate_episode`. It is called from
  `orchestrator.py` (line 621) through `asyncio.to_thread` for every episode. It is documented as
  "never reward shaping". It writes `trace.metrics["loop/*"]` and `trace.info["loop_annotations"]`.
* Contents:
  * word 16-shingle repeats inside the reasoning;
  * tail word 5-gram coverage on reasoning plus content (`text_repetition`);
  * 4x repeated tool-call cycles (`repeated_action_cycle`);
  * 3x repeated (call, identical output) cycles (`unchanged_tool_cycle`).
  An event is flagged only if the repetition persists to the end of the trace.
* Config keys: none. It always runs and changes no training field.
* Frequency (from `monitors/file/traces/stream`): `loop/suspected` is true in 165 of 947 shakedown
  train traces (17%), covering 2.25M tokens in the flagged nodes. In tp8 it is 42 of about 300 traces.
  The only tool-cycle events are 2 recovered `unchanged_tool_cycle` events in tp8.

### 1c. Other related behavior

* `algo/behavior.py` together with `[orchestrator.algo.behavior_penalty] penalty = 0.1` deducts 0.1
  for malformed XML tool calls or for a "reasoning-only voluntary end". The second check requires
  `finish_reason == "stop"` and `not trace.is_truncated`. It does not detect loops.
* A turn that hits `max_completion_tokens = 16384` returns `finish_reason="length"`. The harness
  usually continues: only 152 of 493 length turns were the last turn of their trace. **The truncated
  looping turn is then trained like any other turn, with the trace's advantage.** The only exception
  is when the whole trace is dropped as DegenerateTurn.
* verifiers and verifiers_ucloud contain no loop or repetition logic. The grep found only
  `repetition_penalty` passthrough.

## 2. Measurements (simulated on recorded traces)

Source: train traces in `runs/*/monitors/file/traces/stream/00000.jsonl`, which store per-node
`token_ids`, `mask` and `logprobs`. Shakedown has 947 traces and 23.7M output tokens. tp8 has about
300 traces and 3.6-4.0M tokens; that run was still growing during the analysis.

### 2a. Token-loop detectors on single turns

"Saved" counts the tokens after the detector fires, as a share of all train output tokens. The
"natural-end" column counts fires in turns that ended with `stop` or `tool_calls`, not `length`.

| detector | fires (shake) | length turns caught (shake) | natural-end fires | saved (shake) | saved (tp8) |
|---|---|---|---|---|---|
| 16-gram coverage >= 0.9 over 1024 tokens, min 2048 tokens | 409 | 297/493 | 112 | 13.0% | 7.0% |
| exact periodic tail >= max(4p, 512), p <= 256 | 374 | 270/493 | 104 | 12.3% | 6.5% |
| vLLM built-in, min=16 max=64 count=8 | 544 | 335/493 | 209 | 16.9% | 8.8% |
| vLLM built-in, min=32 max=128 count=6 | 490 | 314/493 | 176 | 15.5% | 8.1% |
| vLLM built-in, min=16 max=256 count=4 | 640 | 358/493 | 282 | 19.2% | 10.1% |
| vLLM built-in, min=1 max=32 count=8 (too aggressive) | 1185 | 423/493 | 762 | 29.1% | 18.1% |

* 90-95% of fires are inside the reasoning, before `</think>` (token 13).
* I decoded random samples of "natural-end" fires, including those in traces with reward 1.0. **All
  were degenerate text**: `WAIT WAIT WAIT…`, `of of of…`, `</</</…`, `!!!!…`, `💵💵💵`, `—— hmm:` cycles.
  The model recovered afterwards. I found no legitimate code or table repetition among the sampled
  fires. Precision is effectively high for min_pattern_size >= 16 and a minimum repeated span of 128
  tokens or more. A pattern size of 1 is not safe.
* Loops have a low NLL. The mean NLL over the last 1024 tokens of a caught length turn is about 0.0.
  So DegenerateTurn cannot catch loops, and the two detectors are complementary.

### 2b. DegenerateTurn: where the tokens go

| | shakedown | tp8 |
|---|---|---|
| all output tokens of DegenerateTurn traces | 55.1% | 32.5% |
| … before the degenerate turn | 18.7% | 17.3% |
| … inside the degenerate turn | 6.5% | 4.8% |
| … **after the degenerate turn** (generated, then discarded) | **29.9%** | **10.3%** |

**Option A: abort between turns.** End the rollout right after a sampled turn that meets the
existing rule. The set of dropped traces is identical to today, so this has no RL effect. It saves
29.9% / 10.3%. 36 of 233 (shakedown) degenerate turns were already the last turn.

**Option B: abort live inside the turn** with a running NLL over the logprobs that vLLM returns. The
"extra traces killed" column counts traces that the post-hoc rule would keep.

| live rule | deg traces caught (shake) | saved (shake) | of which inside deg turn | extra traces killed (shake, rewarded) | extra traces killed (tp8, rewarded) |
|---|---|---|---|---|---|
| cumulative mean > 3 after 100 tokens | 233/233 | 39.4% | 2.9% | 119 (25) | 20 (3) |
| cumulative mean > 3 after 2048 tokens | 208/233 | 36.0% | 3.0% | 75 (10) | 15 (1) |
| window 512 tokens, mean > 3.0 | 227/233 | 43.1% | 2.8% | 167 (28) | 52 (6) |
| window 2048 tokens, mean > 3.5 | 207/233 | 38.0% | 2.8% | 111 (12) | 28 (2) |

Within the degenerate turn itself, live NLL detection saves only about 3% of tokens. The rest of the
apparent gain comes from killing traces earlier, on turns whose *whole-turn* mean stays at or below 3.
High-NLL windows are common in healthy turns: hashes, random IDs, base64, and gibberish that the
model later recovers from. Any live NLL rule therefore **redefines** the filter. It drops 8-20% more
traces, including successful ones. It is not a faster version of the same filter. Not recommended.

### 2c. Combined estimate (A plus an in-engine repetition stop on the turns that remain)

| repetition detector | shakedown: A + rep = total | tp8: A + rep = total | traces newly classified degenerate (shake / tp8) |
|---|---|---|---|
| 16-gram coverage 0.9 | 31.2% + 6.6% = 37.8% | 11.2% + 4.7% = 15.9% | 14 / 5 |
| built-in 16/64/8 | 32.8% + 8.1% = **40.9%** | 11.6% + 6.3% = **17.9%** | 34 / 8 |
| built-in 32/128/6 | 32.2% + 7.7% = 39.9% | 11.5% + 5.5% = 17.0% | 26 / 7 |

There is an interaction to note. Cutting a low-NLL loop tail **raises** the turn's mean NLL. So
"gibberish followed by a loop" turns that pass today can become DegenerateTurn after the cut: 34 of
947 traces with built-in 16/64/8, 4 of them rewarded. See section 3 for how to handle this.

These numbers are first-order. They count decode tokens and ignore the counterfactual of what the
agent does after a shorter turn. Saved tokens sit at long contexts and in straggler rollouts, so the
GPU time and batch-latency savings should be larger than the token share.

## 3. Options and exact hook points

### (a) Inside vLLM

* **Logits processors (v1 API, `vllm.v1.sample.logits_processor`) cannot finish a request.** They
  can only reshape logits, so stopping would mean forcing EOS. A forced token gets an off-policy
  logprob, and RL would then train the model to emit EOS there. Rejected.
* **The right hook is the scheduler stop check.** `vllm/v1/core/sched/scheduler.py`
  `_update_request_with_output` calls `check_stop` in `vllm/v1/core/sched/utils.py` after each
  appended token. vLLM 0.28 **already has** `SamplingParams.repetition_detection =
  RepetitionDetectionParams(min_pattern_size, max_pattern_size, min_count)`. On a match it sets
  `RequestStatus.FINISHED_REPETITION`, which the endpoint returns as `finish_reason="repetition"` with
  `stop_reason="repetition_detected"`. The request ends normally: KV blocks are freed and full blocks
  stay in the prefix cache. All sampled tokens and logprobs up to and including the triggering token
  are returned. The rule fires when the last `p*min_count` tokens are periodic with some period
  p in [min, max]. A loop of base period q is also periodic in every multiple of q, so min=16 still
  catches short-period loops.
* I verified with the venv that `GenerateRequest.model_validate({... "sampling_params":
  {"repetition_detection": {"min_pattern_size":16,"max_pattern_size":64,"min_count":8}}})` parses
  into `RepetitionDetectionParams`. The fast_generate patch keeps `o.finish_reason`, so "repetition"
  passes through unchanged.
* Cost: the check is pure Python on the engine-core scheduler path. Benchmarked per request per step
  (`bench.py`), times 150 sequences:

  | built-in config (min/max/count) | per request | per step at 150 seqs |
  |---|---|---|
  | 16/64/8 | ~7.5 µs | ~1.1 ms |
  | 16/256/4 | ~36 µs | ~5.3 ms |
  | 1/512/4 | ~78 µs | ~11.7 ms |

  The engine runs about 150 sequences at about 3.5k tokens/s, so a step takes about 40 ms. The
  overhead for 16/64/8 is about 2-3%. With async scheduling (the vLLM default when not disabled) part
  of it overlaps with the GPU. Measure before rollout.
* An O(1) alternative: a 16-gram rolling hash with a "same distance as last step" run counter or
  coverage window costs about 0.4 µs per request per step, about 0.07 ms per step. It would go in as
  an `infer_opt` patch that replaces `vllm.v1.core.sched.scheduler.check_stop`, using the same
  `when_imported` pattern as `fast_generate.py`, with state kept on the `Request`. Use it only if the
  built-in overhead proves measurable.

### (b) Streaming per-turn with client abort

* No streaming is used today. `verifiers/v1/clients/train.py::TrainClient.get_response` calls
  `renderers.client.generate`, which does a non-streaming POST to `/inference/v1/generate`.
  `Client.stream` raises `NotImplementedError` for the renderer client. `fast_generate` also skips
  stream requests.
* Disconnecting a non-streaming request makes vLLM abort it. Then `serve_tokens_full_generator`
  returns "Client disconnected" and **all tokens and logprobs are lost**.
* Streaming would need changes in all of these places: renderers client chunk parsing, the
  per-chunk logprob, `sampling_mask` and `routed_experts` accounting, and the fast_generate path.
  Prefix caching is not the issue: computed full blocks stay cached either way. The detection would
  run in the env worker, which is further from the tokens. Strictly worse than (a). Rejected.

### (c) Between turns, in the verifiers interception server

* `verifiers/v1/interception/server.py`, around lines 700-750. Right after
  `session.client.get_response(...)` and `turn.commit(call_response, ...)`, the sampled node with
  `logprobs` and `mask` exists. The server already has a stop path there:
  `session.trace.stop(name)` followed by a 400 "rollout stopped: …". This is the same mechanism as
  `RolloutLimits` (`session.py`: `max_turns`, `max_total_tokens`) and the `@stop` hooks
  (`RolloutSession.response_stops` / `trace_stops`).
* This is the right place for the **DegenerateTurn abort** (option A). It costs one mean over at
  most 16k floats per turn.
* Cross-turn tool-call cycles are rare in our data (section 1b), so a live cycle stop is not worth
  doing now. If needed later, it is a `trace_stops`-style check over the last N committed tool calls.

## 4. RL correctness requirements

Principle: **a live rule must be exactly the post-hoc rule, run earlier on the same recorded data.**
The post-hoc check then still runs and reaches the same verdict, so training treatment is consistent
by construction.

**DegenerateTurn, between-turn abort.**
* After committing a sampled node, run the existing `find_degenerate_turn` logic on that node, with
  the same `min_tokens` and `max_mean_nll` and the same masked-logprob definition.
* If it fires, call `trace.stop("degenerate_turn")`, return 400 to the harness, and let the rollout
  conclude.
* The orchestrator needs no change. The dispatcher's post-hoc `find_degenerate_turn` finds the same
  turn, adds the `DegenerateTurn` error, and drops the trace exactly as today. Group and baseline
  behavior are unchanged.
* Optional: skip scoring for `stop_condition == "degenerate_turn"`, because the result is discarded
  anyway. That saves sandbox and test time, not GPU. Today `rollout.py` scores stopped traces and
  skips only failed ones. Alternatively, raise a failure so `_failed` skips scoring.

**Repetition stop.** The turn is a truncation by an external rule, just like the 16k cap. Every
emitted token was sampled from the policy, and no token is forced, so logprobs and importance ratios
stay valid. Requirements:
1. **verifiers**:
   * Add `"repetition"` to `FinishReason` (`types.py`) and to `FINISH_REASONS`
     (`dialects/chat.py`). Without this, `response_from_generate` maps it to `None` and the
     interception server reports `"stop"` to the harness. The harness may then treat a half-thought
     as the final answer and end the agent, which changes rollout dynamics compared with today's
     `length`.
   * In `serialize_completion` (`clients/train.py`), send **`"length"` on the wire** for
     `"repetition"`, so harnesses (opencode, pi, bash) behave exactly as they do for a cap hit.
   * Keep `"repetition"` on `ModelCall.finish_reason` in the trace.
   * `Trace.is_truncated` (`trace.py:621`) should treat `"repetition"` like `"length"`.
2. **prime-rl**:
   * `algo/behavior.py` already excludes truncated traces from `reasoning_only_end`. With (1) this
     holds automatically. Otherwise a repetition-ended final turn could wrongly get the 0.1 penalty.
   * Add metrics: count `finish_reason=="repetition"` calls (`loop/live_stopped_turns`,
     `loop/live_stopped_tokens_saved_est`) in `loop_annotations.py` or `metrics.py`, next to
     `stop_condition`.
   * Training treatment stays **as today for `length`**: trained with the trace advantage. The
     shorter tail means less negative gradient on loop tokens per sample. If an explicit loop penalty
     is wanted later, add it as an opt-in reason in `behavior.py` (`flags.repetition_stopped`). That
     is a deliberate reward-shaping change and not needed for correctness.
3. **The DegenerateTurn interaction.**
   * Mean NLL over a repetition-cut turn is higher, so 26-34 more shakedown traces (4 rewarded)
     would be dropped.
   * If that is unwanted, compute the NLL rule only over turns not ended by `"repetition"`, or use
     the pre-loop prefix. Make the identical change in the live check (option A) and in
     `degenerate.py`.
   * Either choice is consistent, as long as live and post-hoc share one function. Recommendation:
     move `find_degenerate_turn`'s per-node predicate into a small shared helper that verifiers can
     call, or duplicate it with a test that pins equality.
4. Evals: put `repetition_detection` only in `orchestrator.train.sampling.extra_body`, so eval
   scores stay comparable. Optionally enable it for evals too, and record it.

## 5. Recommended design

**Step 1: between-turn DegenerateTurn abort (biggest win, no RL change).**
* verifiers fork (`rlrs/verifiers`), around `v1/interception/server.py:745`, right after
  `turn.commit(...)`:
  * if `session.limits.degenerate_turn` is set and the new node meets the rule, call
    `session.trace.stop("degenerate_turn")` and return a 400 error body.
  * Add `degenerate_turn: {min_tokens, max_mean_nll} | None` to `RolloutLimits` (`v1/session.py`)
    and to the agent config that builds it (`v1/agent.py:298`), next to `max_total_tokens`.
* prime-rl: when `orchestrator.degenerate_turns.enabled`, stamp the same values into each train
  env's agent limits when the env config is resolved, so there is one source of truth. Keep the
  dispatcher check as the authority and the backstop.
* Effort: about 0.5-1 day, including a unit test that the live and post-hoc verdicts agree on
  recorded traces. Then bump the verifiers pin.
* Expected saving: about 30% (shakedown) / 10% (tp8) of train decode tokens, and fewer 300k-token
  stragglers.

**Step 2: in-engine repetition stop.**
* Config (`gefion_full_r1.toml`):
  ```toml
  [orchestrator.train.sampling.extra_body.repetition_detection]
  min_pattern_size = 16
  max_pattern_size = 64
  min_count = 8
  ```
  This needs at least 128 exactly periodic tokens. A more conservative choice is 32/128/6, which
  needs at least 192 tokens and adds about 1.6 ms per step.
* verifiers: finish-reason plumbing as in section 4, items 1 and 3.
* prime-rl: metrics, plus the `behavior.py` and `degenerate.py` alignment.
* Effort: about 1 day, plus a short canary run. In the canary, check the engine step-time delta, the
  rate of `finish_reason=repetition`, and decoded samples of stopped turns.
* Expected saving on top of step 1: about 8% (shakedown) / 6% (tp8).

**Step 3 (optional): an O(1) detector.** Do this only if step 2's scheduler overhead shows in the
canary. Add an `infer_opt` patch of `vllm.v1.core.sched.scheduler.check_stop` with a rolling-hash
16-gram detector that sets `FINISHED_REPETITION`, so all downstream plumbing is shared. It also
allows a coverage-style rule that catches near-loops with varying tokens. Effort: about 1 day.

**Not recommended:** in-turn NLL abort (section 2b), streaming abort (section 3b), logits-processor
forced EOS (section 3a), and live tool-cycle stops (section 1b, too rare).

## 6. Risks

* **Counterfactual drift.** Shorter loop turns change what the agent sees next, so rewards and
  trajectories differ from today. This is benign: the cut is the same kind of truncation the 16k cap
  already does. Watch the reward and length curves in the canary.
* **False positives on legitimate repetition**, such as generated tables or repeated test cases in
  tool-call arguments. None were found in the sampled fires. Pattern size 1 or 8 with a low count
  would be unsafe. Keep min_pattern_size >= 16 and a repeated span of 128 tokens or more, and log
  decoded samples of stopped turns.
* **Harness behavior on "repetition".** If the finish reason is not mapped to `length` on the wire,
  some harnesses treat the response as final and end the agent.
* **Scheduler CPU cost** of the built-in check grows linearly with `max_pattern_size - min_pattern_size`.
  Do not use large ranges such as 1/512.
* **More traces dropped as DegenerateTurn after cuts** (section 4, item 3). This is a policy decision,
  not a bug, but it must be made the same way in the live and post-hoc checks.
* **Version coupling.** `RepetitionDetectionParams` is new in vLLM 0.28. Re-check the field names when
  upgrading vLLM. `fast_generate` already passes the finish reason through generically.
