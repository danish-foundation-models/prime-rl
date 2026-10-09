# Throughput-aware rollout scheduling

Enable `orchestrator.concurrency.throughput_balancing = true` with a positive
`lookahead_batches` and trace-count `orchestrator.batch_size`.

```toml
[orchestrator]
batch_size = 128
group_size = 8
max_off_policy_steps = 5
constant_trainer_batch_size = true

[orchestrator.concurrency]
throughput_balancing = true
scheduler_target_mix = 0.5
scheduler_oversampling = 3.0
scheduler_min_oversampling = 0.0
scheduler_max_oversampling = 16.0
scheduler_acceptance_floor = 0.05
scheduler_inventory_factor = 2.0
lookahead_batches = 4
initial_inflight = 256
min_inflight = 128
max_inflight = 512
category_targets = { swe = 72, terminal = 40, sql = 8, search = 8 }
category_batch_tolerance = 2
category_max_surplus = 8
```

Each `[[orchestrator.train.source]]` sets `balance_category`, such as `"swe"`.
Without it the category defaults to the taskset id. Explicit targets must cover
exactly the configured categories and have positive finite weights. They normalize
to fractions; the example means 72/40/8/8 traces per batch, not percentages.
Within each category, source ratios guide best-effort taskset/harness selection;
there is no hard per-taskset quota.
A newly added harness must share its taskset's source weight if that taskset's
relative share should stay unchanged.

The prepared r40 config categorizes ScaleSWE, R2E-Gym, SWE-Lego, MultiSWE,
SWESmith, OpenSWE, and SWE-rebench-v2 as SWE; Terminal-Lego and TMax as terminal;
BIRD as SQL; and OpenSeeker as search. Evaluation sources do not enter these quotas.

## Batch guarantees

The search-free full-run configuration is
`configs/lumi/rl_nemotron_super_r5_128k_16t12i_sc_full100_bs512_r16.toml`.
It runs 100 optimizer steps with batch 512 and group size 8, targeting SWE 240,
terminal 144, SQL 32, tooluse 64, and instruction 32 per batch. Rollout concurrency
starts at 512 and adapts up to 1,024; the minimum remains 128 so existing KV-pressure safeguards can drain the pool. These
limits count whole episodes, including sandbox and tool waits, rather than active
model requests. Category proportions and the five-step policy-lag limit are unchanged.
OpenSeeker/Monid is omitted from training, including its
category target, so search availability cannot block training batches.
BrowseComp-Plus remains an evaluation using local BM25 retrieval; it does not
depend on Monid. When disabling a category, remove both its training sources
and its category target, and redistribute its share so targets sum to the
intended batch size. Keep completed-run configs intact and use a fresh run
name and sandbox ownership label.

All GRPO attempts in a group must return or be accounted for by existing
failure/cancellation handling before advantages are computed. There are no racing
replicas or fastest-eight selection. Finalized groups may contribute traces across
batches, with group advantages already computed. Each trace's compaction branches
stay together. The balance unit is eligible traces, not tokens or GPU time.

With category targets configured, the sink constructs a complete batch before
removing any data. Each category's count is within `category_batch_tolerance`
traces of its real-valued batch target, rounded inward to integer bounds. In the
example: SWE 70–74, terminal 38–42, SQL 6–10, search 6–10, totaling 128. When
sufficient data is ready and there is no mixture debt, it selects exactly 72/40/8/8.
Only ready traces from finalized groups are eligible; source selection within a
category uses weighted deficit and FIFO within each source.

A second bound limits each category's cumulative selected count to its cumulative
target plus `category_max_surplus`. Repeated per-batch deviations cannot keep
favoring fast categories. This is a cumulative surplus bound, not a rolling-window
percentage guarantee. The tolerance is not an instruction to split each GRPO group
before advantage computation.

When the bounds cannot be met, the batch waits without consuming queued data.
Missing terminal work cannot be substituted indefinitely with SQL. This can idle
training if a whole category becomes unavailable; there is no silent timeout that
changes the mixture. Individual tasksets need not appear in every batch. Admission
continues for missing categories while adaptive resource limits permit it.
Counters track selected nonempty trainer payloads, not optimizer acknowledgements.
Planner state and category configuration are saved with train-source checkpoints;
resuming with changed balancing configuration is rejected.

Without explicit category targets, selection is best-effort weighted deficit by
taskset family and does not wait for an absent family. Coarse category guarantees
therefore require `category_targets`, not just `throughput_balancing`.

## Concurrent rollout production

PRIME's adaptive controller owns the active episode limit and reacts to engine
KV usage, queues, and preemptions. The scheduler reads the current dispatcher cap;
it never increases that cap itself. Existing soft draining and hard overload
cancellation remain in force, as do admission rate limits and evaluation reserves.

The allocator implements MiMo §6.3 equations 6 and 7. Its scheduling scopes are
our four coarse categories when category targets are configured, otherwise individual
sources. All members of a category must use the same GRPO group size. Within a
category, normalized configured source ratios drive a second smooth weighted
round-robin; the existing curriculum chooses the actual task. A low-acceptance
individual taskset is not promoted into a separate required category.

For scope i, B_i is its target retained group-equivalents per batch, r_i is its
estimated useful fraction, and t_i is its estimated duration. Source estimates use
moving averages (duration coefficient 0.2, useful fraction coefficient 0.1), initialized
to 600 seconds and 1. Scope estimates average sources using their configured group
weights. The useful fraction counts eligible traces per attempted group size, so
partial successful groups are represented without pretending every group yields eight.
The acceptance floor is 0.05; it bounds generation-demand inflation to 20x.

Expected generation demand is m_i = B_i / r_i. The pending-group budget is
(1 + p_i) m_i, with p_i = clip(c t_i - 1, p_min, p_max). Bisection solves c so the
demand-weighted mean oversampling equals the selected global ratio. Budgets round
up to whole groups, at least one, and apply to unretired groups. They do not replace
the active-episode cap. Ready category buffers have separate thresholds.

The base mean oversampling is 3, with scope bounds 0–16. To avoid counting completed
group members as active permits, the mean can increase with the adaptive cap: we
request at least `scheduler_inventory_factor * active_cap` expected unfinished
episode reservations, subject to the oversampling ceiling. This headroom rule is
our integration choice; MiMo specifies a global mean but does not prescribe this
PRIME-specific relationship. Integer budgets and different group sizes can make the
realized reservation total differ from the requested total.

Within these budgets, scheduling uses smooth weighted round-robin with weights
alpha * B_i / r_i + (1-alpha) * max(B_i - A_i, 0) / r_i, where A_i is ready accepted
group-equivalents and alpha is 0.5. Thus a filled current-batch share retains
long-term generation demand, but an unfilled category gets more priority. Ready
inventory and resource guards still limit prefetch. Blocked scopes do not bank launch
credit. The initial active-cap worth of admissions uses t_i * m_i weights for a
steady-state occupancy estimate. Without measured priors, the first fill begins from
the common duration/yield defaults and learns as work completes; it is not an oracle
startup allocation. Credits, estimates, and startup progress are checkpointed.

Live-policy episode durations subtract intervals when the dispatcher is paused for
policy replacement. This estimates active rollout time; tool work can overlap a
policy pause, so it is not an exact model/environment duty-cycle measurement.
Frozen-source episodes do not subtract live-policy update pauses. Predictive KV/rank
placement and startup replay are not part of this change.

The ready target is `batch_size * lookahead_batches` (512 in the example), divided
across categories. Fast categories stop admitting new groups when their ready share
is full. A late burst can overshoot it; that does not globally block missing-category
refill. Token payload has a separate admission guard at `ready_target * seq_len`.
Outstanding group reservations plus ready traces have another ceiling of
`ready_target + max_group_size * active_concurrency`. These are admission thresholds,
not byte-exact memory limits: already admitted rollouts and bounded output queues
can overshoot them. Existing groups continue finishing when new admission stops.
The token guard takes precedence over mixture refill; a full memory buffer with
no feasible batch requires operator attention rather than violating the mixture.
Current environments have one trainable trace per episode; multi-agent trace
amplification needs additional budgeting.

Trainer handoff retains its prefetch bound. Rollout dispatch can continue while a
batch awaits training within resource and five-step policy-lag limits. Policy updates
still pause dispatch, and stale results are removed. The collector currently awaits
packing/handoff; the independent dispatcher can continue into a bounded result queue,
so a long handoff can still cause backpressure. Larger rollout batches increase the
number of packed microbatches accumulated per update; they do not increase the
configured 128K sequence limit or per-microbatch GPU activation shape.

The full-run config enables the existing startup evaluation with
`orchestrator.eval.skip_first_step = false`. It uses the existing asynchronous
evaluation scheduling and concurrency budget without a startup training barrier.
Regular evaluations run every 12 optimizer steps; the heavier SWE-Bench,
Terminal-Bench-2, and Senior SWE-Bench evaluations run every 25 steps.
Checkpoints are saved every 5 steps, retaining the latest two and every 25-step
milestone. This approximately preserves evaluation and checkpoint cadence by
training samples at batch size 512.

## Monitoring and validation

Compare trainer `time/wait_for_batch`, inference requests running/waiting, and
actual trainer activity. Episode concurrency includes tool/provisioning/verification
waits and is not the same as active model-request concurrency.

- `scheduler/ready_traces`, `partial_group_episodes`, `buffered_tokens`: inventory.
- `scheduler/blocked/*`: resource or ready-inventory admission limits.
- `scheduler/batch_balance_blocked`: enough total data may exist without a valid mix.
- `scheduler/batch_shortfall/*`: missing traces to meet category minima.
- `scheduler/batch_share/*`, `trained_share/*`, `target_share/*`, `debt/*`: mixture.
- `scheduler/batch_share_total_variation`: per-batch mixture error (zero is exact).
- `scheduler/duration_seconds/*`, `usable_yield/*`: source estimates.
- `scheduler/group_budget/*`, `launch_share/*`: MiMo scope budgets and selection weights.
- `scheduler/mean_oversampling`: selected demand-weighted oversampling ratio.
- `scheduler/blocked/source_group_budget`: unretired groups exhausted scope budgets.

Scheduler tests cover skew, missing categories, cumulative surplus, recovery,
checkpoint layout, and oversized ready buffers. Before submission, run native group,
compaction, stale-filtering and policy-lag checks and the launcher's `--dry-run`.
Measure throughput and mixture together. Synthetic simulations do not predict GPU
speedup or prove behavior under sandbox failures.

## Relevant design reference

The allocator and selector implement equations 6 and 7 from
[MiMo-V2.6 §6.3, pp. 29–31](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL/blob/main/MiMo_V2_6_technical_report.pdf).
Category targets, partial-group useful fractions, adaptive-cap headroom, and integer
rounding are explicit integration choices. The report's predictive per-rank dispatch,
startup replay, and distributed payload storage are separate mechanisms, not enabled
by this scheduler. Benchmark live throughput and mixture before claiming an improvement.

## Shared smoothed task replay

For `curriculum.sampler.type = "difficulty_pool"`, select
`strategy = "smoothed_replay"` to use recent valid task outcomes instead of hard
reward bins. Set `shared_group` to the same explicit taskset identity across
harness sources. Outcomes for matching content-based task keys are propagated
once to each peer; sampler RNGs and admission gates remain independent. Use
separate seeds for the harness sources so initial exploration covers more tasks.
Infrastructure failures do not update task difficulty.

The estimate is `(1 + sum(recent_rewards)) / (2 + count(recent_rewards))`, using
at most `history_size = 64` valid rollout rewards. Replay weights are
`max(0.05, 4*p*(1-p))`, so 1/8 and 2/8 results retain useful weight and uncertain
zero-success tasks are not excluded. This is a sampling heuristic, not a
confidence interval for independent Bernoulli trials.

The r11 setting `replay_fraction = 0.2` reserves up to 20% of draws for observed tasks. The
fraction ramps as `min(1, observed_tasks/replay_warmup_tasks)`, with
`replay_warmup_tasks = 16`, to avoid a tiny early replay pool dominating. Other
draws explore unseen tasks, falling back to all available tasks once every task
has observations. Category targets and adaptive rollout concurrency still belong
to the existing scheduler. Quarantined tasks are excluded from both paths.

History and per-source RNG state are checkpointed. Legacy latest-mean checkpoints
are treated as one weak observation per task, not a fabricated group of trials.
The default `strategy = "pools"` preserves the explicit threshold/weight policy.

## Short Hetzner integration run

`configs/lumi/rl_nemotron_super_r5_128k_16t4i_sc_hetzner_smoke3_r1.toml`
keeps the production 16-node trainer, SC/ECHO, 128K context, final SFT input,
all 32 training sources, and optimized NCCL inference stack. It requests four
inference nodes and three hours, with three updates at batch 128/group 8,
128 initial / 256 maximum inflight episodes, and a one-step policy-lag limit.
All 14 eval sources are capped to one example, at startup and every two steps.
Checkpoints are saved every two steps and at the final step, retaining the latest.

Success requires actual optimizer updates, changed-policy inference following
weight transfer, successful checkpoint writing, and completed valid rollouts/evals.
A short run cannot establish every source's reliability or production throughput.
Managed rollout registration is required; warm retention is expected and is not
an infrastructure failure. Use a distinct run label and monitor/cleanup ownership
from the 100-step run. The long queued run is not replaced by this smoke.

For the 960-node-minute `dev-g` budget, use
`configs/lumi/rl_nemotron_super_r5_128k_16t2i_sc_hetzner_smoke2_dev_r1.toml`:
16 trainer + 2 inference nodes for 53 minutes, two updates at batch 64/group 8,
64 initial / 128 maximum episodes, and checkpoint/eval interval one. Agent tasks
are capped at 20 turns, 16,384 total tokens and ten-minute rollouts for this smoke.
All sources remain configured, but two batches cannot establish coverage of every
source. This tests the 128K-capable stack without proving full-length memory behavior.
