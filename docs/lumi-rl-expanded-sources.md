# Expanded agentic training sources

Use `configs/lumi/rl_nemotron_super_r5_128k_16t8i_pilot_r10.toml` for the broad pilot:
ScaleSWE, R2E-Gym, SWE-Lego, MultiSWE, TMax, Terminal-Lego, SWE-Smith, OpenSWE,
and SWE-rebench-v2. The config uses full publisher task pools with equal source
weights, batch64/group8, adaptive128–512 episodes, 16 trainer nodes and eight
optimized inference replicas. Preserve held-out evaluation exclusions.

Do not require an individual gold solution or baseline/gold check for task
admission. TMax supplies hidden verifiers for its tasks. Representative environment
checks establish integration; image-build and provisioning errors are infrastructure
failures, not zero-reward training examples. Keep optional diagnostic allowlists
separate from full-pool configs.

## Image preparation

Run `deps/prime-envs/tools/ucloud_index_pools.py SCALE_DIR --output NEW_INDEX`
inside the shared runtime. The indexer consumes pinned publisher sources, uses
bulk Git blob reads for file-based task sets, and records recipes in SQLite.
`image_recipe_db` and `image_build_cache` enable UCloud preparation on first use.
Builds are shared across a rollout group and processes, with a fixed lock inode
per recipe and persistent receipts. A separate executor prevents cold builds from
blocking the asyncio executor used by live rollouts. Never modify an active index.

Reserve managed builders before launch. Cold builds allow7200 seconds and source
setup allows7500 seconds. Successful image builds are reused; failed builds retain
diagnostics. Only environment inputs and dependency bootstrap code enter image
contexts. Hidden verifier tests and oracle patches remain outside images.

Source requirements:

- TMax: pin the task Git commit and set `task_root` to its complete export. Preserve
  per-task resources, hidden verifiers and separate-verifier semantics.
- Terminal-Lego: use the pinned HF snapshot and hydrate environment LFS inputs on
  demand. Bake verifier dependencies without executing candidate programs or tests.
  The offline wrapper verifies the original script hash and preserves the runtime
  test body, including candidate-program execution and inline environment settings.
- SWE-Smith: prepare all task branches per repository image before relay-only
  execution. Capture the evaluation parent before the agent runs. Report missing
  grader profiles instead of silently shrinking the pool.
- OpenSWE: actually download the gated JSONL. Preserve its task Dockerfile and
  intended Conda environment; unusual COPY layouts use the pinned source repository
  as build context. Set explicit workdir overrides for nonstandard repository paths.
- SWE-rebench: use the original public `docker.io/swerebenchv2/...` images behind
  Prime's aliases. Preserve the hidden test patch and language-specific grading.
- MultiSWE: warm dependencies in its task image, preserving the baseline source tree.

All source runtimes must honor relay-only networking. Harbor defaults must not
silently enable unrestricted egress. Prime registry aliases are not directly
pullable by UCloud; never replace a task image with a generic environment.

## Runtime and checks

SWE-Smith's external dependency layer is selected by
`PRIME_RL_ENVIRONMENT_SITE_PACKAGES` and requires a `.ready` marker. The supported
layer contains `swesmith[validate]` at9b74ac0, `swebench==4.1.0`, `ghapi==1.0.6`, and
`fastcore==1.7.29`. Install narrowly with `uv pip --no-deps --target`; do not shadow
the shared HTTP stack. Export the layer path before `lumi_job_container.sh` builds
PYTHONPATH. Verify profile imports and the shared SIF after dependency changes.

Check full native task loading and retain counts, revisions and image receipts.
Use separate processes for concurrent Hugging Face loads: threaded loads can race
on tqdm's global lock. Harbor task parsing overlaps file reads with deterministic
ordered worker pools. A launcher dry-run does not establish image availability or
verifier correctness; retain representative live provisioning checks.

Keep `creates_per_sec = 0.0` in this pilot. Gateway admission and bounded rollout
concurrency provide pacing. If explicitly enabling the optional shared limiter,
use non-append read/write on its fixed inode, overwrite then truncate under lock;
never rename or unlink a live bucket. Legacy NUL repair must preserve its timestamp.

UCloud capacity preparation accepts at most100 slots per request. Reservations
express desired capacity, not physical readiness. Monitor actual admission errors,
image builds, completed rollouts and trainer updates. Release only owned named
reservations and clean only sandboxes matched to the exact run label.

The warm reservation uses 100 parkable slots of 4 CPUs, 4096 MiB RAM and
10240 MiB writable disk. This matches the common SWE task shape and does not
override individual task limits. The gateway additionally reserves parking
storage: at this size it charges 19520 MiB disk per slot, including memory
backing, private pages and fixed overhead. Larger tasks may need additional
backend capacity. Reservation renewal must retain this shape.

## Harbor metadata cache

Use `ucloud_cache_harbor.py CONFIG_TOML COUNTS_JSON OUTPUT_DIR` before allocating
GPUs for full TMax/Terminal-Lego pools. Set each source's `task_data_file` to its
resulting JSONL. The cache holds the complete native HarborData, including hidden
verifier settings and original task paths; it does not select or change tasks.
Export verifies row counts and exact serialized-data hashes after round-trip.
Loading rejects configuration mismatches and incomplete/nonsequential records.
Task files remain at their pinned paths for setup and grading.

Compute-node Lustre path resolution can make per-task Harbor parsing much slower
than a warm login-node check, even with parallel reads. Cached loading avoids
those repeated metadata operations. The r7 config uses the full metadata caches;
prefer it for the current pilot.

A queued pilot can be observed with `scripts/lumi_watch_rl_pilot.py` from host
Python (`uv run --offline --no-project --python /usr/bin/python3.11 python ...`).
Slurm commands belong on the host; the operational SIF does not provide `squeue`.
The observer invokes `lumi_prepare_pilot_capacity.py` through the SIF only for
provider reservations, when the job starts. Reservation IDs must include the
Slurm job ID, preventing accidental reuse across pilots. It records completed
trace counts, environment errors and the first optimizer-update log, including
whether its loss is finite. On job exit it releases only those exact reservation
IDs; it never deletes sandboxes/images or resubmits jobs. With the explicit
`--stop-on-infra-failure` option it can cancel only its specified job.

## Startup ordering and task decoding

Synchronize the startup policy with the inference replicas before loading task
pools. Trainer rank zero waits for the inference rendezvous while the other
trainer ranks enter a collective; task loading must not delay this handshake.
Register dispatcher/evaluation hooks after the pools load and notify them of the
already-synchronized policy, without transferring startup weights a second time.

Materialize sources sequentially off the asyncio event loop. Concurrent Python
threads decoding Arrow rows contend heavily for the GIL and share datasets/tqdm
state. SWE tasksets use `iter_dataset_rows` to decode bounded batches of256 with
native feature decoding, preserving order, fields, filters and task identities.
Harbor sources continue to use the verified complete metadata caches. Compare
full task counts and representative row equality before submitting a new pilot.

## Terminal-Lego preparation checks

Refresh repaired recipes into a new index; never edit an index used by a job:

```bash
uv run python deps/prime-envs/tools/ucloud_refresh_terminal.py OLD_INDEX NEW_INDEX --scripts ORIGINAL_SCRIPTS_JSON
```

The script JSON maps pinned `task_NNNNN/tests/test.sh` paths to original contents.
The refresh preserves every source row and invalidates Terminal-Lego prepared-image
shortcuts. Recipe hashes select new build-cache entries; keep failed receipts for
diagnosis. A declared `directories = ["task_file"]` restores an absent empty workspace
that Git cannot represent. It never synthesizes missing input files. Literal Docker
COPY sources are checked locally before a managed build is requested.

Verifier provisioning uses an explicit installer classifier, preserving conditionals,
candidate execution, working-directory changes, test invocations and reward logic
at runtime. Unclassified shell stays at runtime, never executes as bootstrap. Some
publisher tasks still need task-specific offline dependency/input preparation: a
successful syntax check is not proof that every task can run without network access.
The image fingerprint includes the wrapper version as well as original script content;
images prepared with a different wrapper must be rebuilt. Bashlex is provided by the
shared runtime and declared by terminal-lego.

Difficulty-pool sampling caches immutable task identities on initialization. Do not
rehash complete task records per draw; this blocks dispatch on large publisher pools.
Keep task order, weights, RNG calls and checkpoint reward state unchanged.

## Observation-only loop annotations

Completed train and eval traces receive `info.loop_annotations` and `metrics["loop/..."]`
before recording and aggregation. Analysis runs off the event loop. It never changes
rewards, advantages, admission, termination or sampling. These are review heuristics,
not proof that a model is stuck.

The detector scans both `reasoning_content` and `content`, including text content
blocks, in generation order. The joint ending matters: an answer field does not by
itself prove recovery. Text flags require a repetitive suffix on the last sampled
assistant message. Fenced code, block quotations and XML function payloads are excluded
from this prose check to avoid flagging before/after code edits.

Literal text repetition requires at least 64 words covered by third occurrences of
normalized five-word phrases, with at least 50% coverage in both the last 128 and last
32 prose words. A separate, weaker `low_novelty` candidate catches rephrased arguments:
at least 768 prose words, with previously seen trigrams covering at least 45% of the
last 512 words and 35% of the last 128, and bigrams covering 60% of the last 512.
These are lexical heuristics, not a semantic proof that an argument is circular.
The suffix scan retains at most 8192 prose words per message and reports its offset.

Action cycles follow physical parent links, canonicalize JSON arguments and count
sampled assistant turns. Four repeats of a cycle of one to four action bundles are
flagged even if tool outputs vary or the final response is pending. Three repeats of
identical matched call/output pairs remain stronger evidence. Prompt copies do not
count. Different intervening actions break the cycle. A repeated action with changing
results can be legitimate polling; action candidates deliberately expose that weaker
evidence rather than requiring byte-identical output. Earlier repetition remains
visible but does not flag an otherwise nonrepeating ending.

Cutoffs and errors are shown separately. `agent_completed` is not evidence of recovery:
the underlying model call may have ended with `finish_reason=length`. None of these
flags proves that recovery was impossible. Repetition in code, paraphrases without
lexical overlap, and longer action cycles can remain undetected. Evidence is capped at
20 entries and identifies zero-based nodes. The older 16-word shingle measurements
remain supplemental reasoning metrics; they no longer decide text flags.
`flagged_output_tokens` counts tokens on flagged messages, not measured wasted tokens.

Regression fixtures in `tests/unit/orchestrator/fixtures/loop_review_r9.json` preserve
four manually reviewed pilot messages: two recurring-analysis candidates that exhausted
their token budget and two negative examples with repeated code. These are calibration
examples, not an independent accuracy estimate. Validate changes against additional
held-out traces before using flags for any training or termination decisions.

The dashboard trace metadata shows loop observations and numerical metrics. Existing
traces can be analyzed without changing their original records or scores:

```bash
uv run python scripts/annotate_rollout_loops.py RUN_DIR --write
```

Without `--write` this reports only. With it, the tool exclusively creates a versioned
producer stream under `monitors/file/traces/annotations/loops-vN`; it refuses to overwrite
an existing stream. The companion summary lists counts and example trace IDs. Older-run
backfills add trace annotations, not retroactive changes to recorded training metrics.

## Parking during inference

Every r10 train/eval agent explicitly enables `runtime.parkable = true`.
The verifiers-ucloud integration owns managed process launch and sandbox-bound
relay registration; do not implement a parallel parking path in PRIME-RL.
The named rollout-capacity reservation also requests `parkable=True` so storage
accounting includes checkpoint capacity. Builder reservations are unchanged.

Before submitting, run the owning checkout's `scripts/lumi_probe_parkable_agent.py`
and `scripts/lumi_probe_parkable_harness.py` through `lumi_run_in_container.sh`,
each with a new `--output` path. Require observed `parked` states and successful
post-response continuation, not only `spec.parkable`. The harness probe uses a
local scripted model endpoint; it validates integration, not model quality.


## Relay and image recovery

Relay response commits can wait for a parked sandbox to wake. The UCloud adapter
uses a 120-second response-commit timeout and retries transport failures up to
three times with the same request identity and response bytes. It never retries
generation or arbitrary control mutations. SDK retryable-503 commit handling
remains active. Unrelated control requests retain their shorter timeout.
Read-only managed-job polling tolerates a missing route for at most 120 seconds
and polls every five seconds; it never restarts the managed process. Permanent
route loss still fails the rollout. These are bounded client recoveries, not a
claim that backend route loss is repaired.
Image-cache receipts are checked against the backend build record. An explicit
404 archives the stale receipt and re-submits the identical pinned recipe under
the existing cross-process lock. Other errors do not invalidate the receipt.
Before relaunch, run adapter tests and concurrent real-harness park/resume probes.

The synchronous SDK `SandboxClient` is a request-scoped urllib client, not a
context manager. Use it directly; tests must retain its real lifecycle contract.
A lease-renewal 410 with exactly "request is already completed" is an expected
completion race: let the response forward/commit operation determine success.
Do not suppress other 410s or treat a response-commit 410 as success.


For monitored runs, start `lumi_watch_rl_pilot.py --stop-on-infra-failure
--max-failed-episodes 15 --poll-seconds 15` with explicit job/run paths. It
cancels only that job on fatal component errors or 15 execution failures in the
last 50 eligible episodes. Confirmed image-build failures are excluded from
that window; zero rewards never count. `--tolerate-node-lost` additionally
excludes isolated terminal node-loss episodes from the rate calculation.

`ImageBuildFailure` quarantines the image and recipe-index pair in the training
source. Every matching task across harnesses is removed from finite sampling
pools; in-flight and pending train groups for that image are cancelled with
correct group/permit accounting. Reward/difficulty estimates and category
targets are unchanged. Exclusions live in `quarantined-images.json` and source
checkpoints, survive resume, and do not affect a newly versioned recipe index.
A fully exhausted source stops contributing; if a required category becomes
unfillable, the progress watchdog stops the stalled run rather than altering
the mixture. Other runtime failures do not quarantine task data.

The watchdog also stops after 600 seconds without new prompt/generated tokens,
successful episodes, or optimizer progress, once useful work has started.
Initial startup has a separate 3600-second budget. These are configurable with
`--max-no-progress-seconds` and `--max-startup-seconds`. It uses per-replica token
counters in `metrics.jsonl`, so long tasks that are still generating are not
mistaken for a stall. Prior-attempt traces and metrics are ignored using this
job's Slurm start time.

The eight-replica run additionally uses a conservative aggregate floor of
500 generated tokens/s averaged over 600 seconds, after 1200 seconds of
warmup from the first observed generation. Configure these with
`--min-generation-tps`, `--throughput-window-seconds`, and
`--throughput-warmup-seconds`; set the floor to zero to disable it or choose an
appropriate floor for a different-sized allocation. Weight-transfer pauses,
replica counter resets and changes to the observed replica set reset the
throughput window. Progress from a few short tasks cannot override this floor.
The stop report records the measured rate and threshold. The guard establishes
underperformance, not its root cause; inspect backend/launcher/trainer state
before attributing it to a particular service.

The observer writes `automatic-stop.json` and leaves sandbox deletion to
ownership-checked cleanup. It releases only job-bound reservations after exit.

CPU admission may report either "CPU load" or "CPU pressure". Idempotent file
transfers retry these specific 503 rejections with bounded exponential backoff
(up to eight attempts). Do not replay arbitrary sandbox commands on this basis.

OpenSWE Dockerfiles are normalized from `mirrors.cloud.aliyuncs.com` to
`deb.debian.org` during indexing. The former resolves to an Alibaba-only address
and timed out on the managed builders. Preserve all other task dependencies and
source revisions. Repair existing recipes in a new SQLite index, retain the old
index and failed receipts, and point the next run at the new index only after
checking a representative changed image. A changed recipe gets a new cache key;
never erase failure receipts to force an identical retry.
