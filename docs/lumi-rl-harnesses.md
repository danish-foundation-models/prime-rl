# Multi-harness RL on LUMI

Use the r26 20-step SC pilot config for the final step-2619 model. It retains the
nine complete coding task pools and SQL. Each coding source is represented by
OpenCode, pi, Prime Agent and bash, each with one quarter of the original source
weight. The dispatcher chooses the source before expanding a group of eight, so
all siblings use the same task and harness. SQL retains its native read-only tools.
Source names expose harness-specific reward, errors and curriculum metrics.

The UCloud adapter implements ACP `open_process` using a managed agent job and a
sandbox-local input mailbox. The ACP session and child processes remain inside
the sandbox across parking. Read output only when the sandbox is active; a 400
lifecycle-transition response is retried as a read, never by relaunching the agent.
Managed-log EOF means the current end of available data; only a terminal job ends
the stream. Relay failure supervises pending reads and waits. Cancellation signals
the bridge, which signals the agent process group, before sandbox teardown.

Relay-only setup requires both the pinned offline Python bundle and the selected
harness asset bundle. The Python bundle includes agent-client-protocol 0.12.1.
Agent pins are pi 0.84.1 / pi-acp 0.0.33, Prime Agent 0.9.1 at its upstream harness
commit, and OpenCode 1.18.31. Bundle SHA256 sidecars are checked before staging.
Node setup recognizes the preinstalled pinned binary without invoking apt/curl.
Store large archives under /var/tmp, not the sandbox's small /tmp tmpfs. Export
large build artifacts in bounded chunks; the SDK download path has an output cap.
No runtime dependency downloads are permitted by relay-only policy.

OpenCode uses only the intercepted custom provider, including its small-model
setting. Automatic model-list fetches, upgrades, compaction, title and summary
agents are disabled. Training passes reasoning effort through to harness setup;
the Nemotron renderer controls model thinking and the 128K token budget.

Live acceptance checks require two model waits observed in the parked state,
a tool round-trip that writes and reads a sentinel, captured model turns, and
successful sandbox cleanup. Check Prime Agent's initial kernel startup separately
from steady-state model and tool latency. The probe uses synthetic model responses;
a real pilot must additionally verify training token capture and optimizer steps.

Pilot transfer uses Ring/Simple and the padded weight communicator validated on
16 trainer nodes and eight inference replicas. The optimized profile is pinned to
`artifacts/runtime/nemotron-super-production-reload/2c274f9dbb00008c`; do not mix it with the earlier LL
configuration. Dry-run the native launcher before submitting.

The pilot checkout must also contain
`src/prime_rl/inference/vllm/worker/expert_reload_map.py`: the optimized profile's
weight-transfer module imports this helper from the live PRIME-RL package. Merely
selecting the benchmark profile does not provide that package member. Check both
`expert_reload_map` and `weight_broadcast` imports before allocating GPU nodes.
The SQL compatibility gate includes environment construction and an MCP-backed
submission episode, not only row loading and isolated reward tests.

The production profile is self-contained and contains no benchmark variant index or
profile-output dependency. It pins the measured normal-header variant, enables
Nemotron receive filtering and skips reload-only empty-cache/diagnostic work via
the three `PRIME_RL_RELOAD_*` inference environment flags in r23. Numerical
validation must use this exact profile with those flags, not the benchmark wrapper.

For Slurm dependencies use an explicit `sbatch --dependency=afterok:<job>`
argument and verify `scontrol show job` before relying on the gate. The installed
Slurm does not consume a `SBATCH_DEPENDENCY` environment variable.

For deterministic finite coding pools, set `share_taskset = true` on each train
source. Identical resolved taskset classes and full configs share their read-only
client tasks during collection startup. Source iterators, curriculum state, and
server-side episode task instances remain separate. Infinite loaders are never
shared, and the default remains false for stochastic/custom tasksets. This avoids
four full Arrow decodes and copies when mixing four harnesses.

Interception selects the private guest relay only when the resolved runtime is
network-restricted. Unrestricted evaluation sandboxes retain the SDK public relay
URL: they do not receive the relay-only private hostname mapping. Inspect the
resolved per-task runtime, not just the source config, because task policy can
restrict an otherwise unrestricted source. Preserve the registration path, query
and capability when rewriting the origin.

For transient worker loss, set `env.retries.max_retries = 5` and
`env.retries.include = ["SandboxNodeLost"]` on train and eval sources. The native
server runs `run_slot`, which releases each failed attempt and constructs a fresh
episode/sandbox for the same task. Failed-attempt tokens never join the successful
trajectory; retry errors remain in episode history. Do not replay actions into a
replacement sandbox or resume a lost sandbox incarnation. Use the monitor flag
`--tolerate-node-lost` to count these failures without putting them in the 45/50
stop window. Other terminal failures still stop the pilot; mixed error types are
not silently classified as node loss.

For mixed-duration training, set `orchestrator.concurrency.lookahead_batches = 2`
and `eval_fraction = 0.125`. The dispatcher admits whole groups against weighted
taskset-family budgets, counting ready traces and full outstanding groups through
sink retirement. Harness variants of the same taskset share a budget. Budgets
round up to whole groups; ten equally weighted families with batch64/group8 allow
160 outstanding-or-ready traces (16 per family). Retired failures free admission
immediately; native episode retries handle recoverable attempts. This bounds fast
SQL replenishment but does not enforce an exact mixture in each batch or strict
IID sampling. Do not oversample within a group and accept its fastest answers.
While training admission is open, evaluation uses at most its configured fraction
of current concurrency; during training drain it can borrow unused capacity.
Omitting lookahead_batches preserves the default scheduling behavior.

Streaming agent SDKs must also be tested with TrainClient, not just EvalClient.
The interception server generates through the token-level training path, commits
tokens/logprobs and score-centering top-k evidence once, then returns buffered SSE.
This is not incremental token streaming. Chat tool-call deltas include their
required indices. Never route training streaming requests through the generic
relay, which lacks token-level training capture.

Offline search dependencies live in versioned `search-tools-v1` harness archives.
They add musl ripgrep 15.1.0 and fd 10.3.0 to `/var/tmp/vf-node/bin`, shared by
OpenCode, pi and Prime Agent. Pi must launch with that directory on PATH as well
as invoking Node by absolute path. Offline setup checks both tools before any
model generation (OpenCode requires rg). Online harness installation is unchanged.

Build new bundles with `scripts/lumi_bundle_harness_search_tools.py --source
<original-bundle-directory> --output <new-directory> --ripgrep <upstream-rg.tar.gz>
--fd <upstream-fd.tar.gz>`. The script pins upstream archive checksums, validates
source manifests, and refuses to overwrite existing outputs. Preserve old bundles
for run reproduction. Do not rely on lazy downloads inside relay-only sandboxes.

Run `scripts/lumi_probe_offline_harness_tools.py --harness <opencode|pi|prime-agent>
--output <artifact.json>` in the LUMI runtime. This drives the real harness with
scripted model responses and checks write, file discovery, content search, read,
edit, shell/kernel execution, parking, and teardown. OpenCode uses its actual
glob/grep tools; pi's default exposed toolset uses bash for rg/fd; Prime Agent uses
its persistent Python kernel. It is a dependency/tool smoke test, not a model
quality evaluation or proof that backend concurrency is healthy.

The r26 monitor uses `--max-failed-episodes 15`: it stops its job at 15 failures
in the last 50 eligible episodes, excluding confirmed image-build failures and
tolerated node loss. Failed images are quarantined across training harnesses. This is a failure
threshold, not a reward threshold; successful zero-reward attempts do not count.

Run-bound monitor wrappers should invoke exact-label sandbox cleanup after the
monitor exits successfully with state ENDED. The cleanup must re-read each sandbox
and verify its exact run label before deletion, then list again and record any
remaining sandboxes. Slurm cancellation and reservation release alone do not
delete sandbox instances. Never use a prefix or account-wide deletion filter.

Terminal-Lego uses `task-metadata/terminal-lego-memory2gb.jsonl` in the current
search pilot config: 2 GB RAM with the publisher CPU and 5 GB disk unchanged.
Keep the original `terminal-lego.jsonl` for provenance. Set this in task metadata:
the runtime resolver lets task resources override values equal to runtime defaults,
so `env.agent.runtime.memory = 2` alone does not override the publisher's 1 GB.

The current config is `configs/lumi/rl_nemotron_super_r5_128k_16t8i_sc_pilot_r37_search.toml`.
Training and evaluation allow 60 model turns for coding/search and 20 for SQL,
with 32,768 output tokens and 131,072 total tokens for SQL/search. Coding
harnesses use the compaction budgets below. Coding rollout timeouts are
5,400 seconds and tool timeouts 180 seconds; search rollout timeouts are 2,700
seconds. These agent budgets are separate from the pilot's 20 optimizer steps.

Bash sources in r36 enable context compaction at 98,304 tokens (96K), with
200,000 total task tokens and no separate cumulative output cap. Each inference
context and training sequence remains limited to 131,072 tokens. The task budget
sums branch sequence lengths; it is not a per-context limit or a strict one-compaction
count. The higher turn/time budgets still apply. Pi, Prime Agent and OpenCode
also enable native compaction with 200,000 total task tokens and no cumulative
output cap. All 44 coding entries (training and evaluation) use these budgets;
SQL/search retain their existing 128K task budgets.

Native Bash sends a unique intercepted model-request ID and records explicit
`compaction_attempt` / `compaction` links in a trace-specific sandbox sidecar.
The harness imports those links before training samples are constructed, so
rejected summaries are excluded while accepted summaries and resumed work receive
the task advantage. Keep the live verifiers source with this configuration; the
change is bundled into the Bash program at launch and needs no image rebuild.
However, every changed bundled Python script must have its exact SHA-256 recorded
in a new pinned offline-Python manifest before submission. Validate all configured
script variants against that manifest; external-agent compaction probes alone do
not exercise Bash/search scripts. An old manifest rejects them before execution.
Validation artifacts are under
`artifacts/training/nemotron-super-r5-rl-scale-20260917/compaction-validation-20260922`.
Scripted Bash compaction, GRPO credit, msgpack transport and SC split-gradient
checks pass. Native Pi, Prime Agent and OpenCode binaries have also compacted
and resumed successfully in parkable UCloud sandboxes against a scripted model
endpoint, retaining task files. This does not test real-model summary quality.

Pi and Prime Agent explicitly advertise a 131,072-token model context and use
native settings with 32,768 reserve tokens and 16,384 recent tokens. Prime Agent's
automatic `/refine` review is disabled to avoid unrelated auxiliary model calls.
Pi's pinned agent can recover from context overflow within a continuing tool loop;
its proactive threshold check runs at agent-run boundaries. The training interceptor
therefore leaves a rejected, unsampled context-limit request recoverable only when
harness compaction is enabled. Task-level token and turn limits still terminate
normally. Buffered streaming and non-streaming recovery are both verified.

OpenCode enables native `compaction.auto`, disables separate tool-output pruning,
and clears `OPENCODE_DISABLE_AUTOCOMPACT`. Its pinned 1.18.31 implementation
requires an explicit `limit.input = 131072` alongside `limit.context` to honor
`compaction.reserved = 32768`; otherwise the trigger uses the output-token limit.
Keep both limits. See the pinned upstream `session/overflow.ts` implementation.

External native agents do not currently emit accepted/rejected summary semantic
links. Their physical request graphs still preserve original work, summary sampling
and resumed contexts as separate branches. GRPO assigns one task advantage before
branch splitting. Sampled summaries receive that advantage; copies reused as prompt
context are masked. Unlike Bash's explicit acceptance tracking, a rejected native
summary attempt is not specially excluded. The recorded native graphs are checked
with synthetic token/logprob evidence for once-only sampled-node training and
per-task GRPO credit; this is not a full model/tokenizer integration test.

Run the job-specific monitor wrapper in a named tmux session on the submit host,
for example `tmux new-session -d -s rl-r36-monitor 'cd /scratch/project_465002751/rasmus && bash tmp/watch-search-r36.sh >> tmp/search-r36-monitor.log 2>&1'`.
Check the session and the timestamp in its `status.json` after detaching. A tool
execution session or bare background shell is not a durable monitor lifecycle.
The wrapper retains exact-run ownership checks before sandbox cleanup.

The LUMI RL submit wrapper checks exact bundled script hashes against every
configured offline-Python manifest before rendering or submitting. To refresh
manifests after Python harness edits, run `scripts/lumi_refresh_offline_manifests.py`
inside the shared runtime with `--config OLD.toml --output-config NEW.toml
--output-dir /absolute/new-artifact-directory`. It verifies package archive hashes,
preserves the old artifacts, and writes new manifest references and exact script
sources. It does not install new dependencies: rebuild the package archive if a
script's dependency requirements change. `--config CONFIG.toml --check` is read-only.

Training-source initialization shares task-content identity calculations across
identical read-only task objects for that startup only. Each sampler keeps separate
reward state and RNG. The temporary identity cache is discarded after initialization;
it is not attached to mutable TaskData. Startup runs off the event loop and logs
its duration. This reduces repeated hashing; dataset materialization can still
consume substantial startup time.

The prepared next-run config is
`configs/lumi/rl_nemotron_super_r5_128k_16t8i_sc_pilot_r39_search.toml`.
All 48 training/evaluation entries allow 200 agent model turns, including SQL
and search. Task-token and wall-time budgets remain unchanged and may stop a
rollout sooner; a model turn can include multiple tool calls. The pilot remains
20 optimizer steps. This config does not modify the running r38 attempt.
