---
name: cyber-forge
description: Generate, inspect, validate, evaluate, or train with the Cyber Forge v1 synthetic cyber environment.
---

# Cyber Forge

## Generate

Run from deps/research-environments after installing the local environment:

    uv pip install -e ./environments/cyber_forge_v1
    uv run cyber-forge-synth --output <fresh-directory> --count <n> --seed <seed>

The sequence is addressed by seed and start index. To extend a corpus, write to a fresh
directory and set start-index to the previous end. The same generator version, seed,
index window, and filters reproduce the same tasks. Never edit generated task bundles to
create diversity; change the synthesis index or generator version.

Use repeatable world-pack filters to restrict the eligible template set:

    uv run cyber-forge-synth --output <fresh-directory> --count <n> --seed <seed> \
      --world-pack tenantdesk-saas-v1 \
      --world-pack buildgrid-registry-v1

Repeated values form a union. Families without an implemented template in the selected
packs are omitted, and an empty family/template intersection is an error. The taskset
equivalent is `world_packs = ["tenantdesk-saas-v1", "buildgrid-registry-v1"]`.

`manifest.jsonl` is an ordinary corpus index containing the root seed, synthesis index,
resolved task coordinates, and content hashes. Per-task `task.json` exposes world-pack,
template-lineage, spec-fingerprint, skill, difficulty-profile, required-tool, and public
artifact metadata. Neither contains reference solutions or grader cases. The taskset
materializes only task artifacts into a learner runtime.

Keep corpus mechanics domain-neutral. Do not add secret salts, keyed task IDs, signing
requirements, or cybersecurity-flavored access control to generation. Put cyber behavior
inside task compilers and graders, and enforce learner/scorer visibility through export
and topology boundaries.

Generation must not call closed model APIs. The corpus CLI remains procedural. The
offline guided control plane in `synthesis.client` and `synthesis.pipeline` can propose
quarantined candidates through an OpenAI-compatible open-weight endpoint. It requires
model ID, immutable revision and license provenance, validates the exact served-model
identity, derives phase/attempt seeds from the generation coordinate, records request and
response hashes, and stops on evaluator accept, reject, or its revision budget. A
proposal is not a corpus task: static validation and an independent admission evaluator
remain mandatory, and only an accepted `AdoptedRecipe` may proceed toward deterministic
materialization.

### Guided authoring

Run guided authoring from `deps/research-environments`. First choose and persist a
coordinate:

    uv run cyber-forge-author coordinate \
      --seed 7 \
      --index 0 \
      --world-pack edgefleet-appliance-v1 \
      --output coordinate.json

Generate against the loopback OpenAI-compatible endpoint. Model ID, immutable revision,
and license are required. Supply a weights digest for anything intended for publication:

    uv run cyber-forge-author generate \
      --coordinate coordinate.json \
      --model-id <open-model-id> \
      --served-model-id <exact-response-model-id> \
      --model-revision <immutable-revision> \
      --model-license <license> \
      --weights-sha256 <sha256> \
      --max-phase-repairs 2 \
      --output envelope.json

Use `--base-url` and `--allow-remote` only for an explicitly approved self-hosted
open-weight endpoint. Known closed-model provider hosts are rejected. If authentication
is necessary, pass only `--api-key-env VARIABLE_NAME`; never put a credential value on
the command line or in an authoring document. Remote authenticated endpoints must use
HTTPS.

The default `--response-mode json_schema` sends a strict phase schema. Use
`--response-mode json_object` only as an explicit fallback for a self-hosted server that
does not implement JSON Schema response formatting. Local strict JSON parsing, Pydantic
validation, coordinate binding, and static checks still apply. The response's outer
`model` field must exactly match `--served-model-id`; omit that flag only when it equals
`--model-id`.

Model requests default to `--timeout-seconds 300` and `--max-tokens 65536`. Do not add
a provider-specific flag that disables or otherwise changes model thinking.

`--max-phase-repairs` accepts zero through eight. Valid JSON that fails Pydantic,
coordinate, or coherently assignable static checks receives concise same-phase feedback
and a distinct deterministic seed. Malformed outer or content JSON and
`finish_reason="length"` also consume this budget and retry with the next deterministic
seed. A missing or mismatched served identity, a tool call, or invalid usage telemetry
fails immediately. Successful model runs and failed phase attempts retain request and raw
response hashes plus content hash, finish reason, and prompt/completion token usage when
available in private provenance. Do not move grader/reference content across
phase boundaries: verifier never sees a reference, and reference sees sanitized design
fields plus required milestone IDs. On `revise`, the model first receives only the public
design plus aggregate admission evidence; the host then generates a fresh verifier and a
fresh blinded reference. Parent hash and revision are host-owned. Selected trusted
fixtures may additionally provide a host-owned, phase-scoped constructive source or
witness to the reference author; this is host authoring context, not learner-visible
state, and grader artifacts and verifier-only host evidence remain absent from the
reference phase.

For compatible BuildGrid secure-code coordinates, guided design may select the pinned
inih r62 `FixtureRef`; declare BSD-3-Clause. For EdgeFleet reverse-engineering
coordinates, select `edgefleet-efup-v1` and declare Apache-2.0. Its public projection
contains the stripped vulnerable x86-64 ELF, appliance files, and benign capture. The
patched twin is host-mounted only at its declared `.cyber_forge/grader/fixtures/` path;
construction source and witness files are prompt context and are never materialized.
Copy catalog IDs and semantic digests exactly, choose a safe mount path, and never inline
or reconstruct capsule bytes in model output. A coordinate claiming stripped-binary
access or artifacts must select a capsule with the validated `stripped_binary`
capability.

Run static checks before provisioning quarantine evidence. A failed static report exits
nonzero:

    uv run cyber-forge-author validate-static --input envelope.json --output static-report.json

After an independent runtime writes `evidence.json`, deterministically derive the
verdict:

    uv run cyber-forge-author admit --envelope envelope.json --evidence evidence.json --output admission.json

For a `revise` verdict, make a lineage-bound replacement with the same endpoint flags
used for generation:

    uv run cyber-forge-author revise \
      --envelope envelope.json \
      --admission admission.json \
      --model-id <open-model-id> \
      --served-model-id <exact-response-model-id> \
      --model-revision <immutable-revision> \
      --model-license <license> \
      --weights-sha256 <sha256> \
      --max-phase-repairs 2 \
      --output envelope-r1.json

Only an accepted, matching report can be adopted and emitted. Repeat `--recipe` to
publish several recipes together:

    uv run cyber-forge-author adopt --envelope envelope.json --admission admission.json --output recipe.json
    uv run cyber-forge-author emit --recipe recipe.json --output guided-corpus

Authoring JSON outputs are created with mode 0600 and are never overwritten. On a
terminal `SynthesisPhaseError`, `generate` and `revise` automatically write the structured
failure to mode-0600 `<output>.failure.json` and return 2 instead of writing the requested
output. Use `--failure-output` to choose another fresh path; output and failure paths must
be distinct. Corpus publication also refuses an existing manifest, stores private recipes
with mode 0600, and requires a weights digest on every model run.

For an automated run, use `GuidedSynthesisCoordinator` with a
`QuarantineCandidateEvaluator`. Supply:

- `RuntimeProvisioner(challenge, purpose)`, an async context manager yielding a fresh
  Verifiers `Runtime` that safely supports concurrent independent leases;
- an async solver-cohort callback returning candidate-bound `SolverAttempt` records;
- an async novelty callback, normally `NoveltyCallbackAdapter` over an immutable accepted
  corpus snapshot; and
- an async license callback over the private envelope.

Set `max_parallel_runtimes` for the evaluator to the backend's safe concurrent lease
capacity. It defaults to four and must be between one and 64. Baseline runs alone; oracle
replays and negative controls then share that bound. Evidence order remains replay index
followed by negative-control declaration order, independent of completion order.

Persist the coordinator's `on_progress` checkpoints after proposed, evaluated, revised,
and completed events; they contain private verifier/reference material. Resume through
`coordinator.resume`, not by reconstructing lineage manually. `max_revisions` bounds
complete isolated design/verifier/reference cycles and is distinct from same-phase repair
count. Quarantine gate issues and aggregate metrics become revision evidence; hidden
cases, raw transcripts, and grader bytes do not.

To author multiple independent candidates, pass a precomputed sequence of unique
coordinates to `GuidedSynthesisBatchCoordinator`. Its coordinator factory must return a
fresh `GuidedSynthesisCoordinator` for every coordinate; never reuse a synthesizer,
client, evaluator, or coordinator with mutable per-lineage state. Set
`max_in_flight_candidates` from one through 64. Candidate lineages run concurrently,
while dependent phases within each lineage remain sequential. Returned jobs always match
input order. Inspect `result.failures`: ordinary job exceptions are isolated as bounded
`BatchSynthesisError` records and do not imply the whole batch succeeded. When the error
is a `SynthesisPhaseError`, its optional `synthesis_failure` field retains the structured
coordinate, phase, revision, and private failure records. The optional batch progress
sink receives both coordinate and checkpoint and must be safe for
concurrent calls.

Admission requires an incomplete baseline with a required effect false, at least two
identical successful oracle replays, concrete and distinct negative-control mutations
whose score vectors differ from baseline and fail their named milestones, clean resets,
no leakage, clear licensing, novelty at least 0.15, and the difficulty solver frontier.
The frontier requires at least eight
unique attempts across at least two `(cohort, model_id, model_revision)` solver cohorts,
at least two action fingerprints, a failed attempt with partial milestone and
dependency-depth progress, a difficulty-band solve rate, consistent solved/reward labels,
and median successful tool calls between `max(1, 0.5 * target_min_actions)` and
`1.5 * target_max_actions`. Record turns, tool calls, tokens, duration, failure class,
normalized action fingerprint, model ID, immutable model revision, open-weight license,
and weights SHA-256 for every attempt. Solved attempts require positive reward, milestone
progress, and dependency depth; unsolved attempts cannot claim full reward.

The novelty evaluator compares structural axes, artifacts, milestone graph, tool/service
shape, source structure, and successful action fingerprints to the nearest accepted
candidate. Normalize action traces before fingerprinting: abstract flags, markers,
identifiers, addresses, paths, and harmless spelling differences. Successful-action
evidence participates only when both current and accepted candidates have it.

Load an emitted guided corpus with:

    [taskset]
    id = "cyber-forge-v1"
    guided_manifest = "guided-corpus/manifest.jsonl"
    start_index = 0
    num_tasks = 256

For `guided_manifest`, `start_index` is a zero-based manifest offset, not the procedural
global synthesis index. `num_tasks` is the contiguous slice length; emitted record indexes
remain task sequence metadata. `manifest` and `guided_manifest` are mutually exclusive,
and seed/family/world-pack/difficulty/scale settings do not filter guided records.

## World packs

World packs are the target unit for coherent multi-perspective synthesis. The five
registry IDs are tenantdesk-saas-v1, cloudship-delivery-v1, northstar-branch-v1,
edgefleet-appliance-v1, and buildgrid-registry-v1. The registry has 18 implemented
templates across all twelve families. World and runtime depth varies: current tasks are
independent artifact, program, JSON, answer, payload, chain, or single-container service
challenges, not five complete shared multi-service worlds.

The systems lineages are edgefleet-native-update-chain-v1, an x86-64 protocol-reversing
and callback-exploit replay, and buildgrid-inih-audit-v1, a pinned real C codebase with
digest-checked repair and sanitizer-backed withheld batches. The former requires
file/readelf/objdump/nm/GDB; the latter requires a C compiler. Setup rejects incompatible
images. Inspect `TEMPLATES` or call `eligible_templates` for the complete mapping.

Select corpora with the repeatable `--world-pack` CLI option or the taskset `world_packs`
list rather than editing generated artifacts or deriving a pack from prompt text. Inspect
the registry programmatically through `WORLD_PACKS`, `TEMPLATES`, and
`eligible_templates`; there is no CLI listing command. Registry inspection must not
instantiate worlds or scorer state.

Preserve the task/scorer projection. Current task records contain the pack ID,
template and lineage, family, difficulty, scale, declared skills, realized but
uncalibrated difficulty profile, required executable tools, generator version,
semantic-spec fingerprint, and public artifact hashes. Keep answer-revealing mutations,
flags, withheld cases and labels, reference-solution artifacts, reference traces, replay
markers, and expected state with the scorer. Never expose the compiled
challenge or a future full internal world spec by serializing it into TaskData; maintain
an explicit public projection.

## Validate

Run model-free oracle and setup checks before any live model evaluation:

    uv run --frozen validate cyber-forge-v1 --runtime.type docker --num-tasks 12
    uv run --isolated --no-project --with-editable ./environments/cyber_forge_v1 \
      --with pytest --with pytest-asyncio -- \
      pytest -q ./environments/cyber_forge_v1/tests
    uv run --frozen ruff check ./environments/cyber_forge_v1
    uv run --frozen ruff format --check ./environments/cyber_forge_v1

The isolated full-suite command needs `pytest-asyncio`; adding only `pytest` makes
collection fail on the package's strict `asyncio` marker even though the environment
dependencies themselves are correct. Use `--frozen` for root-project commands: the
current uv version otherwise rewrites unrelated lockfile metadata before running them.

Task.validate is the gold path. It applies the reference solution only inside the disposable
validation runtime. Do not move oracle answers, hidden labels, or verifier cases into
TaskData, prompts, trace info, public task metadata, or public artifacts.

Every new world-pack lineage starts in quarantine. Before admission, require deterministic
world construction and clean reset, reference-solution success through the production
scorer on multiple instances, independent or differential validation where feasible,
benign-functionality and preservation checks, and failure of relevant no-op, hardcoded,
public-example-only, blanket-allow, and blanket-deny mutants. Also check alternate valid
solutions, hidden-case fuzzing, leakage, near-duplicates, flakiness, resource limits, and
network containment. A generated task is not admitted merely because its generator and
oracle agree.

For the inih lineage, retain the digest-before-execution boundary: compile workspace source
and headers only after both exactly match the admitted hashes, and copy the verified bytes
to the scorer work directory first. Constructor/early-exit, workspace-header, and
pristine-upstream-replacement probes must remain in the regression suite. Every required
preservation milestone must transitively depend on a required effect. Guided candidates
with the `enumerate_and_chain` shape or `exploit_chaining` skill must contain a dependency
path through at least three required milestones.

For the EdgeFleet guided fixture, retain all three boundaries: validate ELF64/x86-64 and
the absence of `SHT_SYMTAB` from packaged public bytes; keep construction files out of
both learner and grader materialization; and keep the patched ELF only in grader
artifacts. Regression coverage must replay the trusted exploit with distinct fresh
markers against vulnerable and patched binaries and replay benign traffic against both.
This is one deterministic non-PIE x86-64/glibc target and root-cause kernel, not evidence
of coverage for mitigation bypasses, alternate architectures, or large-codebase
vulnerability discovery. Broader native realism and variety require additional capsules
plus independently validated mutation families.

Keep scorer resource bounds fail-closed: JSON inputs are limited to 256 KiB and submitted
program cases to three seconds and 16 KiB combined output.

For guided candidates, also preserve authoring bounds: safe relative paths up to 256
characters, 2 MiB per decoded file, 16 MiB per candidate, at most 64 non-empty argv
entries per command, command timeouts from 0.1 to 300 seconds, and `network=false`.
Compiled commands use the common 16 KiB combined-output runner and kill their process
group on timeout or overflow. The model client defaults to a 300-second timeout,
65,536 output tokens, a 1 MiB response, JSON depth 64, and 100,000 nodes; the hard
configurable response ceiling is 16 MiB. It does not send a provider-specific control
to disable or otherwise alter model thinking. Background service logs still require an enforced runtime disk quota.

## Topology

Use topology.id cyber-forge-v1 for planner-to-operator runs. Keep planner non-trainable
and operator trainable: the topology-era Prime-RL orchestrator requires exactly one
trainable trace per invocation.

The taskset owns deterministic selection, task setup, and scoring. Harness and runtime
choices remain config-only. Do not instantiate uCloud clients in task code. Shared
multi-runtime target worlds remain a planned host-broker capability.

Guided quarantine follows the same rule. Put Docker/uCloud selection, image digest,
egress policy, quotas, cancellation, credentials, and lease cleanup inside the
`RuntimeProvisioner`; pass that callback to `QuarantineCandidateEvaluator` and solver
cohort code. Do not pass a uCloud SDK client or credential into a generated task.

## uCloud safety

Use bridge networking only because the remote harness needs the model relay, and enforce
relay-only egress outside the SDK. Never use host networking, privileged mode, host
mounts, container runtime sockets, or real credentials.

Production requires an immutable image with a real UID/GID 1000, writable home,
/workspace, /workspace/.vf-transfers, cache, and temp paths. The stock Python image with
user unset is a debug-only compatibility fallback.

Do not promote the current uCloud smoke to adversarial training until the Verifiers
adapter uses a pinned offline harness environment, enforces the SDK disk quota, kills
timed-out remote process groups before scoring, and replaces its argv relay token with a
short-lived per-rollout credential.

Read deps/research-environments/environments/cyber_forge_v1/README.md for the complete
environment contract, family list, reward scheme, and guided open-weight workflow.
Read deps/research-environments/environments/cyber_forge_v1/docs/world-packs.md for the
target world-pack architecture, realism and difficulty models, task/scorer projection, and
admission expectations.
Read deps/research-environments/environments/cyber_forge_v1/docs/systems-track.md for the
low-level skill taxonomy, structural D1-D5 ladders, chain contracts, and explicit
single-runtime limitations.
Read deps/research-environments/environments/cyber_forge_v1/docs/guided-synthesis.md for
phase repair, coordinator/evaluator contracts, admission thresholds, provenance,
guided-manifest loading, and the S1 isolation boundary.
