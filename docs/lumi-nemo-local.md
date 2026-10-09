# Local NeMo training environments

`nemo_local` is a native verifiers v1 taskset and minimal chat-completions harness.
It runs on the CPU side of the Slurm allocation, with `runtime.type = "subprocess"`
and `interception.type = "server"`. Policy calls still use PRIME-RL's inference
router, renderer, and token/logprob capture. No Prime service or UCloud sandbox is
used for these sources. The harness exposes no shell, Python execution, arbitrary
filesystem access, or live external APIs.

Supported kinds:

- `pivot`: exactly one sampled tool call; compare its name and JSON arguments to
  the target and validate against the advertised schema. No tool is executed.
  Append the one-call instruction to an existing system message; Nemotron rejects
  a second system message. Do not prepend another system message to Pivot history.
- `instruction`: final-answer instruction checks. Reasoning is captured normally
  but is not passed to the grader. Empty answers receive zero.
- `calendar`: final-answer calendar constraints, including event identity,
  duration, time windows, and conflicts. Reject duplicate event IDs and invalid
  times before the upstream checks.
- `workplace`: a multi-turn loop over 27 fixed, in-memory simulator functions.
  Each rollout creates fresh state. Verification replays generated calls from
  fresh state and compares with the target state. Verifier targets never enter
  model messages or the harness input file. Tool arguments are schema-validated;
  unknown calls or replay exceptions cannot silently earn reward.

Graders run in a lazily created two-process pool per environment worker. With one
worker per source, the four sources use at most eight grading processes. The
source multiplex limit bounds pending grading work. Simulator tools run inside
that rollout's local harness subprocess. Existing SWE/terminal execution remains
in UCloud. Adding an arbitrary-code tool requires a sandboxed runtime instead.

## Data and provenance

The preparation script pins full Hugging Face commit IDs and stages JSONL ahead
of startup. It writes counts, exclusions, source hashes, and output hashes to
`manifest.json`. It uses only publisher training files. Five percent of stable
identity hashes form a local validation split; all Pivot prefixes of a trajectory
and Calendar variants sharing an initial user message stay together. These are
local monitoring holdouts, not publisher leaderboard results.

The staged pools in `artifacts/training/nemo-local-20260923/data-v2` contain:

| Kind | Train | Holdout |
| --- | ---: | ---: |
| Pivot | 4,546 | 234 |
| Instruction following | 44,110 | 2,281 |
| Calendar | 9,179 | 480 |
| Workplace | 1,029 | 49 |

Pivot excludes 4,701 free-text targets because upstream treats arbitrary messages
as correct, 45 targets for tools absent from the prompt, and 94 targets whose
arguments violate their advertised schemas. Workplace excludes 177 targets with
no resulting state change: a state-only verifier cannot assess those answers.
Instruction grading accepts the publisher's singleton-list keyword fields for
`count:count_increment_word` as the intended single string. No instruction rows
are dropped. Empty calendar targets are rejected if encountered.

Selected upstream code, CSV fixtures, licenses, commit pins, and file hashes live
under `src/nemo_local/_vendor`. NeMo's full server framework is not installed.
Instruction package initialization omits the unused synthesis import, which
otherwise suppresses process-wide logs and warnings. English NLTK tables are
pinned and hash-checked by `nemo_local.assets`; no rollout downloads anything.
The operational SIF already supplies pandas, jsonschema, nltk, langdetect,
immutabledict, and absl. Keep those dependencies when replacing the image.

Run through the normal LUMI container wrapper (`uv run` is supplied by that wrapper):

```bash
./scripts/lumi_job_container.sh python -m nemo_local.assets \
  /scratch/project_465002751/rasmus/.cache/nltk_data
./scripts/lumi_job_container.sh python scripts/lumi_prepare_nemo_local_data.py \
  /scratch/project_465002751/rasmus/artifacts/training/nemo-local-20260923/data-v2
NLTK_DATA=/scratch/project_465002751/rasmus/.cache/nltk_data \
  ./scripts/lumi_job_container.sh python scripts/lumi_validate_nemo_local.py \
  /scratch/project_465002751/rasmus/artifacts/training/nemo-local-20260923/data-v2 \
  --output /scratch/project_465002751/rasmus/artifacts/training/nemo-local-20260923/audit-v2.json
```

The wrapper requires `PROJECT_DIR`, `LUMI_ROOT`, `SIF`, and a prepared
`PRIME_RL_METADATA_DIR`, as described in `lumi.md`. `NLTK_DATA` must be present for
the environment worker and its grading subprocesses, not only for the harness.
The r6 configuration sets it in `[env_vars]`.

## Next training configuration

`configs/lumi/rl_nemotron_super_r5_128k_16t8i_sc_full200_r6.toml` retains the 200-step
run, final SFT checkpoint, SC, group size eight, 16 trainer/eight inference nodes,
128K context, and existing sources. Its batch-128 category targets are 56 SWE,
32 terminal, 8 SQL, 8 search, 16 tool use, and 8 instruction following. Pivot and
Workplace share tool use equally; Calendar and instruction following share the
instruction category equally. Existing scheduler inventory/surplus bounds apply
so these faster tasks cannot run arbitrarily far ahead. All four have 32-task
holdout evaluations every 25 steps. Workplace allows 200 turns; the other three
are single-completion tasks.

Validation artifacts live under `artifacts/training/nemo-local-20260923`:
`audit-v2.json`, `audit-workplace-final.json`, `rendering.json`, `smoke-final/`,
and `dry-run.log`. All 58,864 training tasks loaded through the native taskset,
and 128 representative prompts rendered with the final checkpoint tokenizer. The smoke uses a scripted local HTTP
model with the real runtime, interception, and process-pool grader, including two
concurrent copies of every kind. It validates integration and state separation,
not the policy's capabilities. A GPU-backed model evaluation is still needed to
measure actual difficulty, reward variance, and rollout throughput.

This configuration is prepared for the next run; preparation does not submit it.

Harness CLI values use `--name=value` form. Interception generates URL-safe
random API keys, which can start with `-`; passing these as separate argparse
tokens can make a valid key look like another option and abort before generation.
