# Munin 2.0 RL run history (Gefion, r5)

Munin 2.0 is the **step-100 checkpoint of run r5** (`gefion-super-r6s2638-8t4p4d-tp8-r5`): Nemotron 3
Super (the r6 SFT export, LUMI step 2619), trained with prime-rl on Gefion (DGX H100) with disaggregated
TP8+EP8 prefill/decode inference. r5 ran from step 0 to 160 as five kept segments (Slurm jobs) in one
run directory. Steps 111–160 came after a restart with several changes and are **not** in Munin 2.0;
they matter only for checkpoint merges that use a later step (for example M1 = mean of steps 75, 100
and 125).

**Caveat that applies to everything below.** Every job ran from a *live, dirty* prime-rl working tree
and an editable prime-envs/renderers checkout; no job recorded a commit hash or diff. These branches are
a **best reconstruction**: commit membership per segment comes from the resolved configs each job wrote
(`configs/attempt_N/resolved/*.json` lists every config field the code had at launch), file modification
times, Slurm job start times, and the run log. Where a change has no config field (the LR-on-resume fix,
the P-before-D pause, the surrogate fix) its segment is inferred from those notes and times.

## Repositories and branches

| Repo | Branch (local, to push) | Base | Push to |
|---|---|---|---|
| prime-rl | `munin-2.0/gefion-rl` | `origin/lumi` `138d1b5c` | `danish-foundation-models/prime-rl` |
| prime-envs (research-environments) | `munin-2.0/swe-scoring` | `679dfd3d` (pushed) | `rlrs/research-environments` |
| renderers | `munin-2.0/xml-tool-arg-newlines` | `1ca3e23` (pushed) | `rlrs/renderers` |
| verifiers | `munin-2.0` (= `4f1fabc`) | `29ea1a32` (v0.3.1.post1) | `rlrs/verifiers` |
| router (vllm-router) | `munin-2.0/sticky-load` | `2cae1ab` (upstream `fix/cache-aware-load-accounting-v0.2.0`) | a fork of `PrimeIntellect-ai/router` |

Each repo has local annotated tags `munin-2.0/r5-s000`, `-s036`, `-s051`, `-s061`, `-s111` and
`munin-2.0/step100`, pointing at the commit that was live in that segment. In prime-rl the tags mark the
**source code** state; the Gefion layer (configs, scripts, infer-opt, manifests) and the submodule pins
were committed after the code and are at the branch tip.

## Segment timeline

| Steps | Attempt | Job | Ran (CET) | prime-rl tag → commit | prime-envs | Config |
|---|---|---|---|---|---|---|
| 0–35 | attempt_1 | 535022 | 11-02 05:03 – 11-02 22:18 (NODE_FAIL in step 39; resumed from ckpt 35) | `munin-2.0/r5-s000` → `4ae8ec6b` | `679dfd3d` | `configs/gefion/runs/r5/attempt_1/rl.toml` |
| 36–50 | attempt_7 | 535966 | 11-04 00:44 – 11-04 07:36 | `munin-2.0/r5-s036` → `a041edf7` | `679dfd3d` | `.../attempt_7/rl.toml` |
| 51–60 | attempt_9 | 536005 | 11-04 07:41 – 11-04 13:24 (resumed from ckpt 60) | `munin-2.0/r5-s051` → `a041edf7` | `679dfd3d` | `.../attempt_9/rl.toml` |
| 61–110 | attempt_12 | 536191 | 11-04 14:24 – 11-05 04:42 (entropy guard at step 111; ckpt 110) | `munin-2.0/r5-s061` = `munin-2.0/step100` → `a9da3693` | `679dfd3d` | `.../attempt_12/rl.toml` |
| 111–160 | attempt_17 | 536701 | 11-05 14:50 – 11-06 03:32 (stop watcher after ckpt 160) | `munin-2.0/r5-s111` → `cffda473` | `f994ef6d` | `.../attempt_17/rl.toml` |

The other attempts kept no steps: 535345 ran the stale attempt_2 script and was cancelled at step 37;
535463 hit NODE_FAIL at startup; 535911 trained step 36 at the old LR (discarded, see `ec68379a`);
535952 deadlocked in the startup weight sync (stale broadcast marker); 536164 ran no steps; 536582
(attempt_16) failed at startup on the harbor task cache (see `cffda473`). Attempts 13–15 were renders.

All segments used: verifiers `4f1fabc`, renderers `4c2b30f`, the router build of `5efd49c`, vLLM 0.28.0
with the scheduler patch, the infer-opt patches, temperature 0.8, score centering + TIS (cap 2.0),
plain GRPO with `zero_truncated_reward` and `mask_runaway_turns`, batch 512, and online FP8 inference.

### Steps 0–35 (r5 start)

Code since `origin/lumi`, all in the tree from step 0:

| Commit | Change | First used |
|---|---|---|
| `83ad20be`, `44db3e00` | Live degenerate-turn stop; `repetition` turns judged on their pre-loop tokens; loop metrics (needs verifiers `4c713b5`/`c7c6735`) | r1 |
| `d23e77ad`, `8f903d51` | Concurrency cut on prefill prefix-cache thrash | r1 (3P:5D) |
| `b801a24c` | `zero_truncated_reward`, `mask_runaway_turns` (the r2-collapse fixes) | r3 |
| `7ec8a55e` | Pause prefill before decode on weight updates; `pd_kv_lease_seconds` (120) | r4 (field in r3) |
| `4ae8ec6b` | `loop_annotations` digest with surrogatepass (crashed r4) | r5 |

Config: 8 trainer + 4P:4D nodes, LR 1e-6 constant, Adam eps 1e-8, round-to-nearest bf16, max_off_policy 5,
in-flight 600/512/900, 13 small evals. Dependencies: renderers fix and verifiers `4f1fabc` (both since r3).

### Steps 36–50

New code (`7d393b32` dashboard thinning and `41314262` export fix are tooling, not training):

| Commit | Change |
|---|---|
| `0a5c6273` | `eval.eval_on_resume` (on only in attempt_7) and the `response_cut_off` metric |
| `bde24017` | `concurrency.kv_usage_grow` (0.7) |
| `5813d697` | ruff format (no behaviour change) |
| `ec68379a` | Keep the configured LR on resume (job 535911 had trained step 36 at 1e-6) |
| `a041edf7` | Stochastic bf16 rounding with an fp32 residual in pinned CPU memory |

Config (`9653654f`): **LR 2e-6** and `stochastic_round_bf16`, 6 trainer + 5P:5D, max_inflight 1375, KV
lease 60 s, `collect_inference_metrics` off, big evals (tb2 89×2, tblite 64×2, SWE-bench Verified 128,
interval 24) with the training repetition guard. `render_pd.sh` started clearing stale broadcast markers
at job start.

### Steps 51–60

No code change. Config (`23a25eeb`): max_inflight 1100 (sandbox backend overload at 1375),
max_off_policy_steps 10, eval_on_resume off.

### Steps 61–110 (contains Munin 2.0 = step 100)

| Commit | Change |
|---|---|
| `a9da3693` | `stagger_harnesses`: explore an unseen task on one harness first |

Config (`0d7b917e`): max_inflight 1000, stagger on the 15 bash/pi/opencode sources, 6 ubuntu:16.04
terminal-lego images quarantined (run state). Step 100 was trained by job 536191.

### Steps 111–160 (not in Munin 2.0)

| Repo | Commit | Change |
|---|---|---|
| prime-rl | `8a4c6bdd` | Configurable Adam eps, re-applied after load (**eps 1e-10**) |
| prime-rl | `d9ad96e4` | Task retirement (solved after 16 at 0.85, unsolved after 24); explore-order digest |
| prime-rl | `cffda473` | `exclude_keys_file` (curated exclusions without re-keying tasks) |
| prime-envs | `ad2970ea` | openswe test-patch reset; swesmith drops `ctest --rerun-failed`; `tar -xm` |
| prime-envs | `01a44889` | Per-test outcome logging (reward unchanged) |
| prime-envs | `f994ef6d` | swesmith: cargo cache GC off; CMake cache restore (rs/cpp re-admitted) |

Config (`7b3c736f`, lists in `851ca32a`): max_off_policy_steps 6, SWE group_size 16, openswe ratio 1/6,
exclude keys (openswe 47, scaleswe 35, swe-smith 75 incl. all java, tmax 3, terminal-lego 9,282 easy),
14 h job with `scripts/gefion/stop_after_last_ckpt.py`.

The in-training agentic evals fell after this restart (tb2 0.39 → 0.27 between v96 and v144). The
likely cause: eps 1e-10 together with the other changes raised the effective step about 5×, which
amplified fewer tool calls and longer replies. Replies cut off at the length cap never got a negative
gradient (`mask_runaway_turns`), and stale drops at max_off_policy 6 removed long SWE groups.

## Earlier runs (not part of Munin 2.0)

- **r1** (`…-8t2p6d-tp8-r1`, `…-8t3p5d-tp8-r1`; 10-29 to 10-31): first PD runs. Moved from 2P:6D to
  3P:5D to 4P:4D after prefill cache thrash (hence `sticky_load` and `d23e77ad`), and from temperature
  1.0 to 0.8 after 41–48% of SWE/terminal rollouts derailed. Abandoned at step 15 because its history
  was mixed.
- **r2** (jobs 534618, 534678; 10-31 to 11-01): LR 3e-6, ECHO tool CE. Entropy drift and runaway turns
  led to a regression; stopped at step 20.
- **r3** (job 534713, 11-02): LR 1e-6, GRPO + `b801a24c`. Died at step 2 waiting out a 600 s KV lease.
- **r4** (job 535011, 11-02): adds `7ec8a55e`. Crashed after step 2 on a lone surrogate (`4ae8ec6b`).

Their configs are in `configs/gefion/runs/{r2,r3,r4}/` (`555a3833`).

## Reproducing a segment

1. **prime-rl source:** check out the segment's tag (for Munin 2.0, `munin-2.0/step100`). The tags
   predate the submodule pin commit `f36e001d`, so set the submodules by hand:
   - `deps/verifiers` → rlrs/verifiers `4f1fabc` (the old pin `392d596d` is upstream and is **not**
     what ran);
   - `deps/renderers` → `4c2b30f`;
   - `deps/prime-envs` → `679dfd3d` for steps 0–110, `f994ef6d` for 111+.

   Or work from the branch tip and check out only `src/` and `packages/` at the tag.
2. **Venv** (there is no Gefion venv recipe): `uv sync --all-extras --all-packages` in prime-rl,
   then install by hand on top:
   - verifiers `4f1fabc` (the run used a non-editable install from a checkout);
   - verifiers-ucloud 0.3.7 (`rlrs/verifiers-ucloud@966927c`);
   - ucloud-sandboxes-sdk 0.4.39.

   Then, in site-packages:
   - apply `gefion/infer-opt/vllm-assert/scheduler-p-side-truncation-full-hit.diff` to vLLM 0.28.0's
     `vllm/v1/core/sched/scheduler.py` (the same fix as vllm#53523);
   - replace three files of the vllm-router 0.2.0 wheel with the build of `munin-2.0/sticky-load`
     (see `BUILD-GEFION.md` in that branch).

   Launch with `UV_NO_SYNC=1` so the job does not undo these changes.
3. **Config:** `configs/gefion/runs/r5/attempt_N/rl.toml` is the exact resolved config of the segment.
   It references absolute Gefion paths: the LUMI mirror for model and data, `manifests/…` (copied here
   to `configs/gefion/manifests/`), and `infer-opt` (here `gefion/infer-opt/`, loaded through
   `PYTHONPATH` with `INFER_OPT=ep_compact_moe,fast_generate,fi_moe_chunk,dg_jit_per_proc`).
   It also references the guest relay at 10.42.0.2.

   The generators in `configs/gefion/` are in their final (step-110 restart) state; earlier states
   exist only as the resolved configs.
4. **Render and launch:** `scripts/gefion/render_pd.sh <config>` (it calls `render.sh` and
   `order_nodes_pd.sh`), then `preflight.sh`, then sbatch the rendered `launcher/rl.sbatch`. The render
   patches the sbatch for:
   - the link-local LOCAL_IP;
   - DP=1 ranks;
   - the sticky_load router;
   - broadcast-marker cleanup.

   To resume, set `resume.step`. The stochastic-rounding residual is not checkpointed.
5. **After a cancel:** clean up sandboxes and relay registrations with `scripts/gefion/ucloud/`. It
   needs the UCloud/relay credentials in `.env`, which are not in git.

## Known caveats

- Live working tree: see the top. Two commits mix time and purpose: `5813d697` is formatting only, and
  the dashboard and export commits were in the tree but not in the training path.
- The KV-lease default was changed in place: r3 resolved 600 s and r4/r5 resolved 120 s (60 s from step
  36). The committed code has 120.
- `stop_after_last_ckpt.py`, the cleanup scripts and the merge tools are copies of loose files; the
  originals were run from scratch directories.
- Unfixed: the SFT trainer still ignores the configured LR on resume (only RL is fixed in `ec68379a`),
  and the orchestrator can still trust a stale `.sender_ready`. The render step works around the second
  by deleting `broadcasts/step_*`.
- Under `strategy = "smoothed_replay"` the curriculum `pools` thresholds are ignored.
- `ep_compact_moe` probably falls back to stock under FP8 (not verified in logs).
- Not in any repo: the Hetzner/UCloud credentials (`.env`), checkpoints and models, run logs, the
  run's `quarantined-images.json`, and the oracle-validation scripts referenced by the manifest README.
