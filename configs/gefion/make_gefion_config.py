"""Derive the Gefion full-run config from LUMI's r21 config.

Usage: make_gefion_config.py <r21.toml> <inference_recommended.toml> <out.toml> <lumi-paths.txt>

- LUMI paths under /scratch/project_465002751/ map to MIRROR (the same relative layout), except the
  openswe / swe-smith task-id lists, which use the 2026-10-24 oracle lists.
- The inference block is replaced by inference_recommended.toml (DP8+EP8 online FP8 per node).
- ROCm/LUMI-only settings are dropped; Gefion NCCL/IB, network-interface and cache settings added.
- Every mirrored path is written to <lumi-paths.txt>, to drive the LUMI -> Gefion transfer.
"""
import copy
from collections import Counter
import sys
import tomllib
from pathlib import Path

import tomli_w

LUMI = "/scratch/project_465002751/"
MIRROR = "/dcai/projects/iu_0145/lumi-mirror/"
ORACLE = "/dcai/projects/iu_0145/rasmus/manifests/ucloud-task-ids-20261024-oracle/"
ORACLE_LISTS = {
    LUMI + "rasmus/artifacts/manifests/ucloud-task-ids-20261010b/openswe.task-ids.json": ORACLE + "openswe.task-ids.json",
    # Without the 1,636 ids (24%) whose upstream row has no problem_statement: those rollouts got no task
    # (infer-opt/swesmith-empty-task/NOTE.md), and without 4 tasks whose scorer gave identical exact fixes
    # both 1 and 0 in r2 (manifest README). Same order as the oracle list.
    LUMI + "rasmus/artifacts/manifests/ucloud-task-ids-20261010b/swe-smith-train.task-ids.json":
        ORACLE + "swe-smith-train.nonempty-stable.task-ids.json",
    # Start from the r6 SFT export (copied from LUMI), not r21's r5 step 2619.
    LUMI + "rasmus/artifacts/training/nemotron-super-base-r5-16n-cp8-ep8-bs32-lr1e5-r1/eval-exports/step_2619":
        "/dcai/projects/iu_0145/rasmus/models/nemotron-super-r6-step2638",
}
RUN_NAME = "gefion-super-r6s2638-8t8i-r1"
# Sandbox label marking this run's UCloud runtimes (r21: super-r5-step2619-sc-full200-bs512-r21-20261013a).
SANDBOX_RUN_LABEL = "gefion-super-r6-step2638-r1"
# Since 2026-10-30 the sandboxes run on Hetzner (UCloud's relay job went down); no private DNS there,
# so sandboxes reach the relay gateway by IP instead of UCloud's sandbox-gateway-gateway-1 name.
GUEST_RELAY = {"http://sandbox-gateway-gateway-1:8092": "http://10.42.0.2:8092"}
# Rebooted Gefion nodes carry a duplicate-IP bmc_redfish0 interface; pin every control plane.
IFACE = "enp170s0f1np1"
# 6 + 10 = the project's 16-node QOS limit. r5 ran 8 + 8, but its trainer computed for only 6-9 min of
# each 22-30 min step, rollout-bound; the stochastic-rounding residual moved to CPU so 6 nodes fit.
NUM_TRAIN_NODES = 6
NUM_INFER_REPLICAS = 10  # one DP8+EP8 node each
# r21 ran 24 TP8 replicas with initial/min/max inflight 900/128/2048. The inference nodes could take
# ~1,400 requests each, but the sandbox infrastructure caps the run at ~1,500 rollouts in flight.
# Under PD the prefill nodes' prefix caches bound it: each holds ~12.6M tokens, and at 1,500 sessions
# (~21k-token prompts) on 2 P nodes one P lost its cache to the startup burst and never recovered
# (job 534149: ~900 queued, 17% hits, the other P idle, step 1 took 1 h 32 min). 3 P nodes (~38M
# tokens) with <= 1,100 sessions leave room; a smaller initial burst gives the caches time to warm.
PER_TURN_TOKENS = 16384
# At T=0.8 (turns ~3x shorter) prefill needs ~3.7 computed prompt tokens per generated token and 3 P
# nodes thrashed at ~850 sessions (job 534524); the run moved to 4P:4D with at most 900 sessions.
# Initial burst 600 since r2 step 10: at 800 the burst over-dispatched SWE at v0 while step 1 took
# 51 min, and 173 finished plus 55 in-flight v0 episodes went stale and were dropped at step 7.
# r5 (535022) sat at ~62% decode KV with no queue or preemptions: the 0.6 growth gate bound it.
# 5P:5D since r5 step 35. Max 1375 (1100 x 5/4, scaled with the P nodes' prefix caches) held from
# 04:20 on 2026-11-04, then the sandbox backend degraded (503 exec timeouts, ~480 sandboxes stuck
# creating, 25% of rollouts errored by step 53), so back to r5's 1100.
# 2026-11-04 10:40-13:24 the hosts degraded again at 1100 (image builds out of memory headroom, then
# create pile-ups on 3 of 4 hosts); restarted at step 60 with 1000 after the hosts were repaired.
INFLIGHT = dict(initial_inflight=600, min_inflight=512, max_inflight=1000, kv_usage_grow=0.7)

src, rec_path, out_path, paths_out = sys.argv[1:]
cfg = tomllib.load(open(src, "rb"))
rec = tomllib.load(open(rec_path, "rb"))
mirrored = set()


def remap(x):
    if isinstance(x, dict):
        return {k: remap(v) for k, v in x.items()}
    if isinstance(x, list):
        return [remap(v) for v in x]
    if isinstance(x, str) and x in GUEST_RELAY:
        return GUEST_RELAY[x]
    if isinstance(x, str) and x.startswith("super-r5-step2619-"):
        return SANDBOX_RUN_LABEL
    if isinstance(x, str) and x.startswith(LUMI):
        if x in ORACLE_LISTS:
            return ORACLE_LISTS[x]
        mirrored.add(x)
        return MIRROR + x[len(LUMI):]
    return x


# ROCm / LUMI-only keys.
for k in ("PRIME_RL_SUPER_PROFILE", "PRIME_RL_VLLM_OVERLAY", "PRIME_RL_INFERENCE_PROFILE", "HSA_ENABLE_SDMA",
          "GPU_MAX_HW_QUEUES", "NCCL_DMABUF_ENABLE", "MIOPEN_USER_DB_PATH", "MIOPEN_CUSTOM_CACHE_DIR",
          # LUMI's image-based venv lacked the search/SWE packages (bm25s, swesmith, ...); the Gefion
          # venv has them from uv.lock, and only the LUMI template reads this variable.
          "PRIME_RL_ENVIRONMENT_SITE_PACKAGES"):
    for section in (cfg.get("env_vars", {}), cfg.get("trainer", {}).get("env_vars", {}),
                    cfg.get("inference", {}).get("env_vars", {}), cfg.get("orchestrator", {}).get("env_vars", {})):
        section.pop(k, None)
# The start checkpoint and its conversion stay where r21 points; trainer compile cache goes local.
cfg.get("trainer", {}).get("env_vars", {}).pop("TORCHINDUCTOR_CACHE_DIR", None)
cfg = remap(cfg)


# The relay gives up on a model call from a sandbox after interception.forward_timeout_seconds
# (default 300 s). At 16k tokens per turn and ~20 tok/s per sequence a turn can take ~15 min, so
# allow 30 min. (The bash harness's own client still times out at 600 s; opencode/pi have none.)
def set_forward_timeout(x):
    if isinstance(x, dict):
        for k, v in x.items():
            if k == "interception" and isinstance(v, dict) and v.get("type") == "ucloud":
                v["forward_timeout_seconds"] = 1800.0
            set_forward_timeout(v)
    elif isinstance(x, list):
        for v in x:
            set_forward_timeout(v)


set_forward_timeout(cfg)

cfg["output_dir"] = "/dcai/projects/iu_0145/rasmus/runs"
cfg["run"]["name"] = RUN_NAME
cfg["max_steps"] = 300
cfg["deployment"].update(num_train_nodes=NUM_TRAIN_NODES, num_infer_nodes=1, num_infer_replicas=NUM_INFER_REPLICAS,
                         gpus_per_node=8, orchestrator_on_inference=True)
cfg["slurm"] = {
    "job_name": "gefion-super-r21mix",
    "template_path": "src/prime_rl/templates/multi_node_rl.sbatch.j2",
    "partition": "defq",
    "account": "iu_0145",
    # Whole remaining budget (1500 node-h total) at 16 nodes, after job 534149 (2 h x 16, cancelled),
    # the prefill-routing test 534475 (~1 h x 8) and 534489's first 5 steps (~1.7 h x 16).
    # Remaining budget after r5's first attempt (535022, NODE_FAIL on dgx152 at step 38): ~660 node-h used.
    # After 535345 (cancelled at step 37 for the bf16 rounding fix) and the rounding study: ~685 node-h
    # used; 48 h x 16 leaves ~40 node-h for evals.
    # Step-110 restart: 1,161 node-h used (sacct, 2026-11-05), ~339 left. 14 h x 16 = 224 node-h leaves
    # ~115 node-h for final evals and checkpoint-merging experiments.
    "time": "14:00:00",
    # dgx035: NCCL hang (would stall the weight broadcast); dgx150: CUDA-graph capture hangs;
    # dgx157: excluded by the inference-optimization runs. Nodes without iommu=pt (e.g. dgx060) are
    # allowed but placed on inference (order_nodes.sh).
    "exclude": "dgx035,dgx150,dgx157",
    "pre_run_command": "bash /dcai/projects/iu_0145/rasmus/phase0/full/preflight.sh",
    "cleanup_grace_period": 60,
}

# Trainer: CUDA attention; everything else as r21. The converted weights are regenerated on Gefion,
# and output_dir is replaced, so neither is a LUMI input to transfer.
tm = cfg["trainer"]["model"]
tm["attn"] = "flash_attention_2"
mirrored.discard(LUMI + tm["conversion_dir"][len(MIRROR):])
mirrored.discard(LUMI + "rasmus/artifacts/training")
tm["conversion_dir"] = "/dcai/projects/iu_0145/models/conversion/nemotron-super-r6-step2638"

# Orchestrator in-flight limits for 8 DP8+EP8 inference nodes.
cfg["orchestrator"]["concurrency"].update(INFLIGHT)
# r21: 5. At 6T:5P:5D steps take 6-20 min, so 5 versions is ~1 h and 7-15% of each batch's rollouts
# (the long SWE tail; the rollout timeout is 2 h) were dropped or cancelled as stale from step 41 on.
cfg["orchestrator"]["max_off_policy_steps"] = 10
# 6 from the step-110 restart: at 10 the mean lag grew to ~4.7 and mismatch KL crept from ~0.005 to 0.0136.
cfg["orchestrator"]["max_off_policy_steps"] = 6
# Per-turn completion limit for the multi-turn agent sources (r21: 8192). Single-turn sources keep
# their own 32768 override.
cfg["orchestrator"]["train"]["sampling"]["max_completion_tokens"] = PER_TURN_TOKENS
cfg["orchestrator"]["eval"]["sampling"]["max_completion_tokens"] = PER_TURN_TOKENS
# Train sampling at temperature 0.8 (r21: 1.0). At 1.0, 41-48% of SWE/terminal rollouts in job 534489
# stopped as degenerate turns (reasoning drifting into word salad), rising with entropy over steps 1-4.
# Score centering needs full-vocabulary sampling, so no top-p.
cfg["orchestrator"]["train"]["sampling"]["temperature"] = 0.8
# Eval at 0.8 too since r2 step 10: at 1.0 the SWE evals mostly measured derailment (r2 step-0
# swebench-multilingual 0.00, 58% truncated, while train rollouts at 0.8 solved about half of SWE).
cfg["orchestrator"]["eval"]["sampling"]["temperature"] = 0.8

# Bigger agentic evals (r5 step 35 on): 1 sample x 16 tasks only resolved drops of ~0.3. Each env keeps
# loading the first N tasks in sorted order, so the old 16/32-task sets stay a subset with a v0
# baseline. Every 24 steps with more workers; the zero-signal SWE evals are dropped.
TBLITE_SANDBOX_BROKEN = {"api-endpoint-permission-canonicalizer", "broken-python"}  # loopback hosts setup
# The harbor loader's order: task dirs (with an instruction.md) sorted by path.
TBLITE_ROOT = Path.home() / ".cache/harbor/openthoughts_openthoughts-tblite_1"
TBLITE_ORDER = [p.parent.name for p in sorted(TBLITE_ROOT.rglob("task.toml")) if (p.parent / "instruction.md").is_file()]
TBLITE_TASKS = [t for t in TBLITE_ORDER[:66] if t not in TBLITE_SANDBOX_BROKEN]
assert len(TBLITE_TASKS) == 64
BIG_EVALS = {
    "terminal-bench-2": dict(num_examples=-1, group_size=2),
    "openthoughts-tblite": dict(num_examples=-1, group_size=2),
    "swebench-verified": dict(num_examples=128, group_size=1),
}
ev_cfg = cfg["orchestrator"]["eval"]
ev_cfg["source"] = [s for s in ev_cfg["source"]
                    if s["name"] not in ("swebench-multilingual", "senior-swe-bench-glm53-judge")]
for source in ev_cfg["source"]:
    if source["name"] in BIG_EVALS:
        source.update(BIG_EVALS[source["name"]], interval=24)
        source["serve"]["pool"]["multiplex"] = 32
        if source["name"] == "openthoughts-tblite":
            source["env"]["taskset"]["tasks"] = TBLITE_TASKS
assert {s["name"] for s in ev_cfg["source"]} >= set(BIG_EVALS)
# A resumed run evaluates every env at its checkpoint step. Off for the step-50, -60 and -110 restarts (the interval
# evals ran at step 96 and run again at 120). At step 50 and 60 its ~700 eval
# rollouts would land on a sandbox backend that was failing half of all rollouts (2026-11-04 06:00-07:30).
ev_cfg["eval_on_resume"] = False
# Same harness limits as training: repetition stop on, 16k per call, no nudging. A cut-off final
# response ends the run (logged as "Cut-off" on the eval line).
ev_cfg["sampling"].setdefault("extra_body", {})["repetition_detection"] = {
    "min_pattern_size": 16, "max_pattern_size": 64, "min_count": 8}
# The first run (gefion-super-r6s2638-8t3p5d-tp8-r1, 534489..534552) reached step 15 under mixed settings
# (T=1.0 steps 1-5, a relay outage, two vLLM crashes); the r2 run starts fresh from step 0 with all fixes.
# The venv's vLLM scheduler.py is patched for the NIXL Mamba P-side full-hit assert (infer-opt/vllm-assert/).
# Score centering composed with truncated importance sampling (r21: SC alone). arXiv 2609.20807
# recommends SC + an IS correction "whenever the trainer can move far from the sampler between
# syncs"; we train up to max_off_policy_steps behind an FP8 sampler. Cap 2.0 as in the paper.
cfg["trainer"]["loss"].update(importance_sampling="tis", tis_cap=2.0)
# LR 1e-6, constant, no warmup (r21: 1e-6). At 3e-6 (r2) the policy's entropy on sandbox-agent tokens
# rose every step (terminal 0.40 -> ~0.75) into runaway reasoning turns and SWE/terminal reward fell
# from v6 on; NVIDIA's Nemotron 3 Super SWE-RL stage also used 1e-6 at batch 512.
cfg["trainer"]["optim"]["lr"] = 1e-6
# 2e-6 with stochastic rounding (user decision, 2026-11-03). Under round-to-nearest only ~15% of each
# update reached the served bf16 weights, so r5's effective served LR was ~0.2e-6; r2's 3e-6 was
# likewise attenuated. r2's degeneration also had other causes (ECHO, unclipped SC/TIS), now fixed.
cfg["trainer"]["optim"]["lr"] = 2e-6
# Stochastic rounding of the fp32 master onto the bf16 grid after every step (arXiv 2610.41073).
# The master starts on the bf16 grid (bf16 checkpoint), and round-to-nearest casts left the trainer's
# forward weights and the broadcast weights carrying only ~15% (0.10-0.32) of the master's update by
# step 35; only 4-9% of bf16 weights had changed since step 0 (phase0/scratch_dash/deadzone).
cfg["trainer"]["optim"]["stochastic_round_bf16"] = True
# Adam eps 1e-10 from the step-110 restart (default 1e-8). With RL gradients this small, sqrt(v) < 1e-8 on
# 74-80% of routed-expert weights, so eps shrank expert updates 2.3-2.5x (rarest experts ~4.7x) below
# the nominal LR; dense weights lose 5-35% (optimizer-eps analysis, 2026-11-05). The trainer re-applies
# it after loading the checkpoint, whose optimizer state carries 1e-8.
cfg["trainer"]["optim"]["eps"] = 1e-10
# Plain GRPO on every source (r21/r2: ECHO tool-response CE at alpha 0.02 on SWE, terminal and SQL).
# ECHO's CE weights admitted every group past the zero_advantage gate, so 33-52% of r2's traces were
# CE-only fits to failed rollouts' tool output. Truncated traces score 0, and turns the engine stopped
# on length or repetition (runaway reasoning) leave the rl loss: in r2 they were trained at full
# weight, reinforced in 21% of pi/opencode rollouts, and grew from 5% to 24% of SWE rollouts.
GRPO_FIXES = dict(zero_truncated_reward=True, mask_runaway_turns=True)
cfg["orchestrator"]["algo"].update(GRPO_FIXES)
for source in cfg["orchestrator"]["train"]["source"]:
    algo = source.get("algo")
    if algo is None:
        continue
    if algo.get("type") == "echo":
        algo["type"] = "grpo"
        algo.pop("roles", None)
        algo.pop("filter", None)
    assert algo["type"] == "grpo", algo
    algo.update(GRPO_FIXES)
# Stagger the opencode/pi/bash copies of each agentic taskset (r5 from step 65): an unseen task runs on
# one harness first and reaches the others only if that group was mixed. Its outcome barely depends
# on the harness (AUC 0.88-0.99 from the other harnesses), and the three harnesses used to dispatch
# every new task together, 24 blind rollouts; staggering saves ~25% of rollouts per trainable group.
harness_groups = Counter(s.get("curriculum", {}).get("sampler", {}).get("shared_group")
                         for s in cfg["orchestrator"]["train"]["source"])
for source in cfg["orchestrator"]["train"]["source"]:
    sampler = source.get("curriculum", {}).get("sampler", {})
    if sampler.get("strategy") == "smoothed_replay" and harness_groups[sampler.get("shared_group")] > 1 \
            and source["name"].rsplit("-", 1)[-1] in ("opencode", "pi", "bash"):
        sampler["stagger_harnesses"] = True
# Curated tasks from the step-110 restart (manifests/r5-restart-step110/README.md): swesmith without
# java, whose in-place grading failed 18% of exact gold fixes for an unknown reason (rs and cpp stay, with
# the grading fixes in prime-envs: cargo cache cleanup off, no ctest --rerun-failed, CMake cache restored);
# broken and suspect tasks out everywhere; terminal-lego medium and hard only (easy tasks were 83%
# always-solved). The task lists stay as they were and the sampler excludes the dropped tasks by key: a
# task key hashes the row index, so a shorter list would re-key the tasks and orphan their step-110
# history (job 536582 also failed on it: the terminal-lego and tmax caches pin their task list).
CURATED = "/dcai/projects/iu_0145/rasmus/manifests/r5-restart-step110/"
CURATED_TASKSETS = ("openswe", "scaleswe", "swesmith-env", "tmax", "terminal-lego")
# SWE groups of 16 (others 8): SWE mixed groups are often 1-of-8 or 7-of-8, and a 16-rollout group
# halves the noise of the group baseline and finds more rare successes; the 240-rollout SWE quota
# becomes 15 groups instead of 30.
SWE_GROUP_SIZE = 16
# openswe's 877 validated tasks were all explored by step ~105, so it would be replay-only: half the
# share of scaleswe and swesmith.
SWE_RATIOS = {"openswe": 0.5, "scaleswe": 1.0, "swesmith-env": 1.0}
for source in cfg["orchestrator"]["train"]["source"]:
    taskset = source.get("env", {}).get("taskset", {})
    if taskset.get("id") in CURATED_TASKSETS:
        assert source["curriculum"]["sampler"]["strategy"] == "smoothed_replay"
        source["curriculum"]["sampler"]["exclude_keys_file"] = f"{CURATED}{taskset['id']}.exclude-keys.json"
    if source.get("balance_category") == "swe":
        source["group_size"] = SWE_GROUP_SIZE
        source["ratio"] = SWE_RATIOS[taskset["id"]] / 3
    # Stop replaying decided tasks (task-curation analysis, 2026-11-05): 1,203 tasks had a sampler
    # history of >= 16 with smoothed pass rate >= 0.85, or >= 24 with no pass; at the old 0.05 replay-weight
    # floor they took 13% of terminal and 8% of SWE rollouts, nearly all in all-equal groups.
    sampler = source.get("curriculum", {}).get("sampler", {})
    if sampler.get("strategy") == "smoothed_replay":
        sampler.update(retire_solved_after=16, retire_solved_rate=0.85, retire_unsolved_after=24)
assert sum("exclude_keys_file" in s.get("curriculum", {}).get("sampler", {})
           for s in cfg["orchestrator"]["train"]["source"]) == 3 * len(CURATED_TASKSETS)
# Keep every 25th checkpoint as well as the last two (r2 lost step 10 to keep_last = 2; one is 1.4 TB).
cfg["ckpt"]["keep_interval"] = 25
# vLLM repetition stop (finish_reason "repetition"), train sampling only; eval stays comparable.
cfg["orchestrator"]["train"]["sampling"].setdefault("extra_body", {})["repetition_detection"] = {
    "min_pattern_size": 16, "max_pattern_size": 64, "min_count": 8}
# Live degenerate-turn stop in the env servers (judged on the tokens before a repetition loop).
cfg["orchestrator"]["degenerate_turns"]["live"] = True

# Inference: replace with the recommended per-node block.
inf = copy.deepcopy(rec["inference"])
inf["deployment"] = {"type": "multi_node", "num_nodes": 1, "gpus_per_node": 8}
cfg["inference"] = inf
# Orphaned prefills (cancelled rollouts) hold a weight-update pause until their lease expires.
inf["pd_kv_lease_seconds"] = 60

# r5 resumes in place from its last complete checkpoint (step 110: entropy guard, 2026-11-05).
cfg["resume"] = {"step": 110}
# The per-engine inference rows (~200 KB every 6 s) made metrics.jsonl 1.45 GB by step 26; the
# /metrics poll still feeds the concurrency controller.
cfg["orchestrator"]["collect_inference_metrics"] = False

ev = cfg.setdefault("env_vars", {})
ev.update({
    "NCCL_IB_HCA": "=mlx5_0:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_9:1,mlx5_10:1,mlx5_11:1",
    "NCCL_SOCKET_IFNAME": IFACE,
    "GLOO_SOCKET_IFNAME": IFACE,
    "OMPI_MCA_btl_tcp_if_include": "10.10.17.0/24",
    "OMPI_MCA_oob_tcp_if_include": "10.10.17.0/24",
    "NCCL_DEBUG": "WARN",
    "HF_HOME": "/dcai/projects/iu_0145/rasmus/.cache/huggingface",
    "UV_CACHE_DIR": "/dcai/projects/iu_0145/rasmus/phase0/uvcache",
})
ev.pop("NCCL_DEBUG_SUBSYS", None)
te = cfg["trainer"].setdefault("env_vars", {})
te.update({
    # Per-job compile caches (a shared Triton cache raced and a job-private one broke cross-job hits).
    "TORCHINDUCTOR_CACHE_DIR": "/tmp/$SLURM_JOB_ID/inductor-cache",
    "TRITON_CACHE_DIR": "/tmp/$SLURM_JOB_ID/triton-cache",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
})

with open(out_path, "wb") as f:
    tomli_w.dump(cfg, f)
# Not in the config, but the Harbor task caches (task-metadata-*/terminal-lego.jsonl) point each row's
# task_dir here, and rollouts tar environment/ and tests/ from it.
mirrored.add(LUMI + ".cache/huggingface/terminal-lego-git/PrimeIntellect--Terminal-Lego-15k")
with open(paths_out, "w") as f:
    f.write("\n".join(sorted(mirrored)) + "\n")
print(f"wrote {out_path}; {len(mirrored)} LUMI paths to mirror under {MIRROR} (listed in {paths_out})")
