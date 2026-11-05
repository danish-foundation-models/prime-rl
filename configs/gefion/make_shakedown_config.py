"""Derive a small shakedown run from the full Gefion config (gefion_full_r1.toml).

Same environments, sandboxes, eval suite, FP8 DP8+EP8 inference node and NCCL weight broadcast as
the full run, so its startup and first steps hit the same failure points, but on 4 trainer + 1
inference nodes for a few steps. Trainer difference: the optimizer offloaded to
CPU so the fp32 master weights and Adam state fit on half the GPUs.

Usage: make_shakedown_config.py gefion_full_r1.toml gefion_shakedown_r1.toml
"""

import sys
import tomllib

import tomli_w

src, dst = sys.argv[1], sys.argv[2]
with open(src, "rb") as f:
    cfg = tomllib.load(f)

RUN_NAME = "gefion-super-shakedown-4t1i-r1"
SANDBOX_RUN_LABEL = "gefion-super-shakedown-r1"


def relabel(x):
    if isinstance(x, dict):
        return {k: relabel(v) for k, v in x.items()}
    if isinstance(x, list):
        return [relabel(v) for v in x]
    if x == "gefion-super-r6-step2638-r1":
        return SANDBOX_RUN_LABEL
    return x


cfg = relabel(cfg)
cfg["run"]["name"] = RUN_NAME
cfg["max_steps"] = 3
cfg.pop("resume", None)  # shakedowns start fresh, never from the full run's checkpoint
cfg["deployment"].update(num_train_nodes=4, num_infer_replicas=1)
# Any working node: order_nodes.sh lets this job's trainer use nodes without iommu=pt (slower, not
# broken), so the shakedown does not wait for 4 fixed nodes to free up.
cfg["slurm"].update(job_name="gefion-super-shakedown", time="03:00:00",
                    exclude="dgx035,dgx150,dgx157")
cfg["trainer"]["model"].update(optim_cpu_offload=True)  # EP8 kept: ep must be a multiple of cp (8)

orch = cfg["orchestrator"]
orch["batch_size"] = 128
conc = orch["concurrency"]
conc.update(initial_inflight=128, min_inflight=64, max_inflight=256)
# Full run: swe 240, terminal 128, sql 32, tooluse 56, instruction 32, math 16, toolbench 8 (= 512).
# Scaled to 128 while keeping every category and whole groups of 8.
conc["category_targets"] = dict(swe=48, terminal=32, sql=8, tooluse=16, instruction=8, math=8, toolbench=8)
assert sum(conc["category_targets"].values()) == orch["batch_size"]

with open(dst, "wb") as f:
    tomli_w.dump(cfg, f)
print(f"wrote {dst}")
