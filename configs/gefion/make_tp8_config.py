"""Derive the TP8 shakedown from gefion_shakedown_r1.toml: one TP8+EP8 engine per inference node
instead of DP8+EP8. Profiling showed DP lockstep stalls: one rank's prefill chunk (attention, Mamba,
dense layers on a single GPU) holds every rank in the MoE allgather. Under TP all 8 GPUs share the
chunk. Everything else (envs, sandboxes, trainer, batch) is unchanged so the runs compare directly.

Usage: make_tp8_config.py gefion_shakedown_r1.toml gefion_tp8_r1.toml
"""

import sys
import tomllib

import tomli_w

src, dst = sys.argv[1], sys.argv[2]
with open(src, "rb") as f:
    cfg = tomllib.load(f)


def relabel(x):
    if isinstance(x, dict):
        return {k: relabel(v) for k, v in x.items()}
    if isinstance(x, list):
        return [relabel(v) for v in x]
    # Own sandbox label, so its cleanup cannot touch a concurrent shakedown's sandboxes.
    if x == "gefion-super-shakedown-r1":
        return "gefion-super-tp8-r1"
    return x


cfg = relabel(cfg)
cfg["run"]["name"] = "gefion-super-tp8-4t1i-r1"
# job_name stays gefion-super-shakedown: order_nodes.sh relaxes the iommu=pt check for it.
cfg["slurm"]["job_name"] = "gefion-super-shakedown"

v = cfg["inference"]["vllm"]
v.update(tensor_parallel_size=8, data_parallel_size=1, data_parallel_size_local=1,
         enable_expert_parallel=True)
v.pop("all2all_backend", None)  # no DP: the EP group is the TP group
# Was 176 per DP engine (1408 per node); the shakedown keeps ~130 in flight, the full run ~190 per node.
v["max_num_seqs"] = 384

with open(dst, "wb") as f:
    tomli_w.dump(cfg, f)
print(f"wrote {dst}")
