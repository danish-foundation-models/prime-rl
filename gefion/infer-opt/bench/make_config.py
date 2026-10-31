"""Write a PRIME-RL inference config: the smoke run's resolved config, DP8+EP8 baseline layout, plus overrides.

Usage: make_config.py <out.json> ['{"vllm.use_replayssm": true, "vllm.moe_backend": "flashinfer_cutlass"}']
Dotted keys address nested fields; a null value deletes the key.
"""
import json
import sys

BASE = ("/dcai/projects/iu_0145/rasmus/phase0/smoke/runs/gefion-super-smoke-8t2i-math-r1/configs/attempt_1/"
        "resolved/inference.json")
# The section-11 winner (sweep_configs.py "dp8"): the baseline every variant is compared to.
DP8 = dict(tensor_parallel_size=1, data_parallel_size=8, data_parallel_size_local=8, max_num_seqs=96,
           max_num_batched_tokens=4096, gpu_memory_utilization=0.90, all2all_backend="allgather_reducescatter",
           enable_expert_parallel=True)

out, overrides = sys.argv[1], json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
d = json.load(open(BASE))
d["vllm"].update(DP8)
d["vllm"].pop("worker_cls", None)  # VerifiedSuperWorker rejects DP>1
d["weight_broadcast"] = {"type": "filesystem"}  # no trainer in these jobs
d["kv_cache_offload"] = None  # 8 x 192 GiB of /dev/shm does not fit
for key, val in overrides.items():
    *path, leaf = key.split(".")
    node = d
    for p in path:
        node = node.setdefault(p, {})
    if val is None:
        node.pop(leaf, None)
    else:
        node[leaf] = val
json.dump(d, open(out, "w"), indent=1)
print(out)
