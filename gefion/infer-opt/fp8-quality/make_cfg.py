"""Single-node, non-PD inference config for the FP8-vs-bf16 A/B: the live run's resolved config
(job 534489, attempt_2) with PD, the trainer weight broadcast and KV offload switched off.

  make_cfg.py <out.json> bf16|fp8 [max_num_batched_tokens] [gpu_memory_utilization] [model_dir]

Everything else (TP8+EP8, fp32 lm_head, fp32 router logits, mamba fp32 state, prefix caching
"align", processed_logprobs, max_logprobs 33, infer-opt patches) stays as in the live run.
"""
import json, sys
LIVE = ("/dcai/projects/iu_0145/rasmus/runs/gefion-super-r6s2638-8t3p5d-tp8-r1/configs/attempt_2/resolved/"
        "inference.json")
out, prec = sys.argv[1], sys.argv[2]
d = json.load(open(LIVE))
d["use_pd_kv_transfer"] = False
d["weight_broadcast"] = {"type": "filesystem"}  # no trainer here
d["kv_cache_offload"] = None
d["vllm"]["quantization"] = {"bf16": None, "fp8": "fp8_per_block"}[prec]
if len(sys.argv) > 3: d["vllm"]["max_num_batched_tokens"] = int(sys.argv[3])
if len(sys.argv) > 4: d["vllm"]["gpu_memory_utilization"] = float(sys.argv[4])
if len(sys.argv) > 5: d["vllm"]["model"] = sys.argv[5]
d["env_vars"].pop("UCX_RCACHE_MAX_UNRELEASED", None); d["env_vars"].pop("UCX_MEMTYPE_CACHE", None)  # NIXL only
json.dump(d, open(out, "w"), indent=1)
print(out)
