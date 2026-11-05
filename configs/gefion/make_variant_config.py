"""Derive an inference-layout variant of the shakedown (gefion_shakedown_r1.toml): same envs, trainer
and batch, with only [inference.vllm] keys changed, so variants compare directly under the real
RL load. Each variant gets its own run dir and sandbox label; job_name stays gefion-super-shakedown
(order_nodes.sh relaxes the iommu=pt check for it).

Usage: make_variant_config.py <shakedown.toml> <out.toml> <tag> key=value ...
  e.g. make_variant_config.py gefion_shakedown_r1.toml gefion_dp8psi8_r1.toml dp8psi8 prefill_schedule_interval=8
  Values are parsed as TOML values; key=- removes the key.
"""

import sys
import tomllib

import tomli_w

src, dst, tag, *sets = sys.argv[1:]
with open(src, "rb") as f:
    cfg = tomllib.load(f)


def relabel(x):
    if isinstance(x, dict):
        return {k: relabel(v) for k, v in x.items()}
    if isinstance(x, list):
        return [relabel(v) for v in x]
    # Own sandbox label, so its cleanup cannot touch a concurrent run's sandboxes.
    if x == "gefion-super-shakedown-r1":
        return f"gefion-super-{tag}-r1"
    return x


cfg = relabel(cfg)
cfg["run"]["name"] = f"gefion-super-{tag}-4t1i-r1"
cfg["slurm"]["job_name"] = "gefion-super-shakedown"
v = cfg["inference"]["vllm"]
for s in sets:
    k, val = s.split("=", 1)
    if val == "-":
        v.pop(k, None)
    else:
        v[k] = tomllib.loads(f"x = {val}")["x"]
print({k: v.get(k) for k in ("tensor_parallel_size", "data_parallel_size", "data_parallel_size_local",
                             "max_num_seqs", "max_num_batched_tokens", "prefill_schedule_interval")})

with open(dst, "wb") as f:
    tomli_w.dump(cfg, f)
print(f"wrote {dst}")
