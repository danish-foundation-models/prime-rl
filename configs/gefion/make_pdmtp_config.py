"""Derive the PD + MTP shakedown from gefion_pd_r1.toml (make_pd_config.py): MTP k=1 on both roles.
NIXL puts speculative_config in its handshake compatibility hash, so P and D must both run it; P
computes the drafter layer's KV and it is transferred like any other layer. The gain is on D, which
now only runs decode steps (MTP: 1.24-1.46x on decode-only steps, job 533117).

Usage: make_pdmtp_config.py gefion_pd_r1.toml gefion_pdmtp_r1.toml
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
    # Own sandbox label, so its cleanup cannot touch a concurrent PD run's sandboxes.
    if x == "gefion-super-pd-r1":
        return "gefion-super-pdmtp-r1"
    return x


cfg = relabel(cfg)
cfg["run"]["name"] = "gefion-super-pdmtp-4t1p1d-r1"

inf = cfg["inference"]
inf["vllm"]["speculative_config"] = {"method": "mtp", "num_speculative_tokens": 1}
# MTP needs 1 + k Mamba state blocks per request: half the no-MTP 176 per rank (D ran 16-23 per rank).
inf["deployment"]["decode_vllm_overrides"]["max_num_seqs"] = 96
# No last-block drop on prefix hits (with it, MTP got 0.3% prefix hits on multi-turn loads).
env = inf["env_vars"]
opts = env["INFER_OPT"].split(",")
if "mtp_keep_last_block" not in opts:
    opts.append("mtp_keep_last_block")
env["INFER_OPT"] = ",".join(opts)
# vLLM 0.28 corrupts MTP decoding when prefix caching is on and the batch budget is below 4176.
assert inf["vllm"]["max_num_batched_tokens"] >= 4176

with open(dst, "wb") as f:
    tomli_w.dump(cfg, f)
print(f"wrote {dst}")
