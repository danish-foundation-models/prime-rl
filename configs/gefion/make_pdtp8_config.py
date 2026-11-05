"""Derive the TP8 PD shakedown from gefion_pd_r1.toml (make_pd_config.py): one TP8+EP8 engine per
prefill and per decode node instead of DP8+EP8. NIXL needs equal TP on both roles (the KV block size
depends on TP). The gain is on P: TP8 has 1040-token prefix-cache blocks against DP8's 4160, so less
of each turn is re-prefilled: 1.53 computed prompt tokens per generated token (job 533520) against
2.85 for DP8 PD (job 533326), with the same ~45 tok/s per sequence on D.

Usage: make_pdtp8_config.py gefion_pd_r1.toml gefion_pdtp8_r1.toml
"""

import sys
import tomllib

import tomli_w


def make_tp8(cfg):
    """Switch a PD config from DP8+EP8 to one TP8+EP8 engine per prefill and decode node, in place."""
    inf = cfg["inference"]
    inf["vllm"].update(tensor_parallel_size=8, data_parallel_size=1, data_parallel_size_local=1)
    # all2all_backend stays allgather_reducescatter (unused without DP, but it keeps the PD template's
    # DeepEP defaults out of the per-role overrides).
    # One engine per node instead of 8 x 176; the TP8 colocated run had ~12M tokens of KV per node.
    inf["deployment"]["decode_vllm_overrides"]["max_num_seqs"] = 384
    return cfg


def relabel(x):
    if isinstance(x, dict):
        return {k: relabel(v) for k, v in x.items()}
    if isinstance(x, list):
        return [relabel(v) for v in x]
    # Own sandbox label, so its cleanup cannot touch a concurrent PD run's sandboxes.
    if x == "gefion-super-pd-r1":
        return "gefion-super-pdtp8-r1"
    return x


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    with open(src, "rb") as f:
        cfg = tomllib.load(f)
    cfg = relabel(cfg)
    cfg["run"]["name"] = "gefion-super-pdtp8-4t1p1d-r1"
    make_tp8(cfg)
    with open(dst, "wb") as f:
        tomli_w.dump(cfg, f)
    print(f"wrote {dst}")
