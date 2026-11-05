"""Derive the PD (disaggregated prefill/decode) shakedown from gefion_shakedown_r1.toml: one DP8+EP8
prefill node and one DP8+EP8 decode node instead of a single DP8+EP8 node. Profiling (job 533102)
showed the DP ranks run in lockstep through the MoE allgather, so one rank's prefill chunk stalls the
other 7 ranks' decode; with PD the decode node only runs short steps. Envs, sandboxes, trainer and
batch are unchanged so the runs compare directly.

Deviations from prime-rl's PD defaults, all through the config (no template edits):
- all2all_backend: the PD template hardcodes DeepEP (high-throughput on prefill, low-latency on
  decode). DeepEP is untested on Gefion and its low-latency mode already failed on one node
  (num_rdma_bytes < INT_MAX assert, gefion-phase0-report.md), so both roles keep allgather_reducescatter.
  The per-role overrides are appended after the template's keys in the same JSON object, and the
  later duplicate key wins in json.load.
- decode cudagraph_mode: the template sets FULL_DECODE_ONLY; keep vLLM's default FULL_AND_PIECEWISE
  (as the baseline) so local recompute/preemption prefill on the decode node still runs graphs.
- VLLM_SSM_CONV_STATE_LAYOUT=DS: vLLM 0.28's NixlConnector asserts this layout to transfer Mamba
  conv state (nixl/base_worker.py), and both roles must use the same layout.
- No native CPU KV offload: with PD it would be composed with NIXL through a MultiConnector, an
  untested combination for this hybrid model; the prefill node's whole KV pool is prefix cache anyway.
  offload_fix is dropped from INFER_OPT with it.

Usage: make_pd_config.py gefion_shakedown_r1.toml gefion_pd_r1.toml
"""

import sys
import tomllib

import tomli_w


def make_pd(cfg, n_prefill=1, n_decode=1):
    """Turn a colocated DP8+EP8 config into PD with n_prefill + n_decode nodes, in place. Each node is its
    own 1-node replica (DP8+EP8 within the node): a multi-node replica would be one EP group across
    nodes, with the MoE allgather over InfiniBand on every step."""
    cfg["deployment"].update(num_infer_nodes=n_prefill + n_decode, num_infer_replicas=1)
    inf = cfg["inference"]
    ARB = "allgather_reducescatter"
    inf["deployment"] = {
        "type": "disaggregated",
        "gpus_per_node": 8,
        "prefill_nodes_per_replica": 1,
        "decode_nodes_per_replica": 1,
        "num_prefill_replicas": n_prefill,
        "num_decode_replicas": n_decode,
        "prefill_vllm_overrides": {"all2all_backend": ARB},
        "decode_vllm_overrides": {"all2all_backend": ARB,
                                  "compilation_config": {"cudagraph_mode": "FULL_AND_PIECEWISE"}},
        # The template pins decode to mlx5_0:1 (one NIC for all 8 GPUs); prefill gets every active IB
        # port. decode_env_vars are exported after the pin, so give decode the same Gefion IB ports
        # (NCCL_IB_HCA). Upstream prime-rl dropped the pin (c59badb2: 1.7x faster KV transfers).
        "decode_env_vars": {"UCX_NET_DEVICES": ",".join(f"mlx5_{i}:1" for i in (0, 3, 4, 5, 6, 9, 10, 11))},
    }
    inf.pop("kv_cache_offload", None)
    env = inf["env_vars"]
    env["INFER_OPT"] = ",".join(p for p in env["INFER_OPT"].split(",") if p != "offload_fix")
    env["VLLM_SSM_CONV_STATE_LAYOUT"] = "DS"
    # NIXL/UCX: the bundled UCX of the nixl-cu13 wheel is reported to segfault on P->D transfers
    # (prime-rl docs/advanced.md); a source build is the upstream fix but is not available here. These two
    # are the cheap mitigations used upstream: vLLM's rcache cap (which it cannot set itself once nixl is
    # imported; prime-rl's GLM PD example sets it "else random segfaults") and no memtype cache (as
    # prime-rl's own NIXL weight agent).
    env["UCX_RCACHE_MAX_UNRELEASED"] = "1024"
    env["UCX_MEMTYPE_CACHE"] = "n"
    v = inf["vllm"]
    assert v["all2all_backend"] == ARB and v["data_parallel_size"] == 8 and v["tensor_parallel_size"] == 1
    return cfg


def relabel(x):
    if isinstance(x, dict):
        return {k: relabel(v) for k, v in x.items()}
    if isinstance(x, list):
        return [relabel(v) for v in x]
    # Own sandbox label, so its cleanup cannot touch a concurrent shakedown's sandboxes.
    if x == "gefion-super-shakedown-r1":
        return "gefion-super-pd-r1"
    return x


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    with open(src, "rb") as f:
        cfg = tomllib.load(f)
    cfg = relabel(cfg)
    cfg["run"]["name"] = "gefion-super-pd-4t1p1d-r1"
    # job_name stays gefion-super-shakedown: order_nodes_pd.sh relaxes the iommu=pt check for it.
    cfg["slurm"]["job_name"] = "gefion-super-shakedown"
    make_pd(cfg)
    with open(dst, "wb") as f:
        tomli_w.dump(cfg, f)
    print(f"wrote {dst}")
