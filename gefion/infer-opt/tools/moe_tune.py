"""Microbenchmark + Triton config tuner for Super's routed experts on one DP8+EP8 rank.

Shapes: 64 local of 512 experts, top-22, latent K=1024, N=2688, relu² (non-gated), bf16.
M is the number of *gathered* tokens the rank's MoE sees (8 x per-rank batch with allgather EP).

  moe_tune.py bench  [--configs DIR]   stock TritonExperts body vs infer_opt.ep_moe, CUDA-graphed,
                                       plus max |diff| between them
  moe_tune.py tune   --out DIR         search tile configs for the compact path on all visible GPUs,
                                       write E=64,N=2688,device_name=...json (VLLM_TUNED_CONFIG_FOLDER)
"""
import argparse
import itertools
import json
import os
import sys
import time
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

E_GLOBAL, EP, TOPK, K, N = 512, 8, 22, 1024, 2688
E_LOCAL = E_GLOBAL // EP
M_DECODE = [8, 16, 32, 64, 128, 192, 256, 384, 512, 640, 768, 1024]
M_PREFILL = [1536, 2048, 3072, 4096, 8192, 16384, 32768]


def make_inputs(M: int, seed: int = 0, dev="cuda"):
    g = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn(M, K, device=dev, dtype=torch.bfloat16, generator=g)
    w1 = torch.randn(E_LOCAL, N, K, device=dev, dtype=torch.bfloat16, generator=g) * (K ** -0.5)
    w2 = torch.randn(E_LOCAL, K, N, device=dev, dtype=torch.bfloat16, generator=g) * (N ** -0.5)
    scores = torch.rand(M, E_GLOBAL, device=dev, generator=g)
    tw, tid = scores.topk(TOPK, dim=-1)
    tw = (tw / tw.sum(-1, keepdim=True)).float()
    tid = tid.to(torch.int32)
    emap = torch.full((E_GLOBAL,), -1, dtype=torch.int32, device=dev)
    emap[:E_LOCAL] = torch.arange(E_LOCAL, dtype=torch.int32, device=dev)  # rank 0 owns experts 0..63
    return x, w1, w2, tw, tid, emap


def fake_self():
    from vllm.model_executor.layers.fused_moe.config import FUSED_MOE_UNQUANTIZED_CONFIG

    def moe_problem_size(h, w1, w2, ids):
        return w1.shape[0], h.shape[0], w1.shape[1], w2.shape[1], ids.shape[1]

    return SimpleNamespace(moe_problem_size=moe_problem_size, quant_config=FUSED_MOE_UNQUANTIZED_CONFIG,
                           block_shape=None, w1_bias=None, w2_bias=None)


def workspaces(M):
    ws13 = torch.empty(M * TOPK * max(N, K), device="cuda", dtype=torch.bfloat16)
    ws2 = torch.empty(M * TOPK * max(N, K), device="cuda", dtype=torch.bfloat16)
    return ws13, ws2


def stock_apply(out, x, w1, w2, tw, tid, emap, ws13, ws2):
    """TritonExperts.apply (vLLM 0.28) for bf16 relu² with an expert_map, without LoRA."""
    import torch.nn.functional as F
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        _prepare_expert_assignment, invoke_fused_moe_triton_kernel, try_get_optimal_moe_config)
    from vllm.model_executor.layers.fused_moe.utils import _resize_cache
    from vllm.triton_utils import tl

    M = x.shape[0]
    cfg = try_get_optimal_moe_config(w1.size(), w2.size(), TOPK, None, M)
    c1 = _resize_cache(ws2, (M, TOPK, N))
    c2 = _resize_cache(ws13, (M * TOPK, N))
    c3 = _resize_cache(ws2, (M, TOPK, K))
    sid, eid, npp = _prepare_expert_assignment(tid, cfg, M, TOPK, E_GLOBAL, emap)
    common = dict(compute_type=tl.bfloat16, use_fp8_w8a8=False, use_int8_w8a8=False, use_int8_w8a16=False,
                  use_int4_w4a16=False, per_channel_quant=False, block_shape=None)
    invoke_fused_moe_triton_kernel(x, w1, c1, None, None, None, sid, eid, npp, False, TOPK, cfg, **common)
    inp = c1.view(-1, N)
    F.relu(inp, inplace=True)
    torch.square(inp, out=c2)
    invoke_fused_moe_triton_kernel(c2, w2, c3, None, None, tw, sid, eid, npp, True, 1, cfg, **common)
    ops.moe_sum(c3, out)


def compact(out, x, w1, w2, tw, tid, emap, ws13, ws2):
    from infer_opt.ep_moe import compact_apply
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    compact_apply(fake_self(), out, x, w1, w2, tw, tid, MoEActivation.RELU2_NO_MUL, E_GLOBAL, emap, ws13, ws2, False)


def time_graph(fn, iters=20):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    g.replay()
    torch.cuda.synchronize()
    t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(iters):
        g.replay()
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) / iters * 1000  # us


def bench(a):
    if a.configs:
        os.environ["VLLM_TUNED_CONFIG_FOLDER"] = a.configs
    from vllm.model_executor.layers.fused_moe.fused_moe import get_moe_configs
    get_moe_configs.cache_clear()
    rows = []
    for M in M_DECODE + M_PREFILL:
        x, w1, w2, tw, tid, emap = make_inputs(M)
        ws13, ws2 = workspaces(M)
        o1 = torch.empty(M, K, device="cuda", dtype=torch.bfloat16)
        o2 = torch.empty_like(o1)
        stock_apply(o1, x, w1, w2, tw, tid, emap, ws13, ws2)
        r1 = o1.clone()
        compact(o2, x, w1, w2, tw, tid, emap, ws13, ws2)
        diff = (r1.float() - o2.float()).abs().max().item()
        exact = torch.equal(r1, o2)
        ts = time_graph(lambda: stock_apply(o1, x, w1, w2, tw, tid, emap, ws13, ws2))
        tc = time_graph(lambda: compact(o2, x, w1, w2, tw, tid, emap, ws13, ws2))
        # bytes of expert weights a rank must read once per layer (lower bound for decode)
        wbytes = 2 * E_LOCAL * N * K * 2
        rows.append(dict(M=M, stock_us=round(ts, 1), compact_us=round(tc, 1), speedup=round(ts / tc, 3),
                         max_abs_diff=diff, bitwise_equal=exact, weight_floor_us=round(wbytes / 3.35e12 * 1e6, 1)))
        print(json.dumps(rows[-1]), flush=True)
        del x, w1, w2, ws13, ws2
        torch.cuda.empty_cache()
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1)


def search_space():
    for bm, bn, bk, gm, nw, ns in itertools.product([16, 32, 64, 128], [64, 128, 256], [64, 128], [1, 16],
                                                    [4, 8], [3, 4, 5]):
        smem = ns * (bm * bk + bk * bn) * 2
        if smem > 220 * 1024 or (bm >= 128 and bn >= 256 and nw == 4):
            continue
        yield dict(BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=bk, GROUP_SIZE_M=gm, SPLIT_K=1,
                   num_warps=nw, num_stages=ns)


def tune_worker(rank: int, world: int, Ms: list[int], q):
    torch.cuda.set_device(rank)
    from vllm.model_executor.layers.fused_moe import override_config

    cfgs = [c for i, c in enumerate(search_space()) if i % world == rank]
    res = {}
    for M in Ms:
        x, w1, w2, tw, tid, emap = make_inputs(M)
        ws13, ws2 = workspaces(M)
        o = torch.empty(M, K, device="cuda", dtype=torch.bfloat16)
        best = (1e18, None)
        for c in cfgs:
            try:
                with override_config(c):
                    fn = lambda: compact(o, x, w1, w2, tw, tid, emap, ws13, ws2)  # noqa: E731
                    t = time_graph(fn, iters=5 if M >= 8192 else 20)
            except Exception:  # noqa: BLE001  (out of resources etc.)
                continue
            if t < best[0]:
                best = (t, c)
        res[M] = best
        print(f"[gpu{rank}] M={M} best={best[0]:.1f}us {best[1]}", flush=True)
        del x, w1, w2, ws13, ws2
        torch.cuda.empty_cache()
    q.put(res)


def tune(a):
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    world = torch.cuda.device_count()
    Ms = M_DECODE + M_PREFILL
    procs = [ctx.Process(target=tune_worker, args=(r, world, Ms, q)) for r in range(world)]
    for p in procs:
        p.start()
    results = [q.get() for _ in procs]
    for p in procs:
        p.join()
    best = {}
    for M in Ms:
        t, c = min((r[M] for r in results if r[M][1] is not None), key=lambda x: x[0])
        best[str(M)] = c
        print(f"M={M}: {t:.1f} us {c}")
    from vllm.model_executor.layers.fused_moe.fused_moe import get_config_file_name

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, get_config_file_name(E_LOCAL, N, None))
    best["triton_version"] = __import__("triton").__version__
    json.dump(best, open(path, "w"), indent=1)
    print("wrote", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["bench", "tune"])
    ap.add_argument("--configs", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    bench(a) if a.mode == "bench" else tune(a)
