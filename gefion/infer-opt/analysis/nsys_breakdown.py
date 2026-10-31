"""Per-step GPU time breakdown from an nsys sqlite export of a vLLM DP/EP server.

Usage: nsys_breakdown.py <report.sqlite> [--device N] [--json out.json]

Kernels are bucketed by name. A kernel launched while an NVTX range from infer_opt.nvtx
("lm_head", "sampler*") was open on the launching thread is attributed to that range, which
separates the eager fp32 LM head / logprob kernels from the CUDA-graph model forward.
Step boundaries are the NVTX "step:execute_model" ranges; per-step wall time minus GPU busy
time is the CPU/scheduler gap.
"""
import argparse
import bisect
import json
import re
import sqlite3
from collections import defaultdict

CATS = [
    ("comm:allgather", r"nccl.*allgather|allgather.*nccl"),
    ("comm:reducescatter", r"nccl.*reducescatter"),
    ("comm:allreduce", r"nccl.*allreduce|cross_device_reduce|allreduce"),
    ("comm:other", r"nccl|all2all|dispatch|combine|deep_ep|nvshmem"),
    ("moe:expert_gemm", r"fused_moe_kernel|cutlass.*moe|moe.*gemm|grouped_gemm|cutlass_kernel.*group|moegemm|fused_moe"),
    ("moe:relu2_stock", r"pow_tensor_scalar|launch_clamp_scalar"),
    ("moe:route_permute", r"single_group_topk|_relu2_rows|_combine_local|moe_align|count_and_sort|moe_sum|sort_tokens|topk_softmax|grouped_topk|moe_permute|unpermute|"
                          r"expand|ep_scatter|ep_gather|topkgating|fused_topk|biased_grouped_topk|noaux_tc|routing|"
                          r"compute_identity|moe_fused_mul|finalize"),
    ("mamba:ssm_decode", r"selective_scan_update|selective_state_update|_ssu|ssu_|replay"),
    ("mamba:ssd_prefill", r"chunk_scan|chunk_state|state_passing|bmm_chunk|chunk_cumsum|_ssd|ssd_"),
    ("mamba:conv", r"causal_conv1d|conv1d"),
    ("mamba:other", r"mamba|gated_rms|rmsnorm_gated|layer_norm_fwd"),
    ("attention", r"flash|fmha|attention|reshape_and_cache|paged|kv_cache"),
    ("dense_gemm", r"gemm|gemv|nvjet|cutlass|xmma|cublas|matmul|splitk"),
    ("elementwise/norm", r"triton_|rms|norm|elementwise|vectorized|unrolled|act_and_mul|relu|silu|copy|fill|cat"),
]
CATS = [(c, re.compile(p, re.I)) for c, p in CATS]
SAMPLER_RANGES = ("lm_head", "sampler", "sampler:gather_logprobs", "sampler:sample")


def categorize(name: str) -> str:
    for c, rx in CATS:
        if rx.search(name):
            return c
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("--device", type=int, default=None, help="default: lowest device id")
    ap.add_argument("--json", default=None)
    ap.add_argument("--top", type=int, default=25)
    a = ap.parse_args()
    con = sqlite3.connect(a.db)
    cur = con.cursor()
    strings = dict(cur.execute("select id, value from StringIds"))
    devs = [r[0] for r in cur.execute("select distinct deviceId from CUPTI_ACTIVITY_KIND_KERNEL order by 1")]
    dev = devs[0] if a.device is None else a.device
    kcols = [r[1] for r in cur.execute("pragma table_info(CUPTI_ACTIVITY_KIND_KERNEL)")]
    graph_col = "graphId" if "graphId" in kcols else ("graphNodeId" if "graphNodeId" in kcols else None)
    q = (f"select start, end, demangledName, correlationId, globalPid, "
         f"{graph_col if graph_col else 'NULL'} from CUPTI_ACTIVITY_KIND_KERNEL where deviceId=? order by start")
    kernels = cur.execute(q, (dev,)).fetchall()
    if not kernels:
        raise SystemExit("no kernels")
    pid = kernels[0][4]

    # launch API call (cpu thread + time) for each correlation id in this process
    rt = {}
    for corr, start, tid in cur.execute(
            "select correlationId, start, globalTid from CUPTI_ACTIVITY_KIND_RUNTIME where globalTid/0x1000000 = ?/0x1000000",
            (pid,)):
        rt[corr] = (start, tid)

    # NVTX push/pop ranges on that process
    nv = defaultdict(list)  # tid -> [(start, end, text)]
    has_nvtx = cur.execute("select count(*) from sqlite_master where name='NVTX_EVENTS'").fetchone()[0]
    steps = []
    if has_nvtx:
        for start, end, text, textid, tid in cur.execute(
                "select start, end, text, textId, globalTid from NVTX_EVENTS where end is not null "
                "and globalTid/0x1000000 = ?/0x1000000", (pid,)):
            t = text if text is not None else strings.get(textid, "")
            if t == "step:execute_model":
                steps.append((start, end))
            if t in SAMPLER_RANGES:
                nv[tid].append((start, end, t))
    for v in nv.values():
        v.sort()
    nv_starts = {tid: [r[0] for r in v] for tid, v in nv.items()}

    def nvtx_label(corr):
        if corr not in rt:
            return None
        t, tid = rt[corr]
        v = nv.get(tid)
        if not v:
            return None
        i = bisect.bisect_right(nv_starts[tid], t) - 1
        best = None
        while i >= 0 and v[i][0] <= t:  # innermost open range containing t
            if v[i][1] >= t and (best is None or v[i][0] > best[0]):
                best = v[i]
            if t - v[i][0] > 5e9:
                break
            i -= 1
        return best[2] if best else None

    steps.sort()
    step_starts = [s for s, _ in steps]
    per_cat = defaultdict(float)
    per_kernel = defaultdict(lambda: [0.0, 0])
    in_graph = 0.0
    total = 0.0
    for start, end, nameid, corr, _pid, graph in kernels:
        name = strings.get(nameid, str(nameid))
        dur = (end - start) / 1e6  # ms
        lab = nvtx_label(corr)
        if lab is not None and lab.startswith("lm_head"):
            cat = "lm_head(fp32)"
        elif lab is not None and lab.startswith("sampler"):
            cat = "sampler:" + ("logprobs" if lab.endswith("gather_logprobs") else "sample/other")
        else:
            cat = categorize(name)
        per_cat[cat] += dur
        k = per_kernel[(cat, name[:110])]
        k[0] += dur
        k[1] += 1
        total += dur
        if graph:
            in_graph += dur
    t0, t1 = kernels[0][0], kernels[-1][1]
    # only count complete steps inside the kernel window
    full = [(s, e) for s, e in steps if s >= t0 and e <= t1]
    n_steps = max(1, len(full) - 1)
    span = (full[-1][0] - full[0][0]) / 1e6 if len(full) > 1 else (t1 - t0) / 1e6
    # GPU busy (union) over the step span
    busy, last = 0.0, None
    lo, hi = (full[0][0], full[-1][0]) if len(full) > 1 else (t0, t1)
    for s, e, *_ in kernels:
        s, e = max(s, lo), min(e, hi)
        if e <= s:
            continue
        if last is None or s > last:
            busy += e - s
            last = e
        elif e > last:
            busy += e - last
            last = e
    busy /= 1e6
    scale = (span / (t1 - t0) * 1e6) if span else 1.0
    out = {
        "device": dev, "steps": n_steps, "step_ms": round(span / n_steps, 3),
        "gpu_busy_ms_per_step": round(busy / n_steps, 3),
        "idle_ms_per_step": round((span - busy) / n_steps, 3),
        "cuda_graph_kernel_frac": round(in_graph / total, 3) if graph_col else None,
        "ms_per_step": {c: round(v * scale / n_steps, 3) for c, v in sorted(per_cat.items(), key=lambda x: -x[1])},
        "top_kernels": [(c, n, round(t * scale / n_steps, 3), cnt) for (c, n), (t, cnt) in
                        sorted(per_kernel.items(), key=lambda x: -x[1][0])[:a.top]],
    }
    print(json.dumps(out, indent=1))
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
