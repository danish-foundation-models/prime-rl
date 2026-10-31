"""Compare per-token logprobs of the A/B set's sampled turns under three implementations.

  analyze_scores.py --set data/abset.jsonl --bf16 B.jsonl --fp8 F.jsonl [--bf16-topk B20.jsonl --fp8-topk F20.jsonl]

T = trace logprobs (live run: FP8, PD, decode path), F = FP8 teacher-forced (non-PD prefill path),
B = bf16 teacher-forced. Pairs:
  B-F : pure quantization effect (same path, same tokens)
  F-T : decode/PD path vs prefill path at the same precision (PD/KV-transfer or decode bugs show here)
  B-T : what a bf16 trainer sees against the FP8 sampler (k3 ~ the trainer's "Mismatch KL")
k3 = mean(exp(d) - 1 - d), d = logp_target - logp_sampler.
"""
import argparse, collections, json, math, statistics as st

def load(f):
    return {r["id"]: r for r in map(json.loads, open(f))} if f else {}

def stats(ds):
    ds = [d for d in ds if d is not None and math.isfinite(d)]
    if not ds: return "n=0"
    ab = sorted(abs(d) for d in ds)
    k3 = st.fmean(math.exp(max(min(d, 30), -30)) - 1 - d for d in ds)
    return (f"n={len(ds):8d} k3={k3:.2e} mean={st.fmean(ds):+.4f} mean|d|={st.fmean(ab):.4f} "
            f"p99|d|={ab[int(.99*(len(ab)-1))]:.3f} max|d|={ab[-1]:.2f}")

def main(a):
    S = [r for r in map(json.loads, open(a.set)) if set(r["roles"]) & {"normal", "degen"}]
    B, F = load(a.bf16), load(a.fp8)
    rows = [r for r in S if r["id"] in B and r["id"] in F]
    print(f"{len(rows)} turns scored on both servers (of {len(S)})\n")
    POS = [(0, 1), (1, 16), (16, 256), (256, 4096), (4096, 1 << 30)]
    CTX = [(0, 8000), (8000, 32000), (32000, 64000), (64000, 1 << 30)]
    TAIL = [(-0.1, 1), (-1, -0.1), (-3, -1), (-5, -3), (-10, -5), (-1e9, -10)]
    for role in ("normal", "degen"):
        R = [r for r in rows if role in r["roles"]]
        if not R: continue
        print(f"=== {role} turns: {len(R)}, {sum(r['ntok'] for r in R)} tokens")
        tri = [(t, f, b, i, r) for r in R for i, (t, f, b) in enumerate(zip(r["trace_lp"], F[r["id"]]["lp"], B[r["id"]]["lp"]))
               if f is not None and b is not None]
        for name, fn in (("B-F (quantization)", lambda x: x[2] - x[1]), ("F-T (PD/decode vs prefill)", lambda x: x[1] - x[0]),
                         ("B-T (bf16 trainer vs sampler)", lambda x: x[2] - x[0])):
            print(f"  {name:32s} {stats([fn(x) for x in tri])}")
        print("  by position in turn:")
        for lo, hi in POS:
            sel = [x for x in tri if lo <= x[3] < hi]
            if sel: print(f"    tok {lo:>5d}-{hi if hi < 1<<29 else 'end':<5}  B-F {stats([x[2]-x[1] for x in sel])}\n{'':20s}F-T {stats([x[1]-x[0] for x in sel])}")
        print("  by context length:")
        for lo, hi in CTX:
            sel = [x for x in tri if lo <= x[4]["prompt_len"] < hi]
            if sel: print(f"    ctx {lo//1000:>3d}k-{(str(hi//1000)+'k') if hi < 1<<29 else 'max':<5} B-F {stats([x[2]-x[1] for x in sel])}")
        print("  tail: sampled tokens binned by sampler (trace) logprob -> mean logprob shift")
        for lo, hi in TAIL:
            sel = [x for x in tri if lo <= x[0] < hi]
            if sel: print(f"    trace lp in [{lo:>6.1f},{hi:>5.1f}) n={len(sel):7d}  mean(B-F)={st.fmean(x[2]-x[1] for x in sel):+.4f}  "
                          f"mean(F-T)={st.fmean(x[1]-x[0] for x in sel):+.4f}  mean(B-T)={st.fmean(x[2]-x[0] for x in sel):+.4f}  "
                          f"frac(B<F-1)={sum(x[2] < x[1]-1 for x in sel)/len(sel):.3f}")
        # per-turn mean NLL under each implementation (would bf16 have flagged it?)
        nll = lambda v: -st.fmean(v)
        tm = [(nll(r["trace_lp"]), nll([x for x in F[r["id"]]["lp"] if x is not None]), nll([x for x in B[r["id"]]["lp"] if x is not None])) for r in R]
        print(f"  per-turn mean NLL, median over turns: T={st.median(x[0] for x in tm):.3f} F={st.median(x[1] for x in tm):.3f} B={st.median(x[2] for x in tm):.3f}")
        if role == "degen":
            # does bf16 make the derailment onset less likely than FP8 did?  window = 64 tokens before / after onset
            pre, at = [], []
            for r in R:
                o = r.get("onset")
                if o is None: continue
                f, b = F[r["id"]]["lp"], B[r["id"]]["lp"]
                d = [(bb - ff) if (bb is not None and ff is not None) else 0.0 for ff, bb in zip(f, b)]
                pre.append(sum(d[max(0, o - 64):o])); at.append(sum(d[o:o + 64]))
            if pre:
                print(f"  onset windows (sum of B-F log-ratio, nats): 64 tok before onset median {st.median(pre):+.2f}, "
                      f"64 tok from onset median {st.median(at):+.2f} (mean {st.fmean(at):+.2f}); "
                      f"frac turns with onset window B-F < -2 nats: {sum(x < -2 for x in at)/len(at):.2f}, > +2: {sum(x > 2 for x in at)/len(at):.2f}")
        agree = [F[r["id"]]["top_ids"][i][:1] == B[r["id"]]["top_ids"][i][:1] for r in R for i in range(r["ntok"])
                 if F[r["id"]]["top_ids"][i] and B[r["id"]]["top_ids"][i]]
        if agree: print(f"  argmax agreement B vs F: {sum(agree)/len(agree):.4f}")
        print()
    BK, FK = load(a.bf16_topk), load(a.fp8_topk)
    if BK and FK:
        print("=== top-k (prompt_logprobs=k) on the shorter turns: tail mass and KL(F||B) over F's top-k")
        for role in ("normal", "degen"):
            R = [r for r in rows if role in r["roles"] and r["id"] in BK and r["id"] in FK]
            if not R: continue
            tailB, tailF, kl, entB, entF = [], [], [], [], []
            for r in R:
                for i in range(r["ntok"]):
                    bi, bl, fi, fl = BK[r["id"]]["top_ids"][i], BK[r["id"]]["top_lp"][i], FK[r["id"]]["top_ids"][i], FK[r["id"]]["top_lp"][i]
                    if not bi or not fi: continue
                    tb = max(1e-9, 1 - sum(math.exp(x) for x in bl)); tf = max(1e-9, 1 - sum(math.exp(x) for x in fl))
                    tailB.append(tb); tailF.append(tf)
                    bm = dict(zip(bi, bl)); floor = min(bl) if bl else -30
                    # tokens in F's top-k but not B's get B's k-th logprob as an upper bound -> KL lower bound
                    kl.append(sum(math.exp(x) * (x - bm.get(t, floor)) for t, x in zip(fi, fl)) + tf * math.log(tf / tb))
                    entB.append(-sum(math.exp(x) * x for x in bl)); entF.append(-sum(math.exp(x) * x for x in fl))
            print(f"  {role}: positions={len(kl)} tail mass outside top-k: B mean {st.fmean(tailB):.4f} F mean {st.fmean(tailF):.4f} "
                  f"(F/B {st.fmean(tailF)/st.fmean(tailB):.3f}); top-k entropy B {st.fmean(entB):.4f} F {st.fmean(entF):.4f}; "
                  f"KL_topk(F||B) mean {st.fmean(kl):.2e} p99 {sorted(kl)[int(.99*(len(kl)-1))]:.3f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--set", required=True); p.add_argument("--bf16", required=True); p.add_argument("--fp8", required=True)
    p.add_argument("--bf16-topk"); p.add_argument("--fp8-topk")
    main(p.parse_args())
