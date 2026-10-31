"""Checkpoint comparison (r6 SFT vs NVIDIA post-trained) on the stratified context set.

  analyze_ckpt.py --a r6=results/bf16/gen_ckpt.jsonl --b nvidia=results/nvidia/gen_ckpt.jsonl \
                  [--a-on-b results/nvidia/xscore.jsonl --b-on-a results/bf16/xscore.jsonl]

Per stratum (ck_turn0, ck_lt8k, ck_8_32k, ck_ge32k) and arm: degenerate rate (>=100 tokens, mean NLL > 3),
share of sampled tokens with p < e^-10 and < e^-5 (own T=1 logprobs), median per-sample mean NLL,
mean length, finish reasons; b - a differences with 95% bootstrap CIs over contexts.
Cross-scores (optional): each arm's samples rescored by the other model (score.py --gen):
mean NLL under the other model, and how the other model rates the tokens this arm sampled at p < e^-10.
"""
import argparse, collections, json, math, random, statistics as st

STRATA = ["ck_turn0", "ck_lt8k", "ck_8_32k", "ck_ge32k"]

def load_gen(spec):
    name, f = spec.split("=", 1)
    out = {}
    for x in map(json.loads, open(f)):
        lp = x["lp"]
        x["nll"] = -sum(lp) / len(lp) if lp else 0.0
        x["degen"] = len(lp) >= 100 and x["nll"] > 3.0
        x["n10"] = sum(v < -10 for v in lp); x["n5"] = sum(v < -5 for v in lp)
        out[(x["id"], x["i"])] = x
    return name, out

def metrics(v):
    tok = sum(x["ntok"] for x in v) or 1
    nl = [x["nll"] for x in v if x["ntok"] >= 20]
    return dict(n=len(v), degen=sum(x["degen"] for x in v) / max(1, len(v)), p10=sum(x["n10"] for x in v) / tok,
                p5=sum(x["n5"] for x in v) / tok, med=st.median(nl) if nl else float("nan"),
                len=st.fmean(x["ntok"] for x in v) if v else 0)

def boot(ctx_keys, A, B, key, reps=2000):
    cs = list(ctx_keys); rng = random.Random(0); d = []
    for _ in range(reps):
        s = [k for c in (rng.choice(cs) for _ in cs) for k in ctx_keys[c]]
        d.append(metrics([B[k] for k in s if k in B])[key] - metrics([A[k] for k in s if k in A])[key])
    d.sort(); return d[int(.025 * reps)], d[int(.975 * reps)]

def main(a):
    (na, A), (nb, B) = load_gen(a.a), load_gen(a.b)
    print(f"{na}: {len(A)} samples, {nb}: {len(B)} samples\n")
    print(f"{'stratum':10s} {'arm':8s} {'n':>5s} {'degen':>7s} {'p<e^-10':>8s} {'p<e^-5':>7s} {'medNLL':>7s} {'len':>6s}  finish")
    for s in STRATA + [None]:
        keys = sorted(k for k in set(A) | set(B) if s is None or s in (A.get(k) or B.get(k))["roles"])
        for n, D in ((na, A), (nb, B)):
            v = [D[k] for k in keys if k in D]
            if not v: continue
            m = metrics(v)
            print(f"{s or 'all':10s} {n:8s} {m['n']:5d} {m['degen']:7.2%} {m['p10']:8.4f} {m['p5']:7.4f} {m['med']:7.3f} {m['len']:6.0f}  "
                  f"{dict(collections.Counter(x['finish'] for x in v).most_common(3))}")
        ctx = collections.defaultdict(list)
        for k in keys:
            if k in A and k in B: ctx[k[0]].append(k)
        if ctx:
            for key, fmt in (("degen", "{:+.2%}"), ("p10", "{:+.4f}"), ("med", "{:+.3f}")):
                lo, hi = boot(ctx, A, B, key)
                diff = metrics([B[k] for kk in ctx.values() for k in kk])[key] - metrics([A[k] for kk in ctx.values() for k in kk])[key]
                print(f"{'':10s} {nb}-{na} {key:6s} " + fmt.format(diff) + "  95% CI [" + fmt.format(lo) + ", " + fmt.format(hi) + "]")
        print()
    for label, own, f, other in ((f"{na} samples scored by {nb}", A, a.a_on_b, nb), (f"{nb} samples scored by {na}", B, a.b_on_a, na)):
        if not f: continue
        X = {r["id"]: r["lp"] for r in map(json.loads, open(f))}
        print(f"=== {label}")
        for s in STRATA + [None]:
            pairs = []
            for sid, lp in X.items():
                cid, i = sid.rsplit("#", 1); g = own.get((cid, int(i)))
                if g is None or (s is not None and s not in g["roles"]): continue
                pairs += [(o, x) for o, x in zip(g["lp"], lp) if x is not None]
            if not pairs: continue
            tail = [x for o, x in pairs if o < -10]
            print(f"  {s or 'all':10s} tokens {len(pairs):8d}  NLL own {-st.fmean(o for o, _ in pairs):.3f} vs {other} {-st.fmean(x for _, x in pairs):.3f}  "
                  f"{other} p<e^-10 share {sum(x < -10 for _, x in pairs)/len(pairs):.4f}  "
                  f"own-tail tokens (p<e^-10): n={len(tail)}, {other} median lp {st.median(tail) if tail else float('nan'):.2f}, "
                  f"{other} also <e^-10: {sum(x < -10 for x in tail)/max(1,len(tail)):.2f}")
        print()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True); p.add_argument("--b", required=True)
    p.add_argument("--a-on-b"); p.add_argument("--b-on-a")
    main(p.parse_args())
