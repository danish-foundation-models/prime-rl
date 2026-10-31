"""Degenerate-turn rate of resampled turns, per server, paired by context and seed.

  analyze_gen.py --a bf16=GEN_B.jsonl --b fp8=GEN_F.jsonl
Degenerate = >= 100 tokens and mean NLL > 3 (the run's rule). CI: bootstrap over contexts.
"""
import argparse, collections, json, random, statistics as st

def load(spec):
    name, f = spec.split("=", 1)
    out = {}
    for x in map(json.loads, open(f)):
        lp = x["lp"]; nll = -sum(lp) / len(lp) if lp else 0.0
        x["nll"] = nll; x["degen"] = len(lp) >= 100 and nll > 3.0
        out[(x["id"], x["i"])] = x
    return name, out

def main(a):
    (na, A), (nb, B) = load(a.a), load(a.b)
    keys = sorted(set(A) & set(B))
    print(f"{len(keys)} paired samples ({len(A)} {na}, {len(B)} {nb})")
    for role in ("gen_degen", "gen_normal", None):
        K = [k for k in keys if role is None or role in A[k]["roles"]]
        if not K: continue
        print(f"\n=== {role or 'all'}: {len(K)} samples, {len({k[0] for k in K})} contexts")
        for n, D in ((na, A), (nb, B)):
            v = [D[k] for k in K]; nd = [x["nll"] for x in v if not x["degen"] and x["ntok"] >= 100]
            print(f"  {n:6s} degenerate {sum(x['degen'] for x in v)/len(v):6.2%}  median NLL (non-degen >=100 tok) {st.median(nd) if nd else float('nan'):.3f}  "
                  f"mean len {st.fmean(x['ntok'] for x in v):7.0f}  finish {dict(collections.Counter(x['finish'] for x in v).most_common(4))}")
        both = sum(A[k]["degen"] and B[k]["degen"] for k in K); oa = sum(A[k]["degen"] and not B[k]["degen"] for k in K)
        ob = sum(B[k]["degen"] and not A[k]["degen"] for k in K)
        print(f"  paired: both degenerate {both}, only {na} {oa}, only {nb} {ob}")
        ctx = collections.defaultdict(list)
        for k in K: ctx[k[0]].append(k)
        cs = list(ctx); rng = random.Random(0); diffs = []
        for _ in range(2000):
            s = [k for c in (rng.choice(cs) for _ in cs) for k in ctx[c]]
            diffs.append(sum(B[k]["degen"] for k in s) / len(s) - sum(A[k]["degen"] for k in s) / len(s))
        diffs.sort()
        print(f"  rate({nb}) - rate({na}) = {sum(B[k]['degen'] for k in K)/len(K) - sum(A[k]['degen'] for k in K)/len(K):+.2%}  "
              f"95% CI [{diffs[50]:+.2%}, {diffs[1949]:+.2%}] (bootstrap over contexts)")

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--a", required=True); p.add_argument("--b", required=True)
    main(p.parse_args())
