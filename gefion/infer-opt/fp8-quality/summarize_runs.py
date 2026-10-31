"""Cross-run comparison of degenerate-turn rate and NLL from extract_turns.py tables."""
import collections, re, glob, json, statistics as st, sys
AG = {"swesmith-env", "scaleswe", "openswe", "terminal-lego", "tmax"}
maxstep = int(sys.argv[1]) if len(sys.argv) > 1 else 99
def norm(r):
    m = re.match(r"(.*)-(bash|opencode|pi)$", r["env"])
    if m: r["env"], r["harness"] = m.group(1), m.group(2)
    return r
def load(f):
    return [norm(json.loads(l)) for l in open(f)]
print(f"steps <= {maxstep}; agentic envs only; rollout counted degenerate if any sampled turn has >=100 tok and mean NLL>3")
print(f"{'run':44s} {'roll':>5s} {'degR':>6s} {'turns':>6s} {'degT':>6s} {'medNLL':>6s} {'p90':>5s} {'f10':>7s} {'1stTok':>6s} temp maxtok det")
for f in sorted(glob.glob("data/*.turns.jsonl")):
    rows = [r for r in load(f) if r["env"] in AG and (r["step"] or 0) <= maxstep and r.get("nll") is not None]
    if not rows: continue
    roll = collections.defaultdict(bool)
    for r in rows: roll[r["rid"]] |= r["degen"]
    big = [r for r in rows if r["ntok"] >= 100]
    nl = sorted(r["nll"] for r in big if not r["degen"])
    first = [r["b"][0] for r in rows]
    f10 = st.mean(r["f10"] for r in big)
    name = f.split("/")[-1].replace(".turns.jsonl", "")
    print(f"{name:44s} {len(roll):5d} {sum(roll.values())/len(roll):6.1%} {len(big):6d} {sum(r['degen'] for r in big)/len(big):6.2%} {st.median(nl):6.3f} {nl[int(.9*len(nl))]:5.2f} {f10:7.4f} {st.mean(first):6.3f} {rows[0]['temp']} {rows[0]['max_tokens']} {rows[0]['detector']}")
