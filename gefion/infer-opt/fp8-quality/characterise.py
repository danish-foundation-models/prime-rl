"""Where degenerate turns happen: by harness, env, turn index, context length, preceding tool output,
turn length/onset, finish reason, and within-turn NLL by position (non-degenerate turns)."""
import collections, json, re, statistics as st, sys
f = sys.argv[1]; maxstep = int(sys.argv[2]) if len(sys.argv) > 2 else 99
AG = {"swesmith-env", "scaleswe", "openswe", "terminal-lego", "tmax"}
rows = []
for l in open(f):
    r = json.loads(l)
    m = re.match(r"(.*)-(bash|opencode|pi)$", r["env"])
    if m: r["env"], r["harness"] = m.group(1), m.group(2)
    if r["env"] in AG and (r["step"] or 0) <= maxstep and r.get("nll") is not None and r["ntok"] >= 100:
        rows.append(r)
def tab(title, key, bins=None):
    g = collections.defaultdict(list)
    for r in rows:
        k = key(r)
        if bins: k = next((f"<{b}" for b in bins if k < b), f">={bins[-1]}")
        g[k].append(r)
    print(f"\n{title}:  key  turns  degenerate%  median-NLL(non-degen)  frac>1.5")
    order = sorted(g) if not bins else [f"<{b}" for b in bins] + [f">={bins[-1]}"]
    for k in order:
        v = g.get(k)
        if not v: continue
        nd = [r["nll"] for r in v if not r["degen"]]
        print(f"  {str(k):>10s} {len(v):6d} {sum(r['degen'] for r in v)/len(v):7.2%} {st.median(nd) if nd else float('nan'):7.3f} {sum(x>1.5 for x in nd)/max(1,len(nd)):6.1%}")
print(f, "steps<=", maxstep, "agentic turns >=100 tokens:", len(rows))
tab("harness", lambda r: r["harness"])
tab("env", lambda r: r["env"])
tab("step", lambda r: r["step"])
tab("turn index", lambda r: r["turn"], [1, 2, 4, 8, 16, 32, 64])
tab("prompt tokens (context)", lambda r: r["prompt"] or 0, [4000, 8000, 16000, 32000, 64000, 96000])
tab("preceding tool-output tokens", lambda r: r["prev_tool"], [50, 200, 1000, 4000, 16000])
tab("turn length (tokens)", lambda r: r["ntok"], [256, 1024, 4096, 8192, 16384])
tab("finish reason", lambda r: str(r["finish"]))
deg = [r for r in rows if r["degen"]]
on = [r["onset"] for r in deg if r.get("onset") is not None]
print(f"\ndegenerate turns: {len(deg)}; onset (first 64-tok window >3 nats) found in {len(on)}")
for q in (0.1, 0.25, 0.5, 0.75, 0.9):
    print(f"  onset q{q}: {sorted(on)[int(q*len(on))]}   ntok q{q}: {sorted(r['ntok'] for r in deg)[int(q*len(deg))]}")
print(f"  onset==0 (garbage from the first tokens): {sum(o==0 for o in on)/len(on):.1%};  onset<64: {sum(o<64 for o in on)/len(on):.1%}; onset>=8192: {sum(o>=8192 for o in on)/len(on):.1%}")
print(f"  degenerate turns with ntok>8192 (would be cut by LUMI's 8192 cap): {sum(r['ntok']>8192 for r in deg)/len(deg):.1%}; onset beyond 8192: {sum(o>8192 for o in on)/len(deg):.1%}")
print(f"  finish reasons of degenerate turns: {collections.Counter(r['finish'] for r in deg).most_common()}")
B = ["0", "1-3", "4-15", "16-63", "64-255", "256-1023", "1024-4095", "4096+"]
for lo, hi in ((100, 1024), (1024, 4096), (4096, 1 << 30)):
    sel = [r for r in rows if not r["degen"] and lo <= r["ntok"] < hi]
    print(f"\nnon-degenerate turns {lo}-{hi} tokens (n={len(sel)}): mean NLL by position in turn")
    print("  " + "  ".join(f"{b}:{st.mean([r['b'][i] for r in sel if r['b'][i] is not None]):.3f}" for i, b in enumerate(B) if any(r['b'][i] is not None for r in sel)))
    sel = [r for r in rows if r["degen"] and lo <= r["ntok"] < hi]
    if sel:
        print(f"  degenerate (n={len(sel)}): " + "  ".join(f"{b}:{st.mean([r['b'][i] for r in sel if r['b'][i] is not None]):.3f}" for i, b in enumerate(B) if any(r['b'][i] is not None for r in sel)))
