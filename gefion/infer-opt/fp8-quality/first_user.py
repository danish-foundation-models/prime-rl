"""First user message of each agentic train rollout: how many have no real task (empty problem statement)."""
import collections, json, re, subprocess, sys
T = sys.argv[1]; MAX = int(sys.argv[2]) if len(sys.argv) > 2 else 10**9
def lines():
    import glob
    for f in sorted(glob.glob(f"{T}/stream/*.jsonl*")):
        if f.endswith(".zst"):
            p = subprocess.Popen(["zstd", "-dc", f], stdout=subprocess.PIPE); yield from p.stdout
        else: yield from open(f, "rb")
c = collections.Counter(); deg = collections.Counter(); n = 0; ex = {}
for line in lines():
    try: d = json.loads(line)
    except Exception: continue
    if d["run"]["work"]["type"] != "train": continue
    env = re.sub(r"-(bash|opencode|pi)$", "", d["env"]["name"])
    if env not in ("swesmith-env", "scaleswe", "openswe", "terminal-lego", "tmax"): continue
    t = d["traces"][0]
    u = next((x["message"] for x in t["nodes"] if x["message"].get("role") == "user"), None)
    cont = (u or {}).get("content") or ""
    txt = cont if isinstance(cont, str) else " ".join(p.get("text", "") for p in cont)
    kind = "network-blocked msg" if txt.startswith("Network protocol blocked") else ("short<200ch" if len(txt) < 200 else "task")
    key = (env, kind); c[key] += 1; deg[key] += t.get("stop_condition") == "degenerate_turn"
    ex.setdefault(kind, txt[:300])
    n += 1
    if n >= MAX: break
for k in sorted(c): print(f"{k[0]:14s} {k[1]:20s} {c[k]:5d} degenerate-stop {deg[k]/c[k]:.1%}")
for k, v in ex.items(): print(k, "::", repr(v))
