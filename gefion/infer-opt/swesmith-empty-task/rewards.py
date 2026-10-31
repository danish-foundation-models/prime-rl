"""Reward / stop stats for swesmith train rollouts, empty vs non-empty problem statement."""
import collections, glob, json, subprocess, sys
T = sys.argv[1]
def lines():
    for f in sorted(glob.glob(f"{T}/stream/*.jsonl*")):
        if f.endswith(".zst"): yield from subprocess.Popen(["zstd", "-dc", f], stdout=subprocess.PIPE).stdout
        else: yield from open(f, "rb")
s = collections.defaultdict(lambda: collections.Counter()); tasks = collections.defaultdict(set); gk = None
for line in lines():
    try: d = json.loads(line)
    except Exception: continue
    if d["run"]["work"]["type"] != "train" or not d["env"]["name"].startswith("swesmith-env"): continue
    if gk is None: gk = d["group"]; print("group sample:", json.dumps(gk)[:600])
    empty = not (d["task"]["data"]["row"].get("problem_statement") or "").strip()
    t = d["traces"][0]; r = t["rewards"]
    k = "empty" if empty else "has_task"
    s[k]["n"] += 1; s[k]["solved"] += ((r.get("solved") or {}).get("score") or 0) == 1; s[k]["scored"] += "solved" in r; s[k]["no_reward"] += not r
    s[k]["stop:" + str(t.get("stop_condition"))] += 1; s[k]["ok"] += bool(d.get("ok"))
    tasks[k].add(d["task"]["data"]["name"])
for k, c in s.items(): print(k, "distinct tasks", len(tasks[k]), dict(c))
