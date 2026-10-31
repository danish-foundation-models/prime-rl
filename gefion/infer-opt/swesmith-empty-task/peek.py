import collections, glob, json, subprocess, sys
T = sys.argv[1]
def lines():
    for f in sorted(glob.glob(f"{T}/stream/*.jsonl*")):
        if f.endswith(".zst"): yield from subprocess.Popen(["zstd", "-dc", f], stdout=subprocess.PIPE).stdout
        else: yield from open(f, "rb")
c = collections.Counter(); shown=set(); rew=collections.defaultdict(list)
for line in lines():
    try: d = json.loads(line)
    except Exception: continue
    if d["run"]["work"]["type"] != "train" or not d["env"]["name"].startswith("swesmith-env"): continue
    row = d["task"]["data"]["row"]; empty = not (row.get("problem_statement") or "").strip()
    if not empty: continue
    t = d["traces"][0]; env = d["env"]["name"]
    users = [x["message"] for x in t["nodes"] if x["message"].get("role") == "user"]
    u = users[0] if users else {}
    cont = u.get("content") or ""
    txt = cont if isinstance(cont, str) else " | ".join(p.get("text", "") for p in cont)
    c[(env, txt.startswith("Network"))] += 1
    rew[d["task"]["data"]["name"]].append(t.get("reward"))
    if env not in shown: shown.add(env); print(env, "prompt=", repr(d["task"]["data"]["prompt"]), "\nFIRST USER:", repr(txt[:600]), "\n")
print(c)
for k,v in rew.items(): print(k, len(v), "mean reward", sum(x or 0 for x in v)/len(v))
