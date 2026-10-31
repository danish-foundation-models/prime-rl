"""Per swesmith train rollout: is problem_statement empty, does the first user msg end with the capability notice."""
import collections, glob, json, subprocess, sys
T = sys.argv[1]
def lines():
    for f in sorted(glob.glob(f"{T}/stream/*.jsonl*")):
        if f.endswith(".zst"):
            yield from subprocess.Popen(["zstd", "-dc", f], stdout=subprocess.PIPE).stdout
        else: yield from open(f, "rb")
c = collections.Counter(); names = collections.defaultdict(set); keys = None
for line in lines():
    try: d = json.loads(line)
    except Exception: continue
    if d["run"]["work"]["type"] != "train" or not d["env"]["name"].startswith("swesmith-env"): continue
    row = d["task"]["data"].get("row") or {}
    if keys is None: keys = sorted(d["task"]["data"]); print("task.data keys:", keys)
    t = d["traces"][0]
    u = next((x["message"] for x in t["nodes"] if x["message"].get("role") == "user"), {})
    cont = u.get("content") or ""
    txt = cont if isinstance(cont, str) else " ".join(p.get("text", "") for p in cont)
    empty = not (row.get("problem_statement") or "").strip()
    lang = d["task"]["data"]["name"].split(":")[0]
    c[(lang, empty, txt.rstrip().endswith("provider-side operation."), txt.startswith("Network protocol blocked"))] += 1
    if empty: names[lang].add(d["task"]["data"]["name"])
print("(lang, empty_ps, notice_at_end, notice_only) -> n")
for k in sorted(c): print(k, c[k])
print({l: len(v) for l, v in names.items()}, "distinct empty tasks")
json.dump({l: sorted(v) for l, v in names.items()}, open(sys.argv[2], "w"), indent=1)
