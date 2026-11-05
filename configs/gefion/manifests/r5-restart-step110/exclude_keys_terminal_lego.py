"""terminal-lego: exclude by task name (= task id), since cached rows and a fresh git load differ in paths."""
import json, sys
sys.argv = ["x"]
exec(open("exclude_keys.py").read().split("old, new = ")[0])
old = tasksets(f"{run}/configs/attempt_12/rl.toml")
keep = set(json.load(open(f"{out}/terminal-lego.medium-hard.task-ids.json")))
tasks = load(old["terminal-lego-bash"])
assert len({json.dumps(old[n], sort_keys=True) for n in old if n.startswith("terminal-lego-")}) == 1
names = [t.data.name for t in tasks]
assert keep <= set(names) and len(set(names)) == len(names)
excluded = sorted(t.key for t in tasks if t.data.name not in keep)
assert len(tasks) - len(excluded) == len(keep)
for n in ("terminal-lego-opencode", "terminal-lego-pi", "terminal-lego-bash"):
    hist = set(ck[n]["sampler"]["task_history"])
    print(f"  {n}: history {len(hist)}, in old keys {len(hist & {t.key for t in tasks})}, excluded with history {len(hist & set(excluded))}")
json.dump(excluded, open(f"{out}/terminal-lego.exclude-keys.json", "w"), indent=0)
print(f"terminal-lego: old {len(tasks)} keep {len(keep)} excluded {len(excluded)}")
