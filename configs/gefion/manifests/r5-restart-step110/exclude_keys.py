"""Keys of tasks in the step-110 task lists that the restart lists drop, so the sampler can exclude them
while every other task keeps the key its step-110 history is stored under."""
import hashlib, json, sys, time, tomllib, typing
import torch
from verifiers.v1.utils.loaders import taskset_class

run = "/dcai/projects/iu_0145/rasmus/runs/gefion-super-r6s2638-8t4p4d-tp8-r5"
out = "/dcai/projects/iu_0145/rasmus/manifests/r5-restart-step110"
ck = torch.load(f"{run}/checkpoints/step_110/orchestrator/progress.pt", weights_only=False)["train_source"]["envs"]

def tasksets(path):
    cfg = tomllib.load(open(path, "rb"))
    return {s["name"]: s["env"]["taskset"] for s in cfg["orchestrator"]["train"]["source"]}

def load(ts):
    cls = taskset_class(ts["id"])
    cfgcls = None
    for base in getattr(cls, "__orig_bases__", ()):
        for a in typing.get_args(base):
            if isinstance(a, type) and hasattr(a, "model_validate") and "Config" in a.__name__:
                cfgcls = a
    return list(cls(cfgcls.model_validate(ts)))

def content(task):
    return hashlib.sha256(json.dumps(task.data.model_dump(mode="json", exclude_none=True, exclude={"idx"}),
                                     sort_keys=True).encode()).hexdigest()

old, new = tasksets(f"{run}/configs/attempt_12/rl.toml"), tasksets(f"{run}/configs/attempt_16/rl.toml")
for dataset in sys.argv[1:]:
    names = [n for n in old if n.startswith(dataset + "-")]
    olds = {json.dumps(old[n], sort_keys=True) for n in names}
    assert len(olds) == 1, f"{dataset}: harnesses differ in old taskset config"
    t0 = time.time()
    old_tasks = load(old[names[0]])
    nts = dict(new[names[0]]); nts.pop("task_data_file", None)
    new_tasks = load(nts)
    keep = {content(t) for t in new_tasks}
    excluded = sorted(t.key for t in old_tasks if content(t) not in keep)
    assert len(old_tasks) - len(excluded) == len(new_tasks), (len(old_tasks), len(excluded), len(new_tasks))
    old_keys = {t.key for t in old_tasks}
    for n in names:
        hist = set(ck[n]["sampler"]["task_history"])
        print(f"  {n}: history {len(hist)}, in old keys {len(hist & old_keys)}, excluded with history {len(hist & set(excluded))}")
    path = f"{out}/{dataset}.exclude-keys.json"
    json.dump(excluded, open(path, "w"), indent=0)
    print(f"{dataset}: old {len(old_tasks)} new {len(new_tasks)} excluded {len(excluded)} -> {path} ({time.time()-t0:.0f}s)", flush=True)
