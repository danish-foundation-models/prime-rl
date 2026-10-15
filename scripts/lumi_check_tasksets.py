"""Load every train/eval taskset of a resolved orchestrator config, as Orchestrator setup does.

Catches missing runtime packages before a multi-node RL job spends its allocation on them:

    ./scripts/lumi_run_in_container.sh env HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
      python /workdir/scripts/lumi_check_tasksets.py <run>/configs/<attempt>/resolved/orchestrator.json

Export PRIME_RL_ENVIRONMENT_SITE_PACKAGES first when the config uses that layer.
Judge-backed tasksets need their API keys in the environment.
"""
import sys, time, traceback
from itertools import islice
sys.argv = ["orchestrator", "@", sys.argv[1]]
from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.utils.config import cli
import verifiers.v1 as vf
cfg = cli(OrchestratorConfig)
sources = [("train", s) for s in cfg.train.source] + [("eval", s) for s in (cfg.eval.source if cfg.eval else [])]
seen, failed = set(), 0
for kind, s in sources:
    key = (s.env.taskset.model_dump_json() if hasattr(s.env.taskset, "model_dump_json") else repr(s.env.taskset))
    if key in seen: continue
    seen.add(key)
    t0 = time.time()
    try:
        ts = vf.load_taskset(s.env.taskset)
        n = len(tuple(islice(ts, 2)))
        print(f"OK   {kind:5s} {s.name:40s} {n} rows in {time.time()-t0:.1f}s", flush=True)
    except Exception as e:
        failed += 1
        print(f"FAIL {kind:5s} {s.name:40s} {type(e).__name__}: {str(e)[:200]}", flush=True)
        traceback.print_exc(limit=3)
print(f"{len(seen)} tasksets, {failed} failed")
sys.exit(1 if failed else 0)
