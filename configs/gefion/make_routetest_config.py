"""Derive the prefill-routing test from gefion_shakedown_r1.toml: 4 trainer + 2 prefill + 2 decode
TP8+EP8 nodes, with the full run's per-prefill-node session load (~350 sessions per P, as 1,100 over
3 P). Checks the sticky_load prefill policy (render_pd.sh) and the prefill-thrash concurrency cut on
real engines: P balance, prefix-cache hit rates, sessions moved, and the router's per-worker load
against the engines' own running + waiting (the v0.2.0 load counter leaked on cancelled requests).

Usage: make_routetest_config.py gefion_shakedown_r1.toml gefion_routetest_r1.toml
"""

import ast
import subprocess
import sys
import tomllib

import tomli_w

from make_pd_config import make_pd
from make_pdtp8_config import make_tp8

# make_full_pd_config.py runs on import; read its iommu=pt node list instead.
with open("make_full_pd_config.py") as f:
    IOMMU_PT = next(ast.literal_eval(node.value) for node in ast.parse(f.read()).body
                    if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "IOMMU_PT")

src, dst = sys.argv[1], sys.argv[2]
with open(src, "rb") as f:
    cfg = tomllib.load(f)

assert cfg["deployment"]["num_train_nodes"] == 4
make_pd(cfg, 2, 2)
make_tp8(cfg)
cfg["run"]["name"] = "gefion-super-routetest-4t2p2d-r1"
cfg["max_steps"] = 3
cfg["orchestrator"]["concurrency"].update(initial_inflight=700, min_inflight=350, max_inflight=750)
# Short enough to backfill ahead of the full run's reservation.
cfg["slurm"].update(job_name="gefion-super-routetest", time="01:45:00")
nodes = subprocess.run(["sinfo", "-p", cfg["slurm"]["partition"], "-h", "-N", "-o", "%N"],
                       capture_output=True, text=True, check=True).stdout.split()
cfg["slurm"]["exclude"] = ",".join(sorted(set(nodes) - set(IOMMU_PT)))

with open(dst, "wb") as f:
    tomli_w.dump(cfg, f)
print(f"wrote {dst}")
