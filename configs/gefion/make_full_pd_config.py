"""Derive the full PD run from gefion_full_r1.toml: 8 trainer nodes + 2 prefill + 6 decode nodes, one
TP8+EP8 engine each, instead of 8 colocated DP8+EP8 inference nodes. Same PD settings as the
shakedowns that ran clean (make_pd_config.make_pd, make_pdtp8_config.make_tp8).

TP8 over DP8: same decode speed (job 533520: 22.3 ms steps, ~45 tok/s per sequence, 6.2k generated
tok/s at 140 running; DP8 job 533326: 41-51 tok/s), but TP8's 1040-token prefix-cache blocks (DP8:
4160) cut P's work to 1.53 computed prompt tokens per generated token (DP8: 2.85).
2P:6D: 6 D nodes at 6.2-7.6k generated tok/s x 1.53 over 2 P nodes at ~58k prompt tok/s each puts P
at ~50-60%. See infer-opt/notes/parallelism-and-pd-ratio.md section 5.

Usage: make_full_pd_config.py gefion_full_r1.toml gefion_full_pd_r1.toml [n_prefill n_decode [dp8]]
(the live run is 4 4 since step 10, 5 5 with 6 trainer nodes since step 35)
"""

import subprocess
import sys
import tomllib

import tomli_w

from make_pd_config import make_pd
from make_pdtp8_config import make_tp8

# Nodes seen with iommu=pt (GPUDirect RDMA at full speed). order_nodes_pd.sh fails the job if any of
# its 16 nodes lacks it, so exclude every other defq node and let Slurm only hand out these.
IOMMU_PT = ["dgx045", "dgx046", "dgx047", "dgx048", "dgx049", "dgx050", "dgx051", "dgx052", "dgx053",
            "dgx054", "dgx055", "dgx056", "dgx059", "dgx062", "dgx063", "dgx152", "dgx153", "dgx154",
            "dgx156",
            # iommu=pt by probe: dgx041 and dgx159 (iommu-probe/, 2026-10-30), dgx037 (order_nodes_pd, 2026-11-04).
            "dgx037", "dgx041", "dgx159"]

src, dst = sys.argv[1], sys.argv[2]
n_prefill, n_decode = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (2, 6)
tp8 = not (len(sys.argv) > 5 and sys.argv[5] == "dp8")
with open(src, "rb") as f:
    cfg = tomllib.load(f)

assert cfg["deployment"]["num_train_nodes"] + n_prefill + n_decode == 16
cfg["run"]["name"] = f"gefion-super-r6s2638-8t{n_prefill}p{n_decode}d{'-tp8' if tp8 else ''}-r1"
# r3: restart from step 0 after r2 regressed on SWE/terminal (r2: 534618/534678, stopped at step 20).
if (n_prefill, n_decode) == (4, 4) and tp8:
    cfg["run"]["name"] = "gefion-super-r6s2638-8t4p4d-tp8-r5"
# r5 continues from step 35 as 6T:5P:5D in the same run dir, so it keeps its 8t4p4d name.
if (n_prefill, n_decode) == (5, 5) and tp8:
    cfg["run"]["name"] = "gefion-super-r6s2638-8t4p4d-tp8-r5"
make_pd(cfg, n_prefill, n_decode)
if tp8:
    make_tp8(cfg)
nodes = subprocess.run(["sinfo", "-p", cfg["slurm"]["partition"], "-h", "-N", "-o", "%N"],
                       capture_output=True, text=True, check=True).stdout.split()
# One off-pool node for the trainer (order_nodes_pd.sh MAX_OFFPOOL_TRAIN_NODES): dgx037 ran another
# user's job cleanly on 2026-11-04 while other users held dgx059 and dgx156.
OFFPOOL_TRAIN = ["dgx037"]
cfg["slurm"]["exclude"] = ",".join(sorted(set(nodes) - set(IOMMU_PT) - set(OFFPOOL_TRAIN)))

with open(dst, "wb") as f:
    tomli_w.dump(cfg, f)
print(f"wrote {dst}")
