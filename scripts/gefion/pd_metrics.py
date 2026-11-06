"""Per-node inference load of a PD run, from vLLM's periodic per-engine stats lines.

For each inference node (node_<n>.log + node_<n>_rank<k>.log), over a time window: prompt and
generated tok/s summed over its 8 engines, running/waiting requests, KV usage, and per-sequence decode
tok/s (generated tok/s / running). Also the NIXL "KV Transfer metrics" lines (decode side): number
of transfers, mean of the per-interval avg/p90 transfer times, MB per transfer, and failures.

Each stats line averages over the interval since that engine's previous line, so tokens are
integrated as rate x interval; running/waiting/KV are time-weighted means.

Usage: pd_metrics.py <logs/inference dir> [--start HH:MM:SS] [--end HH:MM:SS] [--last SECONDS]
"""
import argparse
import glob
import os
import re
from collections import defaultdict
from datetime import datetime, timedelta

RX = re.compile(r"(\d\d-\d\d \d\d:\d\d:\d\d) \[loggers\.py:\d+\] Engine (\d+): Avg prompt throughput: ([\d.]+) "
                r"tokens/s, Avg generation throughput: ([\d.]+) tokens/s, Running: (\d+) reqs, Waiting: (\d+) reqs, "
                r"GPU KV cache usage: ([\d.]+)%(?:, Prefix cache hit rate: ([\d.]+)%)?")
KX = re.compile(r"(\d\d-\d\d \d\d:\d\d:\d\d) \[metrics\.py:\d+\] KV Transfer metrics: (.*)")
FAIL = re.compile(r"(\d\d-\d\d \d\d:\d\d:\d\d).*(?:failed to transfer|transfer failed|Failed to|NIXL_ERR|"
                  r"kv load fail|KV load fail|expired)", re.I)


def ts(s):
    return datetime.strptime("2026-" + s, "%Y-%m-%d %H:%M:%S")


ap = argparse.ArgumentParser()
ap.add_argument("dir")
ap.add_argument("--start")
ap.add_argument("--end")
ap.add_argument("--last", type=float, help="window = the last N seconds of logged stats")
a = ap.parse_args()

nodes = defaultdict(list)
for f in sorted(glob.glob(os.path.join(a.dir, "node_*.log"))):
    m = re.match(r"node_(\d+)(?:_rank(\d+))?\.log$", os.path.basename(f))
    if m:
        nodes[int(m.group(1))].append(f)

stats, kv, fails = {}, defaultdict(list), defaultdict(int)
tmax = None
for n, files in nodes.items():
    per = defaultdict(list)
    for f in files:
        for line in open(f, errors="replace"):
            if m := RX.search(line):
                t = ts(m.group(1))
                per[(f, m.group(2))].append((t, float(m.group(3)), float(m.group(4)), int(m.group(5)),
                                             int(m.group(6)), float(m.group(7))))
                tmax = t if tmax is None or t > tmax else tmax
            elif m := KX.search(line):
                kv[n].append((ts(m.group(1)), dict(x.split("=", 1) for x in m.group(2).split(", ") if "=" in x)))
            elif "KV Transfer" not in line and FAIL.search(line) and "nixl" in line.lower():
                fails[n] += 1
    stats[n] = per

if tmax is None:
    raise SystemExit("no stats lines")
day = tmax.strftime("%Y-%m-%d ")
end = datetime.strptime(day + a.end, "%Y-%m-%d %H:%M:%S") if a.end else tmax
start = (datetime.strptime(day + a.start, "%Y-%m-%d %H:%M:%S") if a.start
         else end - timedelta(seconds=a.last or 600))
print(f"window {start:%H:%M:%S} - {end:%H:%M:%S} ({(end - start).total_seconds():.0f} s)")
for n in sorted(stats):
    gen = prm = dur_tot = 0.0
    run = wait = kvu = 0.0
    engines = 0
    for rows in stats[n].values():
        prev = None
        used = False
        for r in rows:
            if prev is not None and start < r[0] <= end:
                dt = (r[0] - prev).total_seconds()
                prm += r[1] * dt
                gen += r[2] * dt
                run += r[3] * dt
                wait += r[4] * dt
                kvu += r[5] * dt
                dur_tot += dt
                used = True
            prev = r[0]
        engines += used
    if not dur_tot:
        print(f"node {n}: no stats in window")
        continue
    span = dur_tot / max(engines, 1)  # mean covered seconds per engine
    R = run / span
    print(f"node {n}: engines={engines} prompt={prm / span:8.0f} tok/s  gen={gen / span:7.0f} tok/s  "
          f"running={R:6.1f}  waiting={wait / span:6.1f}  kv={kvu / dur_tot:5.1f}%  "
          f"per-seq decode={gen / run if run else 0:5.1f} tok/s")
for n in sorted(kv):
    # NIXL lines only (the native offload connector logs its own keys under the same prefix).
    rows = [d for t, d in kv[n] if start < t <= end and "Num successful transfers" in d]
    if not rows:
        continue
    num = sum(int(d["Num successful transfers"]) for d in rows)
    w = lambda k: sum(float(d[k]) * int(d["Num successful transfers"]) for d in rows) / max(num, 1)
    print(f"node {n} KV transfers: {num} in window, avg xfer {w('Avg xfer time (ms)'):.1f} ms, "
          f"mean p90 {w('P90 xfer time (ms)'):.1f} ms, {w('Avg MB per transfer'):.1f} MB each, "
          f"{w('Throughput (MB/s)'):.0f} MB/s per transfer")
for n, c in sorted(fails.items()):
    print(f"node {n}: {c} NIXL failure/expiry log lines (whole log)")
