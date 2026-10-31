"""Node generation throughput over a time window, from vLLM's periodic per-engine stats lines.

Each "Engine 00k: ... Avg generation throughput: X tokens/s" line averages over the interval since
that engine's previous line, so tokens = X * interval even when a blocked API server stretches
the 10 s interval to a minute. Summed over engines, this gives node throughput without relying
on the (possibly unresponsive) /metrics endpoint.

Usage: engine_log_tput.py server.log [--start HH:MM:SS] [--end HH:MM:SS] [--skip-first S]
Without --start/--end it uses the longest run of lines with Running > 0 and no prompt-only burst.
"""
import argparse
import re
from collections import defaultdict
from datetime import datetime

RX = re.compile(r"(\d\d-\d\d \d\d:\d\d:\d\d) \[loggers\.py:\d+\] Engine (\d+): Avg prompt throughput: ([\d.]+) "
                r"tokens/s, Avg generation throughput: ([\d.]+) tokens/s, Running: (\d+) reqs, Waiting: (\d+) reqs, "
                r"GPU KV cache usage: ([\d.]+)%")

ap = argparse.ArgumentParser()
ap.add_argument("log")
ap.add_argument("--start")
ap.add_argument("--end")
ap.add_argument("--skip-first", type=float, default=120.0, help="seconds of a load segment to skip (prefill burst)")
a = ap.parse_args()

rows = defaultdict(list)
for line in open(a.log, errors="replace"):
    m = RX.search(line)
    if m:
        t = datetime.strptime("2026-" + m.group(1), "%Y-%m-%d %H:%M:%S")
        rows[int(m.group(2))].append((t, float(m.group(3)), float(m.group(4)), int(m.group(5)), int(m.group(6)),
                                      float(m.group(7))))


def segments(r):
    """Contiguous stretches with requests running (one per load phase)."""
    seg, out = [], []
    for x in r:
        if x[3] > 0:
            seg.append(x)
        elif seg:
            out.append(seg)
            seg = []
    if seg:
        out.append(seg)
    return out


def hms(s):
    d = rows[0][0][0]
    return datetime.strptime(d.strftime("%Y-%m-%d ") + s, "%Y-%m-%d %H:%M:%S")


if a.start:
    windows = [(hms(a.start), hms(a.end))]
else:
    windows = [(s[0][0], s[-1][0]) for s in segments(rows[0]) if (s[-1][0] - s[0][0]).total_seconds() > 60]

for lo, hi in windows:
    lo2 = lo if a.start else lo.__class__.fromtimestamp(lo.timestamp() + a.skip_first)
    gen = prm = 0.0
    run, wait, kv, n, gaps = [], [], [], 0, []
    for e, r in rows.items():
        for (t0, *_), (t1, p, g, ru, wa, k) in zip(r, r[1:]):
            if t1 <= lo2 or t0 >= hi:
                continue
            dt = (t1 - t0).total_seconds()
            gen += g * dt
            prm += p * dt
            run.append(ru)
            wait.append(wa)
            kv.append(k)
            if e == 0:
                gaps.append(dt)
    dur = (hi - lo2).total_seconds()
    if dur <= 0 or not run:
        continue
    print(f"window {lo2:%H:%M:%S}-{hi:%H:%M:%S} ({dur:.0f}s): gen {gen / dur:,.0f} tok/s, prompt {prm / dur:,.0f} tok/s, "
          f"running/engine {sum(run) / len(run):.1f}, waiting/engine {sum(wait) / len(wait):.1f}, "
          f"kv {sum(kv) / len(kv):.1f}%, stats interval mean {sum(gaps) / max(1, len(gaps)):.1f}s max {max(gaps, default=0):.0f}s")
