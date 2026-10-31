"""Summarise a py-spy --format raw (collapsed stacks) file: busy fraction, top self and inclusive frames.

Usage: pyspy_summary.py pyspy.txt [--top 25]
"""
import argparse
import re
from collections import Counter

ap = argparse.ArgumentParser()
ap.add_argument("path")
ap.add_argument("--top", type=int, default=25)
a = ap.parse_args()

self_c, incl_c, total, idle = Counter(), Counter(), 0, 0
IDLE = re.compile(r"select|epoll|poll \(|_run_once \(asyncio|wait \(threading|sleep|dequeue|recv|zmq", re.I)
for line in open(a.path):
    line = line.rstrip()
    if not line:
        continue
    stack, _, n = line.rpartition(" ")
    n = int(n)
    frames = stack.split(";")
    total += n
    leaf = frames[-1]
    if IDLE.search(leaf):
        idle += n
    self_c[re.sub(r":\d+\)", ")", leaf)] += n
    for f in {re.sub(r":\d+\)", ")", f) for f in frames}:
        incl_c[f] += n
print(f"samples={total} idle-ish leaf={idle / max(1, total):.1%}")
print("--- self")
for f, n in self_c.most_common(a.top):
    print(f"{n / total:6.1%}  {f[:150]}")
print("--- inclusive")
for f, n in incl_c.most_common(a.top + 15):
    print(f"{n / total:6.1%}  {f[:150]}")
