"""Find contiguous runs of tokens whose teacher-forced logprob is far below the sampler's.

A run means the sampler decoded from a corrupted state (seen with MTP in mamba "align" mode at
absolute positions 4097-4176, the tail of the first 4176-token cache block).
Usage: mismatch_bursts.py <gen.json> <tf.json> [--thresh 5]
"""
import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument("gen")
ap.add_argument("tf")
ap.add_argument("--thresh", type=float, default=5.0)
a = ap.parse_args()
S, T = json.load(open(a.gen)), json.load(open(a.tf))
n_seq, total = 0, 0
for i, (s, t) in enumerate(zip(S, T)):
    bad = [p for p in range(len(t)) if t[p] is not None and t[p] - s["lp"][p] < -a.thresh]
    if not bad:
        continue
    bursts, st, pr = [], bad[0], bad[0]
    for p in bad[1:]:
        if p - pr > 20:
            bursts.append((st, pr))
            st = p
        pr = p
    bursts.append((st, pr))
    P = len(s["prompt"])
    big = [(P + x, P + y, y - x + 1) for x, y in bursts if y - x >= 10]
    if big:
        n_seq += 1
        total += sum(b[2] for b in big)
        print(f"seq {i}: bursts (abs start, abs end, len) {big}")
print(f"BURSTS sequences_with_bursts={n_seq}/{len(S)} burst_tokens={total}")
