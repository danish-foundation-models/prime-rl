"""Per-sampled-turn table from a PRIME-RL trace stream (any run).

  python3 extract_turns.py <traces_dir> <out.jsonl> [--max N] [--keep-lp]

One row per sampled assistant turn of a train rollout: env, harness, step, turn index, prompt length,
completion length, finish reason, mean NLL (inference logprobs), NLL by position bucket, the
max 64-token-window NLL, the degenerate criterion (>=100 tokens, mean NLL > 3), the length of
the tool output right before the turn, and the per-config sampling/detector settings.
--keep-lp also stores the raw per-token logprobs (large).
"""
import glob, json, os, subprocess, sys

BUCKETS = [(0, 1), (1, 4), (4, 16), (16, 64), (64, 256), (256, 1024), (1024, 4096), (4096, 1 << 30)]


def lines(tdir):
    for f in sorted(glob.glob(f"{tdir}/stream/*.jsonl*")):
        if f.endswith(".zst"):
            p = subprocess.Popen(["zstd", "-dc", f], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            yield from p.stdout
            p.wait()
        else:
            yield from open(f, "rb")


def main():
    tdir, out = sys.argv[1], sys.argv[2]
    mx = int(sys.argv[sys.argv.index("--max") + 1]) if "--max" in sys.argv else 1 << 60
    keep = "--keep-lp" in sys.argv
    fo = open(out, "w")
    nrec = bad = 0
    for line in lines(tdir):
        if b'"type":"train"' not in line[:3000] and b'"type": "train"' not in line[:3000]:
            # cheap prefilter: run.work.type sits early in the record
            if b'"train"' not in line[:200000]:
                continue
        try:
            d = json.loads(line)
        except Exception:
            bad += 1
            continue
        if d.get("run", {}).get("work", {}).get("type") != "train":
            continue
        nrec += 1
        if nrec > mx:
            break
        t = d["traces"][0]
        cfg = t["agent"]["config"]
        nodes, calls = t["nodes"], t["calls"]
        callby = {c["node"]: c for c in calls if "node" in c}
        err = t["errors"][0]["message"] if t.get("errors") else ""
        base = dict(
            rid=d["id"], env=d["env"]["name"], harness=cfg.get("harness", {}).get("id"),
            step=d["run"]["work"].get("step"), pol=d["run"]["work"].get("policy"),
            stop=t.get("stop_condition"), err=err[:120],
            temp=cfg.get("sampling", {}).get("temperature"), max_tokens=cfg.get("sampling", {}).get("max_tokens"),
            detector=bool(cfg.get("degenerate_turn")),
        )
        ti = 0
        prev_tool = 0
        nsampled = sum(1 for n in nodes if n.get("sampled"))
        for i, n in enumerate(nodes):
            if not n.get("sampled"):
                prev_tool += len(n.get("token_ids") or [])
                continue
            lp = n.get("logprobs") or []
            vals = [x for x in lp if x is not None]
            c = callby.get(i, {})
            row = dict(base, turn=ti, nturns=nsampled, last=(ti == nsampled - 1), node=i,
                       prompt=(c.get("usage") or {}).get("prompt_tokens"), ntok=len(vals),
                       finish=c.get("finish_reason"), prev_tool=prev_tool)
            prev_tool = 0
            ti += 1
            if not vals:
                fo.write(json.dumps(row) + "\n")
                continue
            nll = [-x for x in vals]
            row["nll"] = sum(nll) / len(nll)
            row["b"] = [(sum(nll[a:b]) / len(nll[a:b]) if len(nll[a:b]) else None) for a, b in BUCKETS]
            W = 64
            if len(nll) >= W:
                s = sum(nll[:W]); best = s; onset = 0 if s / W > 3 else None
                for j in range(W, len(nll)):
                    s += nll[j] - nll[j - W]
                    if s > best: best = s
                    if onset is None and s / W > 3: onset = j - W + 1
                row["maxwin"] = best / W
                row["onset"] = onset
            row["degen"] = len(nll) >= 100 and row["nll"] > 3.0
            # fraction of very unlikely tokens (sampled token prob < e^-5, e^-10)
            row["f5"] = sum(x > 5 for x in nll) / len(nll)
            row["f10"] = sum(x > 10 for x in nll) / len(nll)
            if keep:
                row["lp"] = [round(x, 4) for x in vals]
            fo.write(json.dumps(row) + "\n")
    print(f"{tdir}: {nrec} train records, {bad} unreadable lines", file=sys.stderr)


if __name__ == "__main__":
    main()
