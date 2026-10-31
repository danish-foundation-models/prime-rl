"""Build the fixed A/B set of sampled turns from the live run's traces (CPU only).

  python3 build_set.py <traces_dir> <turns.jsonl from extract_turns.py> <out.jsonl> [--ckpt]

Only policy-0 rollouts (pol start = end = 0): their tokens were sampled from exactly the
weights in the start checkpoint, so a bf16 rescore of the checkpoint on disk is the policy's
true bf16 distribution. Agentic envs, sampled turns of >= 100 tokens, prompt + turn <= 131072.

Sets (role field):
  normal  200 non-degenerate turns, 50 per context bucket (<8k, 8-32k, 32-64k, >=64k)   -> scoring
  degen   100 degenerate turns (mean NLL > 3)                                          -> scoring
  gen_degen   150 contexts whose next turn degenerated (the degen 100 + 50 more)       -> generation A/B
  gen_normal  150 contexts of non-degenerate turns, 16k-64k context                    -> generation A/B
--ckpt instead builds the checkpoint-comparison set: 100 random turns per stratum, roles
  ck_turn0 (first turn: system + task), ck_lt8k, ck_8_32k, ck_ge32k (later turns by context length).
Each row: id (rid:node), role(s), prompt_ids, completion_ids, trace_lp (FP8 PD sampler logprobs),
plus metadata (env, harness, prompt length, onset of the first >3-nat 64-token window, ...).
"""
import collections, json, random, re, subprocess, sys, glob

tdir, turns_f, out = sys.argv[1:4]
CKPT = "--ckpt" in sys.argv
AG = {"swesmith-env", "scaleswe", "openswe", "terminal-lego", "tmax"}
rows = []
for l in open(turns_f):
    r = json.loads(l)
    m = re.match(r"(.*)-(bash|opencode|pi)$", r["env"])
    if m: r["env"], r["harness"] = m.group(1), m.group(2)
    if (r["env"] in AG and r.get("nll") is not None and r["ntok"] >= 100 and r.get("pol") == {"start": 0, "end": 0}
            and r["prompt"] and r["prompt"] + r["ntok"] <= 131000):
        rows.append(r)
rng = random.Random(0)
rng.shuffle(rows)
pick = collections.defaultdict(set)  # (rid, node) -> roles
def take(cands, n, role):
    for r in cands[:n]: pick[(r["rid"], r["node"])].add(role)
if CKPT:
    # Checkpoint comparison: random policy-0 turns regardless of outcome (so rates are representative),
    # 100 contexts per stratum; ck_turn0 = first sampled turn (system + task only, fair to any model).
    take([r for r in rows if r["turn"] == 0], 100, "ck_turn0")
    for lo, hi, name in ((0, 8000, "ck_lt8k"), (8000, 32000, "ck_8_32k"), (32000, 1 << 30, "ck_ge32k")):
        take([r for r in rows if r["turn"] > 0 and lo <= r["prompt"] < hi], 100, name)
nd = [] if CKPT else [r for r in rows if not r["degen"]]
for lo, hi in ((0, 8000), (8000, 32000), (32000, 64000), (64000, 1 << 30)):
    take([r for r in nd if lo <= r["prompt"] < hi], 50, "normal")
if not CKPT:
    dg = [r for r in rows if r["degen"]]
    take(dg, 100, "degen"); take(dg, 150, "gen_degen")
    take([r for r in nd if 16000 <= r["prompt"] < 64000 and (r["rid"], r["node"]) not in pick], 150, "gen_normal")
meta = {(r["rid"], r["node"]): r for r in rows}
want = collections.defaultdict(dict)
for (rid, node), roles in pick.items(): want[rid][node] = roles
print(f"picked {len(pick)} turns from {len(want)} rollouts:", collections.Counter(x for v in pick.values() for x in v), file=sys.stderr)

def lines():
    for f in sorted(glob.glob(f"{tdir}/stream/*.jsonl*")):
        if f.endswith(".zst"):
            p = subprocess.Popen(["zstd", "-dc", f], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL); yield from p.stdout
        else: yield from open(f, "rb")
fo = open(out, "w"); done = bad = 0
for line in lines():
    rid = line[7:39].decode(errors="ignore")
    if rid not in want: continue
    d = json.loads(line)
    nodes = d["traces"][0]["nodes"]
    calls = {c["node"]: c for c in d["traces"][0]["calls"] if "node" in c}
    for node, roles in want.pop(rid).items():
        n = nodes[node]; lp = n["logprobs"]; ids = n["token_ids"]
        hdr = len(ids) - len(lp)
        prompt = [t for x in nodes[:node] for t in x["token_ids"]] + ids[:hdr]
        pt = (calls.get(node, {}).get("usage") or {}).get("prompt_tokens")
        if pt != len(prompt) or any(x is None for x in lp):
            bad += 1; print(f"skip {rid}:{node} prompt {len(prompt)} vs usage {pt}", file=sys.stderr); continue
        m = meta[(rid, node)]
        fo.write(json.dumps(dict(id=f"{rid}:{node}", roles=sorted(roles), env=m["env"], harness=m["harness"],
                                 turn=m["turn"], prompt_len=len(prompt), ntok=len(lp), nll=m["nll"], onset=m.get("onset"),
                                 finish=m["finish"], prompt_ids=prompt, completion_ids=ids[hdr:], trace_lp=lp)) + "\n")
        done += 1
    if not want: break
print(f"wrote {done} turns ({bad} skipped) -> {out}", file=sys.stderr)
