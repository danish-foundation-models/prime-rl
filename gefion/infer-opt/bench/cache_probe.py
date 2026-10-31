"""Controlled prefix-cache / host-offload probe on /inference/v1/generate.

One request at a time, each followed by a /metrics diff, so every cache hit can be attributed:
  prefill      A (20000 tokens) -> A + 500            GPU prefix hit after a prefill-only request
  decode       B (1000) + 9000 decoded -> B+out+500   GPU hit on blocks completed during decode
  repeat       A again                                same request twice
  offload      C (20000) -> reset GPU cache -> C+500  host-offload (external) hit
  offload_dec  D (1000) + 9000 decoded -> reset -> D+out+500
The server must run with PRIME_RL_CACHE_DIAGNOSTICS=1 (exposes POST /reset_prefix_cache). All requests of
one test share their first tokens, so dp_affinity keeps them on one DP engine.
"""
import argparse
import json
import random
import re
import time

import httpx

p = argparse.ArgumentParser()
p.add_argument("--url", default="http://localhost:8000")
p.add_argument("--model", required=True)
p.add_argument("--tag", default="probe")
p.add_argument("--long", type=int, default=20000)
p.add_argument("--decode", type=int, default=9000)
p.add_argument("--consistency", action="store_true", help="also compare a warm (prefix-hit) greedy decode "
               "with the same request decoded cold after clearing GPU and host caches")
args = p.parse_args()

KEYS = re.compile(r"^(vllm:(?:prefix_cache_(?:queries|hits)|external_prefix_cache_(?:queries|hits)|num_preemptions)"
                  r"_total|vllm:kv_offload_[a-z_]+?(?:_total|_sum|_count)?)\{([^}]*)\} ([0-9.eE+-]+)$")
c = httpx.Client(timeout=3600)
rng = random.Random(0)


def metrics() -> dict:
    out = {}
    for line in c.get(f"{args.url}/metrics").text.splitlines():
        m = KEYS.match(line)
        if m:
            eng = re.search(r'engine="(\d+)"', m.group(2))
            k = (m.group(1), eng.group(1) if eng else "-")
            out[k] = out.get(k, 0.0) + float(m.group(3))
    return out


def diff(a, b) -> dict:
    d = {}
    for k in b:
        v = b[k] - a.get(k, 0.0)
        if abs(v) > 0:
            d[f"{k[0].replace('vllm:', '')}[e{k[1]}]"] = round(v, 3)
    return d


def toks(n):
    return [rng.randrange(1000, 100000) for _ in range(n)]


def gen(ids, max_tokens, label, greedy=False):
    m0 = metrics()
    t = time.perf_counter()
    body = {"model": args.model, "token_ids": ids,
            "sampling_params": {"max_tokens": max_tokens, "temperature": 0.0 if greedy else 1.0, "ignore_eos": True,
                                "skip_special_tokens": False, **({"logprobs": 1} if greedy else {})}}
    r = c.post(f"{args.url}/inference/v1/generate", json=body)
    r.raise_for_status()
    d = r.json()
    dt = time.perf_counter() - t
    time.sleep(2)  # let engine stats reach the API server
    out = d["choices"][0]["token_ids"]
    usage = d.get("usage") or {}
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    print(json.dumps({"tag": args.tag, "step": label, "prompt": len(ids), "out": len(out), "s": round(dt, 2),
                      "cached_tokens": cached, "metrics": diff(m0, metrics())}), flush=True)
    if greedy:
        return out, [t["logprob"] for t in d["choices"][0]["logprobs"]["content"]]
    return out


def reset(external=False):
    # GPU only unless external (then the host offload tier too)
    r = c.post(f"{args.url}/reset_prefix_cache" + ("?reset_external=true" if external else ""))
    print(json.dumps({"tag": args.tag, "step": "reset_gpu_prefix_cache", "status": r.status_code, "body": r.text[:200]}),
          flush=True)
    time.sleep(2)


A = toks(args.long)
gen(A, 8, "prefill:A")
gen(A + toks(500), 8, "prefill:A+500")
B = toks(1000)
o = gen(B, args.decode, "decode:B")
gen(B + o + toks(500), 8, "decode:B+out+500")
gen(A, 8, "repeat:A")
C = toks(args.long)
gen(C, 8, "offload:C")
time.sleep(5)
reset()
gen(C + toks(500), 8, "offload:C+500")
D = toks(1000)
o = gen(D, args.decode, "offload_dec:D")
time.sleep(5)
reset()
gen(D + o + toks(500), 8, "offload_dec:D+out+500")



def compare(a, b):
    (ta, la), (tb, lb) = a, b
    n = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), len(ta))
    dl = [abs(x - y) for x, y in zip(la[:n], lb[:n])]
    return {"same_prefix_tokens": n, "of": len(ta), "mean_abs_dlogprob": round(sum(dl) / len(dl), 5) if dl else None,
            "max_abs_dlogprob": round(max(dl), 4) if dl else None}


if args.consistency:
    # turn 1: prompt X, sampled decode; turn 2: Y = X + out + tool tokens, greedy, decoded four ways:
    #   gpu   prefix hit on the GPU cache (ends at X's last hash boundary, inside a block)
    #   host  GPU cache cleared, prefix restored from the host offload tier
    #   cold  all caches cleared;  cold2  again (cold vs cold2 is the run-to-run noise floor)
    # Random-token prompts give near-tied logits, so greedy paths can split on noise alone: compare
    # each warm path's agreement with cold against cold2's.
    for trial in range(3):
        X = toks(6000)
        o = gen(X, 3000, f"consistency{trial}:X")
        Y = X + o + toks(700)
        time.sleep(3)
        r = {"gpu": gen(Y, 256, f"consistency{trial}:gpu", greedy=True)}
        time.sleep(5)
        reset()
        r["host"] = gen(Y, 256, f"consistency{trial}:host", greedy=True)
        reset(external=True)
        r["cold"] = gen(Y, 256, f"consistency{trial}:cold", greedy=True)
        reset(external=True)
        r["cold2"] = gen(Y, 256, f"consistency{trial}:cold2", greedy=True)
        for k in ("gpu", "host", "cold2"):
            print(json.dumps({"tag": args.tag, "step": f"consistency{trial}:{k}_vs_cold", **compare(r[k], r["cold"])}),
                  flush=True)
