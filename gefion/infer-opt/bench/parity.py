"""Logprob parity between a stock vLLM server and a patched/reconfigured one.

  parity.py gen   --url U --model M --out ref.json    # seeded sampling with top-33 logprobs (decode path)
  parity.py score --url U --model M --seqs ref.json --out ref_tf.json
                                                      # teacher-forced logprobs of those sequences (prefill path)
  parity.py compare --a ref.json --b var.json [--a-tf ref_tf.json --b-tf var_tf_on_ref.json]

What is compared (Δ = logprob difference of the sampled token, nats):
  decode:   same seeds on both servers; Δ over the common token prefix of each sequence, plus how
            far the sequences stay identical and the top-32 set overlap.
  prefill:  both servers teacher-force the *same* (reference) sequences; Δ per token.
  mismatch: within one server, sampler logprob vs its own teacher-forced logprob of the same
            sequence. This is the train/inference mismatch the RL loss sees (the trainer is a
            third implementation, but a variant should not make this worse).
k3 = mean(exp(Δ) - 1 - Δ) is the usual low-variance KL estimator for the mismatch.
"""
import argparse
import asyncio
import json
import math
import statistics

import httpx

PROMPTS = [
    "Find all real x such that x^4 - 5x^2 + 4 = 0. Show your reasoning.",
    "A bag has 5 red and 7 blue balls. Two are drawn without replacement. What is the probability both are blue?",
    "Prove that the sum of the first n odd numbers is n^2.",
    "Write a Python function that returns the longest palindromic substring of a string, and explain its complexity.",
    "How many positive integers less than 1000 are divisible by 7 but not by 11?",
    "Let f(x) = x^3 - 3x + 1. How many real roots does f have? Justify.",
    "Explain the difference between a process and a thread, with an example where each is preferable.",
    "Compute the determinant of the matrix [[2,1,0],[1,3,1],[0,1,4]].",
    "Solve the recurrence a_n = 3a_{n-1} - 2a_{n-2}, a_0 = 2, a_1 = 3.",
    "A train leaves at 9:00 at 80 km/h; another leaves the same station at 10:30 at 120 km/h on the same track. When does the second catch up?",
    "What is the remainder when 2^100 is divided by 13?",
    "Give a SQL query that returns, for each department, the employee with the highest salary.",
    "Find the area of a triangle with sides 13, 14 and 15.",
    "Describe how a hash map handles collisions, comparing chaining and open addressing.",
    "Find the minimum value of x^2 + y^2 subject to x + 2y = 5.",
    "Is 391 prime? Explain.",
    "Integrate x * e^(2x) dx.",
    "A function in a large codebase occasionally returns stale data under concurrency. Outline how you would debug it.",
    "How many ways can 8 people be seated around a round table if two particular people must not sit together?",
    "Find all integer solutions of x^2 - y^2 = 45.",
    "Explain why the harmonic series diverges.",
    "Write a bash one-liner that finds the 10 largest files under the current directory.",
    "Let a, b, c be positive reals with a+b+c=3. Prove that ab+bc+ca <= 3.",
    "What is the expected number of rolls of a fair die until a 6 appears?",
    "Convert the repeating decimal 0.272727... to a fraction in lowest terms.",
    "Explain gradient descent with momentum and when it helps.",
    "How many trailing zeros does 125! have?",
    "Find the eigenvalues of [[4,1],[2,3]].",
    "A rectangle has perimeter 40. What dimensions maximize its area, and why?",
    "Write a regular expression that matches valid IPv4 addresses and explain it.",
    "Sum the series sum_{k=1}^{inf} 1/(k(k+1)).",
    "Two dice are rolled. What is the probability that the sum is prime?",
]


def tokenize(model: str) -> list[list[int]]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    out = []
    for p in PROMPTS:
        msgs = [{"role": "user", "content": p}]
        try:
            text = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=True, tokenize=False)
        except TypeError:
            text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        ids = tok.encode(text, add_special_tokens=False)
        out.append(list(ids))
    return out


async def post(client, url, body):
    r = await client.post(f"{url}/inference/v1/generate", json=body, timeout=3600)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
    return r.json()


async def gen(a):
    # --prompts-file: real (e.g. i3-math) prompts that end at EOS; otherwise built-in prompts, forced length
    prompts = json.load(open(a.prompts_file))[:a.n_prompts] if a.prompts_file else tokenize(a.model)
    reps = a.reps
    async with httpx.AsyncClient(timeout=3600) as client:
        async def one(i, p):
            sp = {"max_tokens": a.max_tokens, "temperature": 1.0, "seed": 1000 + i, "logprobs": 33,
                  "skip_special_tokens": False, "ignore_eos": a.force_length or not a.prompts_file}
            d = await post(client, a.url, {"model": a.model, "token_ids": p, "sampling_params": sp})
            ch = d["choices"][0]
            lps = [c["logprob"] for c in ch["logprobs"]["content"]]
            tops = [[int(t["token"][9:]) for t in sorted(c["top_logprobs"], key=lambda t: -t["logprob"])[:32]]
                    for c in ch["logprobs"]["content"]]
            return {"prompt": p, "tokens": ch["token_ids"], "lp": lps, "top": tops}

        jobs = [one(r * len(prompts) + i, p) for r in range(reps) for i, p in enumerate(prompts)]
        res = await asyncio.gather(*jobs)
    json.dump(res, open(a.out, "w"))
    print(f"gen: {len(res)} sequences, {sum(len(r['tokens']) for r in res)} tokens -> {a.out}")


async def score(a):
    seqs = json.load(open(a.seqs))
    async with httpx.AsyncClient(timeout=3600) as client:
        async def one(s):
            ids = s["prompt"] + s["tokens"]
            sp = {"max_tokens": 1, "temperature": 1.0, "prompt_logprobs": 1, "skip_special_tokens": False}
            d = await post(client, a.url, {"model": a.model, "token_ids": ids, "sampling_params": sp})
            pl = d["prompt_logprobs"]
            n = len(s["prompt"])
            out = []
            for pos, tid in enumerate(s["tokens"]):
                e = pl[n + pos]
                v = e.get(str(tid)) if e else None
                out.append(None if v is None else v["logprob"])
            return out

        res = await asyncio.gather(*(one(s) for s in seqs))
    json.dump(res, open(a.out, "w"))
    print(f"score: {len(res)} sequences -> {a.out}")


def stats(deltas: list[float]) -> dict:
    if not deltas:
        return {}
    ab = sorted(abs(d) for d in deltas)
    k3 = statistics.fmean(math.exp(d) - 1 - d for d in deltas)
    return {"n": len(deltas), "mean_abs": round(statistics.fmean(ab), 5), "p50_abs": round(ab[len(ab) // 2], 5),
            "p99_abs": round(ab[int(0.99 * (len(ab) - 1))], 5), "max_abs": round(ab[-1], 4),
            "mean": round(statistics.fmean(deltas), 6), "k3": round(k3, 7)}


BUCKETS = [(0, 1024), (1024, 4096), (4096, 8192), (8192, 1 << 30)]


def mismatch_by_pos(seqs, tfs) -> dict:
    """Sampler-vs-teacher-forced mismatch by output position: does decode error grow with length?"""
    out = {}
    for lo, hi in BUCKETS:
        d = [tf[i] - s["lp"][i] for s, tf in zip(seqs, tfs) for i in range(lo, min(hi, len(tf))) if tf[i] is not None]
        if d:
            out[f"{lo}-{hi if hi < 1 << 30 else 'end'}"] = stats(d)
    return out


def compare(a):
    B = json.load(open(a.b))
    out = {}
    if a.b_self_tf and not a.a:  # single-server mode: mismatch only
        TB = json.load(open(a.b_self_tf))
        out["mismatch_b"] = stats([tf - lp for s, t in zip(B, TB) for lp, tf in zip(s["lp"], t) if tf is not None])
        out["mismatch_b_by_pos"] = mismatch_by_pos(B, TB)
        print("PARITY " + json.dumps(out))
        if a.out:
            json.dump(out, open(a.out, "w"), indent=1)
        return
    A = json.load(open(a.a))
    # decode path: common prefix of same-seed sequences
    dec, same_len, tot_len, overlap = [], 0, 0, []
    for x, y in zip(A, B):
        n = 0
        while n < min(len(x["tokens"]), len(y["tokens"])) and x["tokens"][n] == y["tokens"][n]:
            n += 1
        same_len += n
        tot_len += len(x["tokens"])
        dec += [y["lp"][i] - x["lp"][i] for i in range(n)]
        overlap += [len(set(x["top"][i]) & set(y["top"][i])) / max(1, len(x["top"][i])) for i in range(n)]
    out["decode"] = stats(dec)
    out["decode"]["identical_token_frac"] = round(same_len / max(1, tot_len), 4)
    out["decode"]["top32_overlap"] = round(statistics.fmean(overlap), 4) if overlap else None
    if a.a_tf and a.b_tf:  # prefill path on the same (A's) sequences
        TA, TB = json.load(open(a.a_tf)), json.load(open(a.b_tf))
        out["prefill"] = stats([q - p for sa, sb in zip(TA, TB) for p, q in zip(sa, sb) if p is not None and q is not None])
        out["mismatch_a"] = stats([tf - lp for s, t in zip(A, TA) for lp, tf in zip(s["lp"], t) if tf is not None])
    if a.b_self_tf:
        TB = json.load(open(a.b_self_tf))
        out["mismatch_b"] = stats([tf - lp for s, t in zip(B, TB) for lp, tf in zip(s["lp"], t) if tf is not None])
        out["mismatch_b_by_pos"] = mismatch_by_pos(B, TB)
    print("PARITY " + json.dumps(out))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["gen", "score", "compare"])
ap.add_argument("--url", default="http://localhost:8000")
ap.add_argument("--model")
ap.add_argument("--out")
ap.add_argument("--seqs")
ap.add_argument("--max-tokens", type=int, default=1024)
ap.add_argument("--reps", type=int, default=2, help="prompt repetitions (distinct seeds)")
ap.add_argument("--prompts-file", default="")
ap.add_argument("--n-prompts", type=int, default=32)
ap.add_argument("--force-length", action="store_true", help="ignore EOS with --prompts-file too")
ap.add_argument("--a")
ap.add_argument("--b")
ap.add_argument("--a-tf")
ap.add_argument("--b-tf")
ap.add_argument("--b-self-tf")
args = ap.parse_args()
if args.mode == "gen":
    asyncio.run(gen(args))
elif args.mode == "score":
    asyncio.run(score(args))
else:
    compare(args)
