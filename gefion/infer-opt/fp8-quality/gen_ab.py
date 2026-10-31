"""Generation A/B: resample the next turn of fixed contexts, same seeds on every server.

  gen_ab.py --url http://localhost:8000 --set data/abset.jsonl --out OUT.jsonl [--n 8] [--conc 256]
            [--roles gen_degen,gen_normal] [--no-repetition-detection] [--temperature T --top-p P --min-p M]

Sampling mirrors the live train rollouts: temperature 1.0, top_p 1, top_k -1, min_p 0,
max_tokens 16384, repetition_detection {16, 64, 8}, logprobs on. Request i of context c uses
seed hash(c, i), so both servers draw from common random numbers (a paired comparison).
--temperature/--top-p/--min-p run sampling variants (logprobs are then vLLM's processed logprobs,
i.e. of the truncated/tempered distribution, as in the live run).
Per sample: completion length, finish reason, sampled-token logprobs (mean NLL, degenerate flag
computed in analyze_gen.py).
"""
import argparse, asyncio, hashlib, json, sys, time
import httpx


async def post_retry(client, url, body, tries=4):
    """POST with retries on transport errors and 5xx (a single failure must not kill a long run)."""
    for k in range(tries):
        try:
            resp = await client.post(url, json=body)
            if resp.status_code < 500 or k == tries - 1:
                return resp
        except httpx.TransportError:
            if k == tries - 1:
                raise
        await asyncio.sleep(5 * 2 ** k)

async def main(a):
    rows = [r for r in map(json.loads, open(a.set)) if set(r["roles"]) & set(a.roles.split(","))]
    done = set()
    try:
        done = {(x["id"], x["i"]) for x in map(json.loads, open(a.out))}
    except FileNotFoundError:
        pass
    jobs = [(r, i) for r in rows for i in range(a.n) if (r["id"], i) not in done]
    print(f"{len(rows)} contexts x {a.n} = {len(jobs)} samples to go", file=sys.stderr)
    fo = open(a.out, "a"); sem = asyncio.Semaphore(a.conc); t0 = time.time(); n = 0; ntok = 0
    async with httpx.AsyncClient(timeout=7200, limits=httpx.Limits(max_connections=a.conc + 8)) as client:
        async def one(r, i):
            nonlocal n, ntok
            seed = int(hashlib.sha1(f"{r['id']}:{i}".encode()).hexdigest()[:8], 16)
            sp = {"max_tokens": 16384, "temperature": a.temperature, "top_p": a.top_p, "top_k": -1, "min_p": a.min_p, "seed": seed,
                  "logprobs": 1, "skip_special_tokens": False}
            if not a.no_repetition_detection:
                sp["repetition_detection"] = {"min_pattern_size": 16, "max_pattern_size": 64, "min_count": 8}
            async with sem:
                resp = await post_retry(client, f"{a.url}/inference/v1/generate",
                                        {"model": a.model, "token_ids": r["prompt_ids"], "sampling_params": sp})
            if resp.status_code != 200:
                raise RuntimeError(f"{r['id']}: HTTP {resp.status_code}: {resp.text[:400]}")
            ch = resp.json()["choices"][0]
            lp = [round(c["logprob"], 4) for c in ch["logprobs"]["content"]]
            rec = {"id": r["id"], "i": i, "roles": r["roles"], "prompt_len": r["prompt_len"],
                   "finish": ch.get("finish_reason"), "ntok": len(lp), "lp": lp}
            if a.save_ids: rec["ids"] = ch["token_ids"]
            fo.write(json.dumps(rec) + "\n"); fo.flush()
            n += 1; ntok += len(lp)
            if n % 100 == 0: print(f"  {n}/{len(jobs)}  {ntok/(time.time()-t0):.0f} tok/s", file=sys.stderr)
        await asyncio.gather(*(one(r, i) for r, i in jobs))
    print(f"done {n} samples, {ntok} tokens in {time.time()-t0:.0f}s -> {a.out}", file=sys.stderr)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--model", default="/dcai/projects/iu_0145/rasmus/models/nemotron-super-r6-step2638")
    p.add_argument("--set", required=True); p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=8); p.add_argument("--conc", type=int, default=256)
    p.add_argument("--roles", default="gen_degen,gen_normal")
    p.add_argument("--no-repetition-detection", action="store_true")
    p.add_argument("--save-ids", action="store_true", help="store sampled token ids (for cross-scoring)")
    # sampling variants (defaults = the live run before the temperature change)
    p.add_argument("--temperature", type=float, default=1.0); p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--min-p", type=float, default=0.0)
    asyncio.run(main(p.parse_args()))
