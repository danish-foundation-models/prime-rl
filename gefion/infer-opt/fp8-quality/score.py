"""Teacher-forced rescoring of the A/B set's sampled turns on one server (prefill path).

  score.py --url http://localhost:8000 --set data/abset.jsonl --out OUT.jsonl [--k 1] [--max-len N]
           [--roles normal,degen] [--conc 8] [--gen GEN.jsonl --per-context 2]

For each turn: token_ids = prompt_ids + completion_ids, max_tokens 1, prompt_logprobs k.
Writes per completion position: lp (logprob of the sampled token), and the top-k ids/logprobs
(k=1 gives the argmax only). Rows longer than --max-len tokens are skipped (top-k JSON is large).
"""
import argparse, asyncio, json, sys, time
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
    rows = [r for r in map(json.loads, open(a.set)) if set(r["roles"]) & set(a.roles.split(","))
            and r["prompt_len"] + r["ntok"] <= a.max_len]
    if a.gen:  # score another server's generations: completion = sampled ids, first --per-context samples
        ctx = {r["id"]: r for r in rows}
        rows = []
        for g in map(json.loads, open(a.gen)):
            if g["id"] in ctx and g["i"] < a.per_context and g.get("ids"):
                rows.append(dict(id=f"{g['id']}#{g['i']}", prompt_len=ctx[g["id"]]["prompt_len"], ntok=len(g["ids"]),
                                 prompt_ids=ctx[g["id"]]["prompt_ids"], completion_ids=g["ids"]))
    done = set()
    try:
        done = {json.loads(l)["id"] for l in open(a.out)}
    except FileNotFoundError:
        pass
    rows = [r for r in rows if r["id"] not in done]
    print(f"scoring {len(rows)} turns (k={a.k}, {len(done)} already done)", file=sys.stderr)
    fo = open(a.out, "a"); sem = asyncio.Semaphore(a.conc); t0 = time.time(); n = 0
    async with httpx.AsyncClient(timeout=7200) as client:
        async def one(r):
            nonlocal n
            ids = r["prompt_ids"] + r["completion_ids"]
            body = {"model": a.model, "token_ids": ids,
                    "sampling_params": {"max_tokens": 1, "temperature": 1.0, "top_p": 1.0, "prompt_logprobs": a.k,
                                        "skip_special_tokens": False}}
            async with sem:
                resp = await post_retry(client, f"{a.url}/inference/v1/generate", body)
            if resp.status_code != 200:
                raise RuntimeError(f"{r['id']}: HTTP {resp.status_code}: {resp.text[:400]}")
            pl = resp.json()["prompt_logprobs"]
            P = r["prompt_len"]; lp, tid, tlp = [], [], []
            for pos, t in enumerate(r["completion_ids"]):
                e = pl[P + pos] or {}
                v = e.get(str(t))
                lp.append(None if v is None else v["logprob"])
                top = sorted(((int(k), x["logprob"]) for k, x in e.items() if x.get("rank", 0) and x["rank"] <= a.k),
                             key=lambda z: -z[1])[:a.k]
                tid.append([z[0] for z in top]); tlp.append([round(z[1], 5) for z in top])
            fo.write(json.dumps({"id": r["id"], "lp": lp, "top_ids": tid, "top_lp": tlp}) + "\n"); fo.flush()
            n += 1
            if n % 20 == 0: print(f"  {n}/{len(rows)}  {time.time()-t0:.0f}s", file=sys.stderr)
        await asyncio.gather(*(one(r) for r in rows))
    print(f"done {n} turns in {time.time()-t0:.0f}s -> {a.out}", file=sys.stderr)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--model", default="/dcai/projects/iu_0145/rasmus/models/nemotron-super-r6-step2638")
    p.add_argument("--set", required=True); p.add_argument("--out", required=True)
    p.add_argument("--k", type=int, default=1); p.add_argument("--max-len", type=int, default=1 << 30)
    p.add_argument("--roles", default="normal,degen"); p.add_argument("--conc", type=int, default=8)
    p.add_argument("--gen", help="gen_ab.py output (with --save-ids) to score instead of the set's own turns")
    p.add_argument("--per-context", type=int, default=2, help="with --gen: samples per context to score")
    asyncio.run(main(p.parse_args()))
