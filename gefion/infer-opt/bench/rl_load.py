"""Closed-loop RL-shaped load against PRIME-RL's /inference/v1/generate endpoint.

Mirrors what the orchestrator sends (renderers/client.py): token-id prompts, non-streamed,
``logprobs = score_centering_top_k + 1`` (33 for top_k=32), ``skip_special_tokens=False``.
GRPO groups are emulated by sending each prompt ``--group`` times, so prefix caching sees the
same sharing as a real run. Outputs are forced to their sampled length with ``ignore_eos``.

Throughput is measured over a fixed window from the server's Prometheus counters, so the long
tail of a batch does not distort it. Prints one ``RESULT {json}`` line.

Usage: rl_load.py --url http://localhost:8000 --model M --concurrency 768 --duration 120
"""
import argparse
import asyncio
import json
import random
import re
import time

import httpx

p = argparse.ArgumentParser()
p.add_argument("--url", default="http://localhost:8000")
p.add_argument("--metrics-url", default=None, help="vLLM /metrics base URL (the engine port when --url is a router)")
p.add_argument("--session-header", action="store_true", help="send x-session-id per rollout, as the verifiers "
               "clients do (PRIME-RL's vllm-router consistent_hash policy routes on it)")
p.add_argument("--model", required=True)
p.add_argument("--concurrency", type=int, default=768)
p.add_argument("--warmup", type=float, default=60.0, help="seconds before the measurement window")
p.add_argument("--duration", type=float, default=120.0, help="measurement window, seconds")
p.add_argument("--prompt", default="1024:8192", help="prompt length range lo:hi (log-uniform)")
p.add_argument("--output", default="8192:32768", help="output length range lo:hi (log-uniform)")
p.add_argument("--logprobs", type=int, default=33, help="0 = no logprobs field")
p.add_argument("--group", type=int, default=8, help="requests per distinct prompt (GRPO group)")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--turns", type=int, default=1, help="agent-style turns per rollout: each turn re-sends the whole "
               "history (prompt + earlier outputs + tool results), so prefix caching sees real reuse")
p.add_argument("--tool-tokens", default="256:2048", help="synthetic tool-result tokens appended after each turn")
p.add_argument("--max-context", type=int, default=65536, help="end a multi-turn rollout once its context exceeds this")
p.add_argument("--prompts-file", default="", help="JSON list of token-id prompts (e.g. data/i3math_prompts.json); "
               "replaces the synthetic prompts, and outputs end at EOS (max --output hi) instead of a forced length")
p.add_argument("--tag", default="")
p.add_argument("--hook-at", type=float, default=-1, help="seconds into the window to run --hook")
p.add_argument("--hook", default="", help="shell command run once at --hook-at (e.g. start profiler)")
p.add_argument("--post-hook", default="", help="shell command run after the window closes, load still running")
args = p.parse_args()

VOCAB_LO, VOCAB_HI = 1000, 120000  # stay clear of special/control tokens


def loguniform(rng: random.Random, spec: str) -> int:
    lo, hi = (int(x) for x in spec.split(":"))
    if lo == hi:
        return lo
    return int(round(lo * (hi / lo) ** rng.random()))


FILE_PROMPTS = json.load(open(args.prompts_file)) if args.prompts_file else None


class Source:
    """Yields (prompt_ids, max_tokens); each prompt is reused --group times, like GRPO."""

    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.cur, self.left = None, 0

    def next(self):
        if FILE_PROMPTS is not None:
            if self.left == 0:
                self.cur = FILE_PROMPTS[self.rng.randrange(len(FILE_PROMPTS))]
                self.left = args.group
            self.left -= 1
            return self.cur, int(args.output.split(":")[1])
        if self.left == 0:
            n = loguniform(self.rng, args.prompt)
            self.cur = [self.rng.randrange(VOCAB_LO, VOCAB_HI) for _ in range(n)]
            self.left = args.group
        self.left -= 1
        return self.cur, loguniform(self.rng, args.output)


METRIC_RE = re.compile(r"^(vllm:[a-z_]+)(?:\{[^}]*\})?\s+([0-9.eE+-]+)$", re.M)


async def metrics(client: httpx.AsyncClient) -> dict[str, float]:
    # Generous timeout: a CPU-bound API server answers late, and that must not kill the run.
    for attempt in range(5):
        try:
            r = await client.get(f"{args.metrics_url or args.url}/metrics", timeout=600)
            break
        except httpx.HTTPError as e:
            print(f"metrics retry {attempt}: {e!r}", flush=True)
            await asyncio.sleep(5)
    else:
        raise RuntimeError("metrics endpoint unreachable")
    out: dict[str, float] = {}
    for name, val in METRIC_RE.findall(r.text):
        out[name] = out.get(name, 0.0) + float(val)  # sum over DP engines / label sets
        out["n:" + name] = out.get("n:" + name, 0) + 1
    return out


done = {"req": 0, "out": 0, "bytes": 0, "errors": 0, "lat": []}
checked = {"n": 0}


async def generate(client, ids, max_tokens, session=None):
    """One /inference/v1/generate call; returns (completion token ids or None on error, bytes)."""
    sp = {"max_tokens": max_tokens, "temperature": 1.0, "ignore_eos": FILE_PROMPTS is None, "skip_special_tokens": False,
          "seed": random.randrange(1 << 30)}
    if args.logprobs:
        sp["logprobs"] = args.logprobs
    body = {"model": args.model, "token_ids": ids, "sampling_params": sp}
    try:
        headers = {"x-session-id": session} if session else None
        r = await client.post(f"{args.url}/inference/v1/generate", json=body, timeout=7200, headers=headers)
        raw = r.content
    except Exception as e:  # noqa: BLE001
        done["errors"] += 1
        if done["errors"] <= 3:
            print(f"request error: {e!r}", flush=True)
        return None, 0
    if r.status_code != 200:
        done["errors"] += 1
        if done["errors"] <= 3:
            print(f"HTTP {r.status_code}: {raw[:300]!r}", flush=True)
        return None, 0
    return raw, len(raw)


async def worker(i: int, client: httpx.AsyncClient, src: Source, stop: asyncio.Event):
    rng = random.Random(i)
    while not stop.is_set():
        prompt, max_tokens = src.next()
        ctx = list(prompt)
        session = f"w{i}-{rng.getrandbits(48):012x}" if args.session_header else None
        for turn in range(args.turns):
            t = time.perf_counter()
            raw, nbytes = await generate(client, ctx, max_tokens, session)
            if raw is None:
                await asyncio.sleep(1)
                break
            done["req"] += 1
            done["bytes"] += nbytes
            done["lat"].append(time.perf_counter() - t)
            if checked["n"] < 2 or args.turns > 1:  # multi-turn needs the tokens to extend the context
                d = json.loads(raw)
                ch = d["choices"][0]
                out = ch["token_ids"]
                if checked["n"] < 2:
                    checked["n"] += 1
                    k = len(ch["logprobs"]["content"][0]["top_logprobs"]) if args.logprobs and out else 0
                    print(f"check: out={len(out)} (max {max_tokens}) top_logprobs/token={k} bytes={nbytes}", flush=True)
                done["out"] += len(out)
            else:
                m = re.search(rb'"completion_tokens":\s*(\d+)', raw[-400:])
                done["out"] += int(m.group(1)) if m else 0
                break
            if turn == args.turns - 1:
                break
            ctx = ctx + out + [rng.randrange(VOCAB_LO, VOCAB_HI) for _ in range(loguniform(rng, args.tool_tokens))]
            if len(ctx) > args.max_context or stop.is_set():
                break


async def main():
    limits = httpx.Limits(max_connections=args.concurrency + 8, max_keepalive_connections=args.concurrency + 8)
    async with httpx.AsyncClient(limits=limits, timeout=7200) as client:
        stop = asyncio.Event()
        srcs = [Source(args.seed * 1000 + i) for i in range(max(1, args.concurrency // args.group))]
        tasks = [asyncio.create_task(worker(i, client, srcs[i % len(srcs)], stop)) for i in range(args.concurrency)]
        await asyncio.sleep(args.warmup)
        t0 = time.perf_counter()  # window starts when the snapshot is requested, not when it returns
        m0 = await metrics(client)
        samples = []
        hook_done = args.hook_at < 0 or not args.hook
        while (el := time.perf_counter() - t0) < args.duration:
            await asyncio.sleep(min(10.0, args.duration - el))
            m = await metrics(client)
            samples.append((m.get("vllm:num_requests_running", 0), m.get("vllm:num_requests_waiting", 0),
                            m.get("vllm:kv_cache_usage_perc", 0) / max(1, m.get("n:vllm:kv_cache_usage_perc", 1))))
            if not hook_done and time.perf_counter() - t0 >= args.hook_at:
                hook_done = True
                proc = await asyncio.create_subprocess_shell(args.hook)
                await proc.wait()
        m1 = await metrics(client)
        t1 = time.perf_counter()
        if args.post_hook:  # e.g. an nsys capture: steady-state load, but outside the measured window
            proc = await asyncio.create_subprocess_shell(args.post_hook)
            await proc.wait()
        stop.set()
        for t in tasks:
            t.cancel()
    dt = t1 - t0
    d = {k: m1.get(k, 0) - m0.get(k, 0) for k in ("vllm:generation_tokens_total", "vllm:prompt_tokens_total",
                                                   "vllm:num_preemptions_total", "vllm:prefix_cache_hits_total",
                                                   "vllm:prefix_cache_queries_total",
                                                   "vllm:spec_decode_num_accepted_tokens_total",
                                                   "vllm:spec_decode_num_draft_tokens_total",
                                                   "vllm:spec_decode_num_drafts_total")}
    n = max(1, len(samples))
    res = {
        "tag": args.tag, "concurrency": args.concurrency, "prompt": args.prompt, "output": args.output,
        "logprobs": args.logprobs, "window_s": round(dt, 1),
        "gen_tok_s": round(d["vllm:generation_tokens_total"] / dt),
        "prompt_tok_s": round(d["vllm:prompt_tokens_total"] / dt),
        "preemptions": d["vllm:num_preemptions_total"],
        "prefix_hit_rate": round(d["vllm:prefix_cache_hits_total"] / max(1, d["vllm:prefix_cache_queries_total"]), 3),
        "spec_acceptance": round(d["vllm:spec_decode_num_accepted_tokens_total"]
                                 / max(1, d["vllm:spec_decode_num_draft_tokens_total"]), 3),
        "spec_tokens_per_step": round(1 + d["vllm:spec_decode_num_accepted_tokens_total"]
                                      / max(1, d["vllm:spec_decode_num_drafts_total"]), 3),
        "running_avg": round(sum(s[0] for s in samples) / n, 1),
        "waiting_avg": round(sum(s[1] for s in samples) / n, 1),
        "kv_usage_avg": round(sum(s[2] for s in samples) / n, 3),
        "client_done_req": done["req"], "client_errors": done["errors"],
        "client_mean_out": round(done["out"] / max(1, done["req"])),
        "client_mb_s": round(done["bytes"] / 1e6 / (t1 - t0 + args.warmup), 1),
    }
    print("RESULT " + json.dumps(res), flush=True)


asyncio.run(main())
