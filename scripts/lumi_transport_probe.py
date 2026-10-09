"""Exercise 128K serving and production NCCL reload with occupied cache."""

import argparse
import asyncio
import os
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import httpx
import torch
import torch.distributed as dist
from lumi_resident_weights import ResidentWeights
from safetensors import safe_open
from vllm.distributed.utils import StatelessProcessGroup

from prime_rl.utils.weight_broadcast import WeightBroadcastCommunicator


def send_weights(comm, model):
    RESIDENT.send(comm)


async def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"cache_gib": args.cache_gib, "passed": False, "stages": []}

    def record(name, **values):
        report["stages"].append({"name": name, "time": time.time(), **values})
        (args.output / "report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report["stages"][-1]), flush=True)

    async with httpx.AsyncClient(base_url=args.server, timeout=1800) as client:
        for _ in range(240):
            try:
                r = await client.get("/health", timeout=3)
                if r.status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            await asyncio.sleep(5)
        else:
            raise TimeoutError("Inference readiness")
        record("ready")
        async def post(path, body):
            r = await client.post(path, json=body)
            r.raise_for_status()
            return r.json()

        await post("/pause", {})
        init = asyncio.create_task(
            post(
                "/init_broadcaster",
                dict(host=args.sender, port=29651, rank_offset=0, inference_world_size=8, timeout=1800),
            )
        )

        def make_comm():
            torch.cuda.set_device(0)
            pg = StatelessProcessGroup.create(host=args.sender, port=29651, rank=0, world_size=9, store_timeout=1800)
            return WeightBroadcastCommunicator(pg, device=0)

        comm = await asyncio.to_thread(make_comm)
        await init
        record("communicator_ready")

        await post("/resume", {})
        if args.prefix_probe:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                str(Path(__file__).with_name("lumi_super_cache_probe.py")),
                "--model",
                str(args.model),
                "--base-url",
                args.server + "/v1",
                "--output",
                str(args.output / "prefix"),
            )
            if await proc.wait() != 0:
                raise RuntimeError("Prefix probe failed")
            record("prefix_probe_complete")

        if args.prefix_strict:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                str(Path(__file__).with_name("lumi_super_prefix_verify.py")),
                "--model",
                str(args.model),
                "--server",
                args.server,
                "--output",
                str(args.output / "prefix-strict"),
                *(["--host-cache"] if args.host_cache else []),
            )
            if await proc.wait() != 0:
                raise RuntimeError("Isolated prefix parity failed")
            record("prefix_strict_complete")

        async def generate(i, count, salt=None):
            # Exact token counts without tokenization overhead; distinct prompts.
            prompt = [1000 + i] + [1000 + j % 3000 for j in range(131072 - count - 17)]
            return await post(
                "/v1/completions",
                {
                    "model": str(args.model),
                    "prompt": prompt,
                    "temperature": 0,
                    "max_tokens": count,
                    "ignore_eos": True,
                    "cache_salt": salt,
                },
            )

        before = await generate(0, 32, "initial-policy")
        (args.output / "generation-initial.json").write_text(json.dumps(before, indent=2))
        assert before["usage"]["completion_tokens"] == 32
        record("single_128k_initial", usage=before["usage"])

        async def reload_weights(label):
            await post("/pause", {})
            receiving = asyncio.create_task(post("/update_weights", {"weight_dir": "capacity-probe"}))
            t = time.monotonic()
            await asyncio.to_thread(send_weights, comm, args.model)
            await receiving
            await post("/resume", {})
            record(label, seconds=time.monotonic() - t)

        prompts = [
            "Explain why the sky looks blue in three sentences.",
            "Write a Python function that returns the second largest distinct integer in a list.",
            "A shop sells 7 boxes with 12 pencils each, then 9 loose pencils. How many pencils are sold? Explain.",
        ]
        async def semantic_probe():
            results=[]
            for prompt in prompts:
                result=await post("/v1/completions", {"model":str(args.model), "prompt":prompt,
                    "temperature":0, "max_tokens":64, "ignore_eos":True})
                results.append(result["choices"][0]["text"])
            return results
        semantic_reference=await semantic_probe()
        (args.output / "semantic-reference.json").write_text(json.dumps(semantic_reference,indent=2))
        for stage in range(len(json.loads(Path(os.environ["RELOAD_MATRIX_PATH"]).read_text()))):
            await reload_weights(f"reload_stage_{stage}")
            after = await generate(0, 32, f"reload-stage-{stage}")
            assert after["choices"][0]["text"] == before["choices"][0]["text"], "Identical reload changed greedy output"
            semantic_after=await semantic_probe()
            (args.output / f"semantic-stage-{stage}.json").write_text(json.dumps(semantic_after,indent=2))
            assert semantic_after == semantic_reference, "Reload changed semantic probe greedy outputs"
            record(f"parity_stage_{stage}", usage=after["usage"], semantic_prompts=len(prompts))
        # Synthetic per-replica serving sweep; includes prefill and response payload cost.
        for concurrency in (8,32):
            for topk in (0,32):
                async def request(i):
                    body={"model":str(args.model), "prompt":[1000+i]+[1000+j%3000 for j in range(8191)],
                        "temperature":0, "max_tokens":1024, "ignore_eos":True}
                    if topk: body["logprobs"]=topk
                    started=time.monotonic()
                    result=await post("/v1/completions",body)
                    return {"seconds":time.monotonic()-started,"usage":result["usage"]}
                await asyncio.gather(*(request(i) for i in range(concurrency)))
                started=time.monotonic()
                results=await asyncio.gather(*(request(i) for i in range(concurrency)))
                seconds=time.monotonic()-started
                output_tokens=sum(r["usage"]["completion_tokens"] for r in results)
                assert output_tokens==1024*concurrency
                record("serving_sweep",concurrency=concurrency,top_logprobs=topk,
                    seconds=seconds,output_tokens=output_tokens,output_tokens_per_wall_second=output_tokens/seconds,
                    latency_seconds=sorted(r["seconds"] for r in results),prompt_tokens=sum(r["usage"]["prompt_tokens"] for r in results))
        report["passed"] = True
        record("passed")
        (args.output / "done").write_text("passed\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--server", required=True)
    p.add_argument("--sender", required=True)
    p.add_argument("--prefix-probe", action="store_true")
    p.add_argument("--prefix-strict", action="store_true")
    p.add_argument("--host-cache", action="store_true")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cache-gib", type=int, required=True)
    args = p.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    RESIDENT = ResidentWeights(args.model, args.output / "resident-source")
    if dist.get_rank() == 0:
        asyncio.run(main(args))
        RESIDENT.command(0)
    else:
        RESIDENT.serve()
    dist.destroy_process_group()

