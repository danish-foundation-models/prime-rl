"""Exercise 128K serving and production NCCL reload with occupied cache."""

import argparse
import asyncio
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import httpx
import torch
from safetensors import safe_open
from vllm.distributed.utils import StatelessProcessGroup

from prime_rl.utils.weight_broadcast import WeightBroadcastCommunicator


def send_weights(comm, model):
    from prime_rl.transports.weights.nccl import broadcast_integer, broadcast_state_dict

    torch.cuda.set_device(0)
    index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    layers = defaultdict(list)
    for key, filename in index.items():
        if key.startswith("mtp."):
            continue
        if key.startswith("backbone.layers."):
            layer = int(key.split(".")[2])
            if layer >= 88:
                continue
        elif key in {"backbone.embeddings.weight", "backbone.norm_f.weight", "lm_head.weight"}:
            layer = -1
        else:
            raise ValueError(f"Unclassified checkpoint weight: {key}")
        layers[layer].append((key, filename))
    assert len(layers) == 89, len(layers)
    broadcast_integer(len(layers), comm)
    for layer in sorted(layers):
        state = {}
        by_file = defaultdict(list)
        for key, filename in layers[layer]:
            by_file[filename].append(key)
        for filename, keys in by_file.items():
            with safe_open(model / filename, framework="pt", device="cpu") as f:
                for key in keys:
                    state[key] = f.get_tensor(key).cuda()
        broadcast_state_dict(state, comm)
        torch.cuda.synchronize()
        del state
        print(f"SENT_LAYER {layer}", flush=True)


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

        await reload_weights("empty_cache_reload")
        first_reload = await generate(0, 32, "first-reload")
        (args.output / "generation-first-reload.json").write_text(json.dumps(first_reload, indent=2))
        assert first_reload["choices"][0]["text"] == before["choices"][0]["text"], (
            "Initial-policy output changed after identical reload"
        )
        record("single_128k_first_reload", usage=first_reload["usage"])

        concurrency = max(2, int(args.cache_gib * 1024 / 552) - 1)
        tasks = [asyncio.create_task(generate(i + 1, 4096)) for i in range(concurrency)]
        try:
            t = time.monotonic()
            peak = 0.0
            while time.monotonic() - t < 2400:
                response = await client.get("/metrics")
                response.raise_for_status()
                (args.output / "latest-metrics.txt").write_text(response.text)
                values = [
                    float(line.rsplit(" ", 1)[-1])
                    for line in response.text.splitlines()
                    if line.startswith(("vllm:kv_cache_usage_perc{", "vllm:gpu_cache_usage_perc{"))
                ]
                peak = max([peak, *values])
                for task in tasks:
                    if task.done():
                        task.result()
                if peak >= 0.80:
                    break
                if all(task.done() for task in tasks):
                    raise RuntimeError(f"Stress requests ended before sufficient occupancy: {peak}")
                await asyncio.sleep(2)
            else:
                raise TimeoutError(f"Cache occupancy target not reached: {peak}")
            record("cache_occupied", peak_usage=peak, concurrency=concurrency)
            await reload_weights("occupied_cache_reload")
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        for _ in range(30):
            await asyncio.sleep(2)
            response = await client.get("/metrics")
            response.raise_for_status()
            active = [
                float(line.rsplit(" ", 1)[-1])
                for line in response.text.splitlines()
                if line.startswith(("vllm:num_requests_running{", "vllm:num_requests_waiting{"))
            ]
            if len(active) >= 2 and sum(active) == 0:
                break
        else:
            raise RuntimeError("Stress requests did not drain before isolated comparison")
        after = await generate(0, 32, "occupied-reload")
        (args.output / "generation-occupied-reload.json").write_text(json.dumps(after, indent=2))
        assert after["usage"]["completion_tokens"] == 32
        assert after["choices"][0]["text"] == before["choices"][0]["text"], (
            "Greedy output changed after identical reload"
        )
        record("single_128k_after", usage=after["usage"])
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
    asyncio.run(main(p.parse_args()))
