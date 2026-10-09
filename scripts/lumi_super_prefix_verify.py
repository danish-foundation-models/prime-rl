"""Check real prefix-cache hits and cold/warm generation parity over token requests."""

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx
from renderers import Nemotron3RendererConfig, create_renderer
from renderers.base import load_tokenizer
from renderers.client import generate, parse_generate_response


async def main(args):
    tokenizer = load_tokenizer(args.model)
    renderer = create_renderer(tokenizer, Nemotron3RendererConfig(enable_thinking=True))
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"passed": False, "cases": []}
    source = (Path(__file__).resolve().parents[1] / "src/prime_rl/trainer/rl/train.py").read_text()
    # Keep the rendered dialogue boundary while varying the code context length.
    unit = tokenizer.encode(source)
    response_usage = {}

    async def capture_usage(response):
        if response.request.url.path.endswith("/inference/v1/generate"):
            data = parse_generate_response(await response.aread())
            response_usage.clear()
            response_usage.update(data.get("usage") or {})

    async with httpx.AsyncClient(timeout=1200, event_hooks={"response": [capture_usage]}) as http_client:
        from openai import AsyncOpenAI

        async with AsyncOpenAI(
            base_url=args.server + "/v1", api_key="EMPTY", timeout=1200, max_retries=0, http_client=http_client
        ) as client:
            for length in args.lengths:
                prompt = tokenizer.decode((unit * (1 + length // len(unit)))[:length])
                messages = [{"role": "user", "content": prompt + "\nExplain this code briefly."}]
                salt = f"prefix-verify-{time.time_ns()}"
                first = None
                variants = [("cold", salt), ("cold_control", salt + "-control"), ("warm", salt)]
                if args.host_cache:
                    variants.append(("host_restored", salt))
                for variant, case_salt in variants:
                    external_before = 0.0
                    if variant == "host_restored":
                        for _ in range(60):
                            reset = await http_client.post(
                                args.server + "/reset_prefix_cache", params={"reset_external": "false"}
                            )
                            reset.raise_for_status()
                            if reset.json()["success"]:
                                break
                            await asyncio.sleep(1)
                        else:
                            raise RuntimeError("GPU prefix cache did not drain")
                        metrics = await http_client.get(args.server + "/metrics")
                        metrics.raise_for_status()
                        external_before = sum(
                            float(line.rsplit(" ", 1)[1])
                            for line in metrics.text.splitlines()
                            if line.startswith("vllm:external_prefix_cache_hits_total{")
                        )
                    row = await generate(
                        client=client,
                        renderer=renderer,
                        model=args.model,
                        messages=messages,
                        sampling_params={"temperature": 0, "max_tokens": 32},
                        cache_salt=case_salt,
                    )
                    cached = (response_usage.get("prompt_tokens_details") or {}).get("cached_tokens")
                    case = {
                        "length": length,
                        "prompt_tokens": len(row["prompt_ids"]),
                        "variant": variant,
                        "cache_hit_tokens": cached,
                        "ids": row["completion_ids"],
                        "logprobs": row["completion_logprobs"],
                    }
                    if variant == "host_restored":
                        metrics = await http_client.get(args.server + "/metrics")
                        metrics.raise_for_status()
                        external_after = sum(
                            float(line.rsplit(" ", 1)[1])
                            for line in metrics.text.splitlines()
                            if line.startswith("vllm:external_prefix_cache_hits_total{")
                        )
                        case["external_hit_tokens"] = external_after - external_before
                    if first is None:
                        first = case
                    else:
                        case["token_match"] = case["ids"] == first["ids"]
                        case["max_logprob_delta"] = max(
                            (abs(a - b) for a, b in zip(case["logprobs"], first["logprobs"])), default=0
                        )
                    report["cases"].append(case)
                    (args.output / "report.json").write_text(json.dumps(report, indent=2))
                    print(json.dumps({k: v for k, v in case.items() if k not in {"ids", "logprobs"}}), flush=True)
                    if first is not case:
                        assert case["token_match"], case
                    if variant == "host_restored":
                        assert case["external_hit_tokens"] > 0, "Must restore tokens from host memory"
                    if variant == "warm":
                        assert case["cache_hit_tokens"] is not None and case["cache_hit_tokens"] > 0, (
                            "Warm run must actually reuse cached tokens"
                        )
    report["passed"] = True
    (args.output / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host-cache", action="store_true")
    p.add_argument("--model", required=True)
    p.add_argument("--server", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--lengths", type=int, nargs="+", default=[7000, 32000, 130000])
    asyncio.run(main(p.parse_args()))
