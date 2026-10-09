"""Check optimized Super route capture before enabling trainer replay."""

import argparse
import asyncio
import base64
import json
import time
from pathlib import Path

import numpy as np
from openai import AsyncOpenAI
from renderers import Nemotron3RendererConfig, create_renderer
from renderers.base import load_tokenizer
from renderers.client import generate


async def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    config = json.loads((Path(args.model) / "config.json").read_text())
    layer_types = config["layers_block_type"]
    moe_layers = [i for i, kind in enumerate(layer_types) if kind == "moe"]
    other_layers = [i for i, kind in enumerate(layer_types) if kind != "moe"]
    renderer = create_renderer(load_tokenizer(args.model), Nemotron3RendererConfig(enable_thinking=True))
    prompts = [
        ("short", [{"role": "user", "content": "Explain why unit tests are useful."}]),
        (
            "multiturn",
            [
                {"role": "user", "content": "What is a Python dictionary?"},
                {"role": "assistant", "content": "It maps keys to values."},
                {"role": "user", "content": "Give a short example."},
            ],
        ),
        ("long", [{"role": "user", "content": ("def add(x, y): return x + y\n" * 650) + "\nExplain this function."}]),
    ]
    report = {"model": args.model, "moe_layers": moe_layers, "cases": []}
    async with AsyncOpenAI(base_url=args.base_url, api_key="EMPTY", timeout=600, max_retries=0) as client:
        for name, messages in prompts:
            previous = None
            for repeat in range(2):
                started = time.monotonic()
                row = await generate(
                    client=client,
                    renderer=renderer,
                    model=args.model,
                    messages=messages,
                    sampling_params={"temperature": 0, "max_tokens": 64},
                    cache_salt=f"router-probe-{name}",
                )
                payload = row["routed_experts"]
                assert payload is not None, "Inference returned no routed experts"
                routes = np.frombuffer(base64.b64decode(payload["data"]), dtype=payload["dtype"]).reshape(
                    payload["shape"]
                )
                total = len(row["prompt_ids"]) + len(row["completion_ids"])
                start = int(payload.get("start", 0))
                assert start == 0, f"Incomplete cached-prefix routing: {start=}"
                assert routes.shape == (total - 1, len(layer_types), config["num_experts_per_tok"]), routes.shape
                selected = routes[:, moe_layers, :].astype(np.int32)
                assert selected.min() >= 0 and selected.max() < config["n_routed_experts"]
                assert np.all(np.diff(np.sort(selected, axis=-1), axis=-1) > 0), "Missing or duplicate expert choices"
                assert np.all(routes[:, other_layers, :] == 0), "Non-MoE layer routing is not zero"
                assert np.isfinite(row["completion_logprobs"]).all()
                case = {
                    "name": name,
                    "repeat": repeat,
                    "seconds": time.monotonic() - started,
                    "prompt_tokens": len(row["prompt_ids"]),
                    "completion_tokens": len(row["completion_ids"]),
                    "route_shape": list(routes.shape),
                    "route_dtype": str(routes.dtype),
                    "route_bytes": routes.nbytes,
                }
                np.savez_compressed(
                    args.output / f"{name}-{repeat}.npz",
                    routes=routes,
                    token_ids=row["prompt_ids"] + row["completion_ids"],
                    completion_logprobs=row["completion_logprobs"],
                )
                if previous is not None:
                    old_ids, old_probs, old_routes = previous
                    case["repeat_token_match"] = old_ids == row["completion_ids"]
                    case["repeat_routes_match"] = bool(np.array_equal(old_routes, routes))
                    if case["repeat_token_match"]:
                        case["repeat_max_logprob_delta"] = float(
                            np.max(np.abs(old_probs - np.asarray(row["completion_logprobs"])))
                        )
                    assert case["repeat_token_match"], case
                    assert case["repeat_routes_match"], case
                previous = row["completion_ids"], np.asarray(row["completion_logprobs"]), routes.copy()
                report["cases"].append(case)
                (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(case), flush=True)
    report["passed"] = True
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8125/v1")
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(main(parser.parse_args()))
