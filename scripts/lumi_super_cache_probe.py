"""Compare greedy cache-cold and cache-warm generations without router capture."""

import argparse
import asyncio
import json
import time
from pathlib import Path

import numpy as np
from openai import AsyncOpenAI
from renderers import Nemotron3RendererConfig, create_renderer
from renderers.base import load_tokenizer
from renderers.client import generate


async def main(args):
    tokenizer = load_tokenizer(args.model)
    renderer = create_renderer(tokenizer, Nemotron3RendererConfig(enable_thinking=True))
    args.output.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve().parents[1] / "src/prime_rl/trainer/rl/train.py"
    prompts = {
        "repetitive": ("def add(x, y): return x + y\n" * 650) + "\nExplain this function.",
        "code": tokenizer.decode(tokenizer.encode(source.read_text())[:7000])
        + "\nExplain what this training code does.",
    }
    report = []
    async with AsyncOpenAI(base_url=args.base_url, api_key="EMPTY", timeout=600, max_retries=0) as client:
        for name, prompt in prompts.items():
            first = None
            for variant, salt in [
                ("cold_a", name + "-a"),
                ("cold_b", name + "-b"),
                ("warm_a", name + "-a"),
                ("warm_a2", name + "-a"),
            ]:
                started = time.monotonic()
                row = await generate(
                    client=client,
                    renderer=renderer,
                    model=args.model,
                    messages=[{"role": "user", "content": prompt}],
                    sampling_params={"temperature": 0, "max_tokens": 64},
                    cache_salt=salt,
                )
                case = {
                    "name": name,
                    "variant": variant,
                    "seconds": time.monotonic() - started,
                    "prompt_tokens": len(row["prompt_ids"]),
                    "completion_ids": row["completion_ids"],
                    "completion_logprobs": row["completion_logprobs"],
                    "text": tokenizer.decode(row["completion_ids"]),
                }
                if first is None:
                    first = case
                else:
                    case["token_match"] = case["completion_ids"] == first["completion_ids"]
                    common = next(
                        (i for i, (a, b) in enumerate(zip(case["completion_ids"], first["completion_ids"])) if a != b),
                        min(len(case["completion_ids"]), len(first["completion_ids"])),
                    )
                    case["common_completion_tokens"] = common
                    if common:
                        case["common_max_logprob_delta"] = float(
                            np.max(
                                np.abs(
                                    np.asarray(case["completion_logprobs"][:common])
                                    - np.asarray(first["completion_logprobs"][:common])
                                )
                            )
                        )
                report.append(case)
                (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
                print(
                    json.dumps(
                        {k: v for k, v in case.items() if k not in ("completion_ids", "completion_logprobs", "text")}
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--base-url", default="http://127.0.0.1:8125/v1")
    p.add_argument("--output", type=Path, required=True)
    asyncio.run(main(p.parse_args()))
