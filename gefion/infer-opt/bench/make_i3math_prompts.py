"""Tokenize a deterministic sample of i3-math prompts (as prime-envs i3_math builds them) for rl_load.py.

Usage: make_i3math_prompts.py <out.json> [n=512]
"""
import json
import random
import sys

from datasets import load_dataset
from transformers import AutoTokenizer

MODEL = "/dcai/projects/iu_0145/models/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"
INSTRUCTION = "Solve the following math problem. Explain your reasoning and put the final answer in \\boxed{}.\n\n"
out, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 512
rows = load_dataset("PrimeIntellect/INTELLECT-3-RL", "math", split="train")
rows = [r for r in rows if r.get("avg@8_qwen3_4b_instruct_2507") is not None]  # i3_math default filter: [0, 1]
random.Random(0).shuffle(rows)
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
prompts = []
for r in rows[:n]:
    text = tok.apply_chat_template([{"role": "user", "content": INSTRUCTION + r["question"]}],
                                   add_generation_prompt=True, enable_thinking=True, tokenize=False)
    prompts.append(tok.encode(text, add_special_tokens=False))
json.dump(prompts, open(out, "w"))
lens = sorted(len(p) for p in prompts)
print(f"{len(prompts)} prompts, tokens p50={lens[len(lens) // 2]} max={lens[-1]} -> {out}")
