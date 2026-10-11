"""Refuse an RL launch whose inference request slots cannot serve its rollouts.

Every rollout in flight may be waiting on a model call at once, so the replicas'
request slots (num_infer_replicas x vLLM max_num_seqs) must cover the
orchestrator's max_inflight; otherwise rollouts queue inside vLLM while their
sandboxes sit idle.
"""
import json
import sys
import tomllib
from pathlib import Path

resolved = Path(sys.argv[1])
rl = tomllib.loads((resolved.parent / "rl.toml").read_text())
orchestrator = json.loads((resolved / "orchestrator.json").read_text())
inference = json.loads((resolved / "inference.json").read_text())
replicas = int(rl.get("deployment", {}).get("num_infer_replicas", 1))
max_num_seqs = (inference.get("vllm") or {}).get("max_num_seqs")
max_inflight = (orchestrator.get("concurrency") or {}).get("max_inflight")
if max_num_seqs is None or max_inflight is None:
    print("Inference slot check skipped: max_num_seqs or max_inflight is not set")
    sys.exit(0)
slots = replicas * int(max_num_seqs)
if slots < int(max_inflight):
    raise SystemExit(
        f"Inference slots {replicas} x {max_num_seqs} = {slots} < max_inflight {max_inflight}; "
        f"set [inference.vllm] max_num_seqs >= {-(-int(max_inflight) // replicas)}"
    )
print(f"Inference slot check passed: {replicas} x {max_num_seqs} = {slots} >= max_inflight {max_inflight}")
