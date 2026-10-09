"""Stream native Nemotron conversion one layer at a time into a private cache."""
import argparse
import fcntl
import gc
import hashlib
import json
import re
import resource
import tempfile
import time
from collections import defaultdict
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("source", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--threads", type=int, default=16)
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()
assert args.threads > 0 and args.output.name == "prime"
index_path = args.source / "model.safetensors.index.json"
index = json.loads(index_path.read_text())["weight_map"]
assert json.loads((args.source / "config.json").read_text())["model_type"] == "nemotron_h"
assert all((args.source / name).is_file() for name in set(index.values()))
print("CONVERSION_INPUTS", len(index), args.threads, str(args.output), flush=True)
if args.dry_run:
    raise SystemExit(0)

import torch
from safetensors.torch import load, save_file
from prime_rl.trainer.models.nemotron_h.configuration_nemotron_h import NemotronHConfig
from prime_rl.trainer.models.nemotron_h.converting_nemotron_h import conversion_chain
from prime_rl.trainer.models.conversion_ops import Stack

torch.set_num_threads(args.threads)
config = NemotronHConfig.from_pretrained(args.source, local_files_only=True)
ops = conversion_chain(config)
groups = defaultdict(list)
for name in index:
    if name.startswith("mtp."):
        continue
    match = re.match(r"backbone\.layers\.(\d+)\.", name)
    groups[int(match[1]) if match else -1].append(name)
args.output.parent.mkdir(parents=True, exist_ok=True)
with (args.output.parent / ".conversion.lock").open("a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.output.exists():
        if (args.output / ".prime-v1").is_file():
            print("CONVERSION_ALREADY_READY", flush=True)
            raise SystemExit(0)
        raise RuntimeError("Refusing existing unfinished conversion: " + str(args.output))
    stage = Path(tempfile.mkdtemp(prefix=".prime-build-", dir=args.output.parent))
    start = time.monotonic()
    output_index = {}
    checks = []
    total_bytes = 0
    loaded_keys = set()
    for group, names in sorted(groups.items()):
        state = {}
        by_shard = defaultdict(list)
        for name in names:
            by_shard[index[name]].append(name)
        for filename, selected in sorted(by_shard.items()):
            # Whole-file reads avoid fine-grained mmap faults on Lustre. Copies
            # release the shard buffer before the next file is read.
            shard = load((args.source / filename).read_bytes())
            state.update({name: shard[name].clone() for name in selected})
            del shard
        loaded_keys.update(state)
        references = {}
        for op in ops:
            if isinstance(op, Stack):
                source = op.item.replace("model.", "backbone.", 1)
                if source.format(e=0) in state:
                    for expert in (0, config.n_routed_experts - 1):
                        references[(op.stacked, expert)] = state[source.format(e=expert)].clone()
        for op in ops:
            op.hf_to_prime(state)
        for (name, expert), expected in references.items():
            assert torch.equal(state[name][expert], expected), (name, expert)
            checks.append([name, expert])
        references.clear()
        filename = f"model-layer-{group + 1:03d}.safetensors"
        assert not (set(state) & set(output_index))
        assert not any(k.startswith("mtp.") for k in state)
        total_bytes += sum(t.numel() * t.element_size() for t in state.values())
        save_file(state, stage / filename, metadata={"format": "pt"})
        output_index.update(dict.fromkeys(state, filename))
        del state
        gc.collect()
        print("CONVERTED_LAYER", group, round(time.monotonic() - start, 2),
              "max_rss_gib", round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 2), flush=True)
    assert loaded_keys == {k for k in index if not k.startswith("mtp.")}
    assert "model.embed_tokens.weight" in output_index
    assert all((stage / name).is_file() for name in set(output_index.values()))
    (stage / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total_bytes}, "weight_map": output_index}, indent=2) + "\n")
    (stage / "conversion-manifest.json").write_text(json.dumps(dict(
        source=str(args.source), source_index_sha256=hashlib.sha256(index_path.read_bytes()).hexdigest(),
        threads=args.threads, seconds=time.monotonic() - start, tensors=len(output_index),
        checks=checks, strategy="one-layer-at-a-time", source_tensors=len(loaded_keys)), indent=2) + "\n")
    (stage / ".prime-v1").touch()
    stage.rename(args.output)
    print("CONVERSION_READY", str(args.output), round(time.monotonic() - start, 2), flush=True)
