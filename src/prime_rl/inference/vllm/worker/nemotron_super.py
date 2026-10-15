"""vLLM worker that audits a native Nemotron-3-Super checkpoint after TP8/EP8 loading.

Use it with `inference.vllm.worker_cls =
"prime_rl.inference.vllm.worker.nemotron_super.VerifiedSuperWorker"`. After the
initial load it compares representative embedding, LM head, expert and Mamba
tensors on every rank with the checkpoint shards and fails fast on a mismatch.
"""

import json
from pathlib import Path

import torch
from safetensors import safe_open
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.v1.worker.gpu_worker import Worker


class VerifiedSuperWorker(Worker):
    def load_model(self, *, load_dummy_weights=False):
        super().load_model(load_dummy_weights=load_dummy_weights)
        if load_dummy_weights:
            raise RuntimeError("Super checkpoint audit cannot use dummy weights")
        params = dict(self.model_runner.model.named_parameters())
        root = Path(self.model_config.model)
        index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
        rank = get_tensor_model_parallel_rank()
        checks = []

        def check(name, actual, rows=None, columns=None):
            with safe_open(str(root / index[name]), framework="pt", device="cpu") as f:
                tensor = f.get_slice(name)
                if rows is None:
                    expected = tensor[:]
                elif columns is None:
                    expected = tensor[rows]
                else:
                    expected = tensor[rows, columns]
            got = actual.detach().cpu().to(expected.dtype)
            if not torch.equal(got, expected):
                raise RuntimeError(
                    f"Loaded Super weight mismatch: rank={rank} {name} shape={got.shape}/{expected.shape}"
                )
            checks.append(name)

        for name, raw in (("model.embed_tokens.weight", "backbone.embeddings.weight"), ("lm_head.weight", "lm_head.weight")):
            if name not in params:
                raise RuntimeError(f"Missing expected Super parameter {name}; first names: {list(params)[:12]}")
            p = params[name]
            start = rank * p.shape[0]
            check(raw, p[:8], slice(start, start + 8))

        expert_names = [n for n in params if n.endswith("experts.routed_experts.w13_weight")]
        if not expert_names:
            raise RuntimeError("No Super expert tensors found")
        for name in (expert_names[0], expert_names[-1]):
            prefix = name.split(".experts.")[0].replace("model.", "backbone.", 1)
            for suffix, raw_suffix in (("w13_weight", "up_proj.weight"), ("w2_weight", "down_proj.weight")):
                p = params[name.rsplit(".", 1)[0] + "." + suffix]
                for local in (0, p.shape[0] - 1):
                    if p.shape[0] == 64:  # EP8: 64 of 512 experts per rank
                        expert, rows, columns = rank * 64 + local, slice(0, 8), None
                    elif p.shape[0] == 512:  # TP-sharded experts
                        expert = local
                        if suffix == "w13_weight":
                            rows, columns = slice(rank * p.shape[1], rank * p.shape[1] + 8), None
                        else:
                            rows, columns = slice(0, 8), slice(rank * p.shape[2], (rank + 1) * p.shape[2])
                    else:
                        raise RuntimeError("Unsupported Super expert partition")
                    check(f"{prefix}.experts.{expert}.{raw_suffix}", p[local, :8], rows, columns)

        mixers = [
            (n, m) for n, m in self.model_runner.model.named_modules() if hasattr(m, "conv_weights") and hasattr(m, "A")
        ]
        for name, mixer in (mixers[0], mixers[-1]):
            prefix = name.replace("model.", "backbone.", 1)
            for field, sizes in (
                ("in_proj.weight", [8192, 8192, 1024, 1024, 128]),
                ("conv1d.weight", [8192, 1024, 1024]),
                ("conv1d.bias", [8192, 1024, 1024]),
            ):
                local_offset = global_offset = 0
                p = params[name + "." + field]
                for size in sizes:
                    start = global_offset + rank * (size // 8)
                    check(prefix + "." + field, p[local_offset : local_offset + 8], slice(start, start + 8))
                    local_offset += size // 8
                    global_offset += size
            check(prefix + ".norm.weight", mixer.norm.weight, slice(rank * 1024, (rank + 1) * 1024))
            for field in ("D", "dt_bias"):
                check(prefix + "." + field, getattr(mixer, field), slice(rank * 16, (rank + 1) * 16))
            raw = prefix + ".A_log"
            with safe_open(str(root / index[raw]), framework="pt", device="cpu") as f:
                expected = -torch.exp(f.get_slice(raw)[rank * 16 : (rank + 1) * 16].float())
            if not torch.allclose(mixer.A.detach().cpu(), expected, rtol=1e-6, atol=1e-7):
                raise RuntimeError("Mamba A transform mismatch")
            if not torch.equal(mixer.conv_weights, mixer.conv1d.weight.view_as(mixer.conv_weights)):
                raise RuntimeError("Stale Mamba convolution buffer")
            checks.append(raw)
        print("Super native weight audit passed: " + json.dumps({"rank": rank, "checks": checks}), flush=True)
