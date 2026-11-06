"""Merge Nemotron-H checkpoints linearly in fp32 and export an HF checkpoint vLLM can serve.

Usage: merge_export.py OUT_DIR TEMPLATE_HF COEF:SOURCE [COEF:SOURCE ...]

  OUT_DIR      new directory for the merged HF checkpoint (must not exist)
  TEMPLATE_HF  HF checkpoint that supplies config, tokenizer, tensor dtypes and the MTP head
               (MTP is not trained by RL and is copied unchanged from the template)
  SOURCE       a prime-rl DCP trainer checkpoint dir (…/checkpoints/step_N/trainer, fp32 master
               weights) or an HF checkpoint dir (safetensors)

Every backbone tensor of the output is sum_i coef_i * source_i, accumulated in fp32 and cast once
to the template's dtype. Examples:
  export one checkpoint:   1:/…/step_100/trainer
  uniform average:         0.3333333:/…/step_75/trainer 0.3333333:/…/step_100/trainer 0.3333334:/…/step_125/trainer
  task arithmetic:         0.5:NV_HF -0.5:SFT_HF 1:/…/step_100/trainer   (= NV + 0.5(SFT-NV) + (R-SFT))
"""

import json
import re
import shutil
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint import load as dcp_load

from prime_rl.trainer.models.conversion_ops import apply_prime_to_hf
from prime_rl.trainer.models.nemotron_h.configuration_nemotron_h import NemotronHConfig
from prime_rl.trainer.models.nemotron_h.converting_nemotron_h import conversion_chain

DCP_PREFIX = "app.model."


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


class DcpSource:
    def __init__(self, path: Path, ops):
        self.path, self.ops = path, ops
        self.meta = FileSystemReader(str(path)).read_metadata().state_dict_metadata
        self.keys = [k[len(DCP_PREFIX) :] for k in self.meta if k.startswith(DCP_PREFIX)]

    def load(self, group: str) -> dict[str, torch.Tensor]:
        """HF-named fp32 tensors of one group ('extra' or a layer index)."""
        prime = [k for k in self.keys if group_of_prime(k) == group]
        sd = {DCP_PREFIX + k: torch.empty(self.meta[DCP_PREFIX + k].size, dtype=self.meta[DCP_PREFIX + k].properties.dtype) for k in prime}
        dcp_load(sd, storage_reader=FileSystemReader(str(self.path)), no_dist=True)
        sd = {k[len(DCP_PREFIX) :]: v.float() for k, v in sd.items()}
        return apply_prime_to_hf(sd, self.ops)


class HfSource:
    def __init__(self, path: Path):
        self.path = path
        self.weight_map = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]

    def load(self, group: str) -> dict[str, torch.Tensor]:
        keys = [k for k in self.weight_map if not k.startswith("mtp.") and group_of_hf(k) == group]
        by_file: dict[str, list[str]] = {}
        for k in keys:
            by_file.setdefault(self.weight_map[k], []).append(k)
        out = {}
        for fname, ks in by_file.items():
            with safe_open(str(self.path / fname), framework="pt") as f:
                for k in ks:
                    out[k] = f.get_tensor(k).float()
        return out


def group_of_prime(key: str) -> str:
    m = re.match(r"model\.layers\.(\d+)\.", key)
    return m.group(1) if m else "extra"


def group_of_hf(key: str) -> str:
    m = re.match(r"backbone\.layers\.(\d+)\.", key)
    return m.group(1) if m else "extra"


def main():
    out_dir, template = Path(sys.argv[1]), Path(sys.argv[2])
    specs = [(float(s.split(":", 1)[0]), Path(s.split(":", 1)[1])) for s in sys.argv[3:]]
    assert specs, "need at least one COEF:SOURCE"
    assert not out_dir.exists(), f"{out_dir} exists"
    total = sum(c for c, _ in specs)
    assert abs(total - 1) < 1e-6, f"coefficients sum to {total}, not 1"

    config = NemotronHConfig.from_pretrained(str(template))
    ops = conversion_chain(config)
    sources = [(c, DcpSource(p, ops) if (p / ".metadata").is_file() else HfSource(p)) for c, p in specs]
    t_index = json.loads((template / "model.safetensors.index.json").read_text())["weight_map"]
    backbone = [k for k in t_index if not k.startswith("mtp.")]
    groups = ["extra"] + [str(i) for i in range(config.num_hidden_layers)]
    expected = {g: sorted(k for k in backbone if group_of_hf(k) == g) for g in groups}
    assert sum(map(len, expected.values())) == len(backbone)

    tmp = out_dir.with_name(out_dir.name + ".partial")
    tmp.mkdir(parents=True, exist_ok=False)
    dtypes = {}
    for fname in sorted(set(t_index[k] for k in backbone)):
        with safe_open(str(template / fname), framework="pt") as f:
            for k in f.keys():
                if k in t_index and not k.startswith("mtp."):
                    dtypes[k] = f.get_slice(k).get_dtype()
    torch_dtype = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16}

    weight_map = {}
    for g in groups:
        t0 = time.time()
        acc = None
        for coef, src in sources:
            sd = src.load(g)
            assert sorted(sd) == expected[g], f"group {g}: key mismatch in {src.path}: " + str(
                sorted(set(sd) ^ set(expected[g]))[:5]
            )
            if acc is None:
                acc = {k: v.mul_(coef) for k, v in sd.items()}
            else:
                for k, v in sd.items():
                    assert v.shape == acc[k].shape, f"{k}: {tuple(v.shape)} vs {tuple(acc[k].shape)}"
                    acc[k].add_(v, alpha=coef)
            del sd
        fname = f"model-{'extra' if g == 'extra' else 'layer-' + g.zfill(2)}.safetensors"
        save_file({k: v.to(torch_dtype[dtypes[k]]).contiguous() for k, v in acc.items()}, str(tmp / fname), metadata={"format": "pt"})
        weight_map.update({k: fname for k in acc})
        log(f"group {g}: {len(acc)} tensors -> {fname} ({time.time() - t0:.0f}s)")
        del acc

    mtp_files = sorted(set(t_index[k] for k in t_index if k.startswith("mtp.")))
    for fname in mtp_files:
        shutil.copy2(template / fname, tmp / fname)
    weight_map.update({k: t_index[k] for k in t_index if k.startswith("mtp.")})
    assert set(weight_map) == set(t_index)
    (tmp / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}, indent=2))
    for f in template.iterdir():
        if f.is_file() and not f.name.endswith(".safetensors") and f.name not in ("model.safetensors.index.json", "export-check.json", "_READY"):
            shutil.copy2(f, tmp / f.name)
    (tmp / "merge-recipe.json").write_text(
        json.dumps({"template": str(template), "sources": [{"coef": c, "path": str(p)} for c, p in specs]}, indent=2)
    )
    tmp.rename(out_dir)
    log(f"done: {out_dir}")


if __name__ == "__main__":
    main()
