"""Check a merge_export.py output against its recipe on a sample of layers.

Usage: check_merge.py OUT_DIR [LAYER ...]   (default layers: extra 0 1 7 87)
For each sampled tensor, recompute sum_i coef_i * source_i in fp32 from the sources, cast to the
output dtype and require an exact match; also report the relative distance to every source.
"""

import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).parent))
import merge_export as me  # noqa: E402

out = Path(sys.argv[1])
layers = sys.argv[2:] or ["extra", "0", "1", "7", "87"]
recipe = json.loads((out / "merge-recipe.json").read_text())
template = Path(recipe["template"])
config = me.NemotronHConfig.from_pretrained(str(template))
ops = me.conversion_chain(config)
srcs = [
    (s["coef"], me.DcpSource(Path(s["path"]), ops) if (Path(s["path"]) / ".metadata").is_file() else me.HfSource(Path(s["path"])))
    for s in recipe["sources"]
]
wm = json.loads((out / "model.safetensors.index.json").read_text())["weight_map"]
bad = 0
for g in layers:
    loaded = [(c, s.load(g)) for c, s in srcs]
    keys = sorted(loaded[0][1])
    sample = keys if len(keys) < 20 else keys[:: max(1, len(keys) // 12)]
    for k in sample:
        with safe_open(str(out / wm[k]), framework="pt") as f:
            got = f.get_tensor(k)
        # Same operation order as merge_export.py (mul_ then add_ with alpha): with non-dyadic
        # coefficients a different order changes the last fp32 bit and flips rare bf16 roundings.
        acc = None
        for c, sd in loaded:
            acc = sd[k].clone().mul_(c) if acc is None else acc.add_(sd[k], alpha=c)
        want = acc.to(got.dtype)
        exact = torch.equal(got, want)
        bad += not exact
        dists = " ".join(
            f"{((got.float() - sd[k]).norm() / sd[k].norm().clamp_min(1e-12)).item():.2e}" for _, sd in loaded
        )
        if not exact or k == sample[0]:
            print(f"{'OK ' if exact else 'BAD'} {k} {tuple(got.shape)} {got.dtype} rel-dist to sources: {dists}")
    print(f"layer {g}: {len(sample)} sampled tensors checked")
mtp = [k for k in wm if k.startswith("mtp.")]
print(f"mtp tensors: {len(mtp)}; total tensors {len(wm)}")
print("ALL EXACT" if bad == 0 else f"{bad} MISMATCHES")
sys.exit(1 if bad else 0)
