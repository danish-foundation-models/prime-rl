"""Check FP32 Mamba boundary states against an unsplit prefill."""

import json

import torch
from vllm.model_executor.layers.mamba.ops.ssd_combined import mamba_chunk_scan_combined_varlen

torch.cuda.set_device(1)
torch.manual_seed(42)
device = "cuda:1"
n, cut, chunk, heads, dim, state = 7168, 6144, 256, 16, 64, 128
x = torch.randn(n, heads, dim, dtype=torch.bfloat16, device=device) * 0.1
B = torch.randn(n, 1, state, dtype=torch.bfloat16, device=device) * 0.1
C = torch.randn_like(B)
dt = torch.randn(n, heads, dtype=torch.bfloat16, device=device) * 0.1
A = -torch.ones(heads, device=device)
D = torch.ones(heads, device=device)
bias = torch.zeros(heads, device=device)


def run(start, end, initial=None):
    length = end - start
    bounds = torch.arange(0, length + 1, chunk, dtype=torch.int32, device=device)
    if bounds[-1].item() != length:
        bounds = torch.cat([bounds, torch.tensor([length], dtype=torch.int32, device=device)])
    count = len(bounds) - 1
    out = torch.empty_like(x[start:end])
    states = mamba_chunk_scan_combined_varlen(
        x[start:end],
        dt[start:end],
        A,
        B[start:end],
        C[start:end],
        chunk_size=chunk,
        D=D,
        dt_bias=bias,
        dt_softplus=True,
        seq_idx=torch.zeros(count, dtype=torch.int32, device=device),
        cu_seqlens=torch.tensor([0, length], dtype=torch.int32, device=device),
        cu_chunk_seqlens=bounds,
        last_chunk_indices=torch.tensor([count - 1], dtype=torch.int32, device=device),
        initial_states=initial,
        return_intermediate_states=True,
        out=out,
        state_dtype=torch.float32,
    )
    return out, states


full, all_states = run(0, n)
_, prefix = run(0, cut)
resumed, _ = run(cut, n, prefix[-1:].contiguous())
ref = full[cut:].float()
diff = resumed.float() - ref
result = {
    "state_max_abs": (all_states[cut // chunk - 1] - prefix[-1]).abs().max().item(),
    "output_max_abs": diff.abs().max().item(),
    "relative_l2": (diff.norm() / ref.norm()).item(),
    "equal": torch.equal(full[cut:], resumed),
}
print(json.dumps(result), flush=True)
assert result["state_max_abs"] == 0, result
assert result["relative_l2"] < 0.001, result
