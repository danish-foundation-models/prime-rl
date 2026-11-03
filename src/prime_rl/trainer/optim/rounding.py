import zlib

import torch
from torch import nn
from torch.distributed.tensor import DTensor

BF16_DROPPED_BITS = 1 << 16
CHUNK_ELEMENTS = 1 << 26


def stochastic_round_to_bf16_(x: torch.Tensor, generator: torch.Generator) -> None:
    """Round an fp32 tensor in place onto the bf16 grid, up with probability equal to its
    distance from the grid point below divided by the grid spacing (unbiased)."""
    # Adding uniform noise below bit 16 of the fp32 pattern, then truncating those bits,
    # rounds the magnitude up with exactly that probability, for either sign.
    bits = x.view(torch.int32)
    noise = torch.randint(0, BF16_DROPPED_BITS, bits.shape, dtype=torch.int32, device=bits.device, generator=generator)
    bits.add_(noise).bitwise_and_(-BF16_DROPPED_BITS)


class StochasticBF16Rounding:
    """Serve an unbiased bf16 copy of an exact fp32 master (arXiv 2610.41073).

    Round-to-nearest casts drop any accumulated update smaller than half a bf16 step,
    so at small learning rates the bf16 copy the trainer computes with and broadcasts
    stays near its initial values while the fp32 master moves. Here each trainable fp32
    parameter holds the served copy, on the bf16 grid, and a residual holds the rest of
    the master: master = param + residual. After every optimizer step the master is
    re-rounded stochastically, so the forward pass, the loss and the broadcast all use
    the served copy, it is unbiased each step, and it stays within one bf16 step of the
    exact master. The residual is fp32: per-step updates are ~1e-3 of a bf16 step, and a
    bf16 residual's rounding error accumulates to a sizeable fraction of the drift. It lives
    in pinned CPU memory and streams through the GPU once per step, so it costs no GPU memory.

    Residuals are not checkpointed: a resume restarts from the saved served copy, a
    one-time error below one bf16 step per element.

    The random stream is keyed by step, parameter name and the shard's coordinates on
    sharded mesh dims, so replicas of a shard round identically.
    """

    def __init__(self, model: nn.Module, seed: int = 0):
        self.seed = seed
        self.params: list[tuple[str, torch.Tensor, torch.Tensor, tuple[int, ...]]] = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            local = param.to_local() if isinstance(param, DTensor) else param
            if local.dtype != torch.float32 or local.numel() == 0:
                continue
            coords = ()
            if isinstance(param, DTensor):
                coordinate = param.device_mesh.get_coordinate()
                coords = tuple(coordinate[i] for i, placement in enumerate(param.placements) if placement.is_shard())
            residual = torch.zeros(local.numel(), dtype=torch.float32, pin_memory=local.is_cuda)
            self.params.append((name, local, residual, coords))

    @torch.no_grad()
    def step(self, step: int) -> None:
        for name, local, residual, coords in self.params:
            generator = torch.Generator(device=local.device)
            generator.manual_seed(zlib.crc32(f"{self.seed}/{step}/{name}/{coords}".encode()))
            for served, rest in zip(local.view(-1).split(CHUNK_ELEMENTS), residual.split(CHUNK_ELEMENTS)):
                master = served + rest.to(served.device, non_blocking=True)
                served.copy_(master)
                stochastic_round_to_bf16_(served, generator)
                rest.copy_(master.sub_(served))
