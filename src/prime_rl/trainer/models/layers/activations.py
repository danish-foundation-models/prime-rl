from typing import Literal, Protocol

import torch
import torch.nn.functional as F


class Activation(Protocol):
    @staticmethod
    def apply(gate: torch.Tensor | None, up: torch.Tensor) -> torch.Tensor: ...


class ClampedSwiglu(Activation):
    @staticmethod
    def apply(gate: torch.Tensor | None, up: torch.Tensor) -> torch.Tensor:
        assert gate is not None
        gate = gate.clamp(max=7.0)
        up = up.clamp(min=-7.0, max=7.0)
        return (up + 1) * gate * torch.sigmoid(gate * 1.702)


class ClampedSilu(Activation):
    """SwiGLU with the gate clamped to `max=limit` and up to `[-limit, limit]` (DeepSeek V4)."""

    def __init__(self, limit: float) -> None:
        self.limit = limit

    def apply(self, gate: torch.Tensor | None, up: torch.Tensor) -> torch.Tensor:
        assert gate is not None
        gate = gate.clamp(max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        return F.silu(gate) * up


class _Relu2Function(torch.autograd.Function):
    """relu(x)**2 that saves only `x` for backward.

    Eager `F.relu(x).square()` keeps the relu output for backward and builds two
    more full-size buffers in backward. In routed experts these are
    routed-tokens x hidden, so they dominate the activation peak.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return torch.relu(x).square_()

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> torch.Tensor:
        (x,) = ctx.saved_tensors
        return torch.relu(x).mul_(2).mul_(grad)


class Relu2(Activation):
    @staticmethod
    def apply(gate: torch.Tensor | None, up: torch.Tensor) -> torch.Tensor:
        if gate is not None:
            return F.relu(gate).square() * up
        return _Relu2Function.apply(up)


class Silu(Activation):
    @staticmethod
    def apply(gate: torch.Tensor | None, up: torch.Tensor) -> torch.Tensor:
        if gate is None:
            return F.silu(up)
        return F.silu(gate) * up


ActivationType = Literal["silu", "relu2", "clamped_swiglu"]

ActivationDispatch: dict[ActivationType, type[Activation]] = {
    "clamped_swiglu": ClampedSwiglu,
    "silu": Silu,
    "relu2": Relu2,
}
