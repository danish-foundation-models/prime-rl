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


class _Relu2Fn(torch.autograd.Function):
    """relu(x)**2 that saves only its input and allocates one tensor per pass.

    Eager ``F.relu(x).square()`` keeps a separate relu output alive for
    backward and builds two more full-size buffers in backward. In MoE experts
    these tensors are routed-tokens x expert-hidden, so they set the peak.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        out = torch.relu(x)
        return out.square_()

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> torch.Tensor:
        (x,) = ctx.saved_tensors
        grad_x = torch.relu(x)
        return grad_x.mul_(2).mul_(grad)


class Relu2(Activation):
    @staticmethod
    def apply(gate: torch.Tensor | None, up: torch.Tensor) -> torch.Tensor:
        if gate is not None:
            return F.relu(gate).square() * up
        return _Relu2Fn.apply(up)


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
