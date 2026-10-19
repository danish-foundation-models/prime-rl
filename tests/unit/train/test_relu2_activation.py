import pytest
import torch
import torch.nn.functional as F

from prime_rl.trainer.models.layers.activations import Relu2


def _reference(up: torch.Tensor) -> torch.Tensor:
    return F.relu(up).square()


@pytest.mark.parametrize("dtype", [torch.float64, torch.bfloat16])
def test_relu2_matches_eager(dtype):
    torch.manual_seed(0)
    base = torch.randn(4096, 384, dtype=torch.float32) * 3
    grad = torch.randn_like(base).to(dtype)
    up_ref = base.to(dtype).requires_grad_()
    up_new = base.to(dtype).requires_grad_()
    out_ref = _reference(up_ref)
    out_new = Relu2.apply(None, up_new)
    out_ref.backward(grad)
    out_new.backward(grad)
    assert torch.equal(out_new, out_ref)
    assert torch.equal(up_new.grad, up_ref.grad)


def test_relu2_gradcheck():
    up = (torch.randn(64, 32, dtype=torch.float64) + 0.05).requires_grad_()
    assert torch.autograd.gradcheck(lambda x: Relu2.apply(None, x), (up,))


def test_relu2_saves_only_input():
    up = torch.randn(128, 64, requires_grad=True)
    out = Relu2.apply(None, up)
    saved = out.grad_fn.saved_tensors
    assert len(saved) == 1 and saved[0].data_ptr() == up.data_ptr()


def test_relu2_gated_path_unchanged():
    gate = torch.randn(32, 16)
    up = torch.randn(32, 16)
    assert torch.equal(Relu2.apply(gate, up), F.relu(gate).square() * up)
