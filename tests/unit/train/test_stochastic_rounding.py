import torch
from torch import nn

from prime_rl.trainer.optim.rounding import StochasticBF16Rounding


def test_stochastic_rounding_tracks_master_and_is_unbiased():
    torch.manual_seed(0)
    base = (torch.randn(4096) * 0.02).bfloat16().float()
    update = base.abs() * 2e-5 * torch.sign(torch.randn(4096))
    layer = nn.Linear(4096, 1, bias=False)
    layer.weight.data = base.clone().view(1, -1)
    rounding = StochasticBF16Rounding(layer)
    residual = rounding.params[0][2]

    served_sum = torch.zeros(4096)
    steps = 400
    for step in range(steps):
        layer.weight.data += update.view(1, -1)
        rounding.step(step)
        served = layer.weight.data.view(-1)
        assert torch.equal(served, served.bfloat16().float())
        served_sum += served

    master = layer.weight.data.view(-1) + residual.float()
    expected = base.double() + steps * update.double()
    # fp32 accumulation over 400 steps costs ~1e-3 of a bf16 step; a bf16 residual costs ~0.1.
    bf16_step = expected.abs() * 2**-7
    assert ((master.double() - expected).abs() <= 1e-2 * bf16_step).all()
    served_mean = served_sum / steps
    expected_mean = base + update * (steps + 1) / 2
    realized = ((served_mean - base) @ (expected_mean - base)) / ((expected_mean - base) @ (expected_mean - base))
    assert abs(realized.item() - 1) < 0.05
