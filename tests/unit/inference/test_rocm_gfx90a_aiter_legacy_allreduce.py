from unittest.mock import Mock

import pytest
import torch

from prime_rl.inference.vllm.kernels.rocm_gfx90a_aiter_legacy_allreduce import (
    _should_use_legacy_allreduce,
)


@pytest.mark.parametrize("elements", [4096, 32768, 4_194_304])
def test_legacy_allreduce_accepts_valid_tp8_bf16_sizes(elements):
    custom = Mock()
    custom.should_custom_ar.return_value = True
    inp = torch.empty(elements, dtype=torch.bfloat16)

    assert _should_use_legacy_allreduce(custom, 8, inp)


@pytest.mark.parametrize("elements", [4095, 4_194_305])
def test_legacy_allreduce_rejects_sizes_outside_validated_window(elements):
    custom = Mock()
    custom.should_custom_ar.return_value = True
    inp = torch.empty(elements, dtype=torch.bfloat16)

    assert not _should_use_legacy_allreduce(custom, 8, inp)


def test_legacy_allreduce_rejects_unvalidated_inputs():
    custom = Mock()
    custom.should_custom_ar.return_value = True

    assert not _should_use_legacy_allreduce(
        custom,
        4,
        torch.empty(4096, dtype=torch.bfloat16),
    )
    assert not _should_use_legacy_allreduce(
        custom,
        8,
        torch.empty(4096, dtype=torch.float16),
    )
    assert not _should_use_legacy_allreduce(
        custom,
        8,
        torch.empty((4096, 2), dtype=torch.bfloat16)[:, 0],
    )


def test_legacy_allreduce_honors_aiter_capability_check():
    custom = Mock()
    custom.should_custom_ar.return_value = False
    inp = torch.empty(4096, dtype=torch.bfloat16)

    assert not _should_use_legacy_allreduce(custom, 8, inp)
