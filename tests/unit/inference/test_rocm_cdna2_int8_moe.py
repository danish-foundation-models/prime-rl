import pytest
import torch

from prime_rl.inference.vllm.kernels import rocm_cdna2_int8_moe as kernels
from prime_rl.inference.vllm.kernels.rocm_cdna2_int8_moe import (
    GFX90A_INT8_MOE_CONFIGS,
    Gfx90aSiluInt8Config,
    _validate_silu_inputs,
    select_gfx90a_int8_moe_config,
)


def test_silu_input_contract_reports_logical_shape_before_device():
    with pytest.raises(ValueError, match=r"shape \[tokens, 2 \* hidden\]"):
        _validate_silu_inputs(
            torch.zeros((2, 3, 64), dtype=torch.bfloat16),
            1.0e-10,
            Gfx90aSiluInt8Config(),
        )
    with pytest.raises(ValueError, match="positive and even"):
        _validate_silu_inputs(
            torch.zeros((2, 65), dtype=torch.bfloat16),
            1.0e-10,
            Gfx90aSiluInt8Config(),
        )
    with pytest.raises(ValueError, match="requires BF16"):
        _validate_silu_inputs(
            torch.zeros((2, 64), dtype=torch.float32),
            1.0e-10,
            Gfx90aSiluInt8Config(),
        )


def test_silu_input_contract_rejects_unmeasured_fp16():
    with pytest.raises(ValueError, match="requires BF16"):
        _validate_silu_inputs(
            torch.zeros((2, 64), dtype=torch.float16),
            1.0e-10,
            Gfx90aSiluInt8Config(),
        )


def test_silu_input_contract_rejects_cpu_before_launch():
    with pytest.raises(ValueError, match="ROCm GPU"):
        _validate_silu_inputs(
            torch.zeros((2, 64), dtype=torch.bfloat16),
            1.0e-10,
            Gfx90aSiluInt8Config(),
        )


@pytest.mark.parametrize(
    ("config", "message"),
    (
        (Gfx90aSiluInt8Config(block_groups=3), "positive power of two"),
        (Gfx90aSiluInt8Config(num_warps=3), "num_warps"),
        (Gfx90aSiluInt8Config(num_stages=0), "num_stages"),
    ),
)
def test_silu_launch_config_is_fail_closed(config, message):
    with pytest.raises(ValueError, match=message):
        _validate_silu_inputs(
            torch.empty((2, 64), dtype=torch.bfloat16),
            1.0e-10,
            config,
        )


def test_moe_config_dispatch_uses_only_measured_buckets():
    assert select_gfx90a_int8_moe_config(1).as_vllm_config() == {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 1,
        "SPLIT_K": 1,
        "num_warps": 4,
        "num_stages": 2,
    }
    assert select_gfx90a_int8_moe_config(64).group_size_m == 8

    exact = select_gfx90a_int8_moe_config(1024)
    assert exact.as_vllm_config() == {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 1,
        "SPLIT_K": 1,
        "num_warps": 8,
        "num_stages": 2,
    }
    assert select_gfx90a_int8_moe_config(900) is exact
    assert select_gfx90a_int8_moe_config(192) is select_gfx90a_int8_moe_config(128)
    assert select_gfx90a_int8_moe_config(2048).num_warps == 8
    assert select_gfx90a_int8_moe_config(4096).as_vllm_config() == {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
        "SPLIT_K": 1,
        "num_warps": 8,
        "num_stages": 2,
    }


def test_moe_config_table_has_every_measured_decode_and_prefill_m():
    assert tuple(GFX90A_INT8_MOE_CONFIGS) == (
        1,
        8,
        32,
        64,
        128,
        256,
        512,
        1024,
        2048,
        4096,
    )


def test_moe_runtime_symbol_binding_is_idempotent_and_fail_closed(monkeypatch):
    runtime_symbols = tuple(lambda: None for _ in range(4))
    monkeypatch.setattr(kernels, "_VLLM_FUSED_MOE_KERNEL", None)
    monkeypatch.setattr(kernels, "_VLLM_ACTIVATION_QUANTIZER", None)
    monkeypatch.setattr(kernels, "_VLLM_MOE_ALIGN_BLOCK_SIZE", None)
    monkeypatch.setattr(kernels, "_VLLM_MOE_SUM", None)

    kernels.bind_vllm_moe_symbols(*runtime_symbols)
    kernels.bind_vllm_moe_symbols(*runtime_symbols)

    assert kernels._VLLM_FUSED_MOE_KERNEL is runtime_symbols[0]
    assert kernels._VLLM_ACTIVATION_QUANTIZER is runtime_symbols[1]
    assert kernels._VLLM_MOE_ALIGN_BLOCK_SIZE is runtime_symbols[2]
    assert kernels._VLLM_MOE_SUM is runtime_symbols[3]
    with pytest.raises(ValueError, match="different vLLM fused MoE"):
        kernels.bind_vllm_moe_symbols(
            lambda: None,
            runtime_symbols[1],
            runtime_symbols[2],
            runtime_symbols[3],
        )


def test_moe_launch_disables_vllm_fp8_swap_ab(monkeypatch):
    calls = []
    monkeypatch.setattr(kernels.tl, "bfloat16", object(), raising=False)

    class FakeKernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                calls.append((grid, args, kwargs))

            return launch

    monkeypatch.setattr(kernels, "_VLLM_FUSED_MOE_KERNEL", FakeKernel())
    activation = torch.empty((2, 32), dtype=torch.int8)
    weight = torch.empty((1, 64, 32), dtype=torch.int8)
    output = torch.empty((2, 1, 64), dtype=torch.bfloat16)
    activation_scale = torch.empty((2, 1), dtype=torch.float32)
    weight_scale = torch.empty((1, 64, 1), dtype=torch.float32)

    kernels.invoke_gfx90a_block_int8_moe(
        activation,
        weight,
        output,
        activation_scale,
        weight_scale,
        None,
        torch.empty((16,), dtype=torch.int32),
        torch.empty((1,), dtype=torch.int32),
        torch.empty((1,), dtype=torch.int32),
        mul_routed_weight=False,
        top_k=1,
        config=kernels.Gfx90aInt8MoeConfig(block_m=16, block_n=64),
    )

    assert len(calls) == 1
    assert calls[0][2]["use_fp8_w8a8"] is False
    assert calls[0][2]["use_int8_w8a8"] is True
    assert calls[0][2]["SWAP_AB"] is False


def test_moe_launch_uses_local_expert_assignment_capacity(monkeypatch):
    calls = []
    monkeypatch.setattr(kernels.tl, "bfloat16", object(), raising=False)

    class FakeKernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                calls.append((grid, args, kwargs))

            return launch

    monkeypatch.setattr(kernels, "_VLLM_FUSED_MOE_KERNEL", FakeKernel())
    config = kernels.Gfx90aInt8MoeConfig(block_m=16, block_n=64)

    kernels.invoke_gfx90a_block_int8_moe(
        torch.empty((128, 32), dtype=torch.int8),
        torch.empty((16, 64, 32), dtype=torch.int8),
        torch.empty((128, 8, 64), dtype=torch.bfloat16),
        torch.empty((128, 1), dtype=torch.float32),
        torch.empty((16, 64, 1), dtype=torch.float32),
        None,
        torch.empty((4864,), dtype=torch.int32),
        torch.empty((304,), dtype=torch.int32),
        torch.empty((1,), dtype=torch.int32),
        mul_routed_weight=False,
        top_k=8,
        config=config,
        assignment_capacity=1264,
    )

    grid, args, _kwargs = calls[0]
    assert grid({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64}) == (79,)
    assert args[12] == 1264


@pytest.mark.parametrize("assignment_capacity", [0, 17])
def test_moe_launch_rejects_invalid_assignment_capacity(
    monkeypatch,
    assignment_capacity,
):
    monkeypatch.setattr(kernels, "_VLLM_FUSED_MOE_KERNEL", object())

    with pytest.raises(ValueError, match="within sorted_token_ids"):
        kernels.invoke_gfx90a_block_int8_moe(
            torch.empty((2, 32), dtype=torch.int8),
            torch.empty((1, 64, 32), dtype=torch.int8),
            torch.empty((2, 1, 64), dtype=torch.bfloat16),
            torch.empty((2, 1), dtype=torch.float32),
            torch.empty((1, 64, 1), dtype=torch.float32),
            None,
            torch.empty((16,), dtype=torch.int32),
            torch.empty((1,), dtype=torch.int32),
            torch.empty((1,), dtype=torch.int32),
            mul_routed_weight=False,
            top_k=1,
            config=kernels.Gfx90aInt8MoeConfig(block_m=16, block_n=64),
            assignment_capacity=assignment_capacity,
        )


@pytest.mark.parametrize(
    ("num_local_experts", "expected_capacity"),
    (
        (16, 1024 + 16 * 15),
        (8, 1024 + 8 * 15),
    ),
)
def test_local_expert_assignment_capacity_bounds_supported_ep_grids(
    num_local_experts,
    expected_capacity,
):
    sorted_token_ids = torch.empty((4864,), dtype=torch.int32)
    expert_map = torch.empty((256,), dtype=torch.int32)

    capacity = kernels._select_local_expert_assignment_capacity(
        sorted_token_ids,
        num_route_slots=1024,
        num_local_experts=num_local_experts,
        block_size_m=16,
        expert_map=expert_map,
        ignore_invalid_experts=True,
    )

    assert capacity == expected_capacity


def test_local_expert_assignment_capacity_falls_back_on_unproven_layouts():
    sorted_token_ids = torch.empty((4864,), dtype=torch.int32)
    expert_map = torch.empty((256,), dtype=torch.int32)
    common = {
        "sorted_token_ids": sorted_token_ids,
        "num_route_slots": 1024,
        "num_local_experts": 16,
        "block_size_m": 16,
        "expert_map": expert_map,
        "ignore_invalid_experts": True,
    }

    assert kernels._select_local_expert_assignment_capacity(**(common | {"ignore_invalid_experts": False})) is None
    assert kernels._select_local_expert_assignment_capacity(**(common | {"num_local_experts": 256})) is None
    assert (
        kernels._select_local_expert_assignment_capacity(**(common | {"sorted_token_ids": sorted_token_ids[:1263]}))
        is None
    )
    assert (
        kernels._select_local_expert_assignment_capacity(
            **(common | {"sorted_token_ids": sorted_token_ids.to(torch.int64)})
        )
        is None
    )
    assert (
        kernels._select_local_expert_assignment_capacity(
            **(
                common
                | {
                    "sorted_token_ids": torch.empty(
                        (2 * sorted_token_ids.numel(),),
                        dtype=torch.int32,
                    )[::2]
                }
            )
        )
        is None
    )
    assert kernels._select_local_expert_assignment_capacity(**(common | {"expert_map": expert_map[:255]})) is None


@pytest.mark.parametrize(
    ("num_tokens", "message"),
    ((0, "positive"), (4097, "gated through M=4096")),
)
def test_moe_config_dispatch_fails_closed(num_tokens, message):
    with pytest.raises(ValueError, match=message):
        select_gfx90a_int8_moe_config(num_tokens)


def test_moe_workspaces_alias_and_use_the_maximum_not_sum_bound():
    hidden = torch.empty((3, 7), dtype=torch.bfloat16)

    cache1, cache3, output = kernels._allocate_moe_workspaces(
        hidden,
        tokens=3,
        top_k=2,
        hidden_size=7,
        intermediate_size=2,
    )

    assert cache1.shape == (3, 2, 4)
    assert cache3.shape == (3, 2, 7)
    assert output.shape == hidden.shape
    assert cache1.is_contiguous()
    assert cache3.is_contiguous()
    assert cache1.untyped_storage().data_ptr() == cache3.untyped_storage().data_ptr()
    assert cache3.untyped_storage().nbytes() == 3 * 2 * max(4, 7) * 2


def test_fused_moe_zeroes_poisoned_nonlocal_slots_before_sum(monkeypatch):
    tokens, hidden_size, intermediate_size, top_k = 2, 4, 2, 2
    hidden = torch.zeros((tokens, hidden_size), dtype=torch.bfloat16)
    expert_map = torch.tensor([0, -1, 1, -1], dtype=torch.int32)
    config = kernels.Gfx90aInt8MoeConfig(block_m=16, block_n=64)
    monkeypatch.setattr(
        kernels,
        "_validate_fused_moe_inputs",
        lambda *_args: (tokens, hidden_size, intermediate_size, top_k),
    )
    assignment_kwargs = {}

    def fake_assignment(*_args, **kwargs):
        assignment_kwargs.update(kwargs)
        return (
            torch.zeros((64,), dtype=torch.int32),
            torch.zeros((1,), dtype=torch.int32),
            torch.zeros((1,), dtype=torch.int32),
        )

    monkeypatch.setattr(
        kernels,
        "_VLLM_ACTIVATION_QUANTIZER",
        lambda tensor, _group: (
            torch.zeros_like(tensor, dtype=torch.int8),
            torch.ones((tensor.shape[0], 1)),
        ),
    )
    monkeypatch.setattr(
        kernels,
        "_VLLM_FUSED_MOE_KERNEL",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        kernels,
        "_VLLM_MOE_ALIGN_BLOCK_SIZE",
        fake_assignment,
    )
    monkeypatch.setattr(
        kernels,
        "cdna2_silu_and_mul_group_quant_int8",
        lambda tensor, **_kwargs: (
            torch.zeros(
                (tensor.shape[0], intermediate_size),
                dtype=torch.int8,
            ),
            torch.ones((tensor.shape[0], 1)),
        ),
    )

    calls = []

    def fake_invoke(_activation, _weight, output, *_args, **kwargs):
        calls.append(
            (
                kwargs["mul_routed_weight"],
                kwargs["assignment_capacity"],
            )
        )
        if not kwargs["mul_routed_weight"]:
            output.fill_(91)
            return
        assert torch.count_nonzero(output).item() == 0
        output[:, 0].fill_(3)

    monkeypatch.setattr(kernels, "invoke_gfx90a_block_int8_moe", fake_invoke)
    monkeypatch.setattr(
        kernels,
        "_VLLM_MOE_SUM",
        lambda cache, output: output.copy_(cache.sum(dim=1)),
    )

    output = kernels.cdna2_int8_fused_experts(
        hidden,
        torch.empty((1, 4, 4), dtype=torch.int8),
        torch.empty((1, 4, 2), dtype=torch.int8),
        torch.empty((1, 4, 1)),
        torch.empty((1, 4, 1)),
        torch.ones((tokens, top_k)),
        torch.tensor([[0, 3], [1, 2]], dtype=torch.int32),
        expert_map,
        global_num_experts=4,
        config=config,
        device_is_validated=True,
    )

    assert calls == [(False, 19), (True, 19)]
    assert torch.equal(
        assignment_kwargs["topk_ids"],
        torch.tensor([[0, 3], [1, 2]], dtype=torch.int32),
    )
    assert assignment_kwargs["block_size"] == 16
    assert assignment_kwargs["num_experts"] == 4
    assert assignment_kwargs["expert_map"] is expert_map
    assert assignment_kwargs["ignore_invalid_experts"] is True
    assert torch.equal(output, torch.full_like(output, 3))
