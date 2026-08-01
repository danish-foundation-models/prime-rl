import pytest
import torch

from prime_rl.inference.vllm.kernels.rocm_cdna2_int8_linear import (
    GFX90A_INT8_LINEAR_CONFIGS,
    GFX90A_INT8_TOKEN_BUCKETS,
    RUNTIME_WEIGHT_SCALE_DTYPE,
    SOURCE_GROUP_K,
    SOURCE_GROUP_N,
    TARGET_GROUP_K,
    Gfx90aInt8LinearConfig,
    _validate_linear_inputs,
    dequantize_block_weight,
    select_gfx90a_int8_linear_config,
    transcode_ocp_fp8_weight,
    transcode_ocp_fp8_weight_batch,
)


def make_checkpoint_weight(
    out_features: int,
    in_features: int,
    *,
    seed: int = 17,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    padded_n = ((out_features + SOURCE_GROUP_N - 1) // SOURCE_GROUP_N) * SOURCE_GROUP_N
    padded_k = ((in_features + SOURCE_GROUP_K - 1) // SOURCE_GROUP_K) * SOURCE_GROUP_K
    weight = torch.zeros((padded_n, padded_k), dtype=torch.float32)
    weight[:out_features, :in_features] = torch.randn((out_features, in_features), generator=generator)
    blocks = weight.reshape(
        padded_n // SOURCE_GROUP_N,
        SOURCE_GROUP_N,
        padded_k // SOURCE_GROUP_K,
        SOURCE_GROUP_K,
    )
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    scale = blocks.abs().amax(dim=(1, 3)).clamp_min(1.0e-12) / fp8_max
    quantized = (
        (blocks / scale[:, None, :, None]).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn).reshape(padded_n, padded_k)
    )
    return quantized[:out_features, :in_features].contiguous(), scale.contiguous()


def test_transcode_handles_partial_source_and_target_blocks():
    weight, source_scale = make_checkpoint_weight(131, 173)

    quantized, target_scale = transcode_ocp_fp8_weight(weight, source_scale, chunk_rows=17)

    assert quantized.dtype == torch.int8
    assert quantized.shape == weight.shape
    assert target_scale.dtype == RUNTIME_WEIGHT_SCALE_DTYPE
    assert target_scale.shape == (131, 6)
    expected = dequantize_block_weight(
        weight,
        source_scale,
        group_n=SOURCE_GROUP_N,
        group_k=SOURCE_GROUP_K,
    )
    actual = dequantize_block_weight(
        quantized,
        target_scale,
        group_n=1,
        group_k=TARGET_GROUP_K,
    )
    relative_l2 = torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(expected)
    assert relative_l2.item() < 0.01


def test_transcode_is_chunk_size_invariant():
    weight, source_scale = make_checkpoint_weight(257, 289)

    quantized_small, scale_small = transcode_ocp_fp8_weight(weight, source_scale, chunk_rows=7)
    quantized_large, scale_large = transcode_ocp_fp8_weight(weight, source_scale, chunk_rows=256)

    assert torch.equal(quantized_small, quantized_large)
    assert torch.equal(scale_small, scale_large)


def test_transcode_preserves_zero_blocks_with_finite_scales():
    weight = torch.zeros((3, 65), dtype=torch.float8_e4m3fn)
    source_scale = torch.ones((1, 1), dtype=torch.float32)

    quantized, target_scale = transcode_ocp_fp8_weight(weight, source_scale)

    assert torch.count_nonzero(quantized).item() == 0
    assert torch.isfinite(target_scale).all()
    assert torch.all(target_scale > 0)


def test_batched_cpu_reference_matches_independent_matrix_conversion():
    first_weight, first_scale = make_checkpoint_weight(17, 65, seed=101)
    second_weight, second_scale = make_checkpoint_weight(17, 65, seed=102)
    weights = torch.stack((first_weight, second_weight))
    source_scales = torch.stack((first_scale, second_scale))

    batched_weight, batched_scale = transcode_ocp_fp8_weight_batch(
        weights,
        source_scales,
    )
    expected = [transcode_ocp_fp8_weight(weight, scale) for weight, scale in zip(weights, source_scales, strict=True)]

    assert torch.equal(
        batched_weight,
        torch.stack([item[0] for item in expected]),
    )
    assert torch.equal(
        batched_scale,
        torch.stack([item[1] for item in expected]),
    )


def test_transcode_rejects_runtime_scale_overflow():
    weight = torch.full((1, 32), 448.0, dtype=torch.float8_e4m3fn)
    source_scale = torch.full(
        (1, 1),
        torch.finfo(torch.float32).max,
        dtype=torch.float32,
    )

    with pytest.raises(ValueError, match="overflow FP16"):
        transcode_ocp_fp8_weight(weight, source_scale)


def test_transcode_rounds_exact_half_ties_away_from_zero():
    weight = torch.zeros((1, 32), dtype=torch.float8_e4m3fn)
    weight[0, :3] = torch.tensor(
        [0.00390625, 0.001953125, -0.001953125],
        dtype=torch.float8_e4m3fn,
    )
    source_scale = torch.tensor([[31.75]], dtype=torch.float32)

    quantized, target_scale = transcode_ocp_fp8_weight(weight, source_scale)

    assert target_scale[0, 0].item() == 2.0**-10
    assert quantized[0, :3].tolist() == [127, 64, -64]


@pytest.mark.parametrize(
    ("weight", "scale", "kwargs", "message"),
    (
        (
            torch.zeros((3, 64), dtype=torch.float32),
            torch.ones((1, 1)),
            {},
            "OCP torch.float8_e4m3fn",
        ),
        (
            torch.zeros((3, 64), dtype=torch.float8_e4m3fn),
            torch.ones((2, 1)),
            {},
            "weight_scale must have shape",
        ),
        (
            torch.zeros((3, 64), dtype=torch.float8_e4m3fn),
            torch.ones((1, 1)),
            {"chunk_rows": 0},
            "chunk_rows must be a positive integer",
        ),
    ),
)
def test_transcode_rejects_invalid_contracts(weight, scale, kwargs, message):
    with pytest.raises(ValueError, match=message):
        transcode_ocp_fp8_weight(weight, scale, **kwargs)


def test_linear_config_dispatch_uses_only_measured_shapes_and_buckets():
    decode_one = select_gfx90a_int8_linear_config(1, 4096, 2048)
    assert (
        decode_one.block_m,
        decode_one.block_n,
        decode_one.group_size_m,
        decode_one.num_stages,
    ) == (16, 64, 8, 2)

    decode_64 = select_gfx90a_int8_linear_config(64, 4096, 2048)
    assert (
        decode_64.block_m,
        decode_64.block_n,
        decode_64.group_size_m,
        decode_64.num_stages,
    ) == (64, 64, 8, 1)

    exact = select_gfx90a_int8_linear_config(256, 4096, 2048)
    assert (exact.block_m, exact.block_n, exact.group_size_m) == (64, 128, 1)
    assert (exact.num_warps, exact.num_stages) == (4, 2)

    nearest = select_gfx90a_int8_linear_config(300, 4096, 2048)
    assert nearest is exact
    tie_uses_lower_bucket = select_gfx90a_int8_linear_config(192, 2048, 2048)
    exact_128 = select_gfx90a_int8_linear_config(128, 2048, 2048)
    assert tie_uses_lower_bucket is exact_128


def test_linear_config_table_has_every_measured_m_for_every_shape():
    assert len(GFX90A_INT8_LINEAR_CONFIGS) == 7
    assert all(tuple(configs) == GFX90A_INT8_TOKEN_BUCKETS for configs in GFX90A_INT8_LINEAR_CONFIGS.values())


def test_linear_runtime_contract_rejects_unmeasured_fp16_activations():
    with pytest.raises(ValueError, match="requires BF16 activations"):
        _validate_linear_inputs(
            torch.zeros((1, 32), dtype=torch.float16),
            torch.zeros((64, 32), dtype=torch.int8),
            torch.ones((64, 1), dtype=torch.float16),
            None,
            Gfx90aInt8LinearConfig(),
        )


@pytest.mark.parametrize(
    ("m", "n", "k", "message"),
    (
        (0, 4096, 2048, "M must be positive"),
        (2049, 4096, 2048, "gated through M=2048"),
        (128, 7, 2048, "no measured config"),
    ),
)
def test_linear_config_dispatch_fails_closed(m, n, k, message):
    with pytest.raises(ValueError, match=message):
        select_gfx90a_int8_linear_config(m, n, k)
