# SPDX-License-Identifier: Apache-2.0

import functools
import inspect

_DISPATCH_MARKER = "_prime_rl_gfx90a_nemotron_moe_config"
_W1_SHAPE = (64, 2688, 1024)
_W2_SHAPE = (64, 1024, 2688)
_TOP_K = 22
_MIN_TOKENS = 4096
_MAX_TOKENS = 32768
_TUNED_CONFIG = {
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 256,
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 1,
    "SPLIT_K": 1,
    "num_warps": 8,
    "num_stages": 2,
    "waves_per_eu": 1,
    "matrix_instr_nonkdim": 16,
    "kpack": 1,
}


def can_dispatch_nemotron_moe_config(
    w1_shape,
    w2_shape,
    top_k,
    dtype,
    tokens,
    block_shape,
) -> bool:
    return (
        tuple(w1_shape) == _W1_SHAPE
        and tuple(w2_shape) == _W2_SHAPE
        and top_k == _TOP_K
        and dtype is None
        and type(tokens) is int
        and _MIN_TOKENS <= tokens <= _MAX_TOKENS
        and block_shape is None
    )


def wrap_nemotron_moe_config(original):
    if getattr(original, _DISPATCH_MARKER, False):
        return original

    signature = inspect.signature(original)

    @functools.wraps(original)
    def wrapped(
        w1_shape,
        w2_shape,
        top_k,
        dtype,
        M,
        block_shape=None,
    ):
        if can_dispatch_nemotron_moe_config(
            w1_shape,
            w2_shape,
            top_k,
            dtype,
            M,
            block_shape,
        ):
            return _TUNED_CONFIG.copy()
        return original(
            w1_shape,
            w2_shape,
            top_k,
            dtype,
            M,
            block_shape,
        )

    if inspect.signature(wrapped) != signature:
        raise RuntimeError("Nemotron MoE config wrapper changed the vLLM selector signature")
    setattr(wrapped, _DISPATCH_MARKER, True)
    return wrapped
