import torch

from prime_rl.inference.vllm.padded_input_scrub import monkey_patch_vllm_padded_input_scrub

_EXPECTED_SPARSE_VLLM_DISTRIBUTION = "0.24.0+lumi_aif_gfx90a_ee0da84"
_GLM_SPARSE_MLA_TOPK = 2048


def _strict_binary_env(name: str) -> bool:
    """Read an opt-in without silently accepting a misspelled value."""
    import os

    value = os.environ.get(name)
    if value in (None, "0"):
        return False
    if value == "1":
        return True
    raise ValueError(f"{name} must be unset, '0', or '1'; got {value!r}")


def _validate_rocm_sparse_indexer_forward_hip(
    *,
    use_fp4_cache: object,
    q_quant: object,
) -> None:
    """Reject inputs outside vLLM's measured ROCm sparse-indexer ABI."""

    if use_fp4_cache:
        raise NotImplementedError("AMD platform doesn't support fp4 cache yet")
    if not isinstance(q_quant, torch.Tensor):
        raise TypeError(
            "AMD sparse_attn_indexer expects a single FP8 q_quant tensor"
        )


def _installed_vllm_distribution_version() -> str:
    from importlib.metadata import version

    return version("vllm")


def _require_exact_vllm_sparse_patch_abi(
    feature: str,
    symbols: tuple[
        tuple[
            str,
            object,
            tuple[str, ...],
            dict[str, object],
        ],
        ...,
    ],
) -> None:
    """Fail closed when a private vLLM patch boundary has changed."""
    import inspect

    installed = _installed_vllm_distribution_version()
    if installed != _EXPECTED_SPARSE_VLLM_DISTRIBUTION:
        raise RuntimeError(f"{feature} requires vLLM {_EXPECTED_SPARSE_VLLM_DISTRIBUTION!r}; found {installed!r}")

    for symbol_name, symbol, expected_names, expected_defaults in symbols:
        parameters = tuple(inspect.signature(symbol).parameters.values())
        actual_names = tuple(parameter.name for parameter in parameters)
        if actual_names != expected_names:
            raise RuntimeError(f"{feature} requires {symbol_name}{expected_names}; found parameters {actual_names}")
        for parameter in parameters:
            if parameter.kind is not inspect.Parameter.POSITIONAL_OR_KEYWORD:
                raise RuntimeError(
                    f"{feature} requires positional-or-keyword parameter "
                    f"{symbol_name}.{parameter.name}; found {parameter.kind.description}"
                )
            expected_default = expected_defaults.get(
                parameter.name,
                inspect.Parameter.empty,
            )
            if parameter.default != expected_default:
                raise RuntimeError(
                    f"{feature} requires default "
                    f"{symbol_name}.{parameter.name}={expected_default!r}; "
                    f"found {parameter.default!r}"
                )


def _require_sparse_mla_topk_tokens(topk_tokens: object, source: str) -> int:
    """Return the static GLM top-k only when metadata is a Python integer."""
    if type(topk_tokens) is not int or topk_tokens != _GLM_SPARSE_MLA_TOPK:
        raise ValueError(
            f"{source} must provide topk_tokens={_GLM_SPARSE_MLA_TOPK} as a Python int, got {topk_tokens!r}"
        )
    return topk_tokens


def apply_shared_vllm_patches():
    """vLLM general plugin: prime-rl patches that must run in every vLLM process.

    Registered as a ``vllm.general_plugins`` entry-point so it runs automatically
    in every vLLM process, including spawned workers. Note vLLM swallows plugin
    load failures (``load_plugins_by_group`` logs and continues), so a broken
    entry-point target silently skips ALL of these patches.
    """
    _install_cdna2_int8_quantization()
    _patch_lora_key_prefix()
    _patch_qwen35_moe_lora_format()
    monkey_patch_nano_v3_reasoning_parser()
    monkey_patch_qwen3_coder_param_newline_trim()
    monkey_patch_minimax_m2_think_end_passthrough()
    monkey_patch_vllm_padded_input_scrub()
    monkey_patch_return_routed_experts_with_nixl_connector()
    monkey_patch_kv_xfer_finished_tolerate_freed()
    monkey_patch_online_fp8_parameter_cast()


def monkey_patch_online_fp8_parameter_cast():
    """Pass plain tensors to vLLM's compiled block-FP8 caster.

    Layerwise reload restores ``ModelWeightParameter`` objects before online
    quantization. TorchDynamo recursively dispatches that tensor subclass while
    tracing ``per_block_cast_to_fp8``. ``Parameter.data`` shares storage but is
    a plain tensor, matching the input accepted during the initial model load.
    """
    from torch.nn import Parameter
    from vllm.model_executor.layers.quantization.online import fp8

    original_cast = fp8.per_block_cast_to_fp8
    if getattr(original_cast, "_prime_rl_unwraps_parameters", False):
        return

    def _per_block_cast_to_fp8(x, *args, **kwargs):
        if isinstance(x, Parameter):
            x = x.data
        return original_cast(x, *args, **kwargs)

    _per_block_cast_to_fp8._prime_rl_unwraps_parameters = True
    fp8.per_block_cast_to_fp8 = _per_block_cast_to_fp8


def _install_cdna2_int8_quantization() -> None:
    """Register the gfx90a INT8 config before vLLM parses model arguments."""

    import os

    if os.environ.get("PRIME_ROCM_CDNA2_INT8") in (None, "0"):
        return

    from prime_rl.inference.vllm.kernels.rocm_cdna2_int8_quant_config import (
        install_cdna2_int8_quantization,
    )

    install_cdna2_int8_quantization()


def _require_rocm_gfx90a(feature: str) -> str:
    """Fail closed before installing a kernel specialized for CDNA2."""

    if torch.version.hip is None:
        raise RuntimeError(f"{feature} requires a ROCm PyTorch build")
    device_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
    architecture = getattr(properties, "gcnArchName", "").split(":")[0]
    if architecture != "gfx90a":
        raise RuntimeError(
            f"{feature} is validated only on gfx90a; device {device_index} reports {properties.gcnArchName}"
        )
    return architecture


def _cached_rocm_sparse_indexer_context_lens(context_lens):
    from vllm.forward_context import (
        get_forward_context,
        is_forward_context_available,
    )

    if not is_forward_context_available():
        return context_lens.to(device="cpu", non_blocking=False)

    cache = get_forward_context().additional_kwargs.setdefault(
        "_prime_rl_rocm_sparse_indexer_context_lens_cpu_cache",
        {},
    )
    cache_key = (
        context_lens.device,
        context_lens.data_ptr(),
        tuple(context_lens.shape),
        tuple(context_lens.stride()),
        context_lens.storage_offset(),
        context_lens.dtype,
    )
    cached = cache.get(cache_key)
    if cached is None:
        cached = (
            context_lens,
            context_lens.to(device="cpu", non_blocking=False),
        )
        cache[cache_key] = cached
    return cached[1]


def _wrap_rocm_sparse_indexer_bulk_context_lens(original):
    """Replace per-sequence GPU scalar reads with one synchronous host copy.

    The host copy remains incompatible with full graph capture. This wrapper is
    only a speed optimization for eager or genuinely breakable piecewise paths.
    """
    import functools

    if getattr(original, "_prime_rl_bulk_context_lens", False):
        return original

    use_forward_context_cache = _strict_binary_env(
        "PRIME_ROCM_SPARSE_INDEXER_FORWARD_CONTEXT_CACHE"
    )

    @functools.wraps(original)
    def _bulk_context_lens(
        q,
        kv_cache,
        weights,
        context_lens,
        block_tables,
        max_model_len,
    ):
        if q.shape[1] == 1:
            if use_forward_context_cache:
                context_lens = _cached_rocm_sparse_indexer_context_lens(context_lens)
            else:
                context_lens = context_lens.to(device="cpu", non_blocking=False)
        return original(
            q,
            kv_cache,
            weights,
            context_lens,
            block_tables,
            max_model_len,
        )

    _bulk_context_lens._prime_rl_bulk_context_lens = True
    return _bulk_context_lens


def _rocm_sparse_indexer_triton_decode_from_env():
    enabled = _strict_binary_env("PRIME_ROCM_SPARSE_INDEXER_TRITON_DECODE")
    bulk_context_lens = _strict_binary_env(
        "PRIME_ROCM_SPARSE_INDEXER_BULK_CONTEXT_LENS"
    )
    if enabled and bulk_context_lens:
        raise ValueError(
            "PRIME_ROCM_SPARSE_INDEXER_TRITON_DECODE keeps context lengths on "
            "the GPU and cannot be combined with "
            "PRIME_ROCM_SPARSE_INDEXER_BULK_CONTEXT_LENS"
        )
    return enabled


def _rocm_sparse_indexer_prefill_k_chunk_from_env():
    import os

    raw_value = os.environ.get("PRIME_ROCM_SPARSE_INDEXER_PREFILL_K_CHUNK")
    if raw_value is None:
        return None
    try:
        chunk_size = int(raw_value)
    except ValueError as exc:
        raise ValueError("PRIME_ROCM_SPARSE_INDEXER_PREFILL_K_CHUNK must be a positive integer") from exc
    if chunk_size < 1:
        raise ValueError("PRIME_ROCM_SPARSE_INDEXER_PREFILL_K_CHUNK must be a positive integer")
    return chunk_size


def _wrap_rocm_sparse_indexer_prefill_k_chunk(original, chunk_size):
    """Bound the fallback's H x M x N prefill score by slicing K."""
    import functools

    if chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if getattr(original, "_prime_rl_prefill_k_chunk", None) == chunk_size:
        return original

    @functools.wraps(original)
    def _chunked_prefill(
        q,
        kv,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
    ):
        k_fp8, scale = kv
        seq_len_kv = k_fp8.shape[0]
        if seq_len_kv <= chunk_size:
            return original(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke)

        logits_chunks = []
        for chunk_start in range(0, seq_len_kv, chunk_size):
            chunk_end = min(chunk_start + chunk_size, seq_len_kv)
            chunk_len = chunk_end - chunk_start
            chunk_starts = (cu_seqlen_ks - chunk_start).clamp(
                min=0,
                max=chunk_len,
            )
            chunk_ends = (cu_seqlen_ke - chunk_start).clamp(
                min=0,
                max=chunk_len,
            )
            logits_chunks.append(
                original(
                    q,
                    (
                        k_fp8[chunk_start:chunk_end],
                        scale[chunk_start:chunk_end],
                    ),
                    weights,
                    chunk_starts,
                    chunk_ends,
                )
            )
        return torch.cat(logits_chunks, dim=-1)

    _chunked_prefill._prime_rl_prefill_k_chunk = chunk_size
    return _chunked_prefill


def monkey_patch_rocm_sparse_indexer_fallback():
    """Allow vLLM's built-in ROCm sparse-indexer fallback without AITER.

    GLM-5.2 uses the DeepSeek-V3.2 sparse-attention indexer. vLLM 0.24's ROCm
    implementation already contains a gfx90a-compatible fallback: Triton handles
    index-cache packing and PyTorch handles MQA logits when the optional ``aiter``
    package is absent. ``SparseAttnIndexer.forward_hip`` nevertheless rejects the
    call before that implementation can select its fallback.

    Keep this opt-in because the PyTorch logits path is intended for correctness,
    not throughput. Set ``PRIME_ROCM_SPARSE_INDEXER_FALLBACK=1`` to route the HIP
    method to the existing ``rocm_aiter_sparse_attn_indexer`` custom op; with AITER
    disabled, that op selects its own non-AITER implementations.

    ``PRIME_ROCM_SPARSE_INDEXER_BULK_CONTEXT_LENS=1`` optionally replaces the
    plain-decode fallback's per-sequence GPU scalar reads with one synchronous
    host copy per indexer layer. Set
    ``PRIME_ROCM_SPARSE_INDEXER_FORWARD_CONTEXT_CACHE=1`` alongside it to reuse
    one host copy per unique context-length tensor within each forward. Neither
    mode is full-graph compatible.

    ``PRIME_ROCM_SPARSE_INDEXER_TRITON_DECODE=1`` instead selects PRIME-RL's
    fused gfx90a OCP-FP8 paged-MQA kernel for one-token decode. It leaves
    context lengths and block tables on the GPU and is mutually exclusive with
    the bulk-host-context mode.

    ``PRIME_ROCM_SPARSE_INDEXER_PREFILL_K_CHUNK=<positive integer>`` bounds the
    non-AITER prefill fallback's temporary ``H x M x N`` score by slicing its K
    dimension and concatenating the unchanged upstream results.
    """
    import functools
    import os

    if not _strict_binary_env("PRIME_ROCM_SPARSE_INDEXER_FALLBACK"):
        return

    from vllm.logger import init_logger
    from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
    from vllm.platforms import current_platform
    from vllm.utils.torch_utils import _encode_layer_name, direct_register_custom_op
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse

    if not current_platform.is_rocm():
        return

    triton_decode = _rocm_sparse_indexer_triton_decode_from_env()
    prefill_k_chunk = _rocm_sparse_indexer_prefill_k_chunk_from_env()
    signature_symbols = [
        (
            "SparseAttnIndexer.forward_hip",
            SparseAttnIndexer.forward_hip,
            ("self", "hidden_states", "q_quant", "k", "weights"),
            {},
        ),
        (
            "rocm_aiter_sparse_attn_indexer",
            rocm_aiter_mla_sparse.rocm_aiter_sparse_attn_indexer,
            (
                "hidden_states",
                "k_cache_prefix",
                "kv_cache",
                "q_fp8",
                "k",
                "weights",
                "quant_block_size",
                "scale_fmt",
                "topk_tokens",
                "head_dim",
                "max_model_len",
                "total_seq_lens",
                "topk_indices_buffer",
                "skip_k_cache_insert",
            ),
            {"skip_k_cache_insert": False},
        ),
        (
            "rocm_aiter_sparse_attn_indexer_fake",
            rocm_aiter_mla_sparse.rocm_aiter_sparse_attn_indexer_fake,
            (
                "hidden_states",
                "k_cache_prefix",
                "kv_cache",
                "q_fp8",
                "k",
                "weights",
                "quant_block_size",
                "scale_fmt",
                "topk_tokens",
                "head_dim",
                "max_model_len",
                "total_seq_lens",
                "topk_indices_buffer",
                "skip_k_cache_insert",
            ),
            {"skip_k_cache_insert": False},
        ),
    ]
    if triton_decode:
        signature_symbols.append(
            (
                "rocm_fp8_paged_mqa_logits",
                rocm_aiter_mla_sparse.rocm_fp8_paged_mqa_logits,
                (
                    "q_fp8",
                    "kv_cache_fp8",
                    "weights",
                    "context_lens",
                    "block_tables",
                    "schedule_metadata",
                    "max_model_len",
                ),
                {},
            )
        )
    elif _strict_binary_env("PRIME_ROCM_SPARSE_INDEXER_BULK_CONTEXT_LENS"):
        signature_symbols.append(
            (
                "fp8_paged_mqa_logits_torch",
                rocm_aiter_mla_sparse.fp8_paged_mqa_logits_torch,
                (
                    "q",
                    "kv_cache",
                    "weights",
                    "context_lens",
                    "block_tables",
                    "max_model_len",
                ),
                {},
            )
        )
    if prefill_k_chunk is not None:
        signature_symbols.append(
            (
                "fp8_mqa_logits_torch",
                rocm_aiter_mla_sparse.fp8_mqa_logits_torch,
                (
                    "q",
                    "kv",
                    "weights",
                    "cu_seqlen_ks",
                    "cu_seqlen_ke",
                ),
                {},
            )
        )
    _require_exact_vllm_sparse_patch_abi(
        "PRIME_ROCM_SPARSE_INDEXER_FALLBACK",
        tuple(signature_symbols),
    )

    rocm_aiter_sparse_attn_indexer = rocm_aiter_mla_sparse.rocm_aiter_sparse_attn_indexer
    rocm_aiter_sparse_attn_indexer_fake = rocm_aiter_mla_sparse.rocm_aiter_sparse_attn_indexer_fake
    if triton_decode:
        _require_rocm_gfx90a("PRIME_ROCM_SPARSE_INDEXER_TRITON_DECODE")

        from prime_rl.inference.vllm.kernels.rocm_fp8_paged_mqa import (
            wrap_fp8_paged_mqa_logits_gfx90a,
        )

        original_logits = rocm_aiter_mla_sparse.rocm_fp8_paged_mqa_logits
        rocm_aiter_mla_sparse.rocm_fp8_paged_mqa_logits = wrap_fp8_paged_mqa_logits_gfx90a(original_logits)
        init_logger(__name__).warning(
            "Enabled PRIME_ROCM_SPARSE_INDEXER_TRITON_DECODE: using the fused gfx90a OCP-FP8 paged-MQA decode kernel."
        )
    elif os.environ.get("PRIME_ROCM_SPARSE_INDEXER_BULK_CONTEXT_LENS") == "1":
        original_logits = rocm_aiter_mla_sparse.fp8_paged_mqa_logits_torch
        patched_logits = _wrap_rocm_sparse_indexer_bulk_context_lens(original_logits)
        rocm_aiter_mla_sparse.fp8_paged_mqa_logits_torch = patched_logits
        if patched_logits is not original_logits:
            if _strict_binary_env(
                "PRIME_ROCM_SPARSE_INDEXER_FORWARD_CONTEXT_CACHE"
            ):
                init_logger(__name__).warning(
                    "Enabled PRIME_ROCM_SPARSE_INDEXER_BULK_CONTEXT_LENS with "
                    "PRIME_ROCM_SPARSE_INDEXER_FORWARD_CONTEXT_CACHE: copying "
                    "plain-decode context lengths to host once per unique tensor "
                    "per forward; eager or breakable piecewise execution only, "
                    "not full graphs."
                )
            else:
                init_logger(__name__).warning(
                    "Enabled PRIME_ROCM_SPARSE_INDEXER_BULK_CONTEXT_LENS: copying "
                    "plain-decode context lengths to host once per indexer layer; "
                    "eager or breakable piecewise execution only, not full graphs."
                )

    if prefill_k_chunk is not None:
        original_prefill = rocm_aiter_mla_sparse.fp8_mqa_logits_torch
        patched_prefill = _wrap_rocm_sparse_indexer_prefill_k_chunk(
            original_prefill,
            prefill_k_chunk,
        )
        rocm_aiter_mla_sparse.fp8_mqa_logits_torch = patched_prefill
        if patched_prefill is not original_prefill:
            init_logger(__name__).warning(
                "Enabled PRIME_ROCM_SPARSE_INDEXER_PREFILL_K_CHUNK=%d: "
                "chunking the non-AITER prefill fallback's K dimension.",
                prefill_k_chunk,
            )

    if getattr(SparseAttnIndexer.forward_hip, "_prime_rl_non_aiter_fallback", False):
        return

    @functools.wraps(rocm_aiter_sparse_attn_indexer)
    def _bounded_profile_sparse_attn_indexer(
        hidden_states,
        k_cache_prefix,
        kv_cache,
        q_fp8,
        k,
        weights,
        quant_block_size,
        scale_fmt,
        topk_tokens,
        head_dim,
        max_model_len,
        total_seq_lens,
        topk_indices_buffer,
        skip_k_cache_insert=False,
    ):
        from vllm.forward_context import get_forward_context

        # The gfx90a dummy-profile path sizes its decode workspace as
        # q_heads * max_num_batched_tokens * max_model_len (128 GiB for this
        # GLM smoke), although runtime decode rows are bounded by concurrent
        # sequences times speculative tokens. Preserve the upstream prefill
        # and transient-workspace accounting, but give its dummy call the real
        # decode-row bound. Runtime calls retain the original hidden states.
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            max_decode_tokens = int(os.environ.get("PRIME_ROCM_SPARSE_INDEXER_PROFILE_MAX_DECODE_TOKENS", "1"))
            if max_decode_tokens < 1:
                raise ValueError("PRIME_ROCM_SPARSE_INDEXER_PROFILE_MAX_DECODE_TOKENS must be at least 1")
            hidden_states = hidden_states[:max_decode_tokens]

        return rocm_aiter_sparse_attn_indexer(
            hidden_states,
            k_cache_prefix,
            kv_cache,
            q_fp8,
            k,
            weights,
            quant_block_size,
            scale_fmt,
            topk_tokens,
            head_dim,
            max_model_len,
            total_seq_lens,
            topk_indices_buffer,
            skip_k_cache_insert,
        )

    # vLLM registers this op from AiterOps.register_ops_once(), but that method
    # is guarded by ``if_aiter_supported`` even though the op implementation
    # itself has non-AITER paths. Mirror vLLM's registration for this one op.
    if not hasattr(torch.ops.vllm, "rocm_aiter_sparse_attn_indexer"):
        direct_register_custom_op(
            op_name="rocm_aiter_sparse_attn_indexer",
            op_func=_bounded_profile_sparse_attn_indexer,
            mutates_args=["topk_indices_buffer"],
            fake_impl=rocm_aiter_sparse_attn_indexer_fake,
            dispatch_key=current_platform.dispatch_key,
        )

    def _patched_forward_hip(self, hidden_states, q_quant, k, weights):
        _validate_rocm_sparse_indexer_forward_hip(
            use_fp4_cache=self.use_fp4_cache,
            q_quant=q_quant,
        )
        return torch.ops.vllm.rocm_aiter_sparse_attn_indexer(
            hidden_states,
            _encode_layer_name(self.k_cache.prefix),
            self.k_cache.kv_cache,
            q_quant,
            k,
            weights,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
            skip_k_cache_insert=self.skip_k_cache_insert,
        )

    _patched_forward_hip._prime_rl_non_aiter_fallback = True
    SparseAttnIndexer.forward_hip = _patched_forward_hip
    init_logger(__name__).warning(
        "Enabled PRIME_ROCM_SPARSE_INDEXER_FALLBACK: using vLLM's non-AITER "
        "ROCm sparse-indexer path; benchmark before production use."
    )


def _rocm_sparse_mla_prefill_q_chunk_from_env():
    import os

    raw_value = os.environ.get("PRIME_ROCM_SPARSE_MLA_PREFILL_Q_CHUNK")
    if raw_value is None:
        return None
    try:
        chunk_size = int(raw_value)
    except ValueError as exc:
        raise ValueError("PRIME_ROCM_SPARSE_MLA_PREFILL_Q_CHUNK must be a positive integer") from exc
    if chunk_size < 1:
        raise ValueError("PRIME_ROCM_SPARSE_MLA_PREFILL_Q_CHUNK must be a positive integer")
    return chunk_size


def _rocm_sparse_mla_backend_from_env():
    """Return the explicitly selected non-AITER sparse-MLA backend."""
    import os

    reference_enabled = _strict_binary_env(
        "PRIME_ROCM_SPARSE_MLA_REFERENCE_FALLBACK"
    )
    triton_enabled = _strict_binary_env("PRIME_ROCM_SPARSE_MLA_TRITON")
    if reference_enabled and triton_enabled:
        raise ValueError(
            "PRIME_ROCM_SPARSE_MLA_REFERENCE_FALLBACK and PRIME_ROCM_SPARSE_MLA_TRITON are mutually exclusive"
        )
    if triton_enabled:
        if os.environ.get("PRIME_ROCM_SPARSE_MLA_PREFILL_Q_CHUNK") is not None:
            raise ValueError(
                "PRIME_ROCM_SPARSE_MLA_PREFILL_Q_CHUNK only applies to PRIME_ROCM_SPARSE_MLA_REFERENCE_FALLBACK"
            )
        return "triton"
    if reference_enabled:
        return "reference"
    return None


def _reference_mla_sparse_prefill_output(
    reference_mla_sparse_prefill,
    q,
    kv,
    indices,
    scale,
    kv_lora_rank,
    q_chunk_size,
):
    """Evaluate reference sparse MLA with an optional query-axis memory bound."""
    if q_chunk_size is None or q.shape[0] <= q_chunk_size:
        output, _ = reference_mla_sparse_prefill(
            q,
            kv,
            indices,
            scale,
            kv_lora_rank,
        )
        return output

    output_chunks = []
    for chunk_start in range(0, q.shape[0], q_chunk_size):
        chunk_end = min(chunk_start + q_chunk_size, q.shape[0])
        output, _ = reference_mla_sparse_prefill(
            q[chunk_start:chunk_end],
            kv,
            indices[chunk_start:chunk_end],
            scale,
            kv_lora_rank,
        )
        output_chunks.append(output)
    return torch.cat(output_chunks, dim=0)


def _rocm_sparse_mla_runtime_mismatches(vllm_config: object) -> tuple[str, ...]:
    """Validate the complete runtime contract measured for sparse GLM-5.2."""

    from prime_rl.inference.vllm.kernels.rocm_gfx90a_mla_bmm_integration import (
        glm52_mla_runtime_mismatches,
    )

    mismatches = list(glm52_mla_runtime_mismatches(vllm_config))
    parallel_config = getattr(vllm_config, "parallel_config", None)
    expected_topology = {
        "data_parallel_size": 2,
        "pipeline_parallel_size": 1,
        "enable_expert_parallel": True,
    }
    for field, expected in expected_topology.items():
        actual = getattr(parallel_config, field, None)
        if actual != expected:
            mismatches.append(f"{field}={actual!r} (expected {expected!r})")

    cache_config = getattr(vllm_config, "cache_config", None)
    cache_dtype = getattr(cache_config, "cache_dtype", None)
    if cache_dtype not in ("auto", "bfloat16"):
        mismatches.append(f"cache_config.cache_dtype={cache_dtype!r} (expected 'auto' or 'bfloat16')")
    return tuple(mismatches)


def _rocm_sparse_mla_cache_spec_mismatches(kv_cache_spec: object) -> tuple[str, ...]:
    """Validate the BF16 cache storage consumed by the custom sparse kernel."""

    mismatches = []
    cache_dtype = getattr(kv_cache_spec, "dtype", None)
    if cache_dtype is not torch.bfloat16:
        mismatches.append(f"kv_cache_spec.dtype={cache_dtype!r} (expected torch.bfloat16)")
    cache_dtype_str = getattr(kv_cache_spec, "cache_dtype_str", None)
    if cache_dtype_str not in (None, "auto", "bfloat16"):
        mismatches.append(
            f"kv_cache_spec.cache_dtype_str={cache_dtype_str!r} "
            "(expected None, 'auto', or 'bfloat16')"
        )
    return tuple(mismatches)


def monkey_patch_rocm_sparse_mla_fallback():
    """Select a non-AITER sparse-MLA attention backend on ROCm.

    vLLM 0.24 unconditionally selects ``ROCM_AITER_MLA_SPARSE`` for sparse MLA
    models on ROCm. Its indexer has non-AITER implementations, but its metadata
    builder and attention kernel still import AITER. AITER only supports MI3xx
    and newer GPUs, so that backend cannot start on LUMI's gfx90a MI250X GPUs.

    Both opt-in backends keep vLLM's paged sparse-index construction and disable
    only its AITER-specific persistent metadata:

    * ``PRIME_ROCM_SPARSE_MLA_TRITON=1`` selects PRIME-RL's bounded-memory
      gfx90a BF16 kernel. It consumes the ragged index metadata directly,
      materializes only a bounded ``M x 2048`` int32 index workspace, and never
      gathers an ``M x 2048 x 576`` KV tensor.
    * ``PRIME_ROCM_SPARSE_MLA_REFERENCE_FALLBACK=1`` pads the ragged selected
      indices and evaluates vLLM's existing PyTorch reference attention. This
      path is intended for correctness smoke tests, not production throughput.

    The selectors are mutually exclusive.

    ``PRIME_ROCM_SPARSE_MLA_PREFILL_Q_CHUNK=<positive integer>`` optionally
    bounds the reference kernel's ``sq x topk x dqk`` gather by slicing queries
    and their selected-token indices along ``sq``. It is only valid with the
    reference backend. Unset preserves the upstream single-call behavior.
    """
    import functools

    backend = _rocm_sparse_mla_backend_from_env()
    if backend is None:
        return

    from vllm.logger import init_logger
    from vllm.platforms import current_platform
    from vllm.v1.attention.backends.mla import rocm_aiter_mla_sparse

    if not current_platform.is_rocm():
        return

    if backend == "triton":
        _require_rocm_gfx90a("PRIME_ROCM_SPARSE_MLA_TRITON")

    prefill_q_chunk = _rocm_sparse_mla_prefill_q_chunk_from_env() if backend == "reference" else None
    builder_cls = rocm_aiter_mla_sparse.ROCMAiterMLASparseMetadataBuilder
    impl_cls = rocm_aiter_mla_sparse.ROCMAiterMLASparseImpl
    _require_exact_vllm_sparse_patch_abi(
        f"PRIME_ROCM_SPARSE_MLA_{backend.upper()}",
        (
            (
                "ROCMAiterMLASparseMetadataBuilder.__init__",
                builder_cls.__init__,
                ("self", "kv_cache_spec", "layer_names", "vllm_config", "device"),
                {},
            ),
            (
                "ROCMAiterMLASparseMetadataBuilder.build",
                builder_cls.build,
                (
                    "self",
                    "common_prefix_len",
                    "common_attn_metadata",
                    "fast_build",
                ),
                {"fast_build": False},
            ),
            (
                "ROCMAiterMLASparseImpl._forward_mla",
                impl_cls._forward_mla,
                (
                    "self",
                    "layer",
                    "q",
                    "kv_c_and_k_pe_cache",
                    "attn_metadata",
                ),
                {},
            ),
        ),
    )
    installed_backend = getattr(
        builder_cls.__init__,
        "_prime_rl_non_aiter_sparse_mla_backend",
        None,
    )
    if installed_backend is not None:
        if installed_backend != backend:
            raise RuntimeError(
                "ROCm sparse-MLA fallback is already installed with backend "
                f"{installed_backend!r}; cannot switch to {backend!r} in-process"
            )
        return

    original_builder_init = builder_cls.__init__
    original_builder_build = builder_cls.build
    original_impl_forward_mla = impl_cls._forward_mla

    persistent_metadata_attributes = (
        "_mla_work_meta_data",
        "_mla_work_indptr",
        "_mla_work_info_set",
        "_mla_reduce_indptr",
        "_mla_reduce_final_map",
        "_mla_reduce_partial_map",
    )
    generic_builder_attributes = (
        "kv_cache_spec",
        "model_config",
        "model_dtype",
        "device",
        "num_heads",
        "mla_dims",
        "topk_tokens",
        "max_model_len_tensor",
        "dummy_block_table",
        "req_id_per_token_buffer",
        "qo_indptr",
        "paged_kv_last_page_len",
        "paged_kv_indices",
        "paged_kv_indptr",
    )

    def _suppress_aiter_persistent_metadata(builder):
        for attribute in persistent_metadata_attributes:
            setattr(builder, attribute, None)

    @functools.wraps(original_builder_init)
    def _builder_init_without_aiter(
        self,
        kv_cache_spec,
        layer_names,
        vllm_config,
        device,
    ):
        configured_topk = getattr(
            getattr(getattr(vllm_config, "model_config", None), "hf_config", None),
            "index_topk",
            None,
        )
        _require_sparse_mla_topk_tokens(
            configured_topk,
            "vllm_config.model_config.hf_config",
        )
        runtime_mismatches = _rocm_sparse_mla_runtime_mismatches(vllm_config)
        runtime_mismatches += _rocm_sparse_mla_cache_spec_mismatches(kv_cache_spec)
        if runtime_mismatches:
            raise ValueError(
                "ROCm sparse MLA requires the exact GLM-5.2 runtime contract: "
                + "; ".join(runtime_mismatches)
            )

        try:
            original_builder_init(
                self,
                kv_cache_spec,
                layer_names,
                vllm_config,
                device,
            )
        except Exception as exc:
            missing_generic_state = tuple(
                attribute for attribute in generic_builder_attributes if not hasattr(self, attribute)
            )
            if missing_generic_state:
                raise
            if isinstance(exc, ModuleNotFoundError) and exc.name != "aiter":
                raise

            # The exact upstream ABI initializes this complete generic state
            # before entering its optional AITER metadata section.
            self._num_attention_heads = max(16, self.num_heads)
            if not hasattr(self, "_prev_req_extent"):
                self._prev_req_extent = 0
            if not hasattr(self, "_prev_indices_extent"):
                self._prev_indices_extent = 0
            if not hasattr(self, "_prev_metadata_key"):
                self._prev_metadata_key = None
        _suppress_aiter_persistent_metadata(self)
        _require_sparse_mla_topk_tokens(
            self.topk_tokens,
            "ROCMAiterMLASparseMetadataBuilder",
        )

    @functools.wraps(original_builder_build)
    def _build_without_aiter(
        self,
        common_prefix_len,
        common_attn_metadata,
        fast_build=False,
    ):
        _require_sparse_mla_topk_tokens(
            self.topk_tokens,
            "ROCMAiterMLASparseMetadataBuilder",
        )
        _suppress_aiter_persistent_metadata(self)
        # Preinstall the exact key checked by vLLM so its build method always
        # skips the AITER metadata launch while retaining generic sparse state.
        num_reqs = common_attn_metadata.num_reqs
        clamped_seq_lens = rocm_aiter_mla_sparse.np.minimum(
            common_attn_metadata.seq_lens_cpu[:num_reqs].numpy(),
            self.topk_tokens,
        )
        self._prev_metadata_key = (
            common_attn_metadata.num_actual_tokens,
            int(common_attn_metadata.max_query_len),
            self._num_attention_heads,
            clamped_seq_lens.tobytes(),
        )
        return original_builder_build(
            self,
            common_prefix_len,
            common_attn_metadata,
            fast_build,
        )

    if backend == "triton":
        from prime_rl.inference.vllm.kernels.rocm_sparse_mla import (
            rocm_sparse_mla_bf16,
        )

        @functools.wraps(original_impl_forward_mla)
        def _forward_mla(
            self,
            layer,
            q,
            kv_c_and_k_pe_cache,
            attn_metadata,
        ):
            del layer
            _require_sparse_mla_topk_tokens(
                attn_metadata.topk_tokens,
                "ROCMAiterMLASparseMetadata",
            )
            num_tokens = q.shape[0]
            return rocm_sparse_mla_bf16(
                q,
                kv_c_and_k_pe_cache,
                attn_metadata.paged_kv_indices,
                attn_metadata.paged_kv_indptr[: num_tokens + 1],
                self.scale,
                num_heads=self.num_heads,
            )

    else:

        @functools.wraps(original_impl_forward_mla)
        def _forward_mla(
            self,
            layer,
            q,
            kv_c_and_k_pe_cache,
            attn_metadata,
        ):
            del layer
            num_tokens = q.shape[0]
            topk_tokens = _require_sparse_mla_topk_tokens(
                attn_metadata.topk_tokens,
                "ROCMAiterMLASparseMetadata",
            )
            indptr = attn_metadata.paged_kv_indptr[: num_tokens + 1].to(torch.int64)
            lengths = indptr[1:] - indptr[:-1]
            positions = torch.arange(topk_tokens, device=q.device, dtype=torch.int64)
            offsets = indptr[:-1, None] + positions[None, :]
            valid_rows = (
                (indptr[:-1] >= 0)
                & (indptr[1:] >= indptr[:-1])
                & (indptr[1:] <= attn_metadata.paged_kv_indices.shape[0])
                & (lengths <= topk_tokens)
            )
            valid = valid_rows[:, None] & (positions[None, :] < lengths[:, None])
            offsets = torch.where(valid, offsets, torch.zeros_like(offsets))
            indices = attn_metadata.paged_kv_indices[offsets]
            indices = torch.where(valid, indices, torch.full_like(indices, -1))

            # The converted indices address tokens, while vLLM stores the cache
            # as [blocks, block_size, head_size]. Flatten the first two
            # dimensions so the bundled reference implementation sees
            # [tokens, 1, head_size].
            kv = kv_c_and_k_pe_cache.reshape(-1, 1, kv_c_and_k_pe_cache.shape[-1])
            output = _reference_mla_sparse_prefill_output(
                rocm_aiter_mla_sparse.reference_mla_sparse_prefill,
                q,
                kv,
                indices[:, None, :],
                self.scale,
                self.kv_lora_rank,
                prefill_q_chunk,
            )
            output = torch.where(
                valid_rows[:, None, None],
                output,
                torch.zeros_like(output),
            )
            # AITER requires at least 16 query heads and the caller pads smaller
            # TP-local head counts before entering ``_forward_mla``. Preserve
            # the stock method's contract by removing repeated heads again.
            return rocm_aiter_mla_sparse.AiterMLAHelper.get_mla_unpadded_o(
                self.num_heads,
                output,
            )

    _builder_init_without_aiter._prime_rl_non_aiter_sparse_mla_backend = backend
    builder_cls.__init__ = _builder_init_without_aiter
    builder_cls.build = _build_without_aiter
    impl_cls._forward_mla = _forward_mla
    if backend == "triton":
        init_logger(__name__).warning(
            "Enabled PRIME_ROCM_SPARSE_MLA_TRITON: using the bounded-memory gfx90a BF16 sparse-MLA kernel."
        )
    else:
        init_logger(__name__).warning(
            "Enabled PRIME_ROCM_SPARSE_MLA_REFERENCE_FALLBACK: using PyTorch "
            "reference sparse MLA on ROCm; smoke tests only."
        )
        if prefill_q_chunk is not None:
            init_logger(__name__).warning(
                "Enabled PRIME_ROCM_SPARSE_MLA_PREFILL_Q_CHUNK=%d: bounding "
                "the reference sparse-MLA prefill gather along its query "
                "dimension.",
                prefill_q_chunk,
            )


def monkey_patch_rocm_sparse_mla_reference_fallback():
    """Backward-compatible entry point for the non-AITER sparse-MLA selector."""
    monkey_patch_rocm_sparse_mla_fallback()


def monkey_patch_kv_xfer_finished_tolerate_freed():
    """Tolerate KV-transfer finish notifications for already-freed requests.

    In disaggregated P/D (NIXL, optionally + a KV store connector) a request can
    be finished — most often ``FINISHED_ABORTED`` from an off-policy cancel, a
    client disconnect, or a request timeout — while it still has in-flight KV
    transfers. When such a request's ``finished_recving`` and ``finished_sending``
    both land in the same ``Scheduler.update_from_output`` step, the stock
    ``_update_from_kv_xfer_finished`` frees it in the recving branch
    (``_free_blocks`` -> ``del self.requests[req_id]``) and then the sending
    branch hits ``assert req_id in self.requests`` and kills the EngineCore. On a
    DP deployment that one death cascades to every rank via the gloo finish-state
    all-reduce, taking down the whole inference pool.

    The trigger is the abort itself, not weight-update pause/resume: it reproduces
    during normal stepping whenever an aborted request's recv and send complete in
    the same step (observed with zero off-policy cancellations, driven only by
    incidental client-side aborts). Skip already-freed request ids instead of
    asserting — their blocks are freed either way, so dropping the stale
    notification is safe.

    Upstream issue: https://github.com/vllm-project/vllm/issues/46240
    """
    from vllm.logger import init_logger
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.request import RequestStatus

    logger = init_logger("vllm.v1.core.sched.scheduler")

    if getattr(Scheduler._update_from_kv_xfer_finished, "_prime_rl_tolerates_freed", False):
        return

    def _update_from_kv_xfer_finished(self, kv_connector_output):
        if self.connector is not None:
            self.connector.update_connector_output(kv_connector_output)

        for req_id in kv_connector_output.finished_recving or ():
            logger.debug("Finished recving KV transfer for request %s", req_id)
            # Stale notification for a request freed earlier this step (e.g. an
            # aborted request whose send completion freed it). Nothing to do.
            if req_id not in self.requests:
                continue
            req = self.requests[req_id]
            if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                self.finished_recving_kv_req_ids.add(req_id)
            else:
                assert RequestStatus.is_finished(req.status)
                self._free_blocks(self.requests[req_id])
        for req_id in kv_connector_output.finished_sending or ():
            logger.debug("Finished sending KV transfer for request %s", req_id)
            # See above: the recving branch may have already freed an aborted
            # request whose send also completed this step.
            if req_id not in self.requests:
                continue
            self._free_blocks(self.requests[req_id])

    _update_from_kv_xfer_finished._prime_rl_tolerates_freed = True
    Scheduler._update_from_kv_xfer_finished = _update_from_kv_xfer_finished
    logger.warning("Patched Scheduler._update_from_kv_xfer_finished to tolerate freed (aborted) KV-transfer reqs.")


def monkey_patch_nano_v3_reasoning_parser():
    from vllm.reasoning.abs_reasoning_parsers import ReasoningParserManager
    from vllm.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser

    class NanoV3ReasoningParser(DeepSeekR1ReasoningParser):
        def extract_reasoning(self, model_output, request):
            reasoning_content, final_content = super().extract_reasoning(model_output, request)
            chat_template_kwargs = getattr(request, "chat_template_kwargs", None)

            if chat_template_kwargs and chat_template_kwargs.get("enable_thinking") is False and final_content is None:
                reasoning_content, final_content = final_content, reasoning_content

            return reasoning_content, final_content

    ReasoningParserManager.register_module("nano_v3", module=NanoV3ReasoningParser)


def monkey_patch_qwen3_coder_param_newline_trim():
    """Restore vLLM 0.23's single-newline trim for qwen3_coder tool parameters.

    vLLM 0.24's parser engine applies a full ``.strip()`` to every parameter
    value in ``_qwen3_arg_converter``; 0.23's ``Qwen3CoderToolParser`` trimmed
    exactly one leading and one trailing newline. A full strip corrupts
    whitespace-significant string parameters (str_replace-style
    ``old_str``/``new_str``, file content with an indented first line, values
    with intentional trailing newlines), silently changing tool execution and
    rewards for agentic runs. Copy of ``_qwen3_arg_converter`` with only the
    trim changed. Also covers ``nemotron_v3``, which reuses ``qwen3_config``.
    """
    import json

    from vllm.parser import qwen3

    def _trim_one_newline(value: str) -> str:
        if value.startswith("\n"):
            value = value[1:]
        if value.endswith("\n"):
            value = value[:-1]
        return value

    def _patched_arg_converter(raw_args: str, partial: bool) -> str:
        params: dict[str, object] = {}

        for match in qwen3._PARAM_RE.finditer(raw_args):
            name = match.group(1)
            value = match.group(2)
            ## START PATCHED CODE (upstream: value.strip())
            params[name] = _trim_one_newline(value)
            ## END PATCHED CODE

        if partial:
            remaining = qwen3._PARAM_RE.sub("", raw_args)
            m = qwen3._PARTIAL_PARAM_RE.search(remaining)
            if m:
                name = m.group(1)
                value = m.group(2)
                if name:
                    ## START PATCHED CODE (upstream: value.strip())
                    params[name] = _trim_one_newline(value)
                    ## END PATCHED CODE

        return json.dumps(params, ensure_ascii=False)

    qwen3._qwen3_arg_converter = _patched_arg_converter
    # qwen3_config captures the converter when first built; drop any cached configs.
    qwen3.qwen3_config.cache_clear()


def monkey_patch_minimax_m2_think_end_passthrough():
    """Keep the literal ``</think>`` in MiniMax-M2 content on tool-calling turns.

    prime-rl serves MiniMax-M2 with ``reasoning=minimax_m2_append_think``, which
    returns content as ``<think>`` + the full completion so think tags round-trip
    through multi-turn re-serialization. vLLM 0.24's minimax_m2 parser engine
    added a ``(CONTENT, THINK_END) -> no-events`` transition that silently
    swallows the ``</think>`` (0.23's regex tool parser passed it through
    untouched), and it also ``.strip()``s content whenever tool calls are
    present. Drop the transition — the engine emits unmatched terminals as plain
    state content — and disable the content strip.
    """
    import dataclasses
    import functools

    from vllm.parser import minimax_m2
    from vllm.parser.engine.parser_engine_config import ParserState

    original_config = minimax_m2.minimax_m2_config

    @functools.cache
    def _patched_config():
        config = original_config()
        transitions = dict(config.transitions)
        del transitions[(ParserState.CONTENT, "THINK_END")]
        return dataclasses.replace(
            config,
            transitions=transitions,
            strip_content_whitespace_with_tools=False,
        )

    minimax_m2.minimax_m2_config = _patched_config


def monkey_patch_return_routed_experts_with_nixl_connector():
    from vllm import envs
    from vllm.config.vllm import VllmConfig
    from vllm.logger import init_logger

    logger = init_logger(__name__)
    original_post_init = VllmConfig.__post_init__

    if getattr(original_post_init, "_prime_rl_allows_nixl_routed_experts", False):
        return

    def _is_nixl_routed_experts_pd_config(config: VllmConfig) -> bool:
        kv_transfer_config = config.kv_transfer_config
        return (
            config.model_config is not None
            and config.model_config.enable_return_routed_experts
            and kv_transfer_config is not None
            and kv_transfer_config.kv_connector == "NixlConnector"
            and kv_transfer_config.is_kv_transfer_instance
        )

    def _post_init(config: VllmConfig):
        if not _is_nixl_routed_experts_pd_config(config):
            return original_post_init(config)

        if config.parallel_config.pipeline_parallel_size > 1:
            raise ValueError("--enable-return-routed-experts is incompatible with pipeline parallelism (PP > 1).")
        if envs.VLLM_USE_V2_MODEL_RUNNER:
            raise ValueError("VLLM_USE_V2_MODEL_RUNNER does not yet support: routed experts capture")

        # vLLM rejects every KV connector, but our P/D path uses NIXL and
        # stitches prefill/decode routed experts in the router. CPU KV offload
        # remains rejected by prime-rl config validation.
        config.model_config.enable_return_routed_experts = False
        try:
            return original_post_init(config)
        finally:
            config.model_config.enable_return_routed_experts = True

    _post_init._prime_rl_allows_nixl_routed_experts = True
    VllmConfig.__post_init__ = _post_init
    logger.warning("Enabled vLLM routed-experts capture with NIXL connector patch.")


def monkey_patch_strip_routed_experts_from_chat():
    """Drop routed_experts from chat-completions responses.

    routed_experts are only consumed via the serialized ``/generate``
    (serving_tokens) path used for router-replay training, which encodes them as a
    ``{data, shape, start}`` object the PD router can merge. The stock
    chat-completions path instead encodes them as a base64 ``np.save`` *string*,
    which the PD router rejects ("prefill routed_experts must be an object with
    base64 data and shape") and fails every eval rollout (evals go through chat
    completions). ``enable_return_routed_experts`` is a server-wide model-config
    flag with no per-request toggle, so strip the field on the chat path here.
    """
    from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
    from vllm.logger import init_logger

    logger = init_logger(__name__)

    if getattr(OpenAIServingChat.chat_completion_full_generator, "_prime_rl_strips_routed_experts", False):
        return

    _original = OpenAIServingChat.chat_completion_full_generator

    async def _strip(result_generator):
        async for res in result_generator:
            for output in res.outputs:
                output.routed_experts = None
            yield res

    async def _patched(self, request, result_generator, *args, **kwargs):
        return await _original(self, request, _strip(result_generator), *args, **kwargs)

    _patched._prime_rl_strips_routed_experts = True
    OpenAIServingChat.chat_completion_full_generator = _patched
    logger.info(
        "Stripped routed_experts from chat-completions responses (PD router merges only the /generate object form)."
    )


def _patch_qwen35_moe_lora_format():
    """Force Qwen3.5-MoE onto vLLM's 2D per-expert LoRA format.

    vLLM 0.24.0 still defaults ``Qwen3_5MoeForConditionalGeneration.is_3d_moe_weight = True``,
    which makes the LoRA loader expect 3D stacked-expert adapters
    (``base_layer.lora_{A,B}.weight`` / ``lora_{A,B}.weight``, experts folded into the
    rank dim; see ``_stack_moe_lora_weights``). Our trainer instead emits the 2D
    per-expert layout (``{expert_id}.gate_proj.lora_A.weight`` ...) from
    ``MultiLoRAGroupedExperts.state_dict_for_adapter`` -- vLLM only consults that layout
    when ``is_3d_moe_weight`` is False (or ``enable_mixed_moe_lora_format=True``).
    Without this override the adapters fail to load with key/shape mismatches.

    The rest of the old Qwen3.5 LoRA shim (the in_proj_qkvz packed-mapping fix and the
    N-slice ``can_replace_layer`` / ``slice_lora_a`` generalizations for vllm#36372) is
    handled natively by 0.23.0 and was dropped. Remove this too once we either adopt the
    3D stacked save format (like gpt-oss) or start the engine with
    ``enable_mixed_moe_lora_format=True``.
    """
    from vllm.model_executor.models.qwen3_5 import Qwen3_5MoeForConditionalGeneration

    Qwen3_5MoeForConditionalGeneration.is_3d_moe_weight = False


def _patch_lora_key_prefix():
    """Accept both bare-suffix and fully-qualified expert module names in LoRA adapters.

    Copy of vLLM 0.24.0's ``LoRAModel.from_local_checkpoint`` with one change: the
    ``.experts`` branch of ``check_unexpected_modules`` accepts either the bare suffix
    (``down_proj``) or the qualified per-expert name (``experts.N.down_proj``), where
    upstream only accepts the qualified form. Our trainer's 2D per-expert adapters
    (Qwen3.5-MoE) carry names whose qualified form is not in the expected set while
    the bare suffix is; Qwen3-30B-A3B adapters go the other way. Upstream fix
    vllm-project/vllm#38522 was closed unmerged, so this stays.
    """
    from vllm.lora.lora_model import (
        LoRAModel,
        MoEEPLoadSpec,
        PEFTHelper,
        TensorizerConfig,
        WeightsMapper,
        _is_remote_expert_key,
        get_lora_id,
        is_base_embedding_weights,
        os,
        parse_fine_tuned_lora_name,
        safetensors,
    )

    def _patched_from_local_checkpoint(
        cls,
        lora_dir: str,
        expected_lora_modules: set[str],
        peft_helper: PEFTHelper,
        *,
        lora_model_id: int | None = None,
        device: str = "cuda",
        dtype: torch.dtype | None = None,
        model_vocab_size: int | None = None,
        weights_mapper: WeightsMapper | None = None,
        tensorizer_config_dict: dict | None = None,
        skip_prefixes: list[str] | None = None,
        moe_ep_spec: MoEEPLoadSpec | None = None,
    ) -> "LoRAModel":
        """Create a LoRAModel from a local checkpoint.

        Args:
            lora_dir: The local path that has lora data.
            expected_lora_modules: Name of modules that are expected to be
                replaced by lora.
            peft_helper: Loaded lora configuration information.
            lora_model_id: LoRA model id. If not given, automatically set by
                a global counter.
            device: Device where the lora model is loaded.
            dtype: dtype of the lora model weights.
            skip_prefixes: List of module name prefixes to skip during loading.
                Models can define this to skip modules not used in inference
                (e.g., MTP layers). Format: ["mtp."]
            moe_ep_spec: When 2D FusedMoE LoRA modules are present with
                expert parallelism enabled, the (ep_rank, local, global)
                slicing metadata shared across all MoE layers. Non-local
                expert weights are skipped at read time instead of being
                loaded and discarded later.

        Returns:
            Loaded LoRA Model.
        """
        lora_tensor_path = os.path.join(lora_dir, "adapter_model.safetensors")
        lora_bin_file_path = os.path.join(lora_dir, "adapter_model.bin")
        lora_pt_file_path = os.path.join(lora_dir, "adapter_model.pt")

        tensors: dict[str, torch.Tensor] = {}
        unexpected_modules: list[list[str] | str] = []

        def check_unexpected_modules(modules: dict):
            for lora_module in modules.keys():  # noqa
                if is_base_embedding_weights(lora_module):
                    continue
                # Handle PEFT file format where experts.base_layer is the
                # gate_up_proj and experts is the down_proj
                if "base_layer" in lora_module:
                    continue
                # Skip modules based on model-defined prefixes
                if skip_prefixes and cls._should_skip_module(lora_module, skip_prefixes):
                    continue
                module_name, _ = parse_fine_tuned_lora_name(lora_module, weights_mapper)
                # Case for expert lora weights.
                ## START PATCHED CODE (upstream only accepts the qualified form)
                if ".experts" in module_name:
                    expert_suffix = module_name.split(".")[-1]
                    experts_qualified = "experts" + module_name.split(".experts", 1)[-1]
                    if expert_suffix not in expected_lora_modules and experts_qualified not in expected_lora_modules:
                        unexpected_modules.append(module_name)
                ## END PATCHED CODE

                elif module_name.rsplit(".", 1)[-1] not in expected_lora_modules:
                    unexpected_modules.append(module_name)

            if unexpected_modules:
                raise ValueError(
                    f"While loading {lora_dir}, expected"
                    f" target modules in {expected_lora_modules}"
                    f" but received {unexpected_modules}."
                    f" Please verify that the loaded LoRA module is correct"
                )

        if tensorizer_config_dict:
            from tensorizer import TensorDeserializer

            tensorizer_config = TensorizerConfig(**tensorizer_config_dict)
            tensorizer_dir = tensorizer_config.tensorizer_dir
            if tensorizer_dir is None:
                raise ValueError("tensorizer_dir must be set in tensorizer config.")
            lora_tensor_path = os.path.join(tensorizer_dir, "adapter_model.tensors")
            tensorizer_args = tensorizer_config._construct_tensorizer_args()
            tensors = TensorDeserializer(
                lora_tensor_path,
                dtype=tensorizer_config.dtype,
                device=device,
                **tensorizer_args.deserialization_kwargs,
            )
            check_unexpected_modules(tensors)

        elif os.path.isfile(lora_tensor_path):
            # Find unexpected modules.
            # Use safetensor key as a source of truth to find expected modules.
            # in peft if you have target_modules A, B, C and C does not exist
            # in the model it won’t error and model will be trained with A, B
            # loraified. C won’t exist in the safetensor but it will exist in
            # the target_modules of the adapter_config.json.
            unexpected_modules = []
            with safetensors.safe_open(lora_tensor_path, framework="pt") as f:  # type: ignore
                # Load tensors if there are only expected modules.
                check_unexpected_modules(f)
                for module in f.keys():  # noqa
                    if moe_ep_spec is not None and _is_remote_expert_key(module, moe_ep_spec):
                        continue
                    tensors[module] = f.get_tensor(module)
        elif os.path.isfile(lora_bin_file_path) or os.path.isfile(lora_pt_file_path):
            lora_file_path = lora_bin_file_path if os.path.isfile(lora_bin_file_path) else lora_pt_file_path
            tensors = torch.load(lora_file_path, map_location=device, weights_only=True)
            check_unexpected_modules(tensors)
            if moe_ep_spec is not None:
                # `.bin`/`.pt` adapters can't be lazy-loaded, but pruning
                # the dict here still frees the non-local expert tensors
                # before the dtype cast / pin_memory work that follows.
                tensors = {k: v for k, v in tensors.items() if not _is_remote_expert_key(k, moe_ep_spec)}
        else:
            raise ValueError(f"{lora_dir} doesn't contain tensors")

        return cls.from_lora_tensors(
            lora_model_id=get_lora_id() if lora_model_id is None else lora_model_id,
            tensors=tensors,
            peft_helper=peft_helper,
            device=device,
            dtype=dtype,
            model_vocab_size=model_vocab_size,
            weights_mapper=weights_mapper,
            skip_prefixes=skip_prefixes,
        )

    LoRAModel.from_local_checkpoint = classmethod(_patched_from_local_checkpoint)


# Monkeypatch TokenizeParams to fix overly conservative validation
def monkey_patch_tokenize_params_validation():
    """
    Patch TokenizeParams validation to only reject requests where the prompt
    itself exceeds max_model_len, not where prompt + max_tokens > max_model_len.

    Original behavior:
        - Rejects if prompt_len > (max_model_len - max_tokens)

    Patched behavior:
        - Only rejects if prompt_len > max_model_len
        - Lets the engine naturally cap generation at max_model_len
    """
    from vllm.exceptions import VLLMValidationError
    from vllm.renderers.params import TokenizeParams

    def _patched_token_len_check(self, tokenizer, tokens):
        """Only validate that prompt fits in max_model_len, not prompt+max_tokens"""
        if self.max_total_tokens is not None and len(tokens) > self.max_total_tokens:
            raise VLLMValidationError(
                f"The prompt is {len(tokens)} tokens, which exceeds the "
                f"model's maximum context length of {self.max_total_tokens} tokens. "
                f"Please reduce the length of the input prompt.",
                parameter="input_tokens",
                value=len(tokens),
            )
        return tokens

    def _patched_text_len_check(self, tokenizer, text):
        """Only validate text length against max_model_len, not max_input_tokens"""
        if self.max_total_tokens is None or tokenizer is None:
            return text

        if self.truncate_prompt_tokens is None:
            max_chars = self.max_total_tokens * tokenizer.max_chars_per_token
            if len(text) > max_chars:
                raise VLLMValidationError(
                    f"You passed {len(text)} input characters. "
                    f"However, the model's context length is only "
                    f"{self.max_total_tokens} tokens "
                    f"(at most {max_chars} characters). "
                    f"Please reduce the length of the input prompt.",
                    parameter="input_text",
                    value=len(text),
                )
        return text

    def _patched_get_encode_kwargs(self):
        """Use max_total_tokens (max_model_len) instead of max_input_tokens for HF tokenizer truncation.

        The original uses max_input_tokens (= max_model_len - max_tokens) + 1, which causes HuggingFace's
        tokenizer.encode() to left-truncate prompts before _token_len_check even runs.
        """
        max_length = self.truncate_prompt_tokens
        if max_length is not None and max_length < 0:
            max_length = self.max_total_tokens
        elif max_length is None and self.max_total_tokens is not None:
            max_length = self.max_total_tokens + 1

        return dict(
            truncation=max_length is not None,
            max_length=max_length,
            add_special_tokens=self.add_special_tokens,
        )

    TokenizeParams._token_len_check = _patched_token_len_check
    TokenizeParams._text_len_check = _patched_text_len_check
    TokenizeParams.get_encode_kwargs = _patched_get_encode_kwargs


def monkey_patch_minimax_m2_for_lora():
    """Patch vLLM's MiniMaxM2 model for LoRA compatibility.

    These patches are only needed when using LoRA with MiniMax M2 but are safe
    to apply unconditionally (verified with non-LoRA runs). We apply them at
    import time because the worker __init__ runs before the vLLM config is
    available, so we can't check if LoRA is enabled.

    Problem 1 — Gate dtype mismatch:
        vLLM's MiniMaxM2MoE creates the gate (router) with params_dtype=float32
        and casts inputs to float32. When LoRA is enabled, vLLM wraps ALL
        ReplicatedLinear layers (including the gate) with LoRA support. Even
        though our adapter has no gate LoRA weights, the LoRA Triton kernel
        still runs for all wrapped layers when any adapter is active — and it
        asserts inputs are float16/bfloat16. Qwen3 MoE doesn't have this
        problem because its gate uses the model dtype.
        Fix: rebuild the gate as GateLinear with a bf16 weight (out_dtype=float32
        keeps fp32 router logits). vLLM 0.24.0's own forward already drops the
        float32 input cast. FusedMoE also has router_logits_dtype=float32, so
        routing precision is preserved inside the expert dispatch.

    Problem 2 — Adapter key naming mismatch:
        PrimeRL saves adapter keys using its internal naming convention
        (mlp.experts.{j}.gate_proj/down_proj/up_proj), which matches Qwen3 MoE
        but not MiniMax M2. vLLM's MiniMax M2 model expects HF-style keys
        (block_sparse_moe.experts.{j}.w1/w2/w3). For full model weights this
        is handled by vLLM's load_weights(), but LoRA adapters are loaded
        through a separate path (LoRAModel.from_local_checkpoint) that doesn't
        have model-specific key translation.
        Fix: set hf_to_vllm_mapper on the model class so vLLM remaps adapter
        keys during LoRA loading. This attribute is only read by _load_adapter
        in the LoRA worker manager — it has no effect without LoRA.
    """
    from vllm.model_executor.models.minimax_m2 import MiniMaxM2ForCausalLM, MiniMaxM2MoE
    from vllm.model_executor.models.utils import WeightsMapper

    # --- Gate dtype fix (only matters with LoRA, safe without) ---
    _original_init = MiniMaxM2MoE.__init__

    def _patched_init(self, config, quant_config=None, prefix=""):
        _original_init(self, config, quant_config, prefix)
        from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear

        # vLLM 0.24.0 builds the gate as GateLinear with a float32 weight; rebuild it
        # with a bf16 weight (model dtype) so the LoRA Triton kernel's float16/bfloat16
        # assertion passes, keeping out_dtype=float32 so router logits stay fp32 (the
        # GateLinear bf16xbf16->fp32 path).
        self.gate = GateLinear(
            config.hidden_size,
            config.num_local_experts,
            bias=False,
            out_dtype=torch.float32,
            prefix=f"{prefix}.gate",
        )

    MiniMaxM2MoE.__init__ = _patched_init

    # --- Adapter key remapping (only read by vLLM's LoRA adapter loader) ---
    MiniMaxM2ForCausalLM.hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={
            ".mlp.experts.": ".block_sparse_moe.experts.",
            ".gate_proj.": ".w1.",
            ".down_proj.": ".w2.",
            ".up_proj.": ".w3.",
        },
    )


def monkey_patch_harmony_stop_token_propagation():
    """Fix: vLLM doesn't merge harmony stop tokens into per-request SamplingParams.

    The harmony mode sets stop_token_ids (including <|call|> and <|return|>) in
    default_sampling_params at server init, but ChatCompletionRequest.to_sampling_params()
    ignores them, using only self.stop_token_ids (which defaults to []).

    Upstream: https://github.com/vllm-project/vllm/issues/22519
    """
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

    _original_to_sampling_params = ChatCompletionRequest.to_sampling_params

    def _patched_to_sampling_params(self, max_tokens, default_sampling_params):
        params = _original_to_sampling_params(self, max_tokens, default_sampling_params)
        # Merge harmony stop tokens from default_sampling_params
        default_stop_ids = default_sampling_params.get("stop_token_ids", [])
        if default_stop_ids:
            existing = set(params.stop_token_ids or [])
            merged = list(existing | set(default_stop_ids))
            params.stop_token_ids = merged
        return params

    ChatCompletionRequest.to_sampling_params = _patched_to_sampling_params


def monkey_patch_no_moe_lora():
    """This disables LoRA for MoE layers and makes them pick better kernels.

    Otherwise, the oracle will always try to pick TritonExperts.
    For blackwells, we want TRTLLMFlashInfer.
    """
    from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig

    original_post_init = FusedMoEConfig.__post_init__

    def _patched__post_init__(self: FusedMoEConfig):
        original_post_init(self)
        # Disable LoRA for MoE layers. `is_lora_enabled` is only read later during
        # kernel selection (modular_kernel / unquantized oracle), never inside
        # `__post_init__`, so flipping it after the original runs is sufficient.
        self.is_lora_enabled = False

    FusedMoEConfig.__post_init__ = _patched__post_init__


def monkey_patch_fp32_lm_head():
    """Run the lm_head projection with fp32 logits where the backend supports it.

    Uses ``torch.mm(..., out_dtype=torch.float32)`` (PyTorch >= 2.10) so the
    matmul accumulates and emits fp32 directly without zero-padding the bf16
    operands or maintaining a separate fp32 weight copy. This avoids the
    epilogue truncation to bf16 that `F.linear(bf16, bf16)` does, which is
    where lm_head precision actually leaks before the sampler's softmax.

    On ROCm this native ``out_dtype`` path can miss hipBLASLt support for common
    logits shapes and fall back to a much slower GEMM path. In that case we keep
    vLLM's ROCm-specific lm_head kernel and cast the resulting logits to fp32
    before sampling. This is not as strict numerically as fp32 lm_head
    accumulation, but it avoids a severe decode-throughput regression.

    Activated by setting ``additional_config["fp32_lm_head"] = True`` on the
    vLLM namespace; the launcher does this when ``inference.enable_fp32_lm_head``
    is set. The flag is captured once on ``LogitsProcessor.__init__`` (where
    vLLM guarantees a ``set_current_vllm_config()`` context) and stored on the
    instance — reading it from ``_get_logits`` during serving doesn't work
    because vLLM doesn't keep the context set during forwards.

    Tracks vllm-project/vllm#24567 (which uses the operand-upcast approach).
    Per @Jackmin801 on PR #2438, native ``out_dtype=fp32`` mm is more efficient
    and just as correct.
    """
    import torch
    from vllm.config import get_current_vllm_config
    from vllm.logger import init_logger
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.platforms import current_platform

    logger = init_logger(__name__)

    _original_init = LogitsProcessor.__init__
    _original_get_logits = LogitsProcessor._get_logits

    def _patched_init(self, *args, **kwargs):
        _original_init(self, *args, **kwargs)
        vllm_config = get_current_vllm_config()
        additional_config = vllm_config.additional_config or {}
        self._fp32_lm_head_enabled = additional_config.get("fp32_lm_head", False)
        self._fp32_lm_head_use_native_out_dtype = self._fp32_lm_head_enabled and not current_platform.is_rocm()
        if self._fp32_lm_head_enabled:
            if self._fp32_lm_head_use_native_out_dtype:
                logger.warning("fp32 lm_head ENABLED for this LogitsProcessor instance.")
            else:
                logger.warning("fp32 lm_head requested on ROCm; using vLLM ROCm lm_head GEMM plus fp32 logits cast.")

    def _patched_get_logits(self, hidden_states, lm_head, embedding_bias):
        if not getattr(self, "_fp32_lm_head_enabled", False):
            return _original_get_logits(self, hidden_states, lm_head, embedding_bias)

        if not getattr(self, "_fp32_lm_head_use_native_out_dtype", False):
            logits = _original_get_logits(self, hidden_states, lm_head, embedding_bias)
            return None if logits is None else logits.to(torch.float32)

        # Native bf16xbf16 -> fp32 GEMM. torch.mm requires 2D inputs; vLLM v1's
        # generative path passes 2D [num_tokens, hidden_size] hidden_states, but
        # flatten defensively in case some future caller passes 3D.
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        logits = torch.mm(flat, lm_head.weight.t(), out_dtype=torch.float32)
        if embedding_bias is not None:
            logits = logits + embedding_bias.to(torch.float32)
        if hidden_states.dim() > 2:
            logits = logits.reshape(*hidden_states.shape[:-1], -1)

        logits = self._gather_logits(logits)
        if logits is not None:
            logits = logits[..., : self.org_vocab_size]
        return logits

    LogitsProcessor.__init__ = _patched_init
    LogitsProcessor._get_logits = _patched_get_logits
    logger.info("Installed fp32 lm_head patch (native out_dtype=fp32 mm).")


def _align_deepseek_router_logits_dtype(gate, router_logits_dtype):
    """Align a DeepSeek-family gate before FusedMoE backend selection.

    This is the pinned-vLLM backport seam. Upstream should pass the selected
    gate dtype directly while constructing ``FusedMoE``.
    """
    import torch

    if gate is None or not hasattr(gate, "out_dtype"):
        raise RuntimeError(
            "fp32 router logits require DeepseekV2MoE to pass a "
            "GateLinear through FusedMoE(gate=...)"
        )
    if gate.out_dtype is None:
        if router_logits_dtype not in (None, torch.float32):
            raise RuntimeError(
                "FusedMoE router_logits_dtype disagrees with the requested "
                f"FP32 gate: {router_logits_dtype}"
            )
        gate.set_out_dtype(torch.float32)
    if gate.out_dtype == torch.float32:
        if router_logits_dtype not in (None, torch.float32):
            raise RuntimeError(
                "FusedMoE router_logits_dtype disagrees with the FP32 gate: "
                f"{router_logits_dtype}"
            )
        return torch.float32
    return router_logits_dtype


def _require_exact_vllm_fp32_router_patch_abi(fused_moe, deepseek_moe_init) -> None:
    """Validate every pinned constructor parameter touched by the backport."""

    _require_exact_vllm_sparse_patch_abi(
        "fp32 DeepSeek router-logits backport",
        (
            (
                "deepseek_v2.FusedMoE",
                fused_moe,
                (
                    "num_experts",
                    "top_k",
                    "hidden_size",
                    "intermediate_size",
                    "params_dtype",
                    "renormalize",
                    "use_grouped_topk",
                    "num_expert_group",
                    "topk_group",
                    "quant_config",
                    "tp_size",
                    "dp_size",
                    "pcp_size",
                    "prefix",
                    "custom_routing_function",
                    "router",
                    "scoring_func",
                    "routed_scaling_factor",
                    "swiglu_limit",
                    "swiglu_alpha",
                    "swiglu_beta",
                    "e_score_correction_bias",
                    "apply_router_weight_on_input",
                    "activation",
                    "enable_eplb",
                    "num_redundant_experts",
                    "has_bias",
                    "is_sequence_parallel",
                    "expert_mapping",
                    "n_shared_experts",
                    "router_logits_dtype",
                    "gate",
                    "shared_experts",
                    "shared_expert_gate",
                    "routed_input_transform",
                    "routed_output_transform",
                    "apply_routed_scale_to_output",
                    "zero_expert_type",
                    "hash_indices_table",
                    "runner_cls",
                    "runner_args",
                    "routed_experts_cls",
                    "routed_experts_args",
                ),
                {
                    "params_dtype": None,
                    "renormalize": True,
                    "use_grouped_topk": False,
                    "num_expert_group": None,
                    "topk_group": None,
                    "quant_config": None,
                    "tp_size": None,
                    "dp_size": None,
                    "pcp_size": None,
                    "prefix": "",
                    "custom_routing_function": None,
                    "router": None,
                    "scoring_func": "softmax",
                    "routed_scaling_factor": 1.0,
                    "swiglu_limit": None,
                    "swiglu_alpha": None,
                    "swiglu_beta": None,
                    "e_score_correction_bias": None,
                    "apply_router_weight_on_input": False,
                    "activation": "silu",
                    "enable_eplb": False,
                    "num_redundant_experts": 0,
                    "has_bias": False,
                    "is_sequence_parallel": False,
                    "expert_mapping": None,
                    "n_shared_experts": None,
                    "router_logits_dtype": None,
                    "gate": None,
                    "shared_experts": None,
                    "shared_expert_gate": None,
                    "routed_input_transform": None,
                    "routed_output_transform": None,
                    "apply_routed_scale_to_output": False,
                    "zero_expert_type": None,
                    "hash_indices_table": None,
                    "runner_cls": None,
                    "runner_args": None,
                    "routed_experts_cls": None,
                    "routed_experts_args": None,
                },
            ),
            (
                "deepseek_v2.DeepseekV2MoE.__init__",
                deepseek_moe_init,
                (
                    "self",
                    "config",
                    "parallel_config",
                    "quant_config",
                    "prefix",
                ),
                {
                    "quant_config": None,
                    "prefix": "",
                },
            ),
        ),
    )


def monkey_patch_fp32_router_logits():
    """Emit fp32 MoE router logits for DeepSeek-family models (incl. GLM-5.x glm_moe_dsa).

    vLLM's DeepseekV2MoE gate is a GateLinear with a bf16 weight and no out_dtype,
    so router logits are rounded to bf16 before expert scoring. GLM-5.x was trained
    with fp32 routing (Megatron ``--moe-router-dtype fp32``). Current upstream
    vLLM keeps the gate weight in BF16 and requests an FP32 output
    (vllm-project/vllm#47970, which superseded the full weight upcast in #47410).
    This backport requests the same dtype contract. Backends with native
    BF16xBF16-to-FP32 output write the accumulator without a BF16 output
    round-trip; the stock ROCm fallback described below does not.

    On ROCm, vLLM's fallback first stores BF16 and only then casts to FP32.
    ``PRIME_ROCM_GFX90A_GATE_LINEAR=1`` replaces that path for the exact
    GLM-5.2 ``[M, 6144] @ [256, 6144].T`` contract with PRIME-RL's
    BF16-to-FP32 gfx90a kernel. The opt-in fails closed on a changed model,
    device, shape, dtype, token bound, or vLLM ABI. With the opt-in unset,
    vLLM's stock path is unchanged.

    ``PRIME_ROCM_GFX90A_ONE_GROUP_TOPK=1`` also recognizes GLM-5.2's
    ``n_group=topk_group=1`` identity grouping and dispatches directly to
    vLLM's fused biased-sigmoid top-k kernel. It preserves unbiased routing
    weights, normalization, and the runner's separate 2.5 output scale.

    Activated by ``additional_config["fp32_router_logits"] = True``; the launcher
    sets this when ``inference.enable_fp32_router_logits`` is set. The flag is
    read from vLLM's module-construction context. Direct FusedMoE construction
    without that context remains an untouched stock call when no gfx90a kernel
    opt-in is set. An explicitly selected gate dtype (for example AITER's BF16
    contract) is preserved.
    """
    import functools
    import inspect
    import os

    import torch
    from vllm.config import get_current_vllm_config_or_none
    from vllm.logger import init_logger
    from vllm.model_executor.models import deepseek_v2

    logger = init_logger(__name__)
    DeepseekV2MoE = deepseek_v2.DeepseekV2MoE

    init_is_patched = getattr(
        DeepseekV2MoE.__init__,
        "_prime_rl_fp32_router_logits_patch",
        False,
    )
    factory_is_patched = getattr(
        deepseek_v2.FusedMoE,
        "_prime_rl_fp32_router_logits_factory_patch",
        False,
    )
    if init_is_patched and factory_is_patched:
        return
    if init_is_patched or factory_is_patched:
        raise RuntimeError(
            "fp32 router-logits backport is only partially installed; "
            "refusing to stack another constructor patch"
        )

    def _strict_env_enabled(name):
        value = os.environ.get(name)
        if value in (None, "0"):
            return False
        if value == "1":
            return True
        raise ValueError(f"{name} must be unset, '0', or '1'; got {value!r}")

    gfx90a_gate_linear = _strict_env_enabled("PRIME_ROCM_GFX90A_GATE_LINEAR")
    one_group_topk = _strict_env_enabled("PRIME_ROCM_GFX90A_ONE_GROUP_TOPK")

    # DeepseekV2MoE passes gate.out_dtype into FusedMoE while it constructs the
    # FusedMoEConfig and selects a backend. Updating the gate only after
    # DeepseekV2MoE.__init__ returns leaves that configuration at BF16 even
    # though the gate subsequently emits FP32. Set the dtype at the factory
    # boundary instead, where both the gate and FusedMoE observe it before
    # backend selection. Preserve an explicitly selected dtype such as AITER's
    # BF16 contract.
    _original_fused_moe = deepseek_v2.FusedMoE
    _original_init = DeepseekV2MoE.__init__

    @functools.lru_cache(maxsize=1)
    def _validate_pinned_constructor_abi():
        _require_exact_vllm_fp32_router_patch_abi(
            _original_fused_moe,
            _original_init,
        )
        return inspect.signature(_original_fused_moe).bind

    @functools.wraps(_original_fused_moe)
    def _patched_fused_moe(*args, **kwargs):
        vllm_config = get_current_vllm_config_or_none()
        if vllm_config is None:
            if gfx90a_gate_linear or one_group_topk:
                raise RuntimeError(
                    "gfx90a router-kernel opt-ins require a current vLLM "
                    "configuration during FusedMoE construction"
                )
            return _original_fused_moe(*args, **kwargs)
        additional_config = vllm_config.additional_config or {}
        fp32_router_logits = additional_config.get("fp32_router_logits", False)
        if (gfx90a_gate_linear or one_group_topk) and not fp32_router_logits:
            raise RuntimeError(
                "gfx90a router-kernel opt-ins require "
                "additional_config['fp32_router_logits']=True"
            )
        if not fp32_router_logits:
            return _original_fused_moe(*args, **kwargs)

        if gfx90a_gate_linear:
            from prime_rl.inference.vllm.kernels.rocm_gfx90a_gate_linear import (
                validate_glm52_gate_scheduler_config,
            )

            validate_glm52_gate_scheduler_config(
                getattr(vllm_config, "scheduler_config", None)
            )
        bind = _validate_pinned_constructor_abi()
        bound = bind(*args, **kwargs)
        aligned_dtype = _align_deepseek_router_logits_dtype(
            bound.arguments.get("gate"),
            bound.arguments.get("router_logits_dtype"),
        )
        if aligned_dtype is not None:
            bound.arguments["router_logits_dtype"] = aligned_dtype
        return _original_fused_moe(*bound.args, **bound.kwargs)

    _patched_fused_moe._prime_rl_fp32_router_logits_factory_patch = True
    deepseek_v2.FusedMoE = _patched_fused_moe

    def _patched_init(self, *args, **kwargs):
        _original_init(self, *args, **kwargs)
        vllm_config = get_current_vllm_config_or_none()
        if vllm_config is None:
            if gfx90a_gate_linear or one_group_topk:
                raise RuntimeError(
                    "gfx90a router-kernel opt-ins require a current vLLM "
                    "configuration during DeepseekV2MoE construction"
                )
            return
        additional_config = vllm_config.additional_config or {}
        fp32_router_logits = additional_config.get("fp32_router_logits", False)
        if (gfx90a_gate_linear or one_group_topk) and not fp32_router_logits:
            raise RuntimeError("gfx90a router-kernel opt-ins require additional_config['fp32_router_logits']=True")
        if not fp32_router_logits:
            return
        if gfx90a_gate_linear:
            from prime_rl.inference.vllm.kernels.rocm_gfx90a_gate_linear import (
                install_glm52_gfx90a_gate_linear,
            )

            config = args[0] if args else kwargs.get("config")
            install_glm52_gfx90a_gate_linear(self.gate, config)
            logger.warning("Enabled PRIME_ROCM_GFX90A_GATE_LINEAR for an exact GLM-5.2 GateLinear module.")
        elif self.gate.out_dtype is None:
            self.gate.set_out_dtype(torch.float32)
        if one_group_topk:
            from prime_rl.inference.vllm.kernels.rocm_glm52_degenerate_grouped_topk import (
                install_glm52_degenerate_grouped_topk,
            )

            config = args[0] if args else kwargs.get("config")
            install_glm52_degenerate_grouped_topk(self.experts, config)
            logger.warning("Enabled PRIME_ROCM_GFX90A_ONE_GROUP_TOPK for an exact GLM-5.2 router.")
        if gfx90a_gate_linear or one_group_topk:
            router_logits_dtype = getattr(
                getattr(self.experts, "moe_config", None),
                "router_logits_dtype",
                None,
            )
            if self.gate.out_dtype != torch.float32 or router_logits_dtype != torch.float32:
                raise RuntimeError(
                    "gfx90a router kernels require matching FP32 gate and "
                    "FusedMoE router logits; found "
                    f"gate={self.gate.out_dtype}, moe={router_logits_dtype}"
                )

    _patched_init._prime_rl_fp32_router_logits_patch = True
    DeepseekV2MoE.__init__ = _patched_init
    logger.info("Installed fp32 router logits patch (self-gates on additional_config['fp32_router_logits']).")


def monkey_patch_glm52_gfx90a_mla_bmm():
    """Install measured GLM-5.2 MLA BMMs at vLLM's narrow method seams.

    ``PRIME_ROCM_GFX90A_MLA_V_UP=1`` enables the V-up kernel supported by the
    pinned vLLM distribution. ``PRIME_ROCM_GFX90A_MLA_BMM=1`` requires both
    Q-absorb and V-up; it therefore also requires the documented upstream
    ``_q_absorb`` hook. The integration validates the complete model, runtime,
    weight-layout, device, and ABI contract before marking any module.
    """
    import os

    from vllm.logger import init_logger

    logger = init_logger(__name__)

    def _strict_env_enabled(name):
        value = os.environ.get(name)
        if value in (None, "0"):
            return False
        if value == "1":
            return True
        raise ValueError(f"{name} must be unset, '0', or '1'; got {value!r}")

    full_path = _strict_env_enabled("PRIME_ROCM_GFX90A_MLA_BMM")
    v_up_only = _strict_env_enabled("PRIME_ROCM_GFX90A_MLA_V_UP")
    if full_path and v_up_only:
        raise ValueError(
            "PRIME_ROCM_GFX90A_MLA_BMM and PRIME_ROCM_GFX90A_MLA_V_UP "
            "are mutually exclusive"
        )
    if not (full_path or v_up_only):
        return

    from prime_rl.inference.vllm.kernels.rocm_gfx90a_mla_bmm_integration import (
        install_glm52_gfx90a_mla_bmm,
    )

    installed = install_glm52_gfx90a_mla_bmm(require_q_absorb=full_path)
    logger.warning(
        "Enabled GLM-5.2 gfx90a MLA BMM kernels: "
        "q_absorb=%s v_up=%s.",
        installed.q_absorb,
        installed.v_up,
    )


def monkey_patch_dp_coordinator_startup_timeout():
    """Raise the DP coordinator startup timeout from vLLM's hard-coded 120s.

    The coordinator child process is spawned on the DP-rank-0 API server while
    every engine-core rank on the node is importing and loading weights, so its
    own spawn-time re-import can exceed the hard-coded timeout under that CPU/IO
    contention (seen on multi-node disaggregated GLM-5.1 launches). Configurable
    via PRIME_DP_COORDINATOR_STARTUP_TIMEOUT (seconds, default 300).
    """
    import multiprocessing.connection
    import os

    from vllm.v1.engine.coordinator import DPCoordinator

    timeout = float(os.environ.get("PRIME_DP_COORDINATOR_STARTUP_TIMEOUT", "300"))

    def _patched_wait_for_zmq_addrs(self, zmq_addr_pipe):
        try:
            ready = multiprocessing.connection.wait([zmq_addr_pipe, self.proc.sentinel], timeout=timeout)
            if not ready:
                raise RuntimeError(
                    f"DP Coordinator process failed to report ZMQ addresses within {timeout}s during startup."
                )
            try:
                return zmq_addr_pipe.recv()
            except EOFError:
                raise RuntimeError("DP Coordinator process failed during startup.") from None
        finally:
            zmq_addr_pipe.close()

    DPCoordinator._wait_for_zmq_addrs = _patched_wait_for_zmq_addrs
