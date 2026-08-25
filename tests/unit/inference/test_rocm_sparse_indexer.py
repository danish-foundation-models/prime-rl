import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from prime_rl.inference.patches import (
    _require_exact_vllm_sparse_patch_abi,
    _require_rocm_gfx90a,
    _require_sparse_mla_topk_tokens,
    _rocm_sparse_indexer_prefill_k_chunk_from_env,
    _rocm_sparse_indexer_triton_decode_from_env,
    _rocm_sparse_mla_backend_from_env,
    _rocm_sparse_mla_cache_spec_mismatches,
    _rocm_sparse_mla_prefill_q_chunk_from_env,
    _rocm_sparse_mla_runtime_mismatches,
    _strict_binary_env,
    _validate_rocm_sparse_indexer_forward_hip,
    _wrap_rocm_sparse_indexer_bulk_context_lens,
    _wrap_rocm_sparse_indexer_prefill_k_chunk,
    monkey_patch_rocm_sparse_indexer_fallback,
    monkey_patch_rocm_sparse_mla_reference_fallback,
)
from prime_rl.inference.vllm.kernels import (
    rocm_gfx90a_mla_bmm_integration,
)
from prime_rl.inference.vllm.kernels import (
    rocm_sparse_mla as rocm_sparse_mla_kernel,
)


class _ContextLens:
    def __init__(
        self,
        *,
        data_ptr=1024,
        shape=(16, 1),
        stride=(1, 1),
        storage_offset=0,
        device=torch.device("cuda", 0),
        dtype=torch.int32,
    ):
        self.to_calls = []
        self.host_copy = object()
        self.device = device
        self.shape = shape
        self._data_ptr = data_ptr
        self._stride = stride
        self._storage_offset = storage_offset
        self.dtype = dtype

    def to(self, **kwargs):
        self.to_calls.append(kwargs)
        return self.host_copy

    def data_ptr(self):
        return self._data_ptr

    def stride(self):
        return self._stride

    def storage_offset(self):
        return self._storage_offset


def _set_forward_context(monkeypatch, additional_kwargs):
    from vllm import forward_context

    context = SimpleNamespace(additional_kwargs=additional_kwargs)
    monkeypatch.setattr(forward_context, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(forward_context, "get_forward_context", lambda: context)
    return context


def _make_glm52_sparse_vllm_config(
    *,
    model_dtype=torch.bfloat16,
    cache_dtype="auto",
    hf_overrides=None,
    parallel_overrides=None,
):
    hf_fields = {
        "architectures": ["GlmMoeDsaForCausalLM"],
        "model_type": "glm_moe_dsa",
        "hidden_size": 6144,
        "num_attention_heads": 64,
        "num_key_value_heads": 64,
        "head_dim": 192,
        "q_lora_rank": 2048,
        "kv_lora_rank": 512,
        "qk_head_dim": 256,
        "qk_nope_head_dim": 192,
        "qk_rope_head_dim": 64,
        "v_head_dim": 256,
        "num_hidden_layers": 78,
        "index_topk": 2048,
        "quantization_config": {
            "activation_scheme": "dynamic",
            "fmt": "e4m3",
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
        },
    }
    hf_fields.update(hf_overrides or {})
    parallel_fields = {
        "tensor_parallel_size": 8,
        "data_parallel_size": 2,
        "pipeline_parallel_size": 1,
        "decode_context_parallel_size": 1,
        "enable_expert_parallel": True,
    }
    parallel_fields.update(parallel_overrides or {})
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(**hf_fields),
            dtype=model_dtype,
        ),
        parallel_config=SimpleNamespace(**parallel_fields),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=512),
        cache_config=SimpleNamespace(cache_dtype=cache_dtype),
    )


def _make_sparse_kv_cache_spec(
    *,
    dtype=torch.bfloat16,
    cache_dtype_str=None,
):
    return SimpleNamespace(
        block_size=64,
        dtype=dtype,
        cache_dtype_str=cache_dtype_str,
    )


def _install_fake_sparse_mla_vllm(
    monkeypatch,
    reference_mla_sparse_prefill,
    *,
    aiter_available=True,
    aiter_failure=None,
):
    warnings = []

    from prime_rl.inference import patches

    monkeypatch.setattr(
        patches,
        "_require_rocm_gfx90a",
        lambda feature: "gfx90a",
    )
    monkeypatch.setattr(
        patches,
        "_installed_vllm_distribution_version",
        lambda: "0.24.0+lumi_aif_gfx90a_ee0da84",
    )

    class _Logger:
        def warning(self, *args):
            warnings.append(args)

    class _MetadataBuilder:
        init_calls = 0

        def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
            del layer_names
            type(self).init_calls += 1
            self.kv_cache_spec = kv_cache_spec
            self.model_config = vllm_config.model_config
            self.model_dtype = vllm_config.model_config.dtype
            self.device = device
            self.mla_dims = object()
            self.topk_tokens = vllm_config.model_config.hf_config.index_topk
            self.num_heads = 8
            self.max_model_len_tensor = object()
            self.dummy_block_table = object()
            self.req_id_per_token_buffer = object()
            self.qo_indptr = object()
            self.paged_kv_last_page_len = object()
            self.paged_kv_indices = object()
            self.paged_kv_indptr = object()
            self.metadata_launches = 0
            self._prev_req_extent = 0
            self._prev_indices_extent = 0
            self._prev_metadata_key = None
            if not aiter_available:
                raise ModuleNotFoundError("No module named 'aiter'", name="aiter")
            if aiter_failure is not None:
                raise aiter_failure

            self._num_attention_heads = 16
            self._mla_work_meta_data = object()
            self._mla_work_indptr = object()
            self._mla_work_info_set = object()
            self._mla_reduce_indptr = object()
            self._mla_reduce_final_map = object()
            self._mla_reduce_partial_map = object()

        def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
            num_reqs = common_attn_metadata.num_reqs
            clamped_seq_lens = np.minimum(
                common_attn_metadata.seq_lens_cpu[:num_reqs].numpy(),
                self.topk_tokens,
            )
            metadata_key = (
                common_attn_metadata.num_actual_tokens,
                int(common_attn_metadata.max_query_len),
                self._num_attention_heads,
                clamped_seq_lens.tobytes(),
            )
            if metadata_key != self._prev_metadata_key:
                self.metadata_launches += 1
                self._prev_metadata_key = metadata_key
            return common_prefix_len, common_attn_metadata, fast_build

    class _SparseMLAImpl:
        def _forward_mla(
            self,
            layer,
            q,
            kv_c_and_k_pe_cache,
            attn_metadata,
        ):
            raise AssertionError("unpatched sparse MLA implementation called")

    _SparseMLAImpl.metadata_builder_cls = _MetadataBuilder

    class _AiterMLAHelper:
        @staticmethod
        def get_mla_unpadded_o(num_heads, output):
            return output[:, :num_heads]

    rocm_sparse_mla = SimpleNamespace(
        ROCMAiterMLASparseMetadataBuilder=_MetadataBuilder,
        ROCMAiterMLASparseImpl=_SparseMLAImpl,
        AiterMLAHelper=_AiterMLAHelper,
        reference_mla_sparse_prefill=reference_mla_sparse_prefill,
        np=np,
    )

    packages = (
        "vllm",
        "vllm.v1",
        "vllm.v1.attention",
        "vllm.v1.attention.backends",
    )
    for package_name in packages:
        package = ModuleType(package_name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, package_name, package)

    logger_module = ModuleType("vllm.logger")
    logger_module.init_logger = lambda _: _Logger()
    monkeypatch.setitem(sys.modules, "vllm.logger", logger_module)

    platforms_module = ModuleType("vllm.platforms")
    platforms_module.current_platform = SimpleNamespace(is_rocm=lambda: True)
    monkeypatch.setitem(sys.modules, "vllm.platforms", platforms_module)

    mla_module = ModuleType("vllm.v1.attention.backends.mla")
    mla_module.rocm_aiter_mla_sparse = rocm_sparse_mla
    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.attention.backends.mla",
        mla_module,
    )

    def _unconfigured_sparse_kernel(*args, **kwargs):
        raise AssertionError("test sparse MLA kernel was not configured")

    monkeypatch.setattr(
        rocm_sparse_mla_kernel,
        "rocm_sparse_mla_bf16",
        _unconfigured_sparse_kernel,
    )
    _SparseMLAImpl.sparse_kernel_module = rocm_sparse_mla_kernel
    return _SparseMLAImpl, warnings


def _sparse_mla_inputs(num_tokens):
    topk_tokens = 2048
    q = torch.arange(num_tokens * 4 * 5, dtype=torch.float32).view(num_tokens, 4, 5)
    kv_cache = torch.arange(2 * 8 * 5, dtype=torch.float32).view(2, 8, 5)
    metadata = SimpleNamespace(
        topk_tokens=topk_tokens,
        paged_kv_indptr=torch.arange(
            0,
            (num_tokens + 1) * topk_tokens,
            topk_tokens,
            dtype=torch.int32,
        ),
        paged_kv_indices=torch.arange(
            num_tokens * topk_tokens,
            dtype=torch.int32,
        )
        % (kv_cache.shape[0] * kv_cache.shape[1]),
    )
    return q, kv_cache, metadata


def _sparse_mla_common_metadata():
    return SimpleNamespace(
        num_reqs=2,
        num_actual_tokens=7,
        max_query_len=4,
        seq_lens_cpu=torch.tensor([1024, 4096], dtype=torch.int32),
    )


def _sparse_mla_reference(calls):
    def reference(q, kv, indices, scale, kv_lora_rank):
        calls.append(
            {
                "q": q,
                "kv": kv,
                "indices": indices.clone(),
                "scale": scale,
                "kv_lora_rank": kv_lora_rank,
            }
        )
        first_index = indices[:, 0, 0].to(q.dtype).view(-1, 1, 1)
        output = q[..., :kv_lora_rank] + first_index
        return output, q.new_empty((q.shape[0], q.shape[1]))

    return reference


def _sparse_mla_impl(impl_cls):
    impl = impl_cls()
    impl.num_heads = 2
    impl.scale = 0.125
    impl.kv_lora_rank = 2
    return impl


def test_gfx90a_kernel_gate_accepts_only_cdna2(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "7.1")
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda index: SimpleNamespace(gcnArchName="gfx90a:sramecc+:xnack-"),
    )

    assert _require_rocm_gfx90a("test kernel") == "gfx90a"

    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda index: SimpleNamespace(gcnArchName="gfx942:sramecc+:xnack-"),
    )
    with pytest.raises(RuntimeError, match="validated only on gfx90a"):
        _require_rocm_gfx90a("test kernel")


def test_sparse_patch_abi_gate_requires_exact_distribution(monkeypatch):
    def symbol(value, optional=False):
        return value, optional

    monkeypatch.setattr(
        "prime_rl.inference.patches._installed_vllm_distribution_version",
        lambda: "0.24.1",
    )

    with pytest.raises(RuntimeError, match="requires vLLM"):
        _require_exact_vllm_sparse_patch_abi(
            "test sparse patch",
            (("symbol", symbol, ("value", "optional"), {"optional": False}),),
        )


def test_sparse_patch_abi_gate_requires_exact_signature(monkeypatch):
    def symbol(value, changed_default=True):
        return value, changed_default

    monkeypatch.setattr(
        "prime_rl.inference.patches._installed_vllm_distribution_version",
        lambda: "0.24.0+lumi_aif_gfx90a_ee0da84",
    )

    with pytest.raises(RuntimeError, match="requires symbol"):
        _require_exact_vllm_sparse_patch_abi(
            "test sparse patch",
            (("symbol", symbol, ("value", "optional"), {"optional": False}),),
        )


@pytest.mark.parametrize("topk_tokens", [2047, 2049, True, torch.tensor(2048)])
def test_sparse_mla_topk_gate_requires_exact_python_integer(topk_tokens):
    with pytest.raises(ValueError, match="topk_tokens=2048 as a Python int"):
        _require_sparse_mla_topk_tokens(topk_tokens, "test metadata")

    assert _require_sparse_mla_topk_tokens(2048, "test metadata") == 2048


@pytest.mark.parametrize(("value", "expected"), ((None, False), ("0", False), ("1", True)))
def test_sparse_binary_environment_is_strict(monkeypatch, value, expected):
    name = "PRIME_TEST_SPARSE_BINARY_ENV"
    if value is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, value)

    assert _strict_binary_env(name) is expected


@pytest.mark.parametrize("value", ("", "true", "yes", "2", "-1"))
def test_sparse_binary_environment_rejects_ambiguous_values(monkeypatch, value):
    name = "PRIME_TEST_SPARSE_BINARY_ENV"
    monkeypatch.setenv(name, value)

    with pytest.raises(
        ValueError,
        match=rf"{name} must be unset, '0', or '1'",
    ):
        _strict_binary_env(name)


def test_sparse_indexer_forward_boundary_rejects_fp4_without_assert():
    with pytest.raises(
        NotImplementedError,
        match="doesn't support fp4 cache",
    ):
        _validate_rocm_sparse_indexer_forward_hip(
            use_fp4_cache=True,
            q_quant=torch.empty(1),
        )


def test_sparse_indexer_forward_boundary_requires_single_tensor():
    with pytest.raises(
        TypeError,
        match="single FP8 q_quant tensor",
    ):
        _validate_rocm_sparse_indexer_forward_hip(
            use_fp4_cache=False,
            q_quant=(torch.empty(1), torch.empty(1)),
        )

    _validate_rocm_sparse_indexer_forward_hip(
        use_fp4_cache=False,
        q_quant=torch.empty(1),
    )


def test_bulk_context_lens_default_copies_each_plain_decode_call(monkeypatch):
    monkeypatch.delenv(
        "PRIME_ROCM_SPARSE_INDEXER_FORWARD_CONTEXT_CACHE",
        raising=False,
    )
    calls = []

    def original(q, kv_cache, weights, context_lens, block_tables, max_model_len):
        calls.append(context_lens)
        return "result"

    wrapped = _wrap_rocm_sparse_indexer_bulk_context_lens(original)
    context_lens = _ContextLens()

    result = wrapped(
        SimpleNamespace(shape=(16, 1, 32, 128)),
        object(),
        object(),
        context_lens,
        object(),
        32768,
    )

    wrapped(
        SimpleNamespace(shape=(16, 1, 32, 128)),
        object(),
        object(),
        context_lens,
        object(),
        32768,
    )

    assert result == "result"
    assert context_lens.to_calls == [
        {"device": "cpu", "non_blocking": False},
        {"device": "cpu", "non_blocking": False},
    ]
    assert calls == [context_lens.host_copy, context_lens.host_copy]


def test_bulk_context_lens_forward_context_cache_reuses_host_copy(monkeypatch):
    monkeypatch.setenv(
        "PRIME_ROCM_SPARSE_INDEXER_FORWARD_CONTEXT_CACHE",
        "1",
    )
    additional_kwargs = {}
    _set_forward_context(monkeypatch, additional_kwargs)
    calls = []

    def original(q, kv_cache, weights, context_lens, block_tables, max_model_len):
        calls.append(context_lens)
        return "result"

    wrapped = _wrap_rocm_sparse_indexer_bulk_context_lens(original)
    context_lens = _ContextLens()
    args = (
        SimpleNamespace(shape=(16, 1, 32, 128)),
        object(),
        object(),
        context_lens,
        object(),
        32768,
    )

    assert wrapped(*args) == "result"
    assert wrapped(*args) == "result"

    assert context_lens.to_calls == [{"device": "cpu", "non_blocking": False}]
    assert calls == [context_lens.host_copy, context_lens.host_copy]
    cache = additional_kwargs["_prime_rl_rocm_sparse_indexer_context_lens_cpu_cache"]
    cached_source, cached_host = next(iter(cache.values()))
    assert cached_source is context_lens
    assert cached_host is context_lens.host_copy


def test_bulk_context_lens_forward_context_cache_deduplicates_equivalent_views(
    monkeypatch,
):
    monkeypatch.setenv(
        "PRIME_ROCM_SPARSE_INDEXER_FORWARD_CONTEXT_CACHE",
        "1",
    )
    _set_forward_context(monkeypatch, {})
    calls = []

    def original(q, kv_cache, weights, context_lens, block_tables, max_model_len):
        calls.append(context_lens)
        return "result"

    wrapped = _wrap_rocm_sparse_indexer_bulk_context_lens(original)
    first = _ContextLens()
    equivalent_view = _ContextLens()
    q = SimpleNamespace(shape=(16, 1, 32, 128))

    wrapped(q, object(), object(), first, object(), 32768)
    wrapped(q, object(), object(), equivalent_view, object(), 32768)

    assert first.to_calls == [{"device": "cpu", "non_blocking": False}]
    assert equivalent_view.to_calls == []
    assert calls == [first.host_copy, first.host_copy]


def test_bulk_context_lens_forward_context_cache_separates_distinct_tensors(
    monkeypatch,
):
    monkeypatch.setenv(
        "PRIME_ROCM_SPARSE_INDEXER_FORWARD_CONTEXT_CACHE",
        "1",
    )
    _set_forward_context(monkeypatch, {})
    calls = []

    def original(q, kv_cache, weights, context_lens, block_tables, max_model_len):
        calls.append(context_lens)
        return "result"

    wrapped = _wrap_rocm_sparse_indexer_bulk_context_lens(original)
    first = _ContextLens(data_ptr=1024)
    second = _ContextLens(data_ptr=2048)
    q = SimpleNamespace(shape=(16, 1, 32, 128))

    wrapped(q, object(), object(), first, object(), 32768)
    wrapped(q, object(), object(), second, object(), 32768)

    assert first.to_calls == [{"device": "cpu", "non_blocking": False}]
    assert second.to_calls == [{"device": "cpu", "non_blocking": False}]
    assert calls == [first.host_copy, second.host_copy]


def test_bulk_context_lens_forward_context_cache_is_scoped_to_context(monkeypatch):
    monkeypatch.setenv(
        "PRIME_ROCM_SPARSE_INDEXER_FORWARD_CONTEXT_CACHE",
        "1",
    )
    from vllm import forward_context

    contexts = [
        SimpleNamespace(additional_kwargs={}),
        SimpleNamespace(additional_kwargs={}),
    ]
    active_context = [contexts[0]]
    monkeypatch.setattr(forward_context, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(
        forward_context,
        "get_forward_context",
        lambda: active_context[0],
    )

    def original(q, kv_cache, weights, context_lens, block_tables, max_model_len):
        return context_lens

    wrapped = _wrap_rocm_sparse_indexer_bulk_context_lens(original)
    context_lens = _ContextLens()
    q = SimpleNamespace(shape=(16, 1, 32, 128))

    assert wrapped(q, object(), object(), context_lens, object(), 32768) is context_lens.host_copy
    active_context[0] = contexts[1]
    assert wrapped(q, object(), object(), context_lens, object(), 32768) is context_lens.host_copy

    assert context_lens.to_calls == [
        {"device": "cpu", "non_blocking": False},
        {"device": "cpu", "non_blocking": False},
    ]
    assert contexts[0].additional_kwargs is not contexts[1].additional_kwargs


def test_bulk_context_lens_forward_context_cache_falls_back_without_context(
    monkeypatch,
):
    monkeypatch.setenv(
        "PRIME_ROCM_SPARSE_INDEXER_FORWARD_CONTEXT_CACHE",
        "1",
    )
    from vllm import forward_context

    monkeypatch.setattr(forward_context, "is_forward_context_available", lambda: False)
    monkeypatch.setattr(
        forward_context,
        "get_forward_context",
        lambda: pytest.fail("get_forward_context should not be called"),
    )

    def original(q, kv_cache, weights, context_lens, block_tables, max_model_len):
        return context_lens

    wrapped = _wrap_rocm_sparse_indexer_bulk_context_lens(original)
    context_lens = _ContextLens()
    q = SimpleNamespace(shape=(16, 1, 32, 128))

    wrapped(q, object(), object(), context_lens, object(), 32768)
    wrapped(q, object(), object(), context_lens, object(), 32768)

    assert context_lens.to_calls == [
        {"device": "cpu", "non_blocking": False},
        {"device": "cpu", "non_blocking": False},
    ]


def test_bulk_context_lens_delegates_multi_token_decode_unchanged(monkeypatch):
    monkeypatch.setenv(
        "PRIME_ROCM_SPARSE_INDEXER_FORWARD_CONTEXT_CACHE",
        "1",
    )
    calls = []

    def original(q, kv_cache, weights, context_lens, block_tables, max_model_len):
        calls.append(context_lens)
        return "result"

    wrapped = _wrap_rocm_sparse_indexer_bulk_context_lens(original)
    context_lens = _ContextLens()

    wrapped(
        SimpleNamespace(shape=(8, 2, 32, 128)),
        object(),
        object(),
        context_lens,
        object(),
        32768,
    )

    assert context_lens.to_calls == []
    assert calls == [context_lens]


def test_bulk_context_lens_wrapper_is_idempotent():
    def original(q, kv_cache, weights, context_lens, block_tables, max_model_len):
        return context_lens

    wrapped = _wrap_rocm_sparse_indexer_bulk_context_lens(original)

    assert _wrap_rocm_sparse_indexer_bulk_context_lens(wrapped) is wrapped


def test_triton_decode_env_is_opt_in(monkeypatch):
    monkeypatch.delenv("PRIME_ROCM_SPARSE_INDEXER_TRITON_DECODE", raising=False)
    monkeypatch.delenv("PRIME_ROCM_SPARSE_INDEXER_BULK_CONTEXT_LENS", raising=False)

    assert not _rocm_sparse_indexer_triton_decode_from_env()

    monkeypatch.setenv("PRIME_ROCM_SPARSE_INDEXER_TRITON_DECODE", "1")

    assert _rocm_sparse_indexer_triton_decode_from_env()


def test_sparse_indexer_fallback_rejects_ambiguous_opt_in(monkeypatch):
    monkeypatch.setenv("PRIME_ROCM_SPARSE_INDEXER_FALLBACK", "true")

    with pytest.raises(
        ValueError,
        match="PRIME_ROCM_SPARSE_INDEXER_FALLBACK must be unset, '0', or '1'",
    ):
        monkey_patch_rocm_sparse_indexer_fallback()


def test_triton_decode_rejects_bulk_host_context_lens(monkeypatch):
    monkeypatch.setenv("PRIME_ROCM_SPARSE_INDEXER_TRITON_DECODE", "1")
    monkeypatch.setenv("PRIME_ROCM_SPARSE_INDEXER_BULK_CONTEXT_LENS", "1")

    with pytest.raises(
        ValueError,
        match="TRITON_DECODE keeps context lengths on the GPU",
    ):
        _rocm_sparse_indexer_triton_decode_from_env()


@pytest.mark.parametrize(
    "name",
    (
        "PRIME_ROCM_SPARSE_INDEXER_TRITON_DECODE",
        "PRIME_ROCM_SPARSE_INDEXER_BULK_CONTEXT_LENS",
    ),
)
def test_triton_decode_rejects_ambiguous_binary_environment(
    monkeypatch,
    name,
):
    monkeypatch.delenv("PRIME_ROCM_SPARSE_INDEXER_TRITON_DECODE", raising=False)
    monkeypatch.delenv("PRIME_ROCM_SPARSE_INDEXER_BULK_CONTEXT_LENS", raising=False)
    monkeypatch.setenv(name, "true")

    with pytest.raises(ValueError, match=rf"{name} must be unset, '0', or '1'"):
        _rocm_sparse_indexer_triton_decode_from_env()


def test_prefill_k_chunk_env_is_opt_in(monkeypatch):
    monkeypatch.delenv("PRIME_ROCM_SPARSE_INDEXER_PREFILL_K_CHUNK", raising=False)

    assert _rocm_sparse_indexer_prefill_k_chunk_from_env() is None


@pytest.mark.parametrize("value", ["", "zero", "0", "-8"])
def test_prefill_k_chunk_env_requires_positive_integer(monkeypatch, value):
    monkeypatch.setenv("PRIME_ROCM_SPARSE_INDEXER_PREFILL_K_CHUNK", value)

    with pytest.raises(
        ValueError,
        match="PRIME_ROCM_SPARSE_INDEXER_PREFILL_K_CHUNK must be a positive integer",
    ):
        _rocm_sparse_indexer_prefill_k_chunk_from_env()


def test_prefill_k_chunk_env_reads_positive_integer(monkeypatch):
    monkeypatch.setenv("PRIME_ROCM_SPARSE_INDEXER_PREFILL_K_CHUNK", "2048")

    assert _rocm_sparse_indexer_prefill_k_chunk_from_env() == 2048


def test_prefill_k_chunk_delegates_short_k_unchanged():
    q = object()
    k = torch.arange(4)
    scale = torch.arange(4) + 10
    weights = object()
    starts = torch.tensor([0, 1])
    ends = torch.tensor([2, 4])
    calls = []

    def original(*args):
        calls.append(args)
        return "result"

    wrapped = _wrap_rocm_sparse_indexer_prefill_k_chunk(original, chunk_size=4)

    result = wrapped(q, (k, scale), weights, starts, ends)

    assert result == "result"
    assert len(calls) == 1
    assert calls[0][0] is q
    assert calls[0][1][0] is k
    assert calls[0][1][1] is scale
    assert calls[0][2] is weights
    assert calls[0][3] is starts
    assert calls[0][4] is ends


def test_prefill_k_chunk_adjusts_ranges_and_concatenates_in_k_order():
    q = SimpleNamespace(shape=(3, 2, 1))
    k = torch.arange(10).view(10, 1)
    scale = (torch.arange(10) + 100).view(10, 1)
    weights = object()
    starts = torch.tensor([0, 2, 7])
    ends = torch.tensor([3, 8, 10])
    calls = []

    def original(q_arg, kv_arg, weights_arg, starts_arg, ends_arg):
        chunk_k, chunk_scale = kv_arg
        calls.append(
            {
                "q": q_arg,
                "k": chunk_k.clone(),
                "scale": chunk_scale.clone(),
                "weights": weights_arg,
                "starts": starts_arg.clone(),
                "ends": ends_arg.clone(),
            }
        )
        return chunk_k[:, 0].expand(q_arg.shape[0], -1)

    wrapped = _wrap_rocm_sparse_indexer_prefill_k_chunk(original, chunk_size=4)

    logits = wrapped(q, (k, scale), weights, starts, ends)

    assert len(calls) == 3
    assert [call["k"][:, 0].tolist() for call in calls] == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9],
    ]
    assert [call["scale"][:, 0].tolist() for call in calls] == [
        [100, 101, 102, 103],
        [104, 105, 106, 107],
        [108, 109],
    ]
    assert [call["starts"].tolist() for call in calls] == [
        [0, 2, 4],
        [0, 0, 3],
        [0, 0, 0],
    ]
    assert [call["ends"].tolist() for call in calls] == [
        [3, 4, 4],
        [0, 4, 4],
        [0, 0, 2],
    ]
    assert all(call["q"] is q and call["weights"] is weights for call in calls)
    assert logits.tolist() == [
        list(range(10)),
        list(range(10)),
        list(range(10)),
    ]


def test_prefill_k_chunk_wrapper_validates_size_and_is_idempotent():
    def original(q, kv, weights, starts, ends):
        return q

    with pytest.raises(ValueError, match="chunk_size must be a positive integer"):
        _wrap_rocm_sparse_indexer_prefill_k_chunk(original, chunk_size=0)

    wrapped = _wrap_rocm_sparse_indexer_prefill_k_chunk(original, chunk_size=4)

    assert _wrap_rocm_sparse_indexer_prefill_k_chunk(wrapped, chunk_size=4) is wrapped


def test_sparse_mla_prefill_q_chunk_env_is_opt_in(monkeypatch):
    monkeypatch.delenv("PRIME_ROCM_SPARSE_MLA_PREFILL_Q_CHUNK", raising=False)

    assert _rocm_sparse_mla_prefill_q_chunk_from_env() is None


def test_sparse_mla_backend_env_is_opt_in(monkeypatch):
    monkeypatch.delenv("PRIME_ROCM_SPARSE_MLA_REFERENCE_FALLBACK", raising=False)
    monkeypatch.delenv("PRIME_ROCM_SPARSE_MLA_TRITON", raising=False)

    assert _rocm_sparse_mla_backend_from_env() is None


def test_sparse_mla_backend_env_selects_triton(monkeypatch):
    monkeypatch.delenv("PRIME_ROCM_SPARSE_MLA_REFERENCE_FALLBACK", raising=False)
    monkeypatch.setenv("PRIME_ROCM_SPARSE_MLA_TRITON", "1")

    assert _rocm_sparse_mla_backend_from_env() == "triton"


@pytest.mark.parametrize(
    "name",
    (
        "PRIME_ROCM_SPARSE_MLA_REFERENCE_FALLBACK",
        "PRIME_ROCM_SPARSE_MLA_TRITON",
    ),
)
def test_sparse_mla_backend_rejects_ambiguous_binary_environment(
    monkeypatch,
    name,
):
    monkeypatch.delenv("PRIME_ROCM_SPARSE_MLA_REFERENCE_FALLBACK", raising=False)
    monkeypatch.delenv("PRIME_ROCM_SPARSE_MLA_TRITON", raising=False)
    monkeypatch.setenv(name, "true")

    with pytest.raises(ValueError, match=rf"{name} must be unset, '0', or '1'"):
        _rocm_sparse_mla_backend_from_env()


def test_sparse_mla_backend_env_rejects_conflicting_selectors(monkeypatch):
    monkeypatch.setenv("PRIME_ROCM_SPARSE_MLA_REFERENCE_FALLBACK", "1")
    monkeypatch.setenv("PRIME_ROCM_SPARSE_MLA_TRITON", "1")

    with pytest.raises(ValueError, match="mutually exclusive"):
        _rocm_sparse_mla_backend_from_env()


def test_sparse_mla_triton_rejects_reference_query_chunk(monkeypatch):
    monkeypatch.delenv("PRIME_ROCM_SPARSE_MLA_REFERENCE_FALLBACK", raising=False)
    monkeypatch.setenv("PRIME_ROCM_SPARSE_MLA_TRITON", "1")
    monkeypatch.setenv("PRIME_ROCM_SPARSE_MLA_PREFILL_Q_CHUNK", "16")

    with pytest.raises(ValueError, match="only applies"):
        _rocm_sparse_mla_backend_from_env()


@pytest.mark.parametrize("cache_dtype", ["auto", "bfloat16"])
def test_sparse_mla_runtime_contract_accepts_exact_glm52(cache_dtype):
    config = _make_glm52_sparse_vllm_config(cache_dtype=cache_dtype)

    assert rocm_gfx90a_mla_bmm_integration.glm52_mla_runtime_mismatches(config) == ()
    assert _rocm_sparse_mla_runtime_mismatches(config) == ()
    assert _rocm_sparse_mla_cache_spec_mismatches(_make_sparse_kv_cache_spec()) == ()


@pytest.mark.parametrize(
    ("config", "message"),
    (
        (
            _make_glm52_sparse_vllm_config(model_dtype=torch.float16),
            "model_config.dtype",
        ),
        (
            _make_glm52_sparse_vllm_config(hf_overrides={"hidden_size": 4096}),
            "hidden_size",
        ),
        (
            _make_glm52_sparse_vllm_config(
                parallel_overrides={"tensor_parallel_size": 4},
            ),
            "tensor_parallel_size",
        ),
        (
            _make_glm52_sparse_vllm_config(
                parallel_overrides={"data_parallel_size": 1},
            ),
            "data_parallel_size",
        ),
        (
            _make_glm52_sparse_vllm_config(
                parallel_overrides={"pipeline_parallel_size": 2},
            ),
            "pipeline_parallel_size",
        ),
        (
            _make_glm52_sparse_vllm_config(
                parallel_overrides={"enable_expert_parallel": False},
            ),
            "enable_expert_parallel",
        ),
        (
            _make_glm52_sparse_vllm_config(cache_dtype="fp8"),
            "cache_config.cache_dtype",
        ),
    ),
)
def test_sparse_mla_builder_rejects_incompatible_runtime_before_aiter(
    monkeypatch,
    config,
    message,
):
    monkeypatch.setenv("PRIME_ROCM_SPARSE_MLA_TRITON", "1")
    monkeypatch.delenv("PRIME_ROCM_SPARSE_MLA_REFERENCE_FALLBACK", raising=False)
    impl_cls, _ = _install_fake_sparse_mla_vllm(
        monkeypatch,
        _sparse_mla_reference([]),
    )
    monkey_patch_rocm_sparse_mla_reference_fallback()

    with pytest.raises(ValueError, match=message):
        impl_cls.metadata_builder_cls(
            _make_sparse_kv_cache_spec(),
            [],
            config,
            torch.device("cuda"),
        )

    assert impl_cls.metadata_builder_cls.init_calls == 0


@pytest.mark.parametrize(
    ("kv_cache_spec", "message"),
    (
        (
            _make_sparse_kv_cache_spec(dtype=torch.float16),
            "kv_cache_spec.dtype",
        ),
        (
            _make_sparse_kv_cache_spec(cache_dtype_str="fp8_ds_mla"),
            "kv_cache_spec.cache_dtype_str",
        ),
    ),
)
def test_sparse_mla_builder_rejects_non_bf16_cache_before_aiter(
    monkeypatch,
    kv_cache_spec,
    message,
):
    monkeypatch.setenv("PRIME_ROCM_SPARSE_MLA_TRITON", "1")
    monkeypatch.delenv("PRIME_ROCM_SPARSE_MLA_REFERENCE_FALLBACK", raising=False)
    impl_cls, _ = _install_fake_sparse_mla_vllm(
        monkeypatch,
        _sparse_mla_reference([]),
    )
    monkey_patch_rocm_sparse_mla_reference_fallback()

    with pytest.raises(ValueError, match=message):
        impl_cls.metadata_builder_cls(
            kv_cache_spec,
            [],
            _make_glm52_sparse_vllm_config(),
            torch.device("cuda"),
        )

    assert impl_cls.metadata_builder_cls.init_calls == 0


def test_sparse_mla_builder_suppresses_installed_aiter_metadata(monkeypatch):
    monkeypatch.setenv("PRIME_ROCM_SPARSE_MLA_TRITON", "1")
    monkeypatch.delenv("PRIME_ROCM_SPARSE_MLA_REFERENCE_FALLBACK", raising=False)
    impl_cls, _ = _install_fake_sparse_mla_vllm(
        monkeypatch,
        _sparse_mla_reference([]),
        aiter_available=True,
    )
    monkey_patch_rocm_sparse_mla_reference_fallback()

    builder = impl_cls.metadata_builder_cls(
        _make_sparse_kv_cache_spec(),
        [],
        _make_glm52_sparse_vllm_config(),
        torch.device("cuda"),
    )
    persistent_attributes = (
        "_mla_work_meta_data",
        "_mla_work_indptr",
        "_mla_work_info_set",
        "_mla_reduce_indptr",
        "_mla_reduce_final_map",
        "_mla_reduce_partial_map",
    )

    assert all(getattr(builder, attribute) is None for attribute in persistent_attributes)
    builder.build(0, _sparse_mla_common_metadata())
    assert builder.metadata_launches == 0
    assert all(getattr(builder, attribute) is None for attribute in persistent_attributes)


def test_sparse_mla_builder_recovers_installed_aiter_metadata_failure(monkeypatch):
    monkeypatch.setenv("PRIME_ROCM_SPARSE_MLA_TRITON", "1")
    monkeypatch.delenv("PRIME_ROCM_SPARSE_MLA_REFERENCE_FALLBACK", raising=False)
    impl_cls, _ = _install_fake_sparse_mla_vllm(
        monkeypatch,
        _sparse_mla_reference([]),
        aiter_failure=RuntimeError("AITER metadata is unsupported on gfx90a"),
    )
    monkey_patch_rocm_sparse_mla_reference_fallback()

    builder = impl_cls.metadata_builder_cls(
        _make_sparse_kv_cache_spec(),
        [],
        _make_glm52_sparse_vllm_config(),
        torch.device("cuda"),
    )

    assert builder._mla_work_meta_data is None
    builder.build(0, _sparse_mla_common_metadata())
    assert builder.metadata_launches == 0


def test_sparse_mla_builder_recovers_only_absent_aiter_metadata(monkeypatch):
    monkeypatch.setenv("PRIME_ROCM_SPARSE_MLA_TRITON", "1")
    monkeypatch.delenv("PRIME_ROCM_SPARSE_MLA_REFERENCE_FALLBACK", raising=False)
    impl_cls, _ = _install_fake_sparse_mla_vllm(
        monkeypatch,
        _sparse_mla_reference([]),
        aiter_available=False,
    )
    monkey_patch_rocm_sparse_mla_reference_fallback()

    builder = impl_cls.metadata_builder_cls(
        _make_sparse_kv_cache_spec(),
        [],
        _make_glm52_sparse_vllm_config(),
        torch.device("cuda"),
    )

    assert builder._mla_work_meta_data is None
    builder.build(0, _sparse_mla_common_metadata())
    assert builder.metadata_launches == 0


def test_sparse_mla_builder_does_not_hide_unrelated_missing_dependency(monkeypatch):
    monkeypatch.setenv("PRIME_ROCM_SPARSE_MLA_TRITON", "1")
    monkeypatch.delenv("PRIME_ROCM_SPARSE_MLA_REFERENCE_FALLBACK", raising=False)
    impl_cls, _ = _install_fake_sparse_mla_vllm(
        monkeypatch,
        _sparse_mla_reference([]),
        aiter_failure=ModuleNotFoundError(
            "No module named 'unrelated_dependency'",
            name="unrelated_dependency",
        ),
    )
    monkey_patch_rocm_sparse_mla_reference_fallback()

    with pytest.raises(ModuleNotFoundError, match="unrelated_dependency"):
        impl_cls.metadata_builder_cls(
            _make_sparse_kv_cache_spec(),
            [],
            _make_glm52_sparse_vllm_config(),
            torch.device("cuda"),
        )


def test_sparse_mla_triton_patch_forwards_ragged_metadata_without_padding(
    monkeypatch,
):
    monkeypatch.delenv("PRIME_ROCM_SPARSE_MLA_REFERENCE_FALLBACK", raising=False)
    monkeypatch.delenv("PRIME_ROCM_SPARSE_MLA_PREFILL_Q_CHUNK", raising=False)
    monkeypatch.setenv("PRIME_ROCM_SPARSE_MLA_TRITON", "1")
    impl_cls, warnings = _install_fake_sparse_mla_vllm(
        monkeypatch,
        _sparse_mla_reference([]),
    )
    calls = []

    rocm_sparse_mla = impl_cls.sparse_kernel_module

    def fake_triton(q, kv, indices, indptr, scale, *, num_heads):
        calls.append((q, kv, indices, indptr, scale, num_heads))
        return q[:, :num_heads, :2].clone()

    monkeypatch.setattr(rocm_sparse_mla, "rocm_sparse_mla_bf16", fake_triton)

    monkey_patch_rocm_sparse_mla_reference_fallback()
    impl = _sparse_mla_impl(impl_cls)
    q, kv_cache, metadata = _sparse_mla_inputs(num_tokens=7)
    output = impl._forward_mla(None, q, kv_cache, metadata)

    assert torch.equal(output, q[:, : impl.num_heads, :2])
    assert len(calls) == 1
    call_q, call_kv, call_indices, call_indptr, call_scale, call_heads = calls[0]
    assert call_q is q
    assert call_kv is kv_cache
    assert call_indices is metadata.paged_kv_indices
    assert torch.equal(call_indptr, metadata.paged_kv_indptr[: q.shape[0] + 1])
    assert call_scale == impl.scale
    assert call_heads == impl.num_heads
    assert any(warning[0].startswith("Enabled PRIME_ROCM_SPARSE_MLA_TRITON") for warning in warnings)


@pytest.mark.parametrize("topk_tokens", [2047, 2049, True])
def test_sparse_mla_triton_patch_rejects_non_static_builder_topk(
    monkeypatch,
    topk_tokens,
):
    monkeypatch.delenv("PRIME_ROCM_SPARSE_MLA_REFERENCE_FALLBACK", raising=False)
    monkeypatch.setenv("PRIME_ROCM_SPARSE_MLA_TRITON", "1")
    impl_cls, _ = _install_fake_sparse_mla_vllm(
        monkeypatch,
        _sparse_mla_reference([]),
    )

    monkey_patch_rocm_sparse_mla_reference_fallback()
    config = _make_glm52_sparse_vllm_config(
        hf_overrides={"index_topk": topk_tokens},
    )

    with pytest.raises(ValueError, match="topk_tokens=2048 as a Python int"):
        impl_cls.metadata_builder_cls(
            _make_sparse_kv_cache_spec(),
            [],
            config,
            torch.device("cuda"),
        )


def test_sparse_mla_triton_patch_accepts_static_builder_topk(monkeypatch):
    monkeypatch.delenv("PRIME_ROCM_SPARSE_MLA_REFERENCE_FALLBACK", raising=False)
    monkeypatch.setenv("PRIME_ROCM_SPARSE_MLA_TRITON", "1")
    impl_cls, _ = _install_fake_sparse_mla_vllm(
        monkeypatch,
        _sparse_mla_reference([]),
    )

    monkey_patch_rocm_sparse_mla_reference_fallback()
    config = _make_glm52_sparse_vllm_config()

    builder = impl_cls.metadata_builder_cls(
        _make_sparse_kv_cache_spec(),
        [],
        config,
        torch.device("cuda"),
    )

    assert builder.topk_tokens == 2048


@pytest.mark.parametrize("topk_tokens", [2047, 2049, torch.tensor(2048)])
def test_sparse_mla_triton_patch_rejects_non_static_forward_topk(
    monkeypatch,
    topk_tokens,
):
    monkeypatch.delenv("PRIME_ROCM_SPARSE_MLA_REFERENCE_FALLBACK", raising=False)
    monkeypatch.setenv("PRIME_ROCM_SPARSE_MLA_TRITON", "1")
    impl_cls, _ = _install_fake_sparse_mla_vllm(
        monkeypatch,
        _sparse_mla_reference([]),
    )

    monkey_patch_rocm_sparse_mla_reference_fallback()
    impl = _sparse_mla_impl(impl_cls)
    q, kv_cache, metadata = _sparse_mla_inputs(num_tokens=1)
    metadata.topk_tokens = topk_tokens

    with pytest.raises(ValueError, match="topk_tokens=2048 as a Python int"):
        impl._forward_mla(None, q, kv_cache, metadata)


@pytest.mark.parametrize("value", ["", "chunk", "0", "-1"])
def test_sparse_mla_prefill_q_chunk_env_requires_positive_integer(monkeypatch, value):
    monkeypatch.setenv("PRIME_ROCM_SPARSE_MLA_REFERENCE_FALLBACK", "1")
    monkeypatch.setenv("PRIME_ROCM_SPARSE_MLA_PREFILL_Q_CHUNK", value)
    _install_fake_sparse_mla_vllm(
        monkeypatch,
        _sparse_mla_reference([]),
    )

    with pytest.raises(
        ValueError,
        match="PRIME_ROCM_SPARSE_MLA_PREFILL_Q_CHUNK must be a positive integer",
    ):
        monkey_patch_rocm_sparse_mla_reference_fallback()


def test_sparse_mla_prefill_q_chunk_env_reads_positive_integer(monkeypatch):
    monkeypatch.setenv("PRIME_ROCM_SPARSE_MLA_PREFILL_Q_CHUNK", "512")

    assert _rocm_sparse_mla_prefill_q_chunk_from_env() == 512


def test_sparse_mla_prefill_q_chunk_preserves_exact_output_and_bounds_calls(
    monkeypatch,
):
    monkeypatch.setenv("PRIME_ROCM_SPARSE_MLA_REFERENCE_FALLBACK", "1")
    monkeypatch.setenv("PRIME_ROCM_SPARSE_MLA_PREFILL_Q_CHUNK", "3")
    calls = []
    impl_cls, warnings = _install_fake_sparse_mla_vllm(
        monkeypatch,
        _sparse_mla_reference(calls),
    )

    monkey_patch_rocm_sparse_mla_reference_fallback()
    impl = _sparse_mla_impl(impl_cls)
    q, kv_cache, metadata = _sparse_mla_inputs(num_tokens=7)
    output = impl._forward_mla(None, q, kv_cache, metadata)

    first_indices = metadata.paged_kv_indices[:: metadata.topk_tokens].to(q.dtype)
    expected = (q[..., : impl.kv_lora_rank] + first_indices[:, None, None])[
        :,
        : impl.num_heads,
    ]
    assert torch.equal(output, expected)
    assert output.dtype == q.dtype
    assert output.device == q.device
    assert [call["q"].shape[0] for call in calls] == [3, 3, 1]
    assert all(call["q"].shape[0] <= 3 for call in calls)
    assert all(call["indices"].shape == (call["q"].shape[0], 1, metadata.topk_tokens) for call in calls)
    assert torch.equal(
        torch.cat([call["indices"] for call in calls], dim=0)[:, 0],
        metadata.paged_kv_indices.view(7, metadata.topk_tokens),
    )
    assert len({id(call["kv"]) for call in calls}) == 1
    assert all(call["scale"] == impl.scale for call in calls)
    assert all(call["kv_lora_rank"] == impl.kv_lora_rank for call in calls)
    assert any(warning[0].startswith("Enabled PRIME_ROCM_SPARSE_MLA_PREFILL_Q_CHUNK") for warning in warnings)


def test_sparse_mla_reference_patch_makes_overlong_row_empty(monkeypatch):
    monkeypatch.setenv("PRIME_ROCM_SPARSE_MLA_REFERENCE_FALLBACK", "1")
    calls = []
    impl_cls, _ = _install_fake_sparse_mla_vllm(
        monkeypatch,
        _sparse_mla_reference(calls),
    )

    monkey_patch_rocm_sparse_mla_reference_fallback()
    impl = _sparse_mla_impl(impl_cls)
    q, kv_cache, metadata = _sparse_mla_inputs(num_tokens=1)
    metadata.paged_kv_indices = torch.cat((metadata.paged_kv_indices, torch.zeros(1, dtype=torch.int32)))
    metadata.paged_kv_indptr = torch.tensor([0, 2049], dtype=torch.int32)

    output = impl._forward_mla(None, q, kv_cache, metadata)

    assert torch.count_nonzero(output).item() == 0
    assert calls[0]["indices"].eq(-1).all()


@pytest.mark.parametrize("chunk_size", [None, 8])
def test_sparse_mla_prefill_q_chunk_default_and_large_chunk_use_single_call(
    monkeypatch,
    chunk_size,
):
    monkeypatch.setenv("PRIME_ROCM_SPARSE_MLA_REFERENCE_FALLBACK", "1")
    if chunk_size is None:
        monkeypatch.delenv("PRIME_ROCM_SPARSE_MLA_PREFILL_Q_CHUNK", raising=False)
    else:
        monkeypatch.setenv(
            "PRIME_ROCM_SPARSE_MLA_PREFILL_Q_CHUNK",
            str(chunk_size),
        )
    calls = []
    impl_cls, _ = _install_fake_sparse_mla_vllm(
        monkeypatch,
        _sparse_mla_reference(calls),
    )

    monkey_patch_rocm_sparse_mla_reference_fallback()
    impl = _sparse_mla_impl(impl_cls)
    q, kv_cache, metadata = _sparse_mla_inputs(num_tokens=4)
    impl._forward_mla(None, q, kv_cache, metadata)

    assert len(calls) == 1
    assert calls[0]["q"] is q
    assert calls[0]["q"].shape[0] == 4


def test_sparse_mla_prefill_q_chunk_handles_empty_query_with_single_call(
    monkeypatch,
):
    monkeypatch.setenv("PRIME_ROCM_SPARSE_MLA_REFERENCE_FALLBACK", "1")
    monkeypatch.setenv("PRIME_ROCM_SPARSE_MLA_PREFILL_Q_CHUNK", "2")
    calls = []
    impl_cls, _ = _install_fake_sparse_mla_vllm(
        monkeypatch,
        _sparse_mla_reference(calls),
    )

    monkey_patch_rocm_sparse_mla_reference_fallback()
    impl = _sparse_mla_impl(impl_cls)
    q, kv_cache, metadata = _sparse_mla_inputs(num_tokens=0)
    output = impl._forward_mla(None, q, kv_cache, metadata)

    assert output.shape == (0, impl.num_heads, impl.kv_lora_rank)
    assert len(calls) == 1
    assert calls[0]["q"] is q
    assert calls[0]["indices"].shape == (0, 1, metadata.topk_tokens)


def test_sparse_mla_prefill_q_chunk_has_no_cross_forward_state(monkeypatch):
    monkeypatch.setenv("PRIME_ROCM_SPARSE_MLA_REFERENCE_FALLBACK", "1")
    monkeypatch.setenv("PRIME_ROCM_SPARSE_MLA_PREFILL_Q_CHUNK", "2")
    calls = []
    impl_cls, _ = _install_fake_sparse_mla_vllm(
        monkeypatch,
        _sparse_mla_reference(calls),
    )

    monkey_patch_rocm_sparse_mla_reference_fallback()
    impl = _sparse_mla_impl(impl_cls)
    first_q, kv_cache, metadata = _sparse_mla_inputs(num_tokens=5)
    second_q = first_q + 1000

    first_output = impl._forward_mla(None, first_q, kv_cache, metadata)
    second_output = impl._forward_mla(None, second_q, kv_cache, metadata)

    assert [call["q"].shape[0] for call in calls] == [2, 2, 1, 2, 2, 1]
    assert torch.equal(torch.cat([call["q"] for call in calls[:3]]), first_q)
    assert torch.equal(torch.cat([call["q"] for call in calls[3:]]), second_q)
    assert torch.equal(second_output - first_output, torch.full_like(first_output, 1000))
