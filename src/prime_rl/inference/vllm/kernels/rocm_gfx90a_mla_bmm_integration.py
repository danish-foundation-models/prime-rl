# SPDX-License-Identifier: Apache-2.0

"""Fail-closed vLLM integration for the GLM-5.2 gfx90a MLA BMMs.

vLLM already exposes ``MLAAttention._v_up_proj`` as a narrow replacement
boundary. Its Q-absorb BMM is inline in ``forward_impl`` in the pinned
distribution, so replacing it safely requires this minimal upstream hook::

    def _q_absorb(self, q_nope: torch.Tensor, out: torch.Tensor) -> None:
        torch.bmm(q_nope, self.W_UK_T, out=out)

and replacing the inline ``torch.bmm`` call with
``self._q_absorb(mqa_q_nope, mqa_ql_nope)``. PRIME-RL deliberately does not
patch ``torch.bmm``, install a dispatch mode, wrap a weight in a Tensor
subclass, or duplicate ``forward_impl``. The behavior-preserving vLLM change
is recorded in ``docs/vllm-ee0da84-mla-q-absorb-hook.patch``.

Only the measured M=16 and M=32 cells select the custom kernels. Calls outside
those cells retain vLLM's original methods.
"""

from __future__ import annotations

import functools
import inspect
import json
import logging
import os
import socket
from dataclasses import dataclass
from importlib.metadata import version as distribution_version
from typing import Any, Mapping

import torch
import torch.distributed as dist

from prime_rl.inference.vllm.kernels.rocm_gfx90a_mla_bmm import (
    GLM52_MLA_BMM_MAX_TOKENS,
    GLM52_MLA_KV_LORA_RANK,
    GLM52_MLA_LOCAL_HEADS,
    GLM52_MLA_QK_NOPE_HEAD_DIM,
    GLM52_MLA_VALUE_HEAD_DIM,
    GLM52_MLA_WEIGHT_BATCH_STRIDE,
    can_dispatch_gfx90a_mla_bmm,
    get_gfx90a_mla_bmm_config,
    gfx90a_glm52_mla_q_absorb,
    gfx90a_glm52_mla_v_up,
    validate_gfx90a_mla_device,
)

EXPECTED_VLLM_DISTRIBUTION = "0.24.0+lumi_aif_gfx90a_ee0da84"
AUDIT_ENV = "PRIME_ROCM_GFX90A_MLA_AUDIT"
AUDIT_SCHEMA_VERSION = 1
LOGGER = logging.getLogger(f"vllm.{__name__}")

_QK_ROPE_HEAD_DIM = 64
_Q_LORA_RANK = 2048
_GLOBAL_ATTENTION_HEADS = 64
_MLA_MARKER = "_prime_rl_glm52_gfx90a_mla_bmm"
_MLA_AUDIT_MARKER = "_prime_rl_glm52_gfx90a_mla_bmm_audited"
_PROCESS_PATCH_MARKER = "_prime_rl_glm52_gfx90a_mla_process_patch"
_PROCESS_CAPABILITIES_MARKER = "_prime_rl_glm52_gfx90a_mla_process_capabilities"
_PROCESS_AUDIT_ENABLED_MARKER = "_prime_rl_glm52_gfx90a_mla_process_audit_enabled"
_V_UP_PATCH_MARKER = "_prime_rl_glm52_gfx90a_mla_v_up_patch"
_Q_ABSORB_PATCH_MARKER = "_prime_rl_glm52_gfx90a_mla_q_absorb_patch"
_Q_ABSORB_HOOK = "_q_absorb"

# Transformers aliases ``head_dim`` to ``qk_rope_head_dim`` for this model.
# The canonical qk fields below fully specify the MLA dimensions.
_GLM52_MLA_CONFIG_FIELDS: dict[str, object] = {
    "model_type": "glm_moe_dsa",
    "hidden_size": 6144,
    "num_attention_heads": _GLOBAL_ATTENTION_HEADS,
    "num_key_value_heads": _GLOBAL_ATTENTION_HEADS,
    "q_lora_rank": _Q_LORA_RANK,
    "kv_lora_rank": GLM52_MLA_KV_LORA_RANK,
    "qk_head_dim": GLM52_MLA_QK_NOPE_HEAD_DIM + _QK_ROPE_HEAD_DIM,
    "qk_nope_head_dim": GLM52_MLA_QK_NOPE_HEAD_DIM,
    "qk_rope_head_dim": _QK_ROPE_HEAD_DIM,
    "v_head_dim": GLM52_MLA_VALUE_HEAD_DIM,
    "num_hidden_layers": 78,
    "index_topk": 2048,
}
_GLM52_FP8_CONFIG_FIELDS: dict[str, object] = {
    "activation_scheme": "dynamic",
    "fmt": "e4m3",
    "quant_method": "fp8",
    "weight_block_size": (128, 128),
}

_EXPECTED_INIT_ARGS = (
    "self",
    "num_heads",
    "scale",
    "qk_nope_head_dim",
    "qk_rope_head_dim",
    "v_head_dim",
    "q_lora_rank",
    "kv_lora_rank",
    "kv_b_proj",
    "cache_config",
    "quant_config",
    "prefix",
    "attn_backend",
    "use_sparse",
    "indexer",
    "topk_indices_buffer",
    "extra_impl_args",
)
_EXPECTED_FORWARD_IMPL_ARGS = (
    "self",
    "q",
    "k_c_normed",
    "k_pe",
    "kv_cache",
    "attn_metadata",
    "output",
    "output_scale",
    "output_block_scale",
    "quant_group_size",
    "quant_scale_ue8m0",
    "quant_col_major",
    "quant_tma_aligned",
)
_EXPECTED_PROCESS_ARGS = ("self", "act_dtype")
_EXPECTED_V_UP_ARGS = ("self", "x", "out")
_EXPECTED_Q_ABSORB_ARGS = ("self", "q_nope", "out")
_EXPECTED_UNIFIED_MLA_ARGS = (
    "q",
    "kv_c_normed",
    "k_pe",
    "output",
    "layer_name",
    "output_scale",
    "output_block_scale",
    "kv_cache_dummy_dep",
    "quant_group_size",
    "quant_scale_ue8m0",
    "quant_col_major",
    "quant_tma_aligned",
)
_EXPECTED_INIT_DEFAULTS: dict[str, object] = {
    "cache_config": None,
    "quant_config": None,
    "prefix": "",
    "attn_backend": None,
    "use_sparse": False,
    "indexer": None,
    "topk_indices_buffer": None,
}
_EXPECTED_FORWARD_IMPL_DEFAULTS: dict[str, object] = {
    "output_scale": None,
    "output_block_scale": None,
    "quant_group_size": None,
    "quant_scale_ue8m0": None,
    "quant_col_major": None,
    "quant_tma_aligned": None,
}
_EXPECTED_UNIFIED_MLA_DEFAULTS: dict[str, object] = {
    "output_scale": None,
    "output_block_scale": None,
    "kv_cache_dummy_dep": None,
    "quant_group_size": None,
    "quant_scale_ue8m0": None,
    "quant_col_major": None,
    "quant_tma_aligned": None,
}


@dataclass(frozen=True)
class MlaBmmIntegration:
    """Installed vLLM MLA replacement paths."""

    q_absorb: bool
    v_up: bool


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def mla_audit_requested(
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Read the MLA audit opt-in without accepting ambiguous values."""

    value = (os.environ if environ is None else environ).get(AUDIT_ENV)
    if value in (None, "0"):
        return False
    if value == "1":
        return True
    raise ValueError(f"{AUDIT_ENV} must be unset, '0', or '1'; got {value!r}")


def _required_environment_rank(name: str) -> int:
    value = os.environ.get(name)
    if value is None:
        raise RuntimeError(
            f"{AUDIT_ENV}=1 requires an initialized torch.distributed process group or an integer {name}"
        )
    try:
        rank = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    if rank < 0:
        raise ValueError(f"{name} must be non-negative, got {rank}")
    return rank


def _audit_process_fields() -> dict[str, object]:
    if dist.is_available() and dist.is_initialized():
        rank = int(dist.get_rank())
        if not torch.cuda.is_available():
            raise RuntimeError(f"{AUDIT_ENV}=1 requires a current accelerator device after distributed initialization")
        local_rank = int(torch.cuda.current_device())
        identity_source = "torch.distributed"
    else:
        rank = _required_environment_rank("RANK")
        local_rank = _required_environment_rank("LOCAL_RANK")
        identity_source = "environment"

    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "pid": os.getpid(),
        "rank": rank,
        "local_rank": local_rank,
        "identity_source": identity_source,
        "hostname": socket.gethostname(),
    }


def _layer_audit_payload(
    layer: object,
    capabilities: MlaBmmIntegration,
) -> dict[str, object]:
    layer_name = getattr(layer, "layer_name", None)
    _require(
        type(layer_name) is str,
        f"{AUDIT_ENV}=1 requires MLAAttention.layer_name to be a string; got {layer_name!r}",
    )
    return {
        "event": "prime_glm52_gfx90a_mla_bmm_layer",
        "layer_name": layer_name,
        "prefix": layer_name,
        "q_absorb": capabilities.q_absorb,
        "v_up": capabilities.v_up,
        **_audit_process_fields(),
    }


def _normalise_config_value(value: object) -> object:
    if isinstance(value, list):
        return tuple(value)
    return value


def glm52_mla_config_mismatches(config: object) -> tuple[str, ...]:
    """Describe fields that differ from the measured GLM-5.2 FP8 contract."""

    mismatches = []
    architectures = tuple(getattr(config, "architectures", ()) or ())
    if architectures != ("GlmMoeDsaForCausalLM",):
        mismatches.append(f"architectures={architectures!r} (expected ('GlmMoeDsaForCausalLM',))")
    for field, expected in _GLM52_MLA_CONFIG_FIELDS.items():
        actual = getattr(config, field, None)
        if actual != expected:
            mismatches.append(f"{field}={actual!r} (expected {expected!r})")

    quantization_config = getattr(config, "quantization_config", None)
    if not isinstance(quantization_config, Mapping):
        mismatches.append("quantization_config must be a mapping for the GLM-5.2 FP8 checkpoint")
    else:
        for field, expected in _GLM52_FP8_CONFIG_FIELDS.items():
            actual = _normalise_config_value(quantization_config.get(field))
            if actual != expected:
                mismatches.append(f"quantization_config.{field}={actual!r} (expected {expected!r})")
    return tuple(mismatches)


def glm52_mla_runtime_mismatches(vllm_config: object) -> tuple[str, ...]:
    """Validate topology and scheduler state that determine MLA BMM layouts."""

    mismatches = []
    model_config = getattr(vllm_config, "model_config", None)
    if model_config is None:
        return ("model_config is missing",)

    mismatches.extend(glm52_mla_config_mismatches(getattr(model_config, "hf_config", None)))
    dtype = getattr(model_config, "dtype", None)
    if dtype != torch.bfloat16:
        mismatches.append(f"model_config.dtype={dtype!r} (expected torch.bfloat16)")

    parallel_config = getattr(vllm_config, "parallel_config", None)
    tensor_parallel_size = getattr(parallel_config, "tensor_parallel_size", None)
    if tensor_parallel_size != 8:
        mismatches.append(f"tensor_parallel_size={tensor_parallel_size!r} (expected 8)")
    decode_context_parallel_size = getattr(parallel_config, "decode_context_parallel_size", None)
    if decode_context_parallel_size != 1:
        mismatches.append(f"decode_context_parallel_size={decode_context_parallel_size!r} (expected 1)")

    scheduler_config = getattr(vllm_config, "scheduler_config", None)
    max_num_batched_tokens = getattr(scheduler_config, "max_num_batched_tokens", None)
    if type(max_num_batched_tokens) is not int or not (0 < max_num_batched_tokens <= GLM52_MLA_BMM_MAX_TOKENS):
        mismatches.append(
            "max_num_batched_tokens="
            f"{max_num_batched_tokens!r} (expected a positive integer <= {GLM52_MLA_BMM_MAX_TOKENS})"
        )
    return tuple(mismatches)


def glm52_mla_instance_mismatches(layer: object) -> tuple[str, ...]:
    """Validate the attributes that construct the two local-head BMMs."""

    expected_fields = {
        "num_heads": GLM52_MLA_LOCAL_HEADS,
        "qk_nope_head_dim": GLM52_MLA_QK_NOPE_HEAD_DIM,
        "qk_rope_head_dim": _QK_ROPE_HEAD_DIM,
        "v_head_dim": GLM52_MLA_VALUE_HEAD_DIM,
        "q_lora_rank": _Q_LORA_RANK,
        "kv_lora_rank": GLM52_MLA_KV_LORA_RANK,
        "q_pad_num_heads": None,
        "use_sparse": True,
    }
    mismatches = [
        f"{field}={getattr(layer, field, None)!r} (expected {expected!r})"
        for field, expected in expected_fields.items()
        if getattr(layer, field, None) != expected
    ]
    for field in (
        "is_aiter_triton_fp8_bmm_enabled",
        "is_aiter_triton_fp4_bmm_enabled",
    ):
        value = getattr(layer, field, None)
        if value is not None and value is not False:
            mismatches.append(f"{field}={value!r} (expected None or False)")
    return tuple(mismatches)


def glm52_mla_weight_layout_mismatches(layer: object) -> tuple[str, ...]:
    """Validate the BF16 views produced by vLLM post-load processing."""

    mismatches = []
    expected = (
        (
            "W_UK_T",
            (
                GLM52_MLA_LOCAL_HEADS,
                GLM52_MLA_QK_NOPE_HEAD_DIM,
                GLM52_MLA_KV_LORA_RANK,
            ),
            (GLM52_MLA_WEIGHT_BATCH_STRIDE, 512, 1),
            0,
        ),
        (
            "W_UV",
            (
                GLM52_MLA_LOCAL_HEADS,
                GLM52_MLA_KV_LORA_RANK,
                GLM52_MLA_VALUE_HEAD_DIM,
            ),
            (GLM52_MLA_WEIGHT_BATCH_STRIDE, 1, 512),
            GLM52_MLA_QK_NOPE_HEAD_DIM * GLM52_MLA_KV_LORA_RANK,
        ),
    )
    devices = []
    storage_pointers = []
    for name, shape, stride, storage_offset in expected:
        tensor = getattr(layer, name, None)
        if not isinstance(tensor, torch.Tensor):
            mismatches.append(f"{name} is not a torch.Tensor")
            continue
        devices.append(tensor.device)
        storage_pointers.append(tensor.untyped_storage().data_ptr())
        if tuple(tensor.shape) != shape:
            mismatches.append(f"{name}.shape={tuple(tensor.shape)!r} (expected {shape!r})")
        if tensor.stride() != stride:
            mismatches.append(f"{name}.stride={tensor.stride()!r} (expected {stride!r})")
        if tensor.storage_offset() != storage_offset:
            mismatches.append(f"{name}.storage_offset={tensor.storage_offset()!r} (expected {storage_offset!r})")
        if tensor.dtype != torch.bfloat16:
            mismatches.append(f"{name}.dtype={tensor.dtype!r} (expected torch.bfloat16)")
    if len(devices) == 2 and devices[0] != devices[1]:
        mismatches.append(f"W_UK_T.device={devices[0]!r} differs from W_UV.device={devices[1]!r}")
    if len(storage_pointers) == 2 and storage_pointers[0] != storage_pointers[1]:
        mismatches.append("W_UK_T and W_UV do not share their post-load backing storage")
    return tuple(mismatches)


def _require_signature(
    symbol_name: str,
    symbol: Any,
    expected_names: tuple[str, ...],
    expected_defaults: Mapping[str, object] | None = None,
    *,
    last_parameter_kind: Any = inspect.Parameter.POSITIONAL_OR_KEYWORD,
) -> None:
    parameters = tuple(inspect.signature(symbol).parameters.values())
    actual_names = tuple(parameter.name for parameter in parameters)
    _require(
        actual_names == expected_names,
        f"gfx90a MLA BMM requires {symbol_name}{expected_names}; found parameters {actual_names}",
    )
    defaults = expected_defaults or {}
    for index, parameter in enumerate(parameters):
        expected_kind = last_parameter_kind if index == len(parameters) - 1 else inspect.Parameter.POSITIONAL_OR_KEYWORD
        _require(
            parameter.kind is expected_kind,
            f"gfx90a MLA BMM requires {symbol_name}.{parameter.name} to be {expected_kind.description}; "
            f"found {parameter.kind.description}",
        )
        expected_default = defaults.get(parameter.name, inspect.Parameter.empty)
        _require(
            parameter.default == expected_default,
            f"gfx90a MLA BMM requires {symbol_name}.{parameter.name}={expected_default!r}; found {parameter.default!r}",
        )


def _load_vllm_mla_symbols() -> tuple[type[Any], Any, Any]:
    from vllm.config import get_current_vllm_config
    from vllm.model_executor.layers.attention.mla_attention import (
        MLAAttention,
        unified_mla_attention_with_output,
    )

    return MLAAttention, unified_mla_attention_with_output, get_current_vllm_config


def _require_exact_vllm_distribution() -> None:
    installed = distribution_version("vllm")
    _require(
        installed == EXPECTED_VLLM_DISTRIBUTION,
        f"gfx90a MLA BMM requires vLLM {EXPECTED_VLLM_DISTRIBUTION!r}; found {installed!r}",
    )


def _require_exact_vllm_mla_abi(
    mla_type: type[Any],
    unified_mla_attention_with_output: Any,
    *,
    require_q_absorb: bool,
) -> bool:
    _require_exact_vllm_distribution()
    _require_signature(
        "MLAAttention.__init__",
        mla_type.__init__,
        _EXPECTED_INIT_ARGS,
        _EXPECTED_INIT_DEFAULTS,
        last_parameter_kind=inspect.Parameter.VAR_KEYWORD,
    )
    _require_signature(
        "MLAAttention.forward_impl", mla_type.forward_impl, _EXPECTED_FORWARD_IMPL_ARGS, _EXPECTED_FORWARD_IMPL_DEFAULTS
    )
    _require_signature(
        "MLAAttention.process_weights_after_loading", mla_type.process_weights_after_loading, _EXPECTED_PROCESS_ARGS
    )
    _require_signature("MLAAttention._v_up_proj", mla_type._v_up_proj, _EXPECTED_V_UP_ARGS)
    _require_signature(
        "unified_mla_attention_with_output",
        unified_mla_attention_with_output,
        _EXPECTED_UNIFIED_MLA_ARGS,
        _EXPECTED_UNIFIED_MLA_DEFAULTS,
    )

    q_absorb = getattr(mla_type, _Q_ABSORB_HOOK, None)
    if q_absorb is None:
        _require(
            not require_q_absorb,
            "full gfx90a MLA BMM integration requires vLLM "
            "MLAAttention._q_absorb(self, q_nope, out); the pinned vLLM keeps "
            "that torch.bmm inline in forward_impl",
        )
        return False
    _require_signature("MLAAttention._q_absorb", q_absorb, _EXPECTED_Q_ABSORB_ARGS)
    return True


def _validate_layer_after_loading(
    layer: object,
    mla_type: type[Any],
    act_dtype: torch.dtype,
    get_current_vllm_config: Any,
) -> None:
    _require(act_dtype == torch.bfloat16, f"gfx90a MLA BMM requires BF16 post-load weights; got {act_dtype}")
    _require(type(layer) is mla_type, f"gfx90a MLA BMM requires exact {mla_type.__name__}; got {type(layer).__name__}")

    runtime_mismatches = glm52_mla_runtime_mismatches(get_current_vllm_config())
    _require(
        not runtime_mismatches,
        "gfx90a MLA BMM requires the exact GLM-5.2 runtime contract: " + "; ".join(runtime_mismatches),
    )
    instance_mismatches = glm52_mla_instance_mismatches(layer)
    _require(
        not instance_mismatches,
        "gfx90a MLA BMM received an incompatible MLAAttention: " + "; ".join(instance_mismatches),
    )
    layout_mismatches = glm52_mla_weight_layout_mismatches(layer)
    _require(
        not layout_mismatches,
        "gfx90a MLA BMM received incompatible post-load weight views: " + "; ".join(layout_mismatches),
    )

    weight = layer.W_UK_T
    _require(weight.device.type == "cuda", f"gfx90a MLA BMM weights must be on a ROCm GPU; got {weight.device}")
    device_index = weight.device.index
    validate_gfx90a_mla_device(torch.cuda.current_device() if device_index is None else device_index)


def _install_process_weights_patch(
    mla_type: type[Any],
    get_current_vllm_config: Any,
    capabilities: MlaBmmIntegration,
    *,
    audit_enabled: bool,
) -> None:
    original = mla_type.process_weights_after_loading
    if getattr(original, _PROCESS_PATCH_MARKER, False):
        _require(
            getattr(original, _PROCESS_CAPABILITIES_MARKER, None) == capabilities,
            "gfx90a MLA BMM process patch is already installed with different capabilities",
        )
        _require(
            getattr(original, _PROCESS_AUDIT_ENABLED_MARKER, None) is audit_enabled,
            f"gfx90a MLA BMM process patch is already installed with a different {AUDIT_ENV} setting",
        )
        return

    @functools.wraps(original)
    def _patched_process_weights_after_loading(self, act_dtype):
        result = original(self, act_dtype)
        _validate_layer_after_loading(self, mla_type, act_dtype, get_current_vllm_config)
        audit_payload = None
        if audit_enabled and not getattr(self, _MLA_AUDIT_MARKER, False):
            audit_payload = _layer_audit_payload(self, capabilities)
        setattr(self, _MLA_MARKER, True)
        if audit_payload is not None:
            LOGGER.info(json.dumps(audit_payload, sort_keys=True, separators=(",", ":")))
            setattr(self, _MLA_AUDIT_MARKER, True)
        return result

    setattr(_patched_process_weights_after_loading, _PROCESS_PATCH_MARKER, True)
    setattr(_patched_process_weights_after_loading, _PROCESS_CAPABILITIES_MARKER, capabilities)
    setattr(_patched_process_weights_after_loading, _PROCESS_AUDIT_ENABLED_MARKER, audit_enabled)
    mla_type.process_weights_after_loading = _patched_process_weights_after_loading


def _install_v_up_patch(mla_type: type[Any]) -> None:
    original = mla_type._v_up_proj
    if getattr(original, _V_UP_PATCH_MARKER, False):
        return

    @functools.wraps(original)
    def _patched_v_up_proj(self, x, out):
        if not getattr(self, _MLA_MARKER, False):
            return original(self, x, out)

        if not isinstance(x, torch.Tensor) or not isinstance(out, torch.Tensor):
            return original(self, x, out)
        elements_per_token = self.num_heads * self.kv_lora_rank
        output_elements_per_token = self.num_heads * self.v_head_dim
        if (
            x.numel() % elements_per_token != 0
            or out.numel() % output_elements_per_token != 0
            or x.numel() // elements_per_token != out.numel() // output_elements_per_token
        ):
            return original(self, x, out)
        m = x.numel() // elements_per_token
        config = get_gfx90a_mla_bmm_config("v_up", m)
        if config is None:
            return original(self, x, out)

        attention_output = x.view(-1, self.num_heads, self.kv_lora_rank).transpose(0, 1)
        output = out.view(-1, self.num_heads, self.v_head_dim).transpose(0, 1)
        if not can_dispatch_gfx90a_mla_bmm("v_up", attention_output, self.W_UV, output):
            return original(self, x, out)
        gfx90a_glm52_mla_v_up(
            attention_output,
            self.W_UV,
            output,
            config=config,
            device_is_validated=True,
        )
        return None

    setattr(_patched_v_up_proj, _V_UP_PATCH_MARKER, True)
    mla_type._v_up_proj = _patched_v_up_proj


def _install_q_absorb_patch(mla_type: type[Any]) -> None:
    original = getattr(mla_type, _Q_ABSORB_HOOK)
    if getattr(original, _Q_ABSORB_PATCH_MARKER, False):
        return

    @functools.wraps(original)
    def _patched_q_absorb(self, q_nope, out):
        if not getattr(self, _MLA_MARKER, False):
            return original(self, q_nope, out)
        if not isinstance(q_nope, torch.Tensor) or q_nope.ndim != 3:
            return original(self, q_nope, out)
        config = get_gfx90a_mla_bmm_config("q_absorb", q_nope.shape[1])
        if config is None or not can_dispatch_gfx90a_mla_bmm("q_absorb", q_nope, self.W_UK_T, out):
            return original(self, q_nope, out)
        gfx90a_glm52_mla_q_absorb(
            q_nope,
            self.W_UK_T,
            out,
            config=config,
            device_is_validated=True,
        )
        return None

    setattr(_patched_q_absorb, _Q_ABSORB_PATCH_MARKER, True)
    setattr(mla_type, _Q_ABSORB_HOOK, _patched_q_absorb)


def install_glm52_gfx90a_mla_bmm(*, require_q_absorb: bool = True) -> MlaBmmIntegration:
    """Install exact GLM-5.2 MLA replacements at vLLM's narrow method seams.

    ``require_q_absorb=True`` is the production/full-path default. It fails
    before mutating vLLM when the upstream Q-absorb hook is absent.
    ``False`` installs the independently useful V-up replacement only.
    """

    audit_enabled = mla_audit_requested()
    _require_exact_vllm_distribution()
    mla_type, unified_mla_attention_with_output, get_current_vllm_config = _load_vllm_mla_symbols()
    has_q_absorb = _require_exact_vllm_mla_abi(
        mla_type,
        unified_mla_attention_with_output,
        require_q_absorb=require_q_absorb,
    )
    capabilities = MlaBmmIntegration(q_absorb=has_q_absorb, v_up=True)
    _install_process_weights_patch(
        mla_type,
        get_current_vllm_config,
        capabilities,
        audit_enabled=audit_enabled,
    )
    _install_v_up_patch(mla_type)
    if has_q_absorb:
        _install_q_absorb_patch(mla_type)
    return capabilities


__all__ = [
    "EXPECTED_VLLM_DISTRIBUTION",
    "MlaBmmIntegration",
    "glm52_mla_config_mismatches",
    "glm52_mla_instance_mismatches",
    "glm52_mla_runtime_mismatches",
    "glm52_mla_weight_layout_mismatches",
    "install_glm52_gfx90a_mla_bmm",
    "mla_audit_requested",
]
