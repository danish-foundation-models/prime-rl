#!/usr/bin/env bash

# Add the committed gfx90a vLLM source stack to an existing Singularity bind
# array without hiding the compiled extensions from the runtime image.
prime_rl_lumi_add_vllm_source_overlay() {
  local root=${PRIME_RL_VLLM_SOURCE_ROOT:-}
  local expected_rev=${PRIME_RL_VLLM_SOURCE_REV:-}
  local site=/opt/venv/lib/python3.12/site-packages/vllm
  local actual_rev relative

  [[ -n "$root" ]] || return 0
  [[ -d "$root/vllm" ]] || {
    echo "FATAL: PRIME_RL_VLLM_SOURCE_ROOT is not a vLLM checkout: $root" >&2
    return 1
  }

  actual_rev=$(git -C "$root" rev-parse HEAD)
  if [[ -n "$expected_rev" && "$actual_rev" != "$expected_rev"* ]]; then
    echo "FATAL: vLLM source revision mismatch: expected $expected_rev, found $actual_rev" >&2
    return 1
  fi

  local -a files=(
    _aiter_ops.py
    distributed/device_communicators/cuda_communicator.py
    distributed/parallel_state.py
    envs.py
    model_executor/layers/fused_moe/activation.py
    model_executor/layers/fused_moe/expert_map_manager.py
    model_executor/layers/fused_moe/experts/triton_moe.py
    model_executor/layers/fused_moe/rocm_gfx90a_activation.py
    model_executor/layers/fused_moe/router/fused_topk_bias_router.py
    model_executor/layers/fused_moe/router/gate_linear.py
    model_executor/layers/fused_moe/router/grouped_topk_router.py
    model_executor/layers/fused_moe/router/rocm_gfx90a_router_gemm.py
    platforms/rocm.py
    v1/attention/backends/rocm_aiter_fa.py
    v1/attention/ops/prefix_prefill.py
  )

  for relative in "${files[@]}"; do
    [[ -f "$root/vllm/$relative" ]] || {
      echo "FATAL: vLLM source overlay is missing $relative" >&2
      return 1
    }
    SING_BIND_ARGS+=(
      -B "$root/vllm/$relative:$site/$relative:ro"
    )
  done

  local -a directories=(
    model_executor/layers/fused_moe/configs
    model_executor/layers/mamba/ops/configs/selective_state_update
  )
  for relative in "${directories[@]}"; do
    [[ -d "$root/vllm/$relative" ]] || {
      echo "FATAL: vLLM source overlay is missing $relative" >&2
      return 1
    }
    SING_BIND_ARGS+=(
      -B "$root/vllm/$relative:$site/$relative:ro"
    )
  done
  echo "Using vLLM source overlay $actual_rev from $root"
}
