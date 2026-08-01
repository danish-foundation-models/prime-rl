# SPDX-License-Identifier: Apache-2.0

"""Opt-in structured model-load audit for the CDNA2 INT8 path."""

from __future__ import annotations

import json
import os
import socket
from collections.abc import Mapping
from typing import Any

import torch
import torch.distributed as dist
from vllm.logger import init_logger

AUDIT_ENV = "PRIME_ROCM_CDNA2_INT8_AUDIT"
AUDIT_SCHEMA_VERSION = 1
# Worker processes configure an INFO handler on the non-propagating ``vllm``
# logger, while their root logger may filter INFO records.
LOGGER = init_logger(f"vllm.{__name__}")


def cdna2_int8_audit_requested(
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Read the audit opt-in without accepting ambiguous values."""

    value = (os.environ if environ is None else environ).get(AUDIT_ENV)
    if value in (None, "0"):
        return False
    if value == "1":
        return True
    raise ValueError(f"{AUDIT_ENV} must be unset, '0', or '1'; got {value!r}")


def _tensor_fields(
    *,
    source_weight: torch.Tensor,
    runtime_weight: torch.Tensor,
    source_scale: torch.Tensor,
    runtime_scale: torch.Tensor,
) -> dict[str, Any]:
    return {
        "source_weight_dtype": str(source_weight.dtype),
        "runtime_weight_dtype": str(runtime_weight.dtype),
        "source_scale_dtype": str(source_scale.dtype),
        "runtime_scale_dtype": str(runtime_scale.dtype),
        "source_weight_shape": list(source_weight.shape),
        "runtime_weight_shape": list(runtime_weight.shape),
        "source_scale_shape": list(source_scale.shape),
        "runtime_scale_shape": list(runtime_scale.shape),
    }


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


def _process_fields() -> dict[str, Any]:
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


def log_transcode(
    *,
    enabled: bool,
    kind: str,
    prefix: str,
    source_weight: torch.Tensor,
    runtime_weight: torch.Tensor,
    source_scale: torch.Tensor,
    runtime_scale: torch.Tensor,
) -> None:
    """Emit exactly one JSON INFO record for a completed transcode."""

    if not enabled:
        return
    payload = {
        "event": "prime_cdna2_int8_transcode",
        "kind": kind,
        "prefix": prefix,
        **_process_fields(),
        **_tensor_fields(
            source_weight=source_weight,
            runtime_weight=runtime_weight,
            source_scale=source_scale,
            runtime_scale=runtime_scale,
        ),
    }
    LOGGER.info(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def log_exemption(
    *,
    enabled: bool,
    kind: str,
    prefix: str,
    reason: str,
) -> None:
    """Emit one JSON INFO record for an intentional stock-FP8 exemption."""

    if not enabled:
        return
    LOGGER.info(
        json.dumps(
            {
                "event": "prime_cdna2_int8_exemption",
                "kind": kind,
                "prefix": prefix,
                "reason": reason,
                **_process_fields(),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
