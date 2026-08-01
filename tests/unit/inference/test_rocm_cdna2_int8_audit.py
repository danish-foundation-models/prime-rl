from __future__ import annotations

import io
import json
import logging

import pytest
import torch

from prime_rl.inference.vllm.kernels import rocm_cdna2_int8_audit as audit


@pytest.mark.parametrize("value", ("", "true", "yes", "2"))
def test_audit_opt_in_rejects_ambiguous_values(value):
    with pytest.raises(ValueError, match=audit.AUDIT_ENV):
        audit.cdna2_int8_audit_requested({audit.AUDIT_ENV: value})


def test_audit_opt_in_accepts_only_zero_one_or_unset():
    assert audit.cdna2_int8_audit_requested({}) is False
    assert audit.cdna2_int8_audit_requested({audit.AUDIT_ENV: "0"}) is False
    assert audit.cdna2_int8_audit_requested({audit.AUDIT_ENV: "1"}) is True


def test_transcode_audit_is_one_structured_info_line(caplog, monkeypatch):
    monkeypatch.setenv("RANK", "7")
    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.setattr(audit.os, "getpid", lambda: 1234)
    monkeypatch.setattr(audit.socket, "gethostname", lambda: "worker.example")
    monkeypatch.setattr(audit.LOGGER, "handlers", [caplog.handler])
    monkeypatch.setattr(audit.LOGGER, "level", logging.INFO)
    monkeypatch.setattr(audit.LOGGER, "propagate", False)
    source_weight = torch.zeros((2, 32), dtype=torch.float8_e4m3fn)
    runtime_weight = torch.zeros((2, 32), dtype=torch.int8)
    source_scale = torch.ones((1, 1), dtype=torch.float32)
    runtime_scale = torch.ones((2, 1), dtype=torch.float16)

    audit.log_transcode(
        enabled=True,
        kind="linear",
        prefix="model.layers.3.mlp.down_proj",
        source_weight=source_weight,
        runtime_weight=runtime_weight,
        source_scale=source_scale,
        runtime_scale=runtime_scale,
    )

    assert len(caplog.records) == 1
    payload = json.loads(caplog.records[0].message)
    assert payload == {
        "event": "prime_cdna2_int8_transcode",
        "hostname": "worker.example",
        "identity_source": "environment",
        "kind": "linear",
        "local_rank": 3,
        "pid": 1234,
        "prefix": "model.layers.3.mlp.down_proj",
        "rank": 7,
        "runtime_scale_dtype": "torch.float16",
        "runtime_scale_shape": [2, 1],
        "runtime_weight_dtype": "torch.int8",
        "runtime_weight_shape": [2, 32],
        "schema_version": 1,
        "source_scale_dtype": "torch.float32",
        "source_scale_shape": [1, 1],
        "source_weight_dtype": "torch.float8_e4m3fn",
        "source_weight_shape": [2, 32],
    }


def test_disabled_audit_emits_no_line(caplog, monkeypatch):
    monkeypatch.setattr(audit.LOGGER, "handlers", [caplog.handler])
    monkeypatch.setattr(audit.LOGGER, "level", logging.INFO)
    monkeypatch.setattr(audit.LOGGER, "propagate", False)

    audit.log_exemption(
        enabled=False,
        kind="linear",
        prefix="model.layers.3.self_attn.kv_b_proj",
        reason="stock_fp8_post_load_required",
    )

    assert caplog.records == []


def test_audit_prefers_initialized_distributed_identity(caplog, monkeypatch):
    monkeypatch.setenv("RANK", "99")
    monkeypatch.setenv("LOCAL_RANK", "98")
    monkeypatch.setattr(audit.dist, "is_available", lambda: True)
    monkeypatch.setattr(audit.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(audit.dist, "get_rank", lambda: 11)
    monkeypatch.setattr(audit.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(audit.torch.cuda, "current_device", lambda: 3)
    monkeypatch.setattr(audit.LOGGER, "handlers", [caplog.handler])
    monkeypatch.setattr(audit.LOGGER, "level", logging.INFO)
    monkeypatch.setattr(audit.LOGGER, "propagate", False)

    audit.log_exemption(
        enabled=True,
        kind="linear",
        prefix="model.layers.3.self_attn.kv_b_proj",
        reason="stock_fp8_post_load_required",
    )

    payload = json.loads(caplog.records[0].message)
    assert payload["identity_source"] == "torch.distributed"
    assert payload["rank"] == 11
    assert payload["local_rank"] == 3


def test_enabled_audit_rejects_missing_process_identity(monkeypatch):
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.setattr(audit.dist, "is_available", lambda: True)
    monkeypatch.setattr(audit.dist, "is_initialized", lambda: False)

    with pytest.raises(RuntimeError, match="integer RANK"):
        audit.log_exemption(
            enabled=True,
            kind="linear",
            prefix="model.layers.3.self_attn.kv_b_proj",
            reason="stock_fp8_post_load_required",
        )


def test_audit_info_uses_vllm_handler_when_root_filters_info(monkeypatch):
    monkeypatch.setenv("RANK", "7")
    monkeypatch.setenv("LOCAL_RANK", "3")
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(message)s"))

    root_logger = logging.getLogger()
    vllm_logger = logging.getLogger("vllm")
    monkeypatch.setattr(root_logger, "level", logging.WARNING)
    monkeypatch.setattr(vllm_logger, "handlers", [handler])
    monkeypatch.setattr(vllm_logger, "level", logging.INFO)
    monkeypatch.setattr(vllm_logger, "propagate", False)
    monkeypatch.setattr(vllm_logger, "disabled", False)
    monkeypatch.setattr(audit.LOGGER, "handlers", [])
    monkeypatch.setattr(audit.LOGGER, "level", logging.NOTSET)
    monkeypatch.setattr(audit.LOGGER, "propagate", True)
    monkeypatch.setattr(audit.LOGGER, "disabled", False)

    audit.log_exemption(
        enabled=True,
        kind="linear",
        prefix="model.layers.3.self_attn.kv_b_proj",
        reason="stock_fp8_post_load_required",
    )

    lines = stream.getvalue().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["event"] == "prime_cdna2_int8_exemption"
    assert lines[0] == json.dumps(payload, sort_keys=True, separators=(",", ":"))
    assert root_logger.level == logging.WARNING
