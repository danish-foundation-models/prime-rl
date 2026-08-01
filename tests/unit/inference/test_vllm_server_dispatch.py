from argparse import Namespace

import vllm.entrypoints.cli.serve as serve_module
import vllm.entrypoints.openai.api_server as api_server_module
import vllm.entrypoints.openai.dp_supervisor as supervisor_module
from vllm.config import ProfilerConfig

from prime_rl.inference.vllm.server import _normalize_vllm_extra, _run_parsed_server


def test_profiler_config_is_built_before_api_router_reads_namespace():
    config = _normalize_vllm_extra(
        "profiler_config",
        {
            "profiler": "torch",
            "torch_profiler_dir": "/tmp/prime-rl-profile",
            "ignore_frontend": True,
            "max_iterations": 8,
        },
    )

    assert isinstance(config, ProfilerConfig)
    assert config.profiler == "torch"
    assert config.max_iterations == 8


def test_unrelated_vllm_extra_is_unchanged():
    value = {"cudagraph_mode": "PIECEWISE"}

    assert _normalize_vllm_extra("compilation_config", value) is value


def test_multi_port_dp_uses_vllm_supervisor(monkeypatch):
    calls = []

    monkeypatch.setattr(
        serve_module,
        "run_headless",
        lambda _args: calls.append("headless"),
    )
    monkeypatch.setattr(
        serve_module,
        "run_multi_api_server",
        lambda _args: calls.append("multi-api"),
    )
    monkeypatch.setattr(
        supervisor_module,
        "run_dp_supervisor",
        lambda _args: calls.append("dp-supervisor"),
    )
    monkeypatch.setattr(
        api_server_module,
        "run_server",
        lambda _args: calls.append("single-api"),
    )

    _run_parsed_server(
        Namespace(
            headless=False,
            api_server_count=1,
            data_parallel_multi_port_external_lb=True,
        )
    )

    assert calls == ["dp-supervisor"]
