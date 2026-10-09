import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from verifiers.v1.configs.client import EvalClientConfig

from prime_rl.configs.shared import ClientConfig
from prime_rl.orchestrator.clients import (
    AdminPlane,
    _is_retryable_lora_error,
    check_health,
    load_lora_adapter,
    setup_client,
)


def test_is_retryable_lora_error_returns_true_for_404():
    response = MagicMock()
    response.status_code = 404
    error = httpx.HTTPStatusError("Not found", request=MagicMock(), response=response)
    assert _is_retryable_lora_error(error) is True


def test_is_retryable_lora_error_returns_true_for_500():
    response = MagicMock()
    response.status_code = 500
    error = httpx.HTTPStatusError("Server error", request=MagicMock(), response=response)
    assert _is_retryable_lora_error(error) is True


def test_is_retryable_lora_error_returns_false_for_400():
    response = MagicMock()
    response.status_code = 400
    error = httpx.HTTPStatusError("Bad request", request=MagicMock(), response=response)
    assert _is_retryable_lora_error(error) is False


def test_is_retryable_lora_error_returns_false_for_non_http_error():
    assert _is_retryable_lora_error(ValueError("some error")) is False


def test_load_lora_adapter_succeeds_on_first_attempt():
    mock_client = AsyncMock()
    admin_plane = MagicMock(clients=[mock_client])
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_client.post.return_value = mock_response

    asyncio.run(load_lora_adapter(admin_plane, "test-lora", Path("/test/path")))

    mock_client.post.assert_called_once_with(
        "/load_lora_adapter",
        json={"lora_name": "test-lora", "lora_path": "/test/path"},
        timeout=httpx.Timeout(connect=10.0, read=30.0, write=60.0, pool=10.0),
    )


@pytest.mark.parametrize("pause_for_init", [False, True])
def test_admin_plane_initializes_nccl(monkeypatch, pause_for_init):
    monkeypatch.setenv("PRIME_RL_PAUSE_NCCL_INIT", str(int(pause_for_init)))
    admin_plane = AdminPlane(ClientConfig())
    client = AsyncMock()
    response = MagicMock()
    response.raise_for_status.return_value = None
    client.post.return_value = response
    admin_plane.clients = [client]

    asyncio.run(
        admin_plane.initialize_nccl(
            host="trainer",
            port=29501,
            timeout=1200,
            inference_world_size=1,
        )
    )

    paths = [call.args[0] for call in client.post.await_args_list]
    assert paths == (["/pause", "/init_broadcaster", "/resume"] if pause_for_init else ["/init_broadcaster"])
    client.post.assert_any_await(
        "/init_broadcaster",
        json={
            "host": "trainer",
            "port": 29501,
            "rank_offset": 0,
            "inference_world_size": 1,
            "timeout": 1200,
            "quantize_in_weight_transfer": False,
        },
    )
    asyncio.run(admin_plane.aclose())


def test_setup_client_creates_renderer_client():
    from renderers import Qwen3VLRendererConfig

    client_config = ClientConfig(
        base_url="http://worker-a:8000/v1",
        api_key_var="PRIME_API_KEY",
        headers={"X-Test": "test"},
    )

    renderer_settings = Qwen3VLRendererConfig()
    client = setup_client(
        client_config,
        client_type="renderer",
        renderer_config=renderer_settings,
    )

    assert client.type == "train"
    assert client.renderer == renderer_settings
    assert client.renderer_model_name is None
    assert client.base_url == "http://worker-a:8000/v1"
    assert "X-data-parallel-rank" not in client.headers
    assert client.headers["X-Test"] == "test"


def test_check_health_retries_non_success_status():
    client = AsyncMock()
    unavailable = httpx.Response(503, request=httpx.Request("GET", "http://worker/health"))
    healthy = httpx.Response(200, request=httpx.Request("GET", "http://worker/health"))
    client.get.side_effect = [unavailable, healthy]
    client.base_url = httpx.URL("http://worker")

    with patch("prime_rl.orchestrator.clients.asyncio.sleep", new=AsyncMock()):
        asyncio.run(check_health([client], interval=1, timeout=2))

    assert client.get.await_count == 2


def test_setup_client_assigns_renderer_model_name():
    from renderers import Qwen3VLRendererConfig

    client_config = ClientConfig(
        base_url="http://worker-a:8000/v1",
        api_key_var="PRIME_API_KEY",
    )

    client = setup_client(
        client_config,
        client_type="renderer",
        renderer_config=Qwen3VLRendererConfig(),
        renderer_model_name="Qwen/Qwen3-VL-4B-Instruct",
    )

    assert client.renderer_model_name == "Qwen/Qwen3-VL-4B-Instruct"


def test_setup_client_preserves_chat_client_defaults():
    client_config = ClientConfig(
        base_url="http://worker-a:8000/v1",
        api_key_var="PRIME_API_KEY",
    )

    client = setup_client(client_config)

    assert client == EvalClientConfig(
        api_key_var="PRIME_API_KEY",
        base_url="http://worker-a:8000/v1",
        headers={},
    )


@pytest.mark.parametrize("outcome", ["server_error", "timeout", "write_timeout", "success"])
def test_nccl_update_does_not_retry_or_resume_a_failed_receive(outcome):
    requests = []

    def handle(request):
        requests.append(request.url.path)
        if request.url.path == "/update_weights":
            if outcome == "timeout":
                raise httpx.ReadTimeout("Lost update acknowledgement", request=request)
            if outcome == "write_timeout":
                raise httpx.WriteTimeout("Partial request write", request=request)
            if outcome == "server_error":
                return httpx.Response(500, json={"error": "weight receiver OOM"})
        return httpx.Response(200, json={"status": "ok"})

    async def run():
        plane = AdminPlane(ClientConfig())
        await plane.aclose()
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle), base_url="http://engine") as client:
            plane.clients = [client]
            if outcome == "success":
                await plane.update_weights(None, transport="nccl")
            else:
                error = {"timeout": httpx.ReadTimeout, "write_timeout": httpx.WriteTimeout, "server_error": httpx.HTTPStatusError}[outcome]
                with pytest.raises(error):
                    await plane.update_weights(None, transport="nccl")

    asyncio.run(run())
    expected = ["/pause", "/update_weights"]
    if outcome == "success":
        expected.append("/resume")
    assert requests == expected


@pytest.mark.parametrize("status_code", [400, 500, 503])
@pytest.mark.parametrize("pause_for_init", [False, True])
def test_nccl_initialization_propagates_http_failure(status_code, monkeypatch, pause_for_init):
    monkeypatch.setenv("PRIME_RL_PAUSE_NCCL_INIT", str(int(pause_for_init)))

    async def run():
        requests = []

        def respond(request):
            requests.append(request)
            if request.url.path == "/pause":
                return httpx.Response(200, json={})
            return httpx.Response(status_code, json={"error": "broadcaster failed"})

        admin = AdminPlane(ClientConfig())
        await admin.aclose()
        async with httpx.AsyncClient(base_url="http://inference", transport=httpx.MockTransport(respond)) as client:
            admin.clients = [client]
            with pytest.raises(httpx.HTTPStatusError):
                await admin.initialize_nccl(host="trainer", port=29501, timeout=1200, inference_world_size=8)
        assert [request.url.path for request in requests] == (
            ["/pause", "/init_broadcaster"] if pause_for_init else ["/init_broadcaster"]
        )

    asyncio.run(run())


@pytest.mark.parametrize("error_type", [httpx.ConnectTimeout, httpx.ConnectError])
@pytest.mark.parametrize("persistent", [False, True])
def test_nccl_update_retries_only_connection_establishment(error_type, persistent):
    requests = []
    attempts = 0

    def handle(request):
        nonlocal attempts
        requests.append(request.url.path)
        if request.url.path == "/update_weights":
            attempts += 1
            if persistent or attempts == 1:
                raise error_type("Connection not established", request=request)
        return httpx.Response(200, json={"status": "ok"})

    async def run():
        plane = AdminPlane(ClientConfig())
        await plane.aclose()
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle), base_url="http://engine") as client:
            plane.clients = [client]
            if persistent:
                with pytest.raises(error_type):
                    await plane.update_weights(None, transport="nccl")
            else:
                await plane.update_weights(None, transport="nccl")

    asyncio.run(run())
    assert requests == ["/pause"] + ["/update_weights"] * (3 if persistent else 2) + ([] if persistent else ["/resume"])
