import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

from prime_rl.orchestrator.ucloud_capacity import prepare_ucloud_capacity


class FakeImage:
    def __init__(self, reference: str, source: str):
        self.reference = reference
        self.source = source

    @classmethod
    def from_registry(cls, reference: str):
        return cls(reference, "registry")

    @classmethod
    def from_name(cls, name: str):
        return cls(name, "managed")


class FakeAsyncSandboxClient:
    instances = []
    fail_prepare = False

    def __init__(self, base_url, *, api_token=None, **kwargs):
        self.base_url = base_url
        self.api_token = api_token
        self.kwargs = kwargs
        self.prepared = []
        self.pulled = []
        self.closed = False
        FakeAsyncSandboxClient.instances.append(self)

    async def prepare_capacity(self, **kwargs):
        if self.fail_prepare:
            raise TimeoutError("capacity prepare timed out")
        self.prepared.append(kwargs)
        return {}

    async def pull_image(self, image, **kwargs):
        self.pulled.append({"image": image, **kwargs})
        return {}

    async def close(self):
        self.closed = True


def _install_fake_sdk(monkeypatch):
    FakeAsyncSandboxClient.instances.clear()
    FakeAsyncSandboxClient.fail_prepare = False
    monkeypatch.setitem(
        sys.modules,
        "ucloud_sandboxes_sdk",
        SimpleNamespace(AsyncSandboxClient=FakeAsyncSandboxClient, Image=FakeImage),
    )


def _config(env):
    return SimpleNamespace(
        output_dir=Path("/tmp/prime-rl-run"),
        concurrency=SimpleNamespace(max_inflight=8),
        train=SimpleNamespace(source=[env]),
        eval=None,
    )


def _v1_env(runtime, *, workers=2):
    return SimpleNamespace(
        resolved_name="tmax",
        env=SimpleNamespace(
            agent=SimpleNamespace(runtime=runtime),
            taskset=SimpleNamespace(),
        ),
        serve=SimpleNamespace(pool=SimpleNamespace(type="elastic", max_workers=workers)),
    )


def test_prepare_ucloud_capacity_ignores_non_ucloud_v1_source(monkeypatch):
    _install_fake_sdk(monkeypatch)
    env = _v1_env(SimpleNamespace(type="subprocess"))

    asyncio.run(prepare_ucloud_capacity(_config(env)))

    assert FakeAsyncSandboxClient.instances == []


def test_prepare_ucloud_capacity_supports_v1_source(monkeypatch):
    _install_fake_sdk(monkeypatch)
    monkeypatch.setenv("UCLOUD_SANDBOX_API_URL", "https://gateway.example")
    monkeypatch.setenv("UCLOUD_SANDBOX_API_TOKEN", "token")
    runtime = SimpleNamespace(
        type="ucloud",
        image="registry.example/tmax:latest",
        prepare_id=None,
        cpu=2,
        memory=4,
        disk=6,
    )
    env = _v1_env(runtime)

    asyncio.run(prepare_ucloud_capacity(_config(env)))

    client = FakeAsyncSandboxClient.instances[-1]
    assert client.prepared[0]["count"] == 8
    assert client.prepared[0]["cpus"] == 2.0
    assert client.prepared[0]["memory_mb"] == 4096
    assert client.prepared[0]["disk_mb"] == 6144
    assert env.env.agent.runtime.prepare_id == "prime-rl-prime-rl-run-tmax"


def test_prepare_ucloud_capacity_announces_fixed_image(monkeypatch):
    _install_fake_sdk(monkeypatch)
    monkeypatch.setenv("UCLOUD_SANDBOX_API_URL", "https://gateway.example")
    monkeypatch.setenv("UCLOUD_SANDBOX_API_TOKEN", "token")

    env = SimpleNamespace(
        resolved_name="tmax",
        num_workers=2,
        harness=SimpleNamespace(
            runtime=SimpleNamespace(
                type="ucloud",
                image="registry.example/tmax:latest",
                prepare_id=None,
                profile="linux_host",
                user="root",
                enable_cron=True,
                enable_sshd=True,
                keep_alive=True,
                parkable=True,
                writable_paths=["/home/user", "/task", "/app"],
            )
        ),
        args={
            "sandbox_backend": "ucloud",
            "sandbox_cpu_cores": 2,
            "sandbox_memory_gb": 4,
            "sandbox_disk_size_gb": 6,
            "tmax_image": "registry.example/tmax:latest",
        },
    )

    asyncio.run(prepare_ucloud_capacity(_config(env)))

    client = FakeAsyncSandboxClient.instances[-1]
    assert client.base_url == "https://gateway.example"
    assert client.api_token == "token"
    assert client.kwargs["timeout_seconds"] == 300
    assert client.closed is True
    assert client.prepared == [
        {
            "prepare_id": "prime-rl-prime-rl-run-tmax",
            "count": 8,
            "ttl_seconds": 1800,
            "image": client.pulled[0]["image"],
            "cpus": 2.0,
            "memory_mb": 4096,
            "disk_mb": 6144,
            "parkable": True,
        }
    ]
    assert client.pulled == [
        {
            "image": client.prepared[0]["image"],
            "count": 8,
            "cpus": 2.0,
            "memory_mb": 4096,
            "disk_mb": 6144,
        }
    ]
    assert client.prepared[0]["image"].reference == "registry.example/tmax:latest"
    assert client.prepared[0]["image"].source == "registry"
    assert env.harness.runtime.prepare_id == "prime-rl-prime-rl-run-tmax"


def test_prepare_ucloud_capacity_uses_managed_image_id(monkeypatch):
    _install_fake_sdk(monkeypatch)
    monkeypatch.setenv("UCLOUD_SANDBOX_API_URL", "https://gateway.example")
    monkeypatch.setenv("UCLOUD_SANDBOX_API_TOKEN", "token")

    env = SimpleNamespace(
        resolved_name="tmax",
        num_workers=2,
        args={
            "sandbox_backend": "ucloud",
            "tmax_image": "prime-rl-tmax-mini-base-mswe-2-2-8-r6",
        },
    )

    asyncio.run(prepare_ucloud_capacity(_config(env)))

    client = FakeAsyncSandboxClient.instances[-1]
    assert client.prepared[0]["image"].reference == ("prime-rl-tmax-mini-base-mswe-2-2-8-r6")
    assert client.prepared[0]["image"].source == "managed"


def test_prepare_ucloud_capacity_uses_constant_sandbox_image_template(monkeypatch):
    _install_fake_sdk(monkeypatch)
    monkeypatch.setenv("UCLOUD_SANDBOX_API_URL", "https://gateway.example")
    monkeypatch.setenv("UCLOUD_SANDBOX_API_TOKEN", "token")
    monkeypatch.setenv("UCLOUD_SANDBOX_PREWARM_PULL_IMAGES", "false")

    env = SimpleNamespace(
        resolved_name="astropy",
        num_workers=4,
        args={
            "sandbox_backend": "ucloud",
            "sandbox_image_template": "registry.example/astropy:mini-swe",
        },
    )

    asyncio.run(prepare_ucloud_capacity(_config(env)))

    client = FakeAsyncSandboxClient.instances[-1]
    assert len(client.prepared) == 1
    assert client.prepared[0]["image"].reference == "registry.example/astropy:mini-swe"
    assert client.pulled == []


def test_prepare_ucloud_capacity_warns_and_continues_after_prepare_timeout(monkeypatch):
    _install_fake_sdk(monkeypatch)
    FakeAsyncSandboxClient.fail_prepare = True
    monkeypatch.setenv("UCLOUD_SANDBOX_API_URL", "https://gateway.example")
    monkeypatch.setenv("UCLOUD_SANDBOX_API_TOKEN", "token")
    monkeypatch.setenv("UCLOUD_SANDBOX_PREWARM_ATTEMPTS", "1")

    env = SimpleNamespace(
        resolved_name="tmax",
        num_workers=2,
        args={
            "sandbox_backend": "ucloud",
            "tmax_image": "registry.example/tmax:latest",
        },
    )

    asyncio.run(prepare_ucloud_capacity(_config(env)))

    client = FakeAsyncSandboxClient.instances[-1]
    assert client.closed is True
    assert client.prepared == []
    assert client.pulled == []
