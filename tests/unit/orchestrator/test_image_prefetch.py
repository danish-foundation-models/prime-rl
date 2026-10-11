from types import SimpleNamespace

import pytest

from prime_rl.configs.orchestrator import ImagePrefetchConfig
from prime_rl.orchestrator.image_prefetch import ImagePrefetcher


class FakeGateway:
    def __init__(self, states):
        self.states = states
        self.asked = []

    async def ensure_images(self, names, *, timeout_seconds=None):
        self.asked.append(list(names))
        return {name: {"state": self.states[name], "error": "build failed"} for name in names}


class FakeSource:
    def __init__(self, upcoming, envs):
        self.upcoming = upcoming
        self.envs = envs
        self.quarantined = []

    def upcoming_tasks(self, count):
        return {name: tasks[:count] for name, tasks in self.upcoming.items()}

    def quarantine_image(self, env_name, task, reason):
        self.quarantined.append((env_name, task.data.image, reason))


def env(name, runtime_type="ucloud", reference="name"):
    runtime = SimpleNamespace(type=runtime_type, image_reference_type=reference)
    return SimpleNamespace(name=name, config=SimpleNamespace(env=SimpleNamespace(agent=SimpleNamespace(runtime=runtime))))


def task(image):
    return SimpleNamespace(data=SimpleNamespace(image=image))


@pytest.mark.asyncio
async def test_prefetch_ensures_upcoming_gateway_images_and_quarantines_unusable_ones():
    source = FakeSource(
        {
            "swe": [task("a:1"), task("b:1"), task("c:1")],
            "registry": [task("docker.io/x:1")],
            "local": [task("y:1")],
        },
        [env("swe"), env("registry", reference="registry"), env("local", runtime_type="subprocess")],
    )
    gateway = FakeGateway({"a:1": "ready", "b:1": "building", "c:1": "failed"})
    prefetcher = ImagePrefetcher(ImagePrefetchConfig(enabled=True, lookahead_tasks=8), source, client=gateway)

    await prefetcher.prefetch()
    assert gateway.asked == [["a:1", "b:1", "c:1"]]
    assert source.quarantined == [("swe", "c:1", "gateway image failed: build failed")]
    metrics = prefetcher.metrics()
    assert metrics["image_prefetch/ready"] == 1 and metrics["image_prefetch/building"] == 1
    assert metrics["image_prefetch/known_ready"] == 1

    await prefetcher.prefetch()
    assert gateway.asked[-1] == ["b:1", "c:1"]  # A ready name is not asked about again.
