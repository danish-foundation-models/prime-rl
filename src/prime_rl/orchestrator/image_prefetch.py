"""Build upcoming train tasks' sandbox images ahead of time on the UCloud gateway."""

from __future__ import annotations

import asyncio
import time
from collections import Counter

from prime_rl.configs.orchestrator import ImagePrefetchConfig
from prime_rl.orchestrator.train_source import TrainSource
from prime_rl.utils.logger import get_logger

# States after which asking again cannot help: the task image will never start.
UNUSABLE_STATES = ("failed", "unknown")
# The gateway prefixes failures of its own build system (registry, builder timeouts),
# which say nothing about the task's image.
INFRASTRUCTURE_ERROR = "sandbox build infrastructure"


def uses_gateway_image_names(env) -> bool:
    runtime = getattr(getattr(getattr(env.config, "env", None), "agent", None), "runtime", None)
    return getattr(runtime, "type", None) == "ucloud" and getattr(runtime, "image_reference_type", None) == "name"


class ImagePrefetcher:
    """Ensure the images of each env's next unseen tasks, so they are built when sampled.

    The gateway submits missing images for building and answers every name's state;
    asking again only polls. Ready names are not asked about again for a while, and a
    failed or unknown image is quarantined so the sampler skips its tasks, unless the
    gateway's build system failed rather than the image.
    """

    def __init__(self, config: ImagePrefetchConfig, train_source: TrainSource, client=None) -> None:
        self.config = config
        self.train_source = train_source
        self.env_names = {env.name for env in train_source.envs if uses_gateway_image_names(env)}
        self._client = client
        self._ready_until: dict[str, float] = {}
        self._states: Counter[str] = Counter()

    async def run(self) -> None:
        if not self.env_names:
            get_logger().info("Image prefetch has no train envs that name gateway images")
            return
        get_logger().info(f"Prefetching images for {sorted(self.env_names)}")
        while True:
            try:
                await self.prefetch()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # A missed round only delays builds to sampling time.
                get_logger().warning(f"Image prefetch round failed: {type(exc).__name__}: {exc}")
            await asyncio.sleep(self.config.interval_seconds)

    async def prefetch(self) -> None:
        now = time.monotonic()
        wanted: dict[str, list[tuple[str, object]]] = {}
        for env_name, tasks in self.train_source.upcoming_tasks(self.config.lookahead_tasks).items():
            if env_name not in self.env_names:
                continue
            for task in tasks:
                image = getattr(task.data, "image", None)
                if image and self._ready_until.get(image, 0.0) <= now:
                    wanted.setdefault(image, []).append((env_name, task))
        if not wanted:
            self._states = Counter()
            return
        statuses = await self._client_or_create().ensure_images(
            sorted(wanted), timeout_seconds=self.config.request_timeout_seconds
        )
        states: Counter[str] = Counter()
        for name, status in statuses.items():
            state = str(status.get("state"))
            states[state] += 1
            if state == "ready":
                self._ready_until[name] = now + self.config.ready_ttl_seconds
            elif state in UNUSABLE_STATES and INFRASTRUCTURE_ERROR in str(status.get("error") or ""):
                states["infrastructure_failed"] += 1  # Asked again next round, not quarantined.
            elif state in UNUSABLE_STATES:
                reason = f"gateway image {state}: {status.get('error') or 'not in the image index'}"
                for env_name, task in wanted.get(name, []):
                    self.train_source.quarantine_image(env_name, task, reason)
        self._states = states

    def metrics(self) -> dict[str, float]:
        now = time.monotonic()
        metrics = {f"image_prefetch/{state}": float(count) for state, count in self._states.items()}
        metrics["image_prefetch/known_ready"] = float(sum(until > now for until in self._ready_until.values()))
        return metrics

    def _client_or_create(self):
        if self._client is None:
            from ucloud_sandboxes_sdk import AsyncSandboxClient

            self._client = AsyncSandboxClient.from_env(timeout_seconds=self.config.request_timeout_seconds)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None
