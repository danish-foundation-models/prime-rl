import math
import os
import re
from typing import Any

from prime_rl.configs.orchestrator import EnvConfig, OrchestratorConfig
from prime_rl.utils.logger import get_logger


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() not in {"0", "false", "no", "off"}


def _as_int(value: object, default: int) -> int:
    if value is None:
        return default
    return max(1, math.ceil(float(value)))


def _as_float(value: object, default: float) -> float:
    if value is None:
        return default
    return float(value)


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return slug[:80] or "env"


def _ucloud_envs(config: OrchestratorConfig) -> list[EnvConfig]:
    envs: list[EnvConfig] = []
    for env in config.train.env:
        if str(env.args.get("sandbox_backend") or "").lower() == "ucloud":
            envs.append(env)
    if config.eval is not None:
        for env in config.eval.env:
            if str(env.args.get("sandbox_backend") or "").lower() == "ucloud":
                envs.append(env)
    return envs


def _prewarm_count(config: OrchestratorConfig, env: EnvConfig, num_ucloud_train_envs: int) -> int:
    override = os.environ.get("UCLOUD_SANDBOX_PREWARM_COUNT")
    if override:
        return _as_int(override, 1)
    workers = env.num_workers if isinstance(env.num_workers, int) else 1
    if env in config.train.env and num_ucloud_train_envs == 1 and config.max_inflight_rollouts:
        return max(workers, int(config.max_inflight_rollouts))
    return workers


def _resources(args: dict[str, Any]) -> dict[str, Any]:
    return {
        "cpus": _as_float(args.get("sandbox_cpu_cores"), 1.0),
        "memory_mb": _as_int(args.get("sandbox_memory_gb"), 2) * 1024,
        "disk_mb": _as_int(args.get("sandbox_disk_size_gb"), 5) * 1024,
    }


async def prepare_ucloud_capacity(config: OrchestratorConfig) -> None:
    """Send UCloud capacity hints during orchestrator startup.

    This intentionally runs before env servers start creating sandboxes. The
    hints give the sandbox gateway time to scale while the rest of startup
    continues. Create calls still retry capacity-pending responses, but they do
    not submit just-in-time hints.
    """

    if not _env_bool("UCLOUD_SANDBOX_PREWARM", True):
        return

    envs = _ucloud_envs(config)
    if not envs:
        return

    try:
        from ucloud_sandboxes_sdk import AsyncSandboxClient
    except ImportError as exc:
        raise RuntimeError(
            "UCloud sandbox prewarm requires the latest ucloud-sandboxes-sdk."
        ) from exc

    base_url = (
        os.environ.get("UCLOUD_SANDBOX_API_URL")
        or os.environ.get("UCLOUD_SANDBOX_URL")
        or os.environ.get("UCLOUD_SANDBOX_BASE_URL")
    )
    token = os.environ.get("UCLOUD_SANDBOX_API_TOKEN")
    if not base_url or not token:
        raise RuntimeError(
            "UCloud sandbox prewarm requires UCLOUD_SANDBOX_API_URL "
            "and UCLOUD_SANDBOX_API_TOKEN."
        )

    ttl_seconds = _as_int(os.environ.get("UCLOUD_SANDBOX_PREWARM_TTL_SECONDS"), 1800)
    run_slug = _slug(config.output_dir.name)
    num_ucloud_train_envs = sum(1 for env in config.train.env if env in envs)
    client = AsyncSandboxClient(
        base_url,
        headers={"Authorization": f"Bearer {token}"},
    )
    try:
        for env in envs:
            count = _prewarm_count(config, env, num_ucloud_train_envs)
            resources = _resources(env.args)
            prepare_id = f"prime-rl-{run_slug}-{_slug(env.resolved_name)}"
            await client.prepare_capacity(
                prepare_id=prepare_id,
                count=count,
                ttl_seconds=ttl_seconds,
                **resources,
            )
            get_logger().info(
                "Prepared UCloud sandbox capacity "
                f"(env={env.resolved_name}, prepare_id={prepare_id}, "
                f"count={count}, cpus={resources['cpus']}, "
                f"memory_mb={resources['memory_mb']}, disk_mb={resources['disk_mb']}, "
                f"ttl_seconds={ttl_seconds})"
            )
    finally:
        await client.close()
