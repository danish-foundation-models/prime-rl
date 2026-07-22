import asyncio
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
        if _uses_ucloud(env):
            envs.append(env)
    if config.eval is not None:
        for env in config.eval.env:
            if _uses_ucloud(env):
                envs.append(env)
    return envs


def _uses_ucloud(env: EnvConfig) -> bool:
    if str(env.args.get("sandbox_backend") or "").lower() == "ucloud":
        return True
    runtime = getattr(getattr(env, "harness", None), "runtime", None)
    if getattr(runtime, "type", None) == "ucloud":
        return True
    if getattr(runtime, "type", None) != "prime":
        return False
    args = _capacity_args(env)
    if args.get("ucloud_prewarm"):
        return True
    return any("ucloud" in image.lower() for image in _prewarm_images(args))


def _capacity_args(env: EnvConfig) -> dict[str, Any]:
    args = dict(env.args)
    taskset = getattr(env, "taskset", None)
    runtime = getattr(getattr(env, "harness", None), "runtime", None)
    for source, mapping in (
        (
            taskset,
            {
                "sandbox_cpu_cores": "sandbox_cpu_cores",
                "sandbox_memory_gb": "sandbox_memory_gb",
                "sandbox_disk_size_gb": "sandbox_disk_size_gb",
                "tmax_image": "tmax_image",
                "sandbox_image_template": "sandbox_image_template",
                "ucloud_prewarm_image": "ucloud_prewarm_image",
                "ucloud_prewarm_images": "ucloud_prewarm_images",
            },
        ),
        (
            runtime,
            {
                "sandbox_cpu_cores": "cpu",
                "sandbox_memory_gb": "memory",
                "sandbox_disk_size_gb": "disk",
                "image": "image",
                "profile": "profile",
                "user": "user",
                "enable_cron": "enable_cron",
                "enable_sshd": "enable_sshd",
                "keep_alive": "keep_alive",
                "writable_paths": "writable_paths",
            },
        ),
    ):
        if source is None:
            continue
        for arg_key, attr in mapping.items():
            value = getattr(source, attr, None)
            if value is not None and arg_key not in args:
                args[arg_key] = value
    return args


def _prewarm_count(config: OrchestratorConfig, env: EnvConfig, num_ucloud_train_envs: int) -> int:
    override = os.environ.get("UCLOUD_SANDBOX_PREWARM_COUNT")
    if override:
        return _as_int(override, 1)
    pool = getattr(env, "pool", None)
    pool_workers = getattr(pool, "num_workers", None)
    env_workers = getattr(env, "num_workers", None)
    workers = pool_workers if isinstance(pool_workers, int) else env_workers if isinstance(env_workers, int) else 1
    if env in config.train.env and num_ucloud_train_envs == 1 and config.max_inflight_rollouts:
        return max(workers, int(config.max_inflight_rollouts))
    return workers


def _resources(args: dict[str, Any]) -> dict[str, Any]:
    return {
        "cpus": _as_float(args.get("sandbox_cpu_cores"), 1.0),
        "memory_mb": _as_int(args.get("sandbox_memory_gb"), 2) * 1024,
        "disk_mb": _as_int(args.get("sandbox_disk_size_gb"), 5) * 1024,
    }


def _string_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def _constant_template(template: object) -> str | None:
    if not isinstance(template, str) or not template:
        return None
    # Templates that depend on the per-task image cannot be resolved before the
    # env server loads the taskset. A constant template is already a concrete
    # derived image and can be announced during orchestrator startup.
    if "{" in template or "}" in template:
        return None
    return template


def _prewarm_images(args: dict[str, Any]) -> list[str]:
    images: list[str] = []
    for key in (
        "ucloud_prewarm_images",
        "ucloud_prewarm_image",
        "tmax_image",
        "sandbox_image",
        "docker_image",
        "image",
    ):
        images.extend(_string_list(args.get(key)))
    template = _constant_template(args.get("sandbox_image_template"))
    if template is not None:
        images.append(template)

    seen: set[str] = set()
    unique: list[str] = []
    for image in images:
        image = image.strip()
        if image and image not in seen:
            unique.append(image)
            seen.add(image)
    return unique


def _image_prepare_id(base_prepare_id: str, image: str, num_images: int) -> str:
    if num_images == 1:
        return base_prepare_id
    return f"{base_prepare_id}-{_slug(image)}"


def _set_runtime_prepare_id(
    env: EnvConfig, *, prepare_id: str, image_name: str | None
) -> None:
    harness = getattr(env, "harness", None)
    runtime = getattr(harness, "runtime", None)
    if getattr(runtime, "type", None) != "ucloud":
        return
    runtime_image = getattr(runtime, "image", None)
    if image_name and runtime_image and image_name != runtime_image:
        return
    if getattr(runtime, "prepare_id", None):
        return
    model_copy = getattr(runtime, "model_copy", None)
    if callable(model_copy):
        harness.runtime = model_copy(update={"prepare_id": prepare_id})
    else:
        runtime.prepare_id = prepare_id
    get_logger().info(
        "Assigned UCloud prepared capacity to runtime "
        f"(env={env.resolved_name}, prepare_id={prepare_id})"
    )


async def _retry_ucloud_call(
    description: str,
    call,
    *,
    attempts: int,
    delay_seconds: float,
) -> bool:
    for attempt in range(1, attempts + 1):
        try:
            await call()
            return True
        except Exception as exc:
            if attempt >= attempts:
                get_logger().warning(
                    f"{description} failed after {attempts} attempt(s); continuing: {exc}"
                )
                return False
            get_logger().warning(
                f"{description} failed on attempt {attempt}/{attempts}; retrying: {exc}"
            )
            await asyncio.sleep(delay_seconds)
    return False


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
        from ucloud_sandboxes_sdk import AsyncSandboxClient, Image
    except ImportError as exc:
        raise RuntimeError(
            "UCloud sandbox prewarm requires the latest ucloud-sandboxes-sdk."
        ) from exc
    try:
        from ucloud_sandboxes_sdk.client import SandboxApiError
    except ImportError:
        SandboxApiError = Exception

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
    timeout_seconds = _as_int(os.environ.get("UCLOUD_SANDBOX_PREWARM_TIMEOUT_SECONDS"), 300)
    attempts = _as_int(os.environ.get("UCLOUD_SANDBOX_PREWARM_ATTEMPTS"), 3)
    retry_delay_seconds = _as_float(os.environ.get("UCLOUD_SANDBOX_PREWARM_RETRY_DELAY_SECONDS"), 5.0)
    pull_images = _env_bool("UCLOUD_SANDBOX_PREWARM_PULL_IMAGES", True)
    run_slug = _slug(config.output_dir.name)
    num_ucloud_train_envs = sum(1 for env in config.train.env if env in envs)
    client = AsyncSandboxClient(
        base_url,
        api_token=token,
        timeout_seconds=timeout_seconds,
    )
    try:
        for env in envs:
            count = _prewarm_count(config, env, num_ucloud_train_envs)
            args = _capacity_args(env)
            resources = _resources(args)
            prepare_id = f"prime-rl-{run_slug}-{_slug(env.resolved_name)}"
            images = _prewarm_images(args)
            image_objects = [(image, Image.from_registry(image)) for image in images]
            if not image_objects:
                image_objects = [(None, None)]
            for image_name, image in image_objects:
                image_prepare_id = _image_prepare_id(prepare_id, image_name or "", len(image_objects))
                prepared = await _retry_ucloud_call(
                    "UCloud sandbox capacity prewarm "
                    f"(env={env.resolved_name}, image={image_name}, count={count})",
                    lambda: client.prepare_capacity(
                        prepare_id=image_prepare_id,
                        count=count,
                        ttl_seconds=ttl_seconds,
                        image=image,
                        **resources,
                    ),
                    attempts=attempts,
                    delay_seconds=retry_delay_seconds,
                )
                if prepared and pull_images and image is not None:
                    try:
                        await client.pull_image(
                            image,
                            count=count,
                            **resources,
                        )
                    except SandboxApiError as exc:
                        get_logger().warning(
                            "UCloud image pull prewarm failed; continuing after prepare_capacity "
                            f"(env={env.resolved_name}, image={image_name}, count={count}): {exc}"
                        )
                if not prepared:
                    continue
                _set_runtime_prepare_id(
                    env, prepare_id=image_prepare_id, image_name=image_name
                )
                image_part = f", image={image_name}" if image_name else ""
                get_logger().info(
                    "Prepared UCloud sandbox capacity "
                    f"(env={env.resolved_name}, prepare_id={image_prepare_id}, "
                    f"count={count}, cpus={resources['cpus']}, "
                    f"memory_mb={resources['memory_mb']}, disk_mb={resources['disk_mb']}, "
                    f"ttl_seconds={ttl_seconds}{image_part}, "
                    f"pull_image={pull_images and image is not None})"
                )
    finally:
        await client.close()
