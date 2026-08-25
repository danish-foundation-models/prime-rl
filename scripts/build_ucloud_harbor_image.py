#!/usr/bin/env python3
"""Build one exported Harbor task image in the UCloud managed registry."""

import argparse
import os
import sys
import time
from pathlib import Path


def _build_with_status(client, image, timeout_seconds: int) -> dict:
    def on_status(build: dict) -> None:
        build_id = build.get("build_id") or build.get("id") or build.get("image_id")
        print(
            f"build status: id={build_id} status={build.get('status')} updated_at={build.get('updated_at')}",
            flush=True,
        )
        if log_tail := str(build.get("log_tail") or "").strip():
            print(log_tail[-3000:], flush=True)

    deadline = time.monotonic() + timeout_seconds
    attempts = 0
    while True:
        attempts += 1
        try:
            return client.build_image(
                image,
                timeout_seconds=timeout_seconds,
                poll_interval_seconds=10,
                on_status=on_status,
            )
        except Exception as exc:
            status_code = int(getattr(exc, "status_code", 0) or 0)
            if status_code not in {502, 503, 504} or time.monotonic() >= deadline:
                raise
            print(f"builder not ready after attempt {attempts}: {exc}", flush=True)
            time.sleep(20)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    task_dir = args.task_dir.expanduser().resolve()
    context = task_dir / "environment"
    dockerfile = context / "Dockerfile"
    if not dockerfile.is_file():
        raise FileNotFoundError(f"Harbor task Dockerfile not found: {dockerfile}")
    if args.dry_run:
        print(f"task_dir={task_dir}")
        print(f"context={context}")
        print(f"image_id={args.image_id}")
        return 0

    sdk_source = Path(__file__).resolve().parents[1] / "deps/ucloud-sandboxes-sdk/src"
    if sdk_source.is_dir():
        sys.path.insert(0, str(sdk_source))
    from ucloud_sandboxes_sdk import Image, SandboxClient

    base_url = (
        os.environ.get("UCLOUD_SANDBOX_API_URL")
        or os.environ.get("UCLOUD_SANDBOX_URL")
        or os.environ.get("UCLOUD_SANDBOX_BASE_URL")
    )
    token = os.environ.get("UCLOUD_SANDBOX_API_TOKEN")
    if not base_url or not token:
        raise RuntimeError("UCLOUD_SANDBOX_API_URL and UCLOUD_SANDBOX_API_TOKEN are required")

    client = SandboxClient(
        base_url,
        api_token=token,
        timeout_seconds=max(300, args.timeout_seconds),
    )
    prepare_builder = getattr(client, "prepare_builder", None)
    if callable(prepare_builder):
        prepare_builder(count=1, ttl_seconds=1200)
    image = Image.from_dockerfile(
        name=args.image_id,
        context_path=context,
        dockerfile="Dockerfile",
        labels={
            "created_by": "prime-rl",
            "purpose": "harbor-task",
            "harbor_task": task_dir.name,
        },
    )
    started = time.perf_counter()
    result = _build_with_status(client, image, args.timeout_seconds)
    if not result.get("image", {}).get("available_to_sandboxes"):
        raise RuntimeError(f"UCloud build is not available to sandboxes: {result}")
    print(f"built in {time.perf_counter() - started:.1f}s")
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
