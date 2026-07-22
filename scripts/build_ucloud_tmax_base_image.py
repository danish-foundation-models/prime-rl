#!/usr/bin/env python3
"""Build a reusable UCloud base image for TMax terminal tasks."""

import argparse
import os
import re
import tempfile
import textwrap
import time
from pathlib import Path

DEFAULT_BASE_IMAGE = "ucloud-sandbox-registry:5000/prime-rl/mini-swe-python311:mswe-2.2.8"
DEFAULT_TAG = "ucloud-sandbox-registry:5000/prime-rl/tmax-mini-base:mswe-2.2.8-r5"
DEFAULT_IMAGE_ID = "prime-rl-tmax-mini-base-mswe-2-2-8-r5"
DEFAULT_MINI_VERSION = "2.2.8"


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-").lower()[:96] or "tmax"


def _dockerfile(base_image: str) -> str:
    apt_packages = " ".join(
        [
            "bash",
            "build-essential",
            "ca-certificates",
            "cargo",
            "coreutils",
            "cron",
            "curl",
            "ffmpeg",
            "fonts-dejavu-core",
            "g++",
            "gawk",
            "gcc",
            "git",
            "golang-go",
            "imagemagick",
            "jq",
            "make",
            "openssl",
            "python3",
            "python3-cryptography",
            "python3-matplotlib",
            "python3-networkx",
            "python3-numpy",
            "python3-pandas",
            "python3-pil",
            "python3-pip",
            "python3-pytest",
            "python3-requests",
            "python3-scipy",
            "python3-sklearn",
            "python3-venv",
            "rustc",
            "sqlite3",
            "sudo",
            "tesseract-ocr",
            "unzip",
            "wget",
            "zip",
        ]
    )
    return textwrap.dedent(
        f"""\
        FROM {base_image}

        ENV DEBIAN_FRONTEND=noninteractive \\
            PIP_CONFIG_FILE=/dev/null \\
            PIP_INDEX_URL=https://pypi.org/simple \\
            PIP_BREAK_SYSTEM_PACKAGES=1

        RUN set -eux; \\
            apt-get -o Acquire::Retries=3 update -qq; \\
            apt-get -o Acquire::Retries=3 install -y -qq {apt_packages}; \\
            id user >/dev/null 2>&1 || useradd -m -s /bin/bash user; \\
            mkdir -p /logs/agent /mini-swe-agent /home/user /app /task /opt/rust /usr/local/go /opt/tmax-python; \\
            printf '%s\\n' \\
              'try:' \\
              '    from scapy.config import conf, Conf' \\
              'except Exception:' \\
              '    pass' \\
              'else:' \\
              '    conf.ipv6_enabled = False' \\
              '    Conf.ipv6_enabled = False' \\
              >/opt/tmax-python/sitecustomize.py; \\
            printf '%s\\n' 'user ALL=(ALL) NOPASSWD:ALL' >/etc/sudoers.d/90-prime-rl-user; \\
            chmod 0440 /etc/sudoers.d/90-prime-rl-user; \\
            /usr/local/bin/mini --help >/dev/null; \\
            chown -R user:user /home/user /app /logs /mini-swe-agent /task /opt/rust /usr/local/go /opt/tmax-python || true; \\
            chmod -R a+rwX /home/user /app /logs /mini-swe-agent /task /opt /usr/local; \\
            rm -rf /var/lib/apt/lists/*
        """
    )


def _build_with_status(client, image, timeout_seconds: int, retry_interval_seconds: float) -> dict:
    def on_status(build: dict) -> None:
        build_id = build.get("build_id") or build.get("id") or build.get("image_id")
        print(
            f"build status: id={build_id} status={build.get('status')} "
            f"updated_at={build.get('updated_at')}",
            flush=True,
        )
        log_tail = str(build.get("log_tail") or "").strip()
        if log_tail:
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
            time.sleep(retry_interval_seconds)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--image-id", default=DEFAULT_IMAGE_ID)
    parser.add_argument("--base-image", default=DEFAULT_BASE_IMAGE)
    parser.add_argument("--mini-version", default=DEFAULT_MINI_VERSION)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--retry-interval-seconds", type=float, default=20.0)
    parser.add_argument(
        "--push",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Push the built tag so sandbox nodes can pull it.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dockerfile = _dockerfile(args.base_image)
    if args.dry_run:
        print(dockerfile)
        return 0

    from ucloud_sandboxes_sdk import Image, SandboxClient

    base_url = os.environ.get("UCLOUD_SANDBOX_API_URL")
    token = os.environ.get("UCLOUD_SANDBOX_API_TOKEN")
    if not base_url or not token:
        raise RuntimeError("UCLOUD_SANDBOX_API_URL and UCLOUD_SANDBOX_API_TOKEN are required.")

    image_id = args.image_id or _safe_id(args.tag)
    with tempfile.TemporaryDirectory(prefix="ucloud-tmax-base-") as tmp:
        context = Path(tmp)
        (context / "Dockerfile").write_text(dockerfile)
        client = SandboxClient(
            base_url,
            api_token=token,
            timeout_seconds=max(300, args.timeout_seconds),
        )
        prepare_builder = getattr(client, "prepare_builder", None)
        if callable(prepare_builder):
            prepare_builder(count=1, ttl_seconds=1200)
        image = Image.from_dockerfile(
            name=image_id,
            tag=args.tag,
            context_path=context,
            dockerfile="Dockerfile",
            push=args.push,
            labels={
                "created_by": "prime-rl",
                "purpose": "tmax-base",
                "base_image": args.base_image,
                "mini_swe_agent_version": args.mini_version,
            },
        )
        started = time.perf_counter()
        result = _build_with_status(
            client,
            image,
            args.timeout_seconds,
            args.retry_interval_seconds,
        )
    print(f"built in {time.perf_counter() - started:.1f}s")
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
