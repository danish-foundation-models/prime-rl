#!/usr/bin/env python3
"""Build a UCloud sandbox image with mini-SWE-agent preinstalled."""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path


DEFAULT_MINI_VERSION = "2.2.8"
DEFAULT_UV_IMAGE = "ghcr.io/astral-sh/uv:0.8.17"


def _safe_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-").lower()
    return cleaned[:96] or "mini-swe-agent"


def _dockerfile(
    base_image: str,
    mini_version: str,
    uv_image: str,
    *,
    offline_wheelhouse: bool = False,
) -> str:
    copy_wheelhouse = ""
    install_command = f"uv pip install --python /opt/mini-swe-agent/venv/bin/python mini-swe-agent=={mini_version}"
    if offline_wheelhouse:
        copy_wheelhouse = "COPY wheelhouse/ /opt/mini-swe-agent/wheelhouse/\n"
        install_command = (
            "uv pip install --python /opt/mini-swe-agent/venv/bin/python "
            "--no-index --find-links /opt/mini-swe-agent/wheelhouse "
            f"mini-swe-agent=={mini_version}"
        )
    return textwrap.dedent(
        f"""\
        FROM {uv_image} AS uv
        FROM {base_image}

        COPY --from=uv /uv /usr/local/bin/uv
        {copy_wheelhouse.rstrip()}
        RUN set -eux; \\
            export DEBIAN_FRONTEND=noninteractive; \\
            export PIP_CONFIG_FILE=/dev/null; \\
            export UV_CACHE_DIR=/tmp/uv-cache; \\
            export UV_CONCURRENT_DOWNLOADS=1; \\
            export UV_HTTP_TIMEOUT=300; \\
            export UV_LINK_MODE=copy; \\
            PYTHON_BIN="$(command -v python3 || command -v python || true)"; \\
            if [ -z "$PYTHON_BIN" ]; then \\
              apt-get -o Acquire::Retries=3 update -qq; \\
              apt-get -o Acquire::Retries=3 install -y -qq python3 python3-venv ca-certificates; \\
              PYTHON_BIN="$(command -v python3)"; \\
            fi; \\
            if ! "$PYTHON_BIN" -c 'import sys; raise SystemExit(sys.version_info < (3, 10))'; then \\
              uv python install 3.11; \\
              PYTHON_BIN="$(uv python find 3.11)"; \\
            fi; \\
            mkdir -p /opt/mini-swe-agent /logs/agent /mini-swe-agent; \\
            uv venv --python "$PYTHON_BIN" /opt/mini-swe-agent/venv; \\
            for attempt in 1 2 3 4 5; do \\
              {install_command} && break; \\
              if [ "$attempt" = 5 ]; then exit 1; fi; \\
              sleep 20; \\
            done; \\
            printf '%s\\n' '#!/usr/bin/env sh' \\
              'exec /opt/mini-swe-agent/venv/bin/python -m minisweagent.run.mini "$@"' \\
              > /usr/local/bin/mini; \\
            chmod +x /usr/local/bin/mini; \\
            /usr/local/bin/mini --help >/dev/null; \\
            rm -rf "$UV_CACHE_DIR"
        """
    )


def _prepare_wheelhouse(target: Path, mini_version: str, python: str) -> None:
    uv = shutil.which("uv")
    if not uv:
        raise RuntimeError("uv is required to prepare the mini-SWE-agent wheelhouse")
    env = dict(os.environ)
    env.setdefault("UV_CACHE_DIR", "/scratch/project_465002183/.cache/uv")
    env["UV_NO_CONFIG"] = "1"
    subprocess.run(
        [
            uv,
            "run",
            "--no-project",
            "--python",
            python,
            "--with",
            "pip",
            "python",
            "-m",
            "pip",
            "download",
            "--only-binary=:all:",
            "--dest",
            str(target),
            f"mini-swe-agent=={mini_version}",
        ],
        env=env,
        check=True,
    )


def _build_with_retries(client, image: object, timeout_seconds: int, retry_interval_seconds: float) -> dict:
    deadline = time.monotonic() + timeout_seconds
    attempts = 0

    def on_status(build: dict[str, object]) -> None:
        build_id = build.get("build_id") or build.get("id") or build.get("image_id")
        status = build.get("status")
        updated_at = build.get("updated_at")
        node = build.get("node") or build.get("node_id") or build.get("builder")
        header = f"build status: id={build_id} status={status} updated_at={updated_at}"
        if node:
            header += f" node={node}"
        print(header, flush=True)
        log_tail = str(build.get("log_tail") or "").strip()
        if log_tail:
            print(log_tail[-2000:], flush=True)

    while True:
        attempts += 1
        try:
            try:
                return client.build_image(
                    image,
                    timeout_seconds=timeout_seconds,
                    poll_interval_seconds=10,
                    on_status=on_status,
                )
            except TypeError:
                return client.build_image(image)
        except Exception as exc:
            status_code = int(getattr(exc, "status_code", 0) or 0)
            body = getattr(exc, "body", None)
            message = str(exc).lower()
            retryable_timeout = status_code == 0 and "timed out" in message
            if (
                status_code not in {502, 503, 504}
                and not retryable_timeout
            ) or time.monotonic() >= deadline:
                raise
            detail = str(body or exc).replace("\n", " ")
            if len(detail) > 500:
                detail = detail[:500] + "..."
            print(
                f"builder not ready after attempt {attempts}: {detail}; retrying",
                file=sys.stderr,
            )
            time.sleep(retry_interval_seconds)


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-image", required=True, help="Base SWE task image.")
    parser.add_argument("--tag", required=True, help="Derived image tag to build.")
    parser.add_argument("--image-id", help="Gateway image id. Defaults from --tag.")
    parser.add_argument("--mini-version", default=DEFAULT_MINI_VERSION)
    parser.add_argument(
        "--offline-wheelhouse",
        action="store_true",
        help="Download wheels locally and install from COPYed wheelhouse during Docker build.",
    )
    parser.add_argument(
        "--wheel-python",
        default="/usr/bin/python3.11",
        help="Host Python version used for resolving the offline wheelhouse.",
    )
    parser.add_argument(
        "--uv-image",
        default=DEFAULT_UV_IMAGE,
        help="Image stage used to copy the uv binary into the build.",
    )
    parser.add_argument("--push", action="store_true", help="Ask gateway to push the built tag.")
    parser.add_argument("--dry-run", action="store_true", help="Print Dockerfile without building.")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--retry-interval-seconds", type=float, default=20.0)
    args = parser.parse_args()

    dockerfile = _dockerfile(
        args.base_image,
        args.mini_version,
        args.uv_image,
        offline_wheelhouse=args.offline_wheelhouse,
    )
    if args.dry_run:
        print(dockerfile)
        return 0

    try:
        from ucloud_sandboxes_sdk import Image, SandboxClient
    except ImportError as exc:
        print(f"Failed to import ucloud_sandboxes_sdk: {exc}", file=sys.stderr)
        return 1

    base_url = (
        os.environ.get("UCLOUD_SANDBOX_API_URL")
        or os.environ.get("UCLOUD_SANDBOX_URL")
        or os.environ.get("UCLOUD_SANDBOX_BASE_URL")
    )
    token = os.environ.get("UCLOUD_SANDBOX_API_TOKEN")
    if not base_url or not token:
        print(
            "Image build requires UCLOUD_SANDBOX_API_URL and UCLOUD_SANDBOX_API_TOKEN.",
            file=sys.stderr,
        )
        return 1

    image_id = args.image_id or _safe_id(args.tag)
    with tempfile.TemporaryDirectory(prefix="ucloud-mini-swe-image-") as tmp:
        context = Path(tmp)
        if args.offline_wheelhouse:
            print("preparing offline wheelhouse", flush=True)
            _prepare_wheelhouse(context / "wheelhouse", args.mini_version, args.wheel_python)
        (context / "Dockerfile").write_text(dockerfile)
        client = SandboxClient(
            base_url,
            headers={"Authorization": f"Bearer {token}"},
            timeout_seconds=max(300, args.timeout_seconds),
        )
        image = Image.from_dockerfile(
            name=image_id,
            tag=args.tag,
            context_path=str(context),
            dockerfile="Dockerfile",
            push=args.push,
            labels={
                "created_by": "prime-rl",
                "purpose": "mini-swe-agent-v2",
                "base_image": args.base_image,
                "mini_swe_agent_version": args.mini_version,
            },
        )
        result = _build_with_retries(
            client,
            image,
            timeout_seconds=args.timeout_seconds,
            retry_interval_seconds=args.retry_interval_seconds,
        )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
