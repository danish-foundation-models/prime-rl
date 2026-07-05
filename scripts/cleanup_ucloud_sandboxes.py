#!/usr/bin/env python3
"""Terminate UCloud sandboxes tagged by a PRIME-RL config."""

import argparse
import sys
import tomllib
from pathlib import Path
from typing import Any


def _ucloud_labels(config_path: Path) -> list[str]:
    with config_path.open("rb") as f:
        config = tomllib.load(f)

    orchestrator = config.get("orchestrator")
    if not isinstance(orchestrator, dict):
        orchestrator = config

    labels: list[str] = []

    sections = [orchestrator.get("train", {})]
    if isinstance(orchestrator.get("eval"), dict):
        sections.append(orchestrator["eval"])

    for section in sections:
        envs = section.get("env", []) if isinstance(section, dict) else []
        if not isinstance(envs, list):
            continue
        for env in envs:
            if not isinstance(env, dict):
                continue
            args = env.get("args", {})
            if not isinstance(args, dict):
                continue
            if str(args.get("sandbox_backend") or "").lower() != "ucloud":
                continue

            raw_labels = args.get("labels") or []
            if isinstance(raw_labels, str):
                raw_labels = [raw_labels]
            if isinstance(raw_labels, list):
                labels.extend(str(label) for label in raw_labels)

    return sorted(set(labels))


def _sandbox_id(record: dict[str, Any]) -> str | None:
    spec = record.get("spec")
    if isinstance(spec, dict) and spec.get("id"):
        return str(spec["id"])
    if record.get("id"):
        return str(record["id"])
    return None


def _sandbox_labels(record: dict[str, Any]) -> dict[str, Any]:
    spec = record.get("spec")
    if isinstance(spec, dict) and isinstance(spec.get("labels"), dict):
        return spec["labels"]
    labels = record.get("labels")
    return labels if isinstance(labels, dict) else {}


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, help="Resolved PRIME-RL TOML config.")
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        help="UCloud label to clean. Can be provided multiple times.",
    )
    parser.add_argument("--dry-run", action="store_true", help="List without deleting.")
    args = parser.parse_args()

    labels = list(args.label)
    if args.config:
        labels.extend(_ucloud_labels(args.config))

    labels = sorted(set(labels))
    if not labels:
        print("No UCloud labels found to clean.", file=sys.stderr)
        return 0

    try:
        from ucloud_sandboxes_sdk import SandboxClient
    except ImportError as exc:
        print(f"Failed to import ucloud_sandboxes_sdk: {exc}", file=sys.stderr)
        return 1

    import os

    base_url = (
        os.environ.get("UCLOUD_SANDBOX_API_URL")
        or os.environ.get("UCLOUD_SANDBOX_URL")
        or os.environ.get("UCLOUD_SANDBOX_BASE_URL")
    )
    token = os.environ.get("UCLOUD_SANDBOX_API_TOKEN")
    if not base_url or not token:
        print(
            "UCloud cleanup requires UCLOUD_SANDBOX_API_URL and UCLOUD_SANDBOX_API_TOKEN.",
            file=sys.stderr,
        )
        return 1

    client = SandboxClient(base_url, headers={"Authorization": f"Bearer {token}"})

    total = 0
    if labels:
        label_set = set(labels)
        targets: list[str] = []
        for record in client.list_sandboxes():
            if not isinstance(record, dict):
                continue
            sandbox_id = _sandbox_id(record)
            sandbox_labels = _sandbox_labels(record)
            if sandbox_id and any(sandbox_labels.get(label) == "true" for label in label_set):
                targets.append(sandbox_id)

        print(f"Found {len(targets)} UCloud sandbox(es) for labels: {', '.join(labels)}")
        for sandbox_id in targets:
            total += 1
            if args.dry_run:
                print(f"would delete sandbox {sandbox_id}")
                continue
            print(f"deleting sandbox {sandbox_id}")
            try:
                client.delete_sandbox(sandbox_id)
            except Exception as exc:
                print(f"failed to delete sandbox {sandbox_id}: {exc}", file=sys.stderr)

    print(f"Total matched UCloud resource(s): {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
