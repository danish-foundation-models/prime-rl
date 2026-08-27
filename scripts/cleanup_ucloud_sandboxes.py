#!/usr/bin/env python3
"""Terminate UCloud sandboxes tagged by a PRIME-RL config."""

import argparse
import json
import sys
import tomllib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable


def _ucloud_labels(config_path: Path) -> list[str]:
    if config_path.suffix == ".json":
        with config_path.open() as f:
            config = json.load(f)
    else:
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
        envs = section.get("source", section.get("env", [])) if isinstance(section, dict) else []
        if not isinstance(envs, list):
            continue
        for env in envs:
            if not isinstance(env, dict):
                continue
            labels.extend(_ucloud_labels_from_env(env))
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


def _ucloud_labels_from_env(env: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    if isinstance(env.get("env"), dict):
        env = env["env"]
    runtime = env.get("runtime")
    agent = env.get("agent")
    if isinstance(agent, dict):
        runtime = agent.get("runtime", runtime)
    harness = env.get("harness")
    if isinstance(harness, dict):
        runtime = harness.get("runtime", runtime)
    if not isinstance(runtime, dict):
        return labels
    if str(runtime.get("type") or "").lower() != "ucloud":
        return labels
    raw_labels = runtime.get("labels") or []
    if isinstance(raw_labels, str):
        raw_labels = [raw_labels]
    if isinstance(raw_labels, list):
        labels.extend(str(label) for label in raw_labels)
    return labels


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


def _delete_resources(
    resource_ids: list[str],
    *,
    delete: Callable[[str], object],
    resource_name: str,
    workers: int,
) -> list[str]:
    failures: list[str] = []
    if not resource_ids:
        return failures

    def delete_one(resource_id: str) -> tuple[str, Exception | None]:
        try:
            delete(resource_id)
        except Exception as exc:
            if int(getattr(exc, "status_code", 0) or 0) != 404:
                return resource_id, exc
        return resource_id, None

    with ThreadPoolExecutor(max_workers=min(workers, len(resource_ids))) as executor:
        futures = {}
        for resource_id in resource_ids:
            print(f"deleting {resource_name} {resource_id}")
            futures[executor.submit(delete_one, resource_id)] = resource_id
        for future in as_completed(futures):
            resource_id, error = future.result()
            if error is not None:
                failures.append(resource_id)
                print(
                    f"failed to delete {resource_name} {resource_id}: {error}",
                    file=sys.stderr,
                )
    return sorted(failures)


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, help="Resolved PRIME-RL TOML config.")
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        help="UCloud label to clean. Can be provided multiple times.",
    )
    parser.add_argument(
        "--sandbox-id",
        action="append",
        default=[],
        help="UCloud sandbox ID to delete directly. Can be provided multiple times.",
    )
    parser.add_argument(
        "--sandbox-ids-file",
        type=Path,
        help="File containing UCloud sandbox IDs to delete directly, one per line.",
    )
    parser.add_argument(
        "--prepared-capacity-id",
        action="append",
        default=[],
        help="UCloud prepared-capacity ID to delete. Can be provided multiple times.",
    )
    parser.add_argument(
        "--prepared-capacity-ids-file",
        type=Path,
        help="File containing UCloud prepared-capacity IDs to delete, one per line.",
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=30.0,
        help="Per-request timeout for UCloud API calls.",
    )
    parser.add_argument(
        "--delete-workers",
        type=int,
        default=32,
        help="Maximum concurrent UCloud deletion requests.",
    )
    parser.add_argument(
        "--allow-list-failure",
        action="store_true",
        help="Return success if direct cleanup works but list_sandboxes fails.",
    )
    parser.add_argument("--dry-run", action="store_true", help="List without deleting.")
    args = parser.parse_args()
    if args.delete_workers < 1:
        parser.error("--delete-workers must be at least 1")

    labels = list(args.label)
    if args.config:
        labels.extend(_ucloud_labels(args.config))

    labels = sorted(set(labels))
    if not labels:
        print("No UCloud labels found to clean.", file=sys.stderr)

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

    client = SandboxClient(
        base_url,
        timeout_seconds=args.request_timeout_seconds,
        api_token=token,
    )

    sandbox_ids = list(args.sandbox_id)
    if args.sandbox_ids_file and args.sandbox_ids_file.exists():
        sandbox_ids.extend(
            line.strip()
            for line in args.sandbox_ids_file.read_text().splitlines()
            if line.strip()
        )
    sandbox_ids = sorted(set(sandbox_ids))
    if sandbox_ids:
        print(f"Found {len(sandbox_ids)} explicit UCloud sandbox ID(s) to clean")

    prepared_capacity_ids = list(args.prepared_capacity_id)
    if args.prepared_capacity_ids_file and args.prepared_capacity_ids_file.exists():
        prepared_capacity_ids.extend(
            line.strip()
            for line in args.prepared_capacity_ids_file.read_text().splitlines()
            if line.strip()
        )
    prepared_capacity_ids = sorted(set(prepared_capacity_ids))
    if prepared_capacity_ids:
        print(
            f"Found {len(prepared_capacity_ids)} explicit UCloud prepared-capacity ID(s) "
            "to clean"
        )
    list_failed = False
    if labels:
        label_set = set(labels)
        targets: list[str] = []
        records: list[dict[str, Any]] = []
        try:
            records = client.list_sandboxes()
        except Exception as exc:
            print(f"failed to list UCloud sandboxes by label: {exc}", file=sys.stderr)
            list_failed = True

        for record in records:
            if isinstance(record, dict):
                sandbox_id = _sandbox_id(record)
                sandbox_labels = _sandbox_labels(record)
                if sandbox_id and any(
                    sandbox_labels.get(label) == "true" for label in label_set
                ):
                    targets.append(sandbox_id)

        print(f"Found {len(targets)} UCloud sandbox(es) for labels: {', '.join(labels)}")
        sandbox_ids = sorted(set(sandbox_ids) | set(targets))

    total = len(sandbox_ids) + len(prepared_capacity_ids)
    if args.dry_run:
        for sandbox_id in sandbox_ids:
            print(f"would delete sandbox {sandbox_id}")
        for prepare_id in prepared_capacity_ids:
            print(f"would delete prepared capacity {prepare_id}")
        print(f"Total matched UCloud resource(s): {total}")
        return int(list_failed and not args.allow_list_failure)

    failures = _delete_resources(
        sandbox_ids,
        delete=client.delete_sandbox,
        resource_name="sandbox",
        workers=args.delete_workers,
    )
    failures.extend(
        _delete_resources(
            prepared_capacity_ids,
            delete=client.delete_prepared_capacity,
            resource_name="prepared capacity",
            workers=args.delete_workers,
        )
    )
    print(f"Total matched UCloud resource(s): {total}")
    if failures:
        print(f"Failed to delete {len(failures)} UCloud resource(s)", file=sys.stderr)
    return int(bool(failures) or (list_failed and not args.allow_list_failure))


if __name__ == "__main__":
    raise SystemExit(_main())
