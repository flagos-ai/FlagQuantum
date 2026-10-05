"""Explicitly invoked QBoson optimization and sampling smoke test.

This command can consume provider quota. It never falls back to a local solver
and does not constitute QDiffusion or hardware acceptance unless the provider
task and target identities are available in the pinned SDK response mapping.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from flagquantum.remote.kaiwu import (
    KaiwuCredentials,
    KaiwuSDKClient,
    KaiwuTaskClient,
    KaiwuTaskResult,
    resolve_kaiwu_credentials,
    submit_kaiwu_task,
)

from .verify_environment_lock import SHA256, verify_environment_lock

SCHEMA = "flagquantum.qboson_kaiwu_live_smoke"
ACKNOWLEDGEMENT = "I_ACKNOWLEDGE_QBOSON_QUOTA_USAGE"
_MATRIX = ((0.0, 1.0), (1.0, 0.0))


def _result_record(result: KaiwuTaskResult) -> dict[str, Any]:
    receipt = result.receipt
    return {
        "task_name": receipt.task_name,
        "task_mode": receipt.mode,
        "matrix_sha256": receipt.matrix_sha256,
        "matrix_size": receipt.matrix_size,
        "requested_samples": receipt.requested_samples,
        "returned_samples": len(result.samples),
        "provider_task_id": receipt.provider_task_id,
        "provider_target": receipt.provider_target,
        "raw_status": result.raw_status,
        "fallback_occurred": result.metadata.get("fallback_occurred"),
        "minimum_energy": min(result.energies),
        "maximum_energy": max(result.energies),
        "provider_task_id_available": result.metadata.get(
            "provider_task_id_available", receipt.provider_task_id is not None
        ),
        "provider_target_available": result.metadata.get(
            "provider_target_available", receipt.provider_target is not None
        ),
        "provider_result_schema": result.metadata.get("provider_result_schema"),
    }


def run_live_smoke(
    *,
    client: KaiwuTaskClient,
    task_prefix: str,
    project_no: str,
    timeout: float,
    poll_interval: float,
    environment_lock_sha256: str,
    requested_samples: int = 10,
) -> dict[str, Any]:
    """Run one optimization and one sampling task without fallback."""

    if not task_prefix.strip():
        raise ValueError("task_prefix must be non-empty")
    if not project_no.strip():
        raise ValueError("project_no must be non-empty")
    if SHA256.fullmatch(environment_lock_sha256) is None:
        raise ValueError("environment_lock_sha256 must be a lowercase SHA-256 digest")
    real_provider_transport = type(client) is KaiwuSDKClient
    records: list[dict[str, Any]] = []
    for mode in ("optimization", "sampling"):
        job = submit_kaiwu_task(
            _MATRIX,
            client=client,
            task_name=f"{task_prefix.strip()}-{mode}",
            mode=mode,
            requested_samples=requested_samples,
            project_no=project_no.strip(),
        )
        result = job.wait(timeout=timeout, poll_interval=poll_interval)
        records.append(_result_record(result))

    provider_identity_complete = all(
        record["provider_task_id_available"] is True
        and record["provider_target_available"] is True
        and isinstance(record["provider_task_id"], str)
        and bool(record["provider_task_id"].strip())
        and isinstance(record["provider_target"], str)
        and bool(record["provider_target"].strip())
        for record in records
    )
    smoke_passed = all(
        record["raw_status"].strip().lower()
        in {"finished", "completed", "done", "success", "succeed", "succeeded"}
        and record["fallback_occurred"] is False
        and math.isfinite(record["minimum_energy"])
        and math.isfinite(record["maximum_energy"])
        for record in records
    )
    hardware_acceptance = (
        real_provider_transport and smoke_passed and provider_identity_complete
    )
    return {
        "schema": SCHEMA,
        "version": "1.0",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "transport": "kaiwu_cim" if real_provider_transport else "injected_test",
        "real_provider_evidence": real_provider_transport,
        "qboson_hardware_used": real_provider_transport,
        "project_no": project_no.strip(),
        "environment_lock_sha256": environment_lock_sha256,
        "tasks": records,
        "live_provider_smoke_passed": smoke_passed,
        "provider_identity_complete": provider_identity_complete,
        "hardware_acceptance": hardware_acceptance,
        "fallback_occurred": False,
        "limitations": [
            "This smoke test does not execute QDiffusion or A800 tensor work.",
            "Hardware acceptance remains false without provider-reported task and target identities.",
            "This record does not establish performance, quantum advantage, or production maturity.",
        ],
    }


def _contains_forbidden_value(value: object, forbidden_values: tuple[str, ...]) -> bool:
    if isinstance(value, str):
        return any(forbidden and forbidden in value for forbidden in forbidden_values)
    if isinstance(value, dict):
        return any(
            _contains_forbidden_value(key, forbidden_values)
            or _contains_forbidden_value(item, forbidden_values)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_forbidden_value(item, forbidden_values) for item in value)
    return False


def _write_private_json(
    path: Path,
    payload: dict[str, Any],
    *,
    forbidden_values: tuple[str, ...] = (),
) -> None:
    if _contains_forbidden_value(payload, forbidden_values):
        raise RuntimeError("refusing to persist a smoke record containing credentials")
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(encoded)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--environment-lock", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--project-no", required=True)
    parser.add_argument("--task-prefix", required=True)
    parser.add_argument("--expected-sdk-version", default="1.3.1")
    parser.add_argument("--requested-samples", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--poll-interval", type=float, default=60.0)
    parser.add_argument("--acknowledge-provider-cost", required=True)
    arguments = parser.parse_args()
    if arguments.acknowledge_provider_cost != ACKNOWLEDGEMENT:
        parser.error(
            "--acknowledge-provider-cost must equal "
            f"{ACKNOWLEDGEMENT!r}; no task was submitted"
        )

    _, environment_lock_sha256 = verify_environment_lock(arguments.environment_lock)
    user_id, sdk_code = resolve_kaiwu_credentials()
    credentials = KaiwuCredentials(user_id=user_id, sdk_code=sdk_code)
    client = KaiwuSDKClient(
        checkpoint_dir=arguments.checkpoint_dir,
        credentials=credentials,
        expected_version=arguments.expected_sdk_version,
    )
    payload = run_live_smoke(
        client=client,
        task_prefix=arguments.task_prefix,
        project_no=arguments.project_no,
        timeout=arguments.timeout,
        poll_interval=arguments.poll_interval,
        environment_lock_sha256=environment_lock_sha256,
        requested_samples=arguments.requested_samples,
    )
    _write_private_json(
        arguments.output,
        payload,
        forbidden_values=(user_id, sdk_code),
    )
    print(f"Private smoke record written to {arguments.output}")
    if not payload["hardware_acceptance"]:
        print("Hardware acceptance remains closed; inspect the recorded limitations.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
