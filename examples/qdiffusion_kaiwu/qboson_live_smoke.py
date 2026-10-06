"""Explicitly invoked QBoson optimization and sampling smoke test.

This command can consume provider quota. It never falls back to a local solver
and does not constitute QDiffusion or hardware acceptance unless the provider
task and target identities are available in the pinned SDK response mapping.
"""

from __future__ import annotations

import argparse
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from examples.qdiffusion_kaiwu.failure_evidence import redacted_failure_record
from examples.qdiffusion_kaiwu.private_io import (
    validate_private_directory,
    validate_private_json_output_path,
    write_private_json_exclusive,
)
from examples.qdiffusion_kaiwu.provider_inputs import normalize_provider_identifier
from examples.qdiffusion_kaiwu.sdk_approval import (
    load_sdk_approval,
    verify_approved_kaiwu_distribution,
)
from flagquantum.remote.kaiwu import (
    KaiwuCredentials,
    KaiwuRemoteJob,
    KaiwuSDKClient,
    KaiwuTaskClient,
    KaiwuTaskResult,
    resolve_kaiwu_credentials,
    submit_kaiwu_task,
)
from flagquantum.remote.kaiwu.jobs import _freeze_matrix, _matrix_sha256

from .verify_environment_lock import SHA256, verify_environment_lock

SCHEMA = "flagquantum.qboson_kaiwu_live_smoke"
ACKNOWLEDGEMENT = "I_ACKNOWLEDGE_QBOSON_QUOTA_USAGE"
SMOKE_MATRIX = ((0.0, 1.0), (1.0, 0.0))
SMOKE_MATRIX_SHA256 = _matrix_sha256(_freeze_matrix(SMOKE_MATRIX))
SMOKE_TASK_FIELDS = frozenset(
    {
        "receipt_schema",
        "task_name",
        "task_mode",
        "matrix_sha256",
        "matrix_size",
        "requested_samples",
        "project_no",
        "submitted_at",
        "returned_samples",
        "samples",
        "energies",
        "provider_task_id",
        "provider_target",
        "raw_status",
        "fallback_occurred",
        "minimum_energy",
        "maximum_energy",
        "provider_task_id_available",
        "provider_target_available",
        "provider_result_schema",
    }
)
SMOKE_RECORD_FIELDS = frozenset(
    {
        "schema",
        "version",
        "recorded_at",
        "transport",
        "real_provider_evidence",
        "qboson_hardware_used",
        "qboson_target",
        "project_no",
        "environment_lock_sha256",
        "sdk_approval_sha256",
        "tasks",
        "run_completed",
        "failure",
        "live_provider_smoke_passed",
        "provider_identity_complete",
        "hardware_acceptance",
        "fallback_occurred",
        "secrets_redacted",
        "limitations",
    }
)
SMOKE_LIMITATIONS = (
    "This smoke test does not execute QDiffusion or A800 tensor work.",
    "Hardware acceptance remains false without provider-reported task and target identities.",
    "This record does not establish performance, quantum advantage, or production maturity.",
)


def _result_record(result: KaiwuTaskResult) -> dict[str, Any]:
    receipt = result.receipt
    return {
        "receipt_schema": receipt.schema,
        "task_name": receipt.task_name,
        "task_mode": receipt.mode,
        "matrix_sha256": receipt.matrix_sha256,
        "matrix_size": receipt.matrix_size,
        "requested_samples": receipt.requested_samples,
        "project_no": receipt.project_no,
        "submitted_at": receipt.submitted_at,
        "returned_samples": len(result.samples),
        "samples": [list(sample) for sample in result.samples],
        "energies": [float(energy) for energy in result.energies],
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


def _attempt_record(job: KaiwuRemoteJob) -> dict[str, Any]:
    """Retain a submitted task identity when no validated result is available."""

    receipt = job.receipt
    return {
        "receipt_schema": receipt.schema,
        "task_name": receipt.task_name,
        "task_mode": receipt.mode,
        "matrix_sha256": receipt.matrix_sha256,
        "matrix_size": receipt.matrix_size,
        "requested_samples": receipt.requested_samples,
        "project_no": receipt.project_no,
        "submitted_at": receipt.submitted_at,
        "returned_samples": None,
        "samples": None,
        "energies": None,
        "provider_task_id": receipt.provider_task_id,
        "provider_target": receipt.provider_target,
        "raw_status": job.raw_status,
        "fallback_occurred": False,
        "minimum_energy": None,
        "maximum_energy": None,
        "provider_task_id_available": receipt.provider_task_id is not None,
        "provider_target_available": receipt.provider_target is not None,
        "provider_result_schema": None,
    }


def run_live_smoke(
    *,
    client: KaiwuTaskClient,
    task_prefix: str,
    project_no: str,
    timeout: float,
    poll_interval: float,
    environment_lock_sha256: str,
    sdk_approval_sha256: str,
    requested_samples: int = 10,
) -> dict[str, Any]:
    """Run one optimization and one sampling task without fallback."""

    task_prefix = normalize_provider_identifier(task_prefix, label="task_prefix")
    project_no = normalize_provider_identifier(project_no, label="project_no")
    if SHA256.fullmatch(environment_lock_sha256) is None:
        raise ValueError("environment_lock_sha256 must be a lowercase SHA-256 digest")
    if SHA256.fullmatch(sdk_approval_sha256) is None:
        raise ValueError("sdk_approval_sha256 must be a lowercase SHA-256 digest")
    real_provider_transport = type(client) is KaiwuSDKClient
    records: list[dict[str, Any]] = []
    failure: dict[str, str] | None = None
    for mode in ("optimization", "sampling"):
        job: KaiwuRemoteJob | None = None
        try:
            job = submit_kaiwu_task(
                SMOKE_MATRIX,
                client=client,
                task_name=f"{task_prefix}-{mode}",
                mode=mode,
                requested_samples=requested_samples,
                project_no=project_no,
            )
            result = job.wait(timeout=timeout, poll_interval=poll_interval)
            records.append(_result_record(result))
        except (Exception, KeyboardInterrupt) as exc:
            if job is not None:
                records.append(_attempt_record(job))
            failure = redacted_failure_record(exc)
            break

    provider_task_ids = [
        record.get("provider_task_id")
        for record in records
        if isinstance(record.get("provider_task_id"), str)
    ]
    provider_targets = {
        record.get("provider_target")
        for record in records
        if isinstance(record.get("provider_target"), str)
    }
    provider_identity_complete = (
        len(records) == 2
        and len(provider_task_ids) == 2
        and len(set(provider_task_ids)) == 2
        and len(provider_targets) == 1
        and all(
        record["provider_task_id_available"] is True
        and record["provider_target_available"] is True
        and isinstance(record["provider_task_id"], str)
        and bool(record["provider_task_id"].strip())
        and isinstance(record["provider_target"], str)
        and bool(record["provider_target"].strip())
        for record in records
        )
    )
    smoke_passed = (
        failure is None
        and len(records) == 2
        and all(
            record["raw_status"].strip().lower()
            in {"finished", "completed", "done", "success", "succeed", "succeeded"}
            and record["fallback_occurred"] is False
            and math.isfinite(record["minimum_energy"])
            and math.isfinite(record["maximum_energy"])
            for record in records
        )
    )
    provider_use_proven = bool(
        real_provider_transport
        and any(
            isinstance(record["provider_task_id"], str)
            and bool(record["provider_task_id"].strip())
            and isinstance(record["provider_target"], str)
            and bool(record["provider_target"].strip())
            and type(record["returned_samples"]) is int
            and record["returned_samples"] > 0
            and isinstance(record["raw_status"], str)
            and record["raw_status"].strip().lower()
            in {"finished", "completed", "done", "success", "succeed", "succeeded"}
            and record["fallback_occurred"] is False
            and math.isfinite(record["minimum_energy"])
            and math.isfinite(record["maximum_energy"])
            for record in records
        )
    )
    hardware_acceptance = (
        real_provider_transport and smoke_passed and provider_identity_complete
    )
    return {
        "schema": SCHEMA,
        "version": "1.0",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "transport": "kaiwu_cim" if real_provider_transport else "injected_test",
        "real_provider_evidence": provider_use_proven,
        "qboson_hardware_used": provider_use_proven,
        "qboson_target": (
            next(iter(provider_targets)) if provider_identity_complete else None
        ),
        "project_no": project_no,
        "environment_lock_sha256": environment_lock_sha256,
        "sdk_approval_sha256": sdk_approval_sha256,
        "tasks": records,
        "run_completed": failure is None,
        "failure": failure,
        "live_provider_smoke_passed": smoke_passed,
        "provider_identity_complete": provider_identity_complete,
        "hardware_acceptance": hardware_acceptance,
        "fallback_occurred": False,
        "secrets_redacted": True,
        "limitations": list(SMOKE_LIMITATIONS),
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
    write_private_json_exclusive(path, payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--environment-lock", required=True, type=Path)
    parser.add_argument("--sdk-approval", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--project-no", required=True)
    parser.add_argument("--task-prefix", required=True)
    parser.add_argument("--expected-sdk-version", choices=("1.3.1",), default="1.3.1")
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
    try:
        arguments.project_no = normalize_provider_identifier(
            arguments.project_no, label="--project-no"
        )
        arguments.task_prefix = normalize_provider_identifier(
            arguments.task_prefix, label="--task-prefix"
        )
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    validate_private_json_output_path(arguments.output)
    validate_private_directory(
        arguments.checkpoint_dir, label="Kaiwu checkpoint directory"
    )

    sdk_approval, sdk_approval_sha256 = load_sdk_approval(arguments.sdk_approval)
    environment_record, environment_lock_sha256 = verify_environment_lock(
        arguments.environment_lock
    )
    verify_approved_kaiwu_distribution(environment_record, sdk_approval)
    user_id, sdk_code = resolve_kaiwu_credentials()
    credentials = KaiwuCredentials(user_id=user_id, sdk_code=sdk_code)
    os.environ.pop("QBOSON_USER_ID", None)
    os.environ.pop("QBOSON_SDK_CODE", None)
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
        sdk_approval_sha256=sdk_approval_sha256,
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
