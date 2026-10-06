"""Resume one retained Kaiwu task identity without calling FlagQuantum submit.

The pinned SDK may still contact QBoson when it queries the documented
``task_name + ising_matrix`` identity. An earlier ambiguous failure may have
occurred before the provider accepted that identity, so this command requires
the same explicit quota acknowledgement as a new live submission.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from examples.qdiffusion_kaiwu.failure_evidence import redacted_failure_record
from examples.qdiffusion_kaiwu.private_io import (
    read_private_bytes,
    validate_private_directory,
    validate_private_json_output_path,
)
from examples.qdiffusion_kaiwu.provider_inputs import normalize_provider_identifier
from examples.qdiffusion_kaiwu.qboson_live_smoke import (
    ACKNOWLEDGEMENT,
    _attempt_record,
    _result_record,
    _write_private_json,
)
from examples.qdiffusion_kaiwu.sdk_approval import (
    load_sdk_approval,
    verify_approved_kaiwu_distribution,
    verify_approved_project_assignment,
)
from examples.qdiffusion_kaiwu.verify_environment_lock import (
    SHA256,
    verify_environment_lock,
)
from flagquantum.remote.kaiwu import (
    KaiwuCredentials,
    KaiwuSDKClient,
    KaiwuTaskClient,
    KaiwuTaskMode,
    resolve_kaiwu_credentials,
    restore_kaiwu_job,
)

SCHEMA = "flagquantum.qboson_kaiwu_resume"
MAX_RECEIPT_BYTES = 64 * 1024 * 1024
RESUME_RECORD_FIELDS = frozenset(
    {
        "schema",
        "version",
        "recorded_at",
        "transport",
        "environment_lock_sha256",
        "sdk_approval_sha256",
        "task",
        "resume_completed",
        "failure",
        "provider_result_validated",
        "provider_identity_complete",
        "real_provider_evidence",
        "qboson_hardware_used",
        "flagquantum_submit_api_called",
        "fallback_occurred",
        "secrets_redacted",
        "limitations",
    }
)
RESUME_LIMITATIONS = (
    "This command queries one retained task identity and never calls the "
    "FlagQuantum submit API.",
    "The pinned SDK solve operation may contact QBoson and can consume quota "
    "if the earlier provider outcome was ambiguous.",
    "This diagnostic record does not establish system, application, "
    "performance, or production acceptance.",
)


def run_resume(
    *,
    client: KaiwuTaskClient,
    recovery_receipt: Path,
    expected_mode: KaiwuTaskMode,
    project_no: str,
    timeout: float,
    poll_interval: float,
    environment_lock_sha256: str,
    sdk_approval_sha256: str,
) -> dict[str, Any]:
    """Restore and poll one existing task without invoking ``client.submit``."""

    project_no = normalize_provider_identifier(project_no, label="project_no")
    for label, digest in (
        ("environment_lock_sha256", environment_lock_sha256),
        ("sdk_approval_sha256", sdk_approval_sha256),
    ):
        if SHA256.fullmatch(digest) is None:
            raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    if expected_mode not in {"optimization", "sampling"}:
        raise ValueError("expected_mode must be optimization or sampling")

    job = restore_kaiwu_job(recovery_receipt, client=client)
    if job.receipt.mode != expected_mode:
        raise ValueError("retained Kaiwu task mode does not match --mode")
    if job.receipt.project_no != project_no:
        raise ValueError("retained Kaiwu project does not match --project-no")

    failure: dict[str, str] | None = None
    result = None
    try:
        result = job.wait(timeout=timeout, poll_interval=poll_interval)
        task = _result_record(result)
    except (Exception, KeyboardInterrupt) as exc:
        task = _attempt_record(job)
        failure = redacted_failure_record(exc)

    real_provider_transport = type(client) is KaiwuSDKClient
    provider_identity_complete = bool(
        result is not None
        and isinstance(result.receipt.provider_task_id, str)
        and bool(result.receipt.provider_task_id.strip())
        and isinstance(result.receipt.provider_target, str)
        and bool(result.receipt.provider_target.strip())
    )
    provider_use_proven = bool(
        real_provider_transport
        and provider_identity_complete
        and result is not None
        and result.metadata.get("fallback_occurred") is False
    )
    return {
        "schema": SCHEMA,
        "version": "1.0",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "transport": "kaiwu_cim" if real_provider_transport else "injected_test",
        "environment_lock_sha256": environment_lock_sha256,
        "sdk_approval_sha256": sdk_approval_sha256,
        "task": task,
        "resume_completed": failure is None,
        "failure": failure,
        "provider_result_validated": result is not None,
        "provider_identity_complete": provider_identity_complete,
        "real_provider_evidence": provider_use_proven,
        "qboson_hardware_used": provider_use_proven,
        "flagquantum_submit_api_called": False,
        "fallback_occurred": False,
        "secrets_redacted": True,
        "limitations": list(RESUME_LIMITATIONS),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--recovery-receipt", required=True, type=Path)
    parser.add_argument("--environment-lock", required=True, type=Path)
    parser.add_argument("--sdk-approval", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--project-no", required=True)
    parser.add_argument("--mode", choices=("optimization", "sampling"), required=True)
    parser.add_argument("--expected-sdk-version", choices=("1.3.1",), default="1.3.1")
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--poll-interval", type=float, default=60.0)
    parser.add_argument("--acknowledge-provider-cost", required=True)
    arguments = parser.parse_args()
    if arguments.acknowledge_provider_cost != ACKNOWLEDGEMENT:
        parser.error(
            "--acknowledge-provider-cost must equal "
            f"{ACKNOWLEDGEMENT!r}; no task was queried"
        )
    try:
        arguments.project_no = normalize_provider_identifier(
            arguments.project_no, label="--project-no"
        )
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))

    validate_private_json_output_path(arguments.output)
    validate_private_directory(
        arguments.checkpoint_dir, label="Kaiwu checkpoint directory"
    )
    read_private_bytes(
        arguments.recovery_receipt,
        label="Kaiwu recovery receipt",
        max_bytes=MAX_RECEIPT_BYTES,
    )
    sdk_approval, sdk_approval_sha256 = load_sdk_approval(arguments.sdk_approval)
    verify_approved_project_assignment(arguments.project_no, sdk_approval)
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
    payload = run_resume(
        client=client,
        recovery_receipt=arguments.recovery_receipt,
        expected_mode=arguments.mode,
        project_no=arguments.project_no,
        timeout=arguments.timeout,
        poll_interval=arguments.poll_interval,
        environment_lock_sha256=environment_lock_sha256,
        sdk_approval_sha256=sdk_approval_sha256,
    )
    _write_private_json(
        arguments.output,
        payload,
        forbidden_values=(user_id, sdk_code),
    )
    print(f"Private resume record written to {arguments.output}")
    if not payload["resume_completed"]:
        print("Task recovery remains incomplete; inspect the redacted record.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
