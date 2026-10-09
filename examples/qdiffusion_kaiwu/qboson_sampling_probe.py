"""Run one explicitly authorized QBoson sampling development probe.

This command can consume exactly ten provider sampling credits. It cannot
request optimization, does not fall back to a local solver, and never
constitutes Phase 2 or QDiffusion hardware acceptance.
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
)
from examples.qdiffusion_kaiwu.provider_inputs import normalize_provider_identifier
from examples.qdiffusion_kaiwu.provider_resources import (
    assess_provider_budget,
    load_provider_resources,
    provider_resource_valid_until,
)
from examples.qdiffusion_kaiwu.qboson_live_smoke import (
    SMOKE_MATRIX,
    _attempt_record,
    _result_record,
    _write_private_json,
)
from examples.qdiffusion_kaiwu.qboson_optimization_probe import (
    EXPECTED_KAIWU_VERSION,
    _verify_pinned_kaiwu_distribution,
)
from examples.qdiffusion_kaiwu.verify_environment_lock import (
    SHA256,
    verify_environment_lock,
)
from flagquantum.remote.kaiwu import (
    KaiwuCredentials,
    KaiwuRemoteJob,
    KaiwuSDKClient,
    KaiwuTaskClient,
    resolve_kaiwu_credentials,
    submit_kaiwu_task,
)

SCHEMA = "flagquantum.qboson_kaiwu_sampling_probe"
ACKNOWLEDGEMENT = "I_ACKNOWLEDGE_TEN_QBOSON_SAMPLING_CREDITS"
REQUESTED_SAMPLES = 10
LIMITATIONS = (
    "This development probe is not the Phase 2 optimization-and-sampling smoke test.",
    "This probe does not execute QDiffusion or A800 tensor work.",
    "Kaiwu 1.3.1 does not expose a documented provider task ID or target mapping.",
    "This record does not establish hardware acceptance, performance, quantum advantage, or production maturity.",
)


def run_sampling_probe(
    *,
    client: KaiwuTaskClient,
    task_name: str,
    project_no: str | None,
    timeout: float,
    poll_interval: float,
    environment_lock_sha256: str,
    provider_resources_sha256: str,
) -> dict[str, Any]:
    """Run one ten-sample task and return a non-acceptance evidence record."""

    task_name = normalize_provider_identifier(task_name, label="task_name")
    if project_no is not None:
        project_no = normalize_provider_identifier(project_no, label="project_no")
    if SHA256.fullmatch(environment_lock_sha256) is None:
        raise ValueError("environment_lock_sha256 must be a lowercase SHA-256 digest")
    if SHA256.fullmatch(provider_resources_sha256) is None:
        raise ValueError("provider_resources_sha256 must be a lowercase SHA-256 digest")

    job: KaiwuRemoteJob | None = None
    task: dict[str, Any] | None = None
    failure: dict[str, str] | None = None
    try:
        job = submit_kaiwu_task(
            SMOKE_MATRIX,
            client=client,
            task_name=task_name,
            mode="sampling",
            requested_samples=REQUESTED_SAMPLES,
            project_no=project_no,
        )
        task = _result_record(job.wait(timeout=timeout, poll_interval=poll_interval))
    except (Exception, KeyboardInterrupt) as exc:
        if job is not None:
            task = _attempt_record(job)
        failure = redacted_failure_record(exc)

    completed = bool(
        failure is None
        and task is not None
        and task["task_mode"] == "sampling"
        and isinstance(task["raw_status"], str)
        and task["raw_status"].strip().lower()
        in {"finished", "completed", "done", "success", "succeed", "succeeded"}
        and task["fallback_occurred"] is False
        and task["returned_samples"] == REQUESTED_SAMPLES
        and math.isfinite(task["minimum_energy"])
        and math.isfinite(task["maximum_energy"])
    )
    real_transport = type(client) is KaiwuSDKClient
    return {
        "schema": SCHEMA,
        "version": "1.0",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "transport": "kaiwu_cim" if real_transport else "injected_test",
        "mode": "sampling",
        "maximum_provider_calls": 1,
        "maximum_sampling_credits": REQUESTED_SAMPLES,
        "project_no": project_no,
        "environment_lock_sha256": environment_lock_sha256,
        "provider_resources_sha256": provider_resources_sha256,
        "task": task,
        "run_completed": completed,
        "failure": failure,
        "real_provider_transport_invoked": real_transport and task is not None,
        "hardware_acceptance": False,
        "qdiffusion_acceptance": False,
        "fallback_occurred": False,
        "secrets_redacted": True,
        "limitations": list(LIMITATIONS),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--environment-lock", required=True, type=Path)
    parser.add_argument("--provider-resources", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--project-no")
    parser.add_argument("--task-name", required=True)
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
        arguments.task_name = normalize_provider_identifier(
            arguments.task_name, label="--task-name"
        )
        if arguments.project_no is not None:
            arguments.project_no = normalize_provider_identifier(
                arguments.project_no, label="--project-no"
            )
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))

    validate_private_json_output_path(arguments.output)
    validate_private_directory(
        arguments.checkpoint_dir, label="Kaiwu checkpoint directory"
    )
    provider_resources, provider_resources_sha256 = load_provider_resources(
        arguments.provider_resources
    )
    budget_ready, budget_reason = assess_provider_budget(
        provider_resources,
        mode="sampling",
        required_calls=REQUESTED_SAMPLES,
    )
    if not budget_ready:
        parser.error(f"provider resource gate failed: {budget_reason}")

    environment, environment_lock_sha256 = verify_environment_lock(
        arguments.environment_lock
    )
    _verify_pinned_kaiwu_distribution(environment)
    user_id, sdk_code = resolve_kaiwu_credentials()
    credentials = KaiwuCredentials(user_id=user_id, sdk_code=sdk_code)
    os.environ.pop("QBOSON_USER_ID", None)
    os.environ.pop("QBOSON_SDK_CODE", None)
    client = KaiwuSDKClient(
        checkpoint_dir=arguments.checkpoint_dir,
        credentials=credentials,
        expected_version=EXPECTED_KAIWU_VERSION,
        submission_deadline=provider_resource_valid_until(provider_resources),
    )
    payload = run_sampling_probe(
        client=client,
        task_name=arguments.task_name,
        project_no=arguments.project_no,
        timeout=arguments.timeout,
        poll_interval=arguments.poll_interval,
        environment_lock_sha256=environment_lock_sha256,
        provider_resources_sha256=provider_resources_sha256,
    )
    _write_private_json(
        arguments.output,
        payload,
        forbidden_values=(user_id, sdk_code),
    )
    print(f"Private sampling probe record written to {arguments.output}")
    if not payload["run_completed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
