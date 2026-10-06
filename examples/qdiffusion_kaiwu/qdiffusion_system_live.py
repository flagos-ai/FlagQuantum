"""Run the bounded QDiffusion system slice on A800 plus QBoson CIM.

This command consumes provider quota. It binds the run to a preregistered
configuration and never falls back. A completed run remains failed for hardware
acceptance unless the pinned provider mapping supplies task and target IDs.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import platform
import re
import socket
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from examples.qdiffusion_kaiwu.failure_evidence import redacted_failure_record
from examples.qdiffusion_kaiwu.private_io import (
    read_private_bytes,
    validate_private_directory,
    validate_private_json_output_path,
    write_private_json_exclusive,
)
from examples.qdiffusion_kaiwu.provider_inputs import normalize_provider_identifier
from examples.qdiffusion_kaiwu.provider_resources import (
    assess_provider_budget,
    load_provider_resources,
)
from examples.qdiffusion_kaiwu.qdiffusion_system_development_probe import (
    FULL_REVISION,
    HOSTS,
    _execute_qdiffusion_slice,
    _load_pinned_qdiffusion_api,
)
from examples.qdiffusion_kaiwu.sdk_approval import (
    verify_approved_kaiwu_distribution,
)
from examples.qdiffusion_kaiwu.source_preflight import load_source_preflight
from examples.qdiffusion_kaiwu.strict_json import loads_json_strict
from examples.qdiffusion_kaiwu.validate_acceptance import _validate_config
from examples.qdiffusion_kaiwu.verify_environment_lock import (
    verify_frozen_environment_lock,
)
from flagquantum.ecosystem.kaiwu import KaiwuSampler
from flagquantum.remote.kaiwu import (
    KaiwuCredentials,
    KaiwuSDKClient,
    KaiwuTaskClient,
    resolve_kaiwu_credentials,
)
from flagquantum.version import __version__ as flagquantum_version

ACKNOWLEDGEMENT = "I_ACKNOWLEDGE_QBOSON_QUOTA_USAGE"
SCHEMA = "flagquantum.qboson_qdiffusion_system_live_probe"
_MAX_CONFIG_BYTES = 1024 * 1024


def _load_frozen_config(path: Path) -> tuple[dict[str, Any], str]:
    encoded = read_private_bytes(
        path, label="frozen configuration", max_bytes=_MAX_CONFIG_BYTES
    )
    raw = loads_json_strict(encoded)
    if not isinstance(raw, dict):
        raise ValueError("frozen configuration must be a JSON object")
    errors: list[str] = []
    _validate_config(raw, errors)
    if errors:
        raise ValueError("invalid frozen configuration: " + "; ".join(errors))
    return raw, hashlib.sha256(encoded).hexdigest()


def _validate_lane(
    config: dict[str, Any],
    *,
    execution_host: str,
    source_revision: str,
    plugin_revision: str,
    sdk_version: str,
) -> tuple[str, int, tuple[int, int]]:
    if execution_host not in HOSTS:
        raise ValueError("execution_host must be one of the two declared A800 hosts")
    if FULL_REVISION.fullmatch(source_revision) is None:
        raise ValueError("source_revision must be a full lowercase Git revision")
    if FULL_REVISION.fullmatch(plugin_revision) is None:
        raise ValueError("plugin_revision must be a full lowercase Git revision")
    software = config["software"]
    observed = {
        "source_revision": source_revision,
        "flagquantum_version": flagquantum_version,
        "kaiwu_pytorch_plugin_revision": plugin_revision,
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "kaiwu_sdk_version": sdk_version,
    }
    for field, value in observed.items():
        if software.get(field) != value:
            raise ValueError(f"observed {field} differs from the frozen configuration")
    if execution_host == config["primary_host"]:
        role = "primary"
    elif execution_host == config["replay_host"]:
        role = "portability_replay"
    else:
        raise ValueError("execution_host is not assigned a role in the configuration")
    remote_call_budget = config["remote_call_budget"]
    precision = config["precision_policy"]
    target_range = (precision["target_min"], precision["target_max"])
    return role, remote_call_budget, target_range


def _validate_requested_cuda_device(device_name: str) -> torch.device:
    """Require the single CUDA device authorized by the acceptance protocol."""

    try:
        device = torch.device(device_name)
    except (RuntimeError, TypeError):
        raise ValueError("live QDiffusion system probe requires explicit cuda:0") from None
    if device != torch.device("cuda:0"):
        raise ValueError("live QDiffusion system probe requires explicit cuda:0")
    return device


def _receipt_records(sampler: KaiwuSampler) -> list[dict[str, Any]]:
    return [
        {
            "schema": receipt.schema,
            "task_name": receipt.task_name,
            "matrix_sha256": receipt.matrix_sha256,
            "matrix_size": receipt.matrix_size,
            "mode": receipt.mode,
            "requested_samples": receipt.requested_samples,
            "project_no": receipt.project_no,
            "submitted_at": receipt.submitted_at,
            "provider_task_id": receipt.provider_task_id,
            "provider_target": receipt.provider_target,
        }
        for receipt in sampler.receipts
    ]


def _completed_provider_result_has_identity(sampler: KaiwuSampler) -> bool:
    """Require a validated result, not merely a submitted task receipt."""

    result = sampler.last_result
    return bool(
        result is not None
        and isinstance(result.receipt.provider_task_id, str)
        and bool(result.receipt.provider_task_id.strip())
        and isinstance(result.receipt.provider_target, str)
        and bool(result.receipt.provider_target.strip())
    )


def _precision_evidence_complete(
    sampler: KaiwuSampler,
    receipts: list[dict[str, Any]],
) -> bool:
    """Reconcile live precision evidence, receipts, and sampler transfers."""

    reports = sampler.precision_reports
    evidence = sampler.precision_evidence
    if not reports or len(evidence) != len(reports):
        return False
    for report, item in zip(reports, evidence, strict=True):
        if (
            item.normalized_dtype != report.normalized_dtype
            or item.normalized_min != report.normalized_min
            or item.normalized_max != report.normalized_max
            or item.symmetry_normalization != report.symmetry_normalization
            or item.rounding_policy != report.rounding_policy
            or item.scale_factor != report.scale_factor
            or item.target_min != report.target_min
            or item.target_max != report.target_max
            or item.max_abs_error != report.max_abs_error
            or item.mean_abs_error != report.mean_abs_error
        ):
            return False
        numeric_values = (
            item.normalized_min,
            item.normalized_max,
            item.scale_factor,
            item.max_abs_error,
            item.mean_abs_error,
        )
        if (
            not isinstance(item.original_matrix_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", item.original_matrix_sha256) is None
            or not isinstance(item.submission_matrix_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", item.submission_matrix_sha256) is None
            or item.source_type != "numpy.ndarray"
            or not isinstance(item.source_dtype, str)
            or not item.source_dtype.strip()
            or item.normalized_dtype != "torch.float64"
            or item.symmetry_normalization != "arithmetic_mean"
            or item.rounding_policy != "round_half_to_even"
            or any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                for value in numeric_values
            )
            or not all(math.isfinite(float(value)) for value in numeric_values)
            or type(item.target_min) is not int
            or type(item.target_max) is not int
            or item.normalized_min > item.normalized_max
            or item.target_min >= 0
            or item.target_max <= 0
            or item.scale_factor <= 0
            or item.max_abs_error < 0
            or item.mean_abs_error < 0
            or item.mean_abs_error > item.max_abs_error
        ):
            return False
        maximum_magnitude = max(abs(item.normalized_min), abs(item.normalized_max))
        expected_scale = (
            1.0
            if maximum_magnitude == 0.0
            else min(abs(item.target_min), item.target_max) / maximum_magnitude
        )
        if not math.isclose(
            item.scale_factor, expected_scale, rel_tol=1e-12, abs_tol=1e-12
        ):
            return False
    original_digests = {item.original_matrix_sha256 for item in evidence}
    if len(original_digests) != len(evidence):
        return False
    receipt_digests = {receipt.get("matrix_sha256") for receipt in receipts}
    if {item.submission_matrix_sha256 for item in evidence} != receipt_digests:
        return False
    precision_origins = {
        (
            item.original_matrix_sha256,
            item.submission_matrix_sha256,
            item.source_type,
            item.source_dtype,
        )
        for item in evidence
    }
    transfer_origins = {
        (
            record.original_matrix_sha256,
            record.submission_matrix_sha256,
            record.input_type,
            record.input_dtype,
        )
        for record in sampler.transfer_records
    }
    return precision_origins == transfer_origins


def run_live_system_probe(
    *,
    client: KaiwuTaskClient,
    config: dict[str, Any],
    config_sha256: str,
    execution_host: str,
    observed_hostname: str,
    source_revision: str,
    plugin_revision: str,
    source_preflight_sha256: str,
    transfer_manifest_sha256: str,
    environment_lock_sha256: str,
    sdk_version: str,
    device: torch.device,
    observed_gpu: str,
    project_no: str,
    task_prefix: str,
    requested_samples: int,
    timeout: float,
    poll_interval: float,
    real_provider_transport: bool,
    plugin_root: Path,
) -> dict[str, Any]:
    if requested_samples != config.get("requested_samples"):
        raise ValueError("requested_samples differs from the frozen configuration")
    role, remote_call_budget, target_range = _validate_lane(
        config,
        execution_host=execution_host,
        source_revision=source_revision,
        plugin_revision=plugin_revision,
        sdk_version=sdk_version,
    )
    sampler = KaiwuSampler(
        client=client,
        task_name=task_prefix,
        project_no=project_no,
        requested_samples=requested_samples,
        timeout=timeout,
        poll_interval=poll_interval,
        max_remote_calls=remote_call_budget,
        integer_target_range=target_range,
    )
    failure: dict[str, str] | None = None
    slice_record: dict[str, Any] = {}
    retrieval_resubmitted: bool | None = None
    try:
        slice_record = _execute_qdiffusion_slice(
            device,
            plugin_root=plugin_root,
            sampler=sampler,
            remote_call_budget=remote_call_budget,
        )
        if sampler.last_job is None:
            raise RuntimeError("QDiffusion completed without a recoverable last job")
        call_count = sampler.remote_call_count
        receipt = sampler.last_job.receipt
        repeated = sampler.last_job.result()
        retrieval_resubmitted = not (
            sampler.remote_call_count == call_count and repeated.receipt == receipt
        )
    except (Exception, KeyboardInterrupt) as exc:
        failure = redacted_failure_record(exc)

    receipts = _receipt_records(sampler)
    provider_identity_complete = bool(receipts) and all(
        isinstance(receipt["provider_task_id"], str)
        and bool(receipt["provider_task_id"].strip())
        and isinstance(receipt["provider_target"], str)
        and bool(receipt["provider_target"].strip())
        for receipt in receipts
    )
    targets = {
        receipt["provider_target"]
        for receipt in receipts
        if isinstance(receipt["provider_target"], str)
    }
    run_completed = failure is None
    precision_reports = sampler.precision_reports
    precision_evidence = sampler.precision_evidence
    precision_complete = _precision_evidence_complete(sampler, receipts)
    verified_provider_transport = (
        real_provider_transport and type(client) is KaiwuSDKClient
    )
    provider_use_proven = bool(
        verified_provider_transport
        and sampler.remote_call_count > 0
        and _completed_provider_result_has_identity(sampler)
    )
    tensor_devices_complete = all(
        slice_record.get(field) == "cuda:0"
        for field in ("proposal_device", "energy_device", "generated_device")
    )
    training_values = tuple(
        slice_record.get(field)
        for field in ("objective", "gradient_norm", "parameter_delta_max")
    )
    training_complete = bool(
        all(
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
            for value in training_values
        )
        and float(training_values[1]) > 0.0
        and float(training_values[2]) > 0.0
    )
    system_acceptance = bool(
        run_completed
        and verified_provider_transport
        and provider_use_proven
        and provider_identity_complete
        and precision_complete
        and retrieval_resubmitted is False
        and device == torch.device("cuda:0")
        and "A800" in observed_gpu
        and tensor_devices_complete
        and training_complete
        and slice_record.get("fallback_occurred") is False
        and slice_record.get("token_constraints_passed") is True
    )
    return {
        "schema": SCHEMA,
        "version": "1.0",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "experiment_config_sha256": config_sha256,
        "source_revision": source_revision,
        "flagquantum_version": flagquantum_version,
        "kaiwu_pytorch_plugin_revision": plugin_revision,
        "source_preflight_sha256": source_preflight_sha256,
        "transfer_manifest_sha256": transfer_manifest_sha256,
        "environment_lock_sha256": environment_lock_sha256,
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "kaiwu_sdk_version": sdk_version,
        "execution_host": execution_host,
        "observed_hostname": observed_hostname,
        "run_role": role,
        "requested_cuda_device": str(device),
        "observed_tensor_device": slice_record.get("generated_device"),
        "observed_gpu_model": observed_gpu,
        "transport": "kaiwu_cim" if verified_provider_transport else "injected_test",
        "qboson_hardware_used": provider_use_proven,
        "real_provider_evidence": provider_use_proven,
        "pinned_sdk_client": verified_provider_transport,
        "provider_identity_complete": provider_identity_complete,
        "provider_reported_target": provider_identity_complete,
        "provider_result_schema": (
            sampler.last_result.metadata.get("provider_result_schema")
            if sampler.last_result is not None
            else None
        ),
        "qboson_target": next(iter(targets)) if len(targets) == 1 else None,
        "qboson_task_ids": [
            receipt["provider_task_id"]
            for receipt in receipts
            if isinstance(receipt["provider_task_id"], str)
        ],
        "task_receipts": receipts,
        "sampling_mode": "sampling",
        "requested_samples": requested_samples,
        "returned_samples": (
            len(sampler.last_result.samples)
            if sampler.last_result is not None
            else None
        ),
        "remote_call_budget": remote_call_budget,
        "remote_call_count": sampler.remote_call_count,
        "precision_policy": {
            "name": config["precision_policy"]["name"],
            "target_min": target_range[0],
            "target_max": target_range[1],
            "matrix_count": len(precision_reports),
            "scale_factor_min": (
                min(report.scale_factor for report in precision_reports)
                if precision_reports
                else None
            ),
            "scale_factor_max": (
                max(report.scale_factor for report in precision_reports)
                if precision_reports
                else None
            ),
            "max_abs_error": (
                max(report.max_abs_error for report in precision_reports)
                if precision_reports
                else None
            ),
            "mean_of_matrix_mean_abs_error": (
                sum(report.mean_abs_error for report in precision_reports)
                / len(precision_reports)
                if precision_reports
                else None
            ),
        },
        "precision_evidence_complete": precision_complete,
        "precision_evidence": [asdict(evidence) for evidence in precision_evidence],
        "fallback_occurred": False,
        "retrieval_resubmitted": retrieval_resubmitted,
        "secrets_redacted": True,
        "run_completed": run_completed,
        "failure": failure,
        "training": {
            key: slice_record.get(key)
            for key in ("objective", "gradient_norm", "parameter_delta_max")
        },
        "generation": {
            "generated_tokens": slice_record.get("generated_tokens"),
            "token_constraints_passed": slice_record.get("token_constraints_passed"),
        },
        "transfer_accounting": slice_record.get("transfer_accounting"),
        "acceptance": {
            "system": "pass" if system_acceptance else "fail",
            "application": "not_run",
        },
        "limitations": [
            "This bounded system probe does not run the frozen protein effectiveness experiment.",
            "System acceptance remains failed without provider-reported task and target identities.",
            "Two independent passing host records are required; this is one single-device run.",
            "No performance, distributed, domestic-accelerator, or quantum-advantage claim is made.",
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


def _write_private_redacted_json(
    path: Path, payload: dict[str, Any], *, forbidden_values: tuple[str, ...]
) -> None:
    if _contains_forbidden_value(payload, forbidden_values):
        raise RuntimeError("refusing to write evidence containing a credential value")
    write_private_json_exclusive(path, payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--execution-host", choices=sorted(HOSTS), required=True)
    parser.add_argument("--expected-hostname", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--plugin-revision", required=True)
    parser.add_argument("--plugin-root", required=True, type=Path)
    parser.add_argument("--source-preflight", required=True, type=Path)
    parser.add_argument("--environment-lock", required=True, type=Path)
    parser.add_argument("--provider-resources", required=True, type=Path)
    parser.add_argument("--project-no", required=True)
    parser.add_argument("--task-prefix", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expected-sdk-version", choices=("1.3.1",), default="1.3.1")
    parser.add_argument("--requested-samples", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--poll-interval", type=float, default=60.0)
    parser.add_argument("--acknowledge-provider-cost", required=True)
    arguments = parser.parse_args()
    if not arguments.plugin_root.is_absolute():
        parser.error("--plugin-root must be an absolute path")
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
    config, config_sha256 = _load_frozen_config(arguments.config)
    if arguments.requested_samples != config["requested_samples"]:
        parser.error("--requested-samples differs from the frozen configuration")
    provider_resources, _ = load_provider_resources(arguments.provider_resources)
    resources_ready, resource_reason = assess_provider_budget(
        provider_resources,
        mode="sampling",
        required_calls=config["remote_call_budget"],
    )
    if not resources_ready:
        parser.error(f"provider resource gate failed: {resource_reason}")
    if arguments.expected_hostname != config["host_identities"][
        arguments.execution_host
    ]:
        parser.error("--expected-hostname differs from the frozen host identity")
    observed_hostname = socket.gethostname()
    if observed_hostname != arguments.expected_hostname:
        parser.error("observed hostname differs from --expected-hostname")
    try:
        device = _validate_requested_cuda_device(arguments.device)
    except ValueError as exc:
        parser.error(str(exc))
    if not torch.cuda.is_available():
        parser.error("live QDiffusion system probe requires an observed CUDA device")
    torch.cuda.set_device(device)
    observed_gpu = torch.cuda.get_device_name(device)
    if "A800" not in observed_gpu:
        parser.error("live QDiffusion system probe requires an NVIDIA A800")
    _validate_lane(
        config,
        execution_host=arguments.execution_host,
        source_revision=arguments.source_revision,
        plugin_revision=arguments.plugin_revision,
        sdk_version=arguments.expected_sdk_version,
    )
    source_preflight, source_preflight_sha256 = load_source_preflight(
        arguments.source_preflight,
        execution_host=arguments.execution_host,
        source_revision=arguments.source_revision,
        plugin_revision=arguments.plugin_revision,
        source_root=Path(__file__).resolve().parents[2],
        plugin_root=arguments.plugin_root,
    )
    _load_pinned_qdiffusion_api(arguments.plugin_root)
    environment_record, environment_lock_sha256 = verify_frozen_environment_lock(
        arguments.environment_lock,
        expected_sha256=config["software"]["environment_lock_sha256"],
    )
    verify_approved_kaiwu_distribution(environment_record, config["kaiwu_sdk"])

    user_id, sdk_code = resolve_kaiwu_credentials()
    credentials = KaiwuCredentials(user_id=user_id, sdk_code=sdk_code)
    os.environ.pop("QBOSON_USER_ID", None)
    os.environ.pop("QBOSON_SDK_CODE", None)
    client = KaiwuSDKClient(
        checkpoint_dir=arguments.checkpoint_dir,
        credentials=credentials,
        expected_version=arguments.expected_sdk_version,
    )
    payload = run_live_system_probe(
        client=client,
        config=config,
        config_sha256=config_sha256,
        execution_host=arguments.execution_host,
        observed_hostname=observed_hostname,
        source_revision=arguments.source_revision,
        plugin_revision=arguments.plugin_revision,
        source_preflight_sha256=source_preflight_sha256,
        transfer_manifest_sha256=source_preflight["manifest_sha256"],
        environment_lock_sha256=environment_lock_sha256,
        sdk_version=arguments.expected_sdk_version,
        device=device,
        observed_gpu=observed_gpu,
        project_no=arguments.project_no,
        task_prefix=arguments.task_prefix,
        requested_samples=arguments.requested_samples,
        timeout=arguments.timeout,
        poll_interval=arguments.poll_interval,
        real_provider_transport=True,
        plugin_root=arguments.plugin_root,
    )
    _write_private_redacted_json(
        arguments.output,
        payload,
        forbidden_values=(user_id, sdk_code),
    )
    print(f"Private live-system record written to {arguments.output}")
    if payload["acceptance"]["system"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
