"""Fail-closed validator for QBoson QDiffusion acceptance evidence."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import re
from datetime import datetime, timedelta
from pathlib import Path, PurePosixPath
from statistics import fmean
from typing import Any
from urllib.parse import urlsplit

from examples.qdiffusion_kaiwu.plan_quota import (
    estimate_portability_remote_calls as _estimate_portability_remote_calls,
)
from examples.qdiffusion_kaiwu.plan_quota import (
    estimate_protein_remote_calls as _estimate_protein_remote_calls,
)
from examples.qdiffusion_kaiwu.preflight_protein_artifacts import ARTIFACT_FIELDS
from examples.qdiffusion_kaiwu.preflight_protein_artifacts import (
    PREFLIGHT_SCHEMA as ARTIFACT_PREFLIGHT_COMPONENT_SCHEMA,
)
from examples.qdiffusion_kaiwu.private_io import read_private_bytes
from examples.qdiffusion_kaiwu.provider_resources import (
    SCHEMA as PROVIDER_RESOURCES_COMPONENT_SCHEMA,
)
from examples.qdiffusion_kaiwu.provider_resources import (
    assess_provider_budget,
    assess_provider_resources,
    validate_provider_resource_gate,
    validate_provider_resources,
)
from examples.qdiffusion_kaiwu.qboson_live_smoke import (
    SCHEMA as PROVIDER_SMOKE_COMPONENT_SCHEMA,
)
from examples.qdiffusion_kaiwu.qboson_live_smoke import (
    SMOKE_LIMITATIONS,
    SMOKE_MATRIX,
    SMOKE_MATRIX_SHA256,
    SMOKE_RECORD_FIELDS,
    SMOKE_TASK_FIELDS,
)
from examples.qdiffusion_kaiwu.sdk_approval import (
    SCHEMA as SDK_APPROVAL_COMPONENT_SCHEMA,
)
from examples.qdiffusion_kaiwu.sdk_approval import (
    VERSION as SDK_APPROVAL_COMPONENT_VERSION,
)
from examples.qdiffusion_kaiwu.sdk_approval import (
    validate_sdk_approval_record,
    verify_approved_kaiwu_distribution,
)
from examples.qdiffusion_kaiwu.source_preflight import (
    SCHEMA as SOURCE_PREFLIGHT_COMPONENT_SCHEMA,
)
from examples.qdiffusion_kaiwu.source_preflight import (
    TRANSFER_MANIFEST_SCHEMA,
    validate_common_transfer_manifest,
    validate_source_preflight_record,
    validate_transfer_manifest_record,
)
from examples.qdiffusion_kaiwu.strict_json import loads_json_strict
from examples.qdiffusion_kaiwu.verify_environment_lock import (
    parse_environment_lock_bytes,
)

HOSTS = {"jp-a800-171", "jp-a800-172"}
FULL_REVISION = re.compile(r"[0-9a-f]{40}")
LICENSE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+() -]{0,127}")
UNAPPROVED_LICENSE_IDS = {
    "NOASSERTION",
    "NONE",
    "UNKNOWN",
    "UNLICENSED",
}


def _effective_uid() -> int:
    getter = getattr(os, "geteuid", None)
    if not callable(getter):
        raise ValueError("platform cannot validate acceptance evidence ownership")
    return int(getter())


CONFIG_SCHEMA = "flagquantum.qboson_qdiffusion_config"
RECORD_SCHEMA = "flagquantum.qboson_qdiffusion_acceptance"
MANIFEST_SCHEMA = "flagquantum.qboson_qdiffusion_manifest"
SYSTEM_COMPONENT_SCHEMA = "flagquantum.qboson_qdiffusion_system_live_probe"
PORTABILITY_COMPONENT_SCHEMA = "flagquantum.qboson_qdiffusion_portability_replay"
TRAINING_COMPONENT_SCHEMA = "flagquantum.qboson_qdiffusion_protein_training"
EVALUATION_COMPONENT_SCHEMA = "flagquantum.qboson_qdiffusion_protein_evaluation"
TASK_RECEIPT_SCHEMA = "flagquantum.kaiwu-task.v1"
SYSTEM_COMPONENT_FIELDS = frozenset(
    {
        "schema",
        "version",
        "recorded_at",
        "experiment_config_sha256",
        "source_revision",
        "flagquantum_version",
        "kaiwu_pytorch_plugin_revision",
        "source_preflight_sha256",
        "transfer_manifest_sha256",
        "environment_lock_sha256",
        "provider_resource_gate",
        "python_version",
        "torch_version",
        "kaiwu_sdk_version",
        "execution_host",
        "observed_hostname",
        "run_role",
        "requested_cuda_device",
        "observed_tensor_device",
        "observed_gpu_model",
        "transport",
        "qboson_hardware_used",
        "real_provider_evidence",
        "pinned_sdk_client",
        "provider_identity_complete",
        "provider_reported_target",
        "provider_result_schema",
        "qboson_target",
        "qboson_task_ids",
        "task_receipts",
        "sampling_mode",
        "requested_samples",
        "returned_samples",
        "remote_call_budget",
        "remote_call_count",
        "precision_policy",
        "precision_evidence_complete",
        "precision_evidence",
        "fallback_occurred",
        "retrieval_resubmitted",
        "secrets_redacted",
        "run_completed",
        "failure",
        "training",
        "generation",
        "transfer_accounting",
        "acceptance",
        "limitations",
    }
)
TRAINING_COMPONENT_FIELDS = frozenset(
    {
        "schema",
        "version",
        "recorded_at",
        "experiment_config_sha256",
        "artifact_preflight_sha256",
        "artifact_inputs_unchanged",
        "source_revision",
        "flagquantum_version",
        "kaiwu_pytorch_plugin_revision",
        "source_preflight_sha256",
        "transfer_manifest_sha256",
        "environment_lock_sha256",
        "provider_resource_gate",
        "python_version",
        "torch_version",
        "kaiwu_sdk_version",
        "execution_host",
        "observed_hostname",
        "observed_gpu_model",
        "requested_cuda_device",
        "seed",
        "transport",
        "pinned_sdk_client",
        "real_provider_evidence",
        "qboson_hardware_used",
        "provider_reported_target",
        "qboson_target",
        "qboson_task_ids",
        "sampling_mode",
        "requested_samples",
        "fallback_occurred",
        "secrets_redacted",
        "run_completed",
        "failure",
        "remote_call_budget",
        "protein_remote_call_budget_per_seed",
        "estimated_worst_case_remote_calls",
        "remote_call_count",
        "task_receipts",
        "provider_identity_complete",
        "precision_report_count",
        "precision_policy",
        "precision_evidence_complete",
        "precision_evidence",
        "run_directory_name",
        "trained_energy_checkpoint_name",
        "trained_energy_checkpoint_sha256",
        "workflow_artifacts",
        "acceptance",
        "limitations",
    }
)
EVALUATION_COMPONENT_FIELDS = frozenset(
    {
        "schema",
        "version",
        "recorded_at",
        "experiment_config_sha256",
        "training_record_sha256",
        "source_revision",
        "flagquantum_version",
        "kaiwu_pytorch_plugin_revision",
        "source_preflight_sha256",
        "transfer_manifest_sha256",
        "environment_lock_sha256",
        "python_version",
        "torch_version",
        "kaiwu_sdk_version",
        "execution_host",
        "observed_hostname",
        "observed_gpu_model",
        "observed_tensor_device",
        "seed",
        "evaluation_model_sha256",
        "artifact_inputs_unchanged",
        "baseline_metrics",
        "guided_metrics",
        "secrets_redacted",
        "provider_quota_consumed",
        "acceptance",
    }
)
PORTABILITY_COMPONENT_FIELDS = frozenset(
    {
        "schema",
        "version",
        "recorded_at",
        "experiment_config_sha256",
        "artifact_preflight_sha256",
        "artifact_inputs_unchanged",
        "training_record_sha256",
        "source_revision",
        "flagquantum_version",
        "kaiwu_pytorch_plugin_revision",
        "source_preflight_sha256",
        "transfer_manifest_sha256",
        "environment_lock_sha256",
        "provider_resource_gate",
        "python_version",
        "torch_version",
        "kaiwu_sdk_version",
        "execution_host",
        "observed_hostname",
        "observed_gpu_model",
        "requested_cuda_device",
        "observed_tensor_device",
        "transport",
        "pinned_sdk_client",
        "qboson_hardware_used",
        "real_provider_evidence",
        "provider_identity_complete",
        "provider_reported_target",
        "qboson_target",
        "qboson_task_ids",
        "task_receipts",
        "sampling_mode",
        "requested_samples",
        "returned_samples",
        "remote_call_budget",
        "remote_call_count",
        "fallback_occurred",
        "retrieval_resubmitted",
        "secrets_redacted",
        "precision_policy",
        "precision_evidence",
        "precision_evidence_complete",
        "trained_energy_checkpoint_sha256",
        "artifacts",
        "fixture",
        "run_completed",
        "failure",
        "acceptance",
        "limitations",
    }
)
COMPONENT_FIELDS_BY_SCHEMA = {
    SYSTEM_COMPONENT_SCHEMA: SYSTEM_COMPONENT_FIELDS,
    TRAINING_COMPONENT_SCHEMA: TRAINING_COMPONENT_FIELDS,
    EVALUATION_COMPONENT_SCHEMA: EVALUATION_COMPONENT_FIELDS,
    PORTABILITY_COMPONENT_SCHEMA: PORTABILITY_COMPONENT_FIELDS,
}
PRECISION_POLICY_FIELDS = frozenset(
    {
        "name",
        "target_min",
        "target_max",
        "matrix_count",
        "scale_factor_min",
        "scale_factor_max",
        "max_abs_error",
        "mean_of_matrix_mean_abs_error",
    }
)
SYSTEM_TRAINING_FIELDS = frozenset(
    {"objective", "gradient_norm", "parameter_delta_max"}
)
SYSTEM_GENERATION_FIELDS = frozenset(
    {"generated_tokens", "token_constraints_passed"}
)
TRANSFER_ACCOUNTING_FIELDS = frozenset(
    {
        "matrix_origin_device",
        "sampler_boundaries",
        "returned_sample_target_device",
    }
)
SAMPLER_BOUNDARY_FIELDS = frozenset(
    {
        "input_type",
        "input_device",
        "input_dtype",
        "matrix_shape",
        "original_matrix_sha256",
        "submission_matrix_sha256",
        "canonical_device",
        "canonical_dtype",
        "submission_storage",
        "returned_storage",
        "returned_dtype",
        "returned_shape",
        "cache_hit",
    }
)
WORKFLOW_ARTIFACT_FIELDS = frozenset(
    {
        "test_fasta",
        "baseline_fasta",
        "guided_fasta",
        "training_history",
        "sequence_metrics",
        "baseline_quality",
        "guided_quality",
    }
)
WORKFLOW_ARTIFACT_IDENTITY_FIELDS = frozenset({"relative_path", "sha256"})
PORTABILITY_ARTIFACT_FIELDS = frozenset(
    {
        "dataset_sha256",
        "base_checkpoint_sha256",
        "tokenizer_sha256",
        "evaluation_model_sha256",
        "trained_energy_checkpoint_sha256",
    }
)
PORTABILITY_FIXTURE_FIELDS = frozenset(
    {
        "training_seed",
        "index",
        "steps",
        "energy_objective",
        "generated_length",
        "generated_sha256",
        "token_constraints_passed",
    }
)
SYSTEM_LIMITATIONS = (
    "This bounded system probe does not run the frozen protein effectiveness experiment.",
    "System acceptance remains failed without provider-reported task and target identities.",
    "Two independent passing host records are required; this is one single-device run.",
    "No performance, distributed, domestic-accelerator, or quantum-advantage claim is made.",
)
TRAINING_LIMITATIONS = (
    "This record covers one protein-training seed only.",
    "ESM2 evaluation and two-host system acceptance are separate gates.",
    "No performance, distributed, domestic-accelerator, or quantum-advantage claim is made.",
)
PORTABILITY_LIMITATIONS = (
    "This is one fixed replay fixture, not a second training run.",
    "Final acceptance also requires both system gates and all primary-host seeds.",
    "No performance, distributed, domestic-accelerator, or quantum-advantage claim is made.",
)
FINAL_ACCEPTANCE_LIMITATIONS = (
    "This record establishes development acceptance only.",
    "The two host records are independent single-device runs and do not establish distributed execution.",
    "These runs use NVIDIA A800 devices and do not establish domestic-accelerator support.",
    "Application evidence is limited to the frozen seeds, sample size, artifacts, and recorded environment.",
    "No performance superiority, production support, quantum advantage, or scalability claim is made.",
)
TASK_RECEIPT_FIELDS = frozenset(
    {
        "schema",
        "task_name",
        "matrix_sha256",
        "matrix_size",
        "mode",
        "requested_samples",
        "project_no",
        "submitted_at",
        "provider_task_id",
        "provider_target",
    }
)
PRECISION_EVIDENCE_FIELDS = frozenset(
    {
        "original_matrix_sha256",
        "submission_matrix_sha256",
        "source_type",
        "source_dtype",
        "normalized_dtype",
        "normalized_min",
        "normalized_max",
        "symmetry_normalization",
        "rounding_policy",
        "scale_factor",
        "target_min",
        "target_max",
        "max_abs_error",
        "mean_abs_error",
    }
)
METRIC_NAMES = (
    "mean_cosine_distance",
    "median_cosine_distance",
    "mean_l2_distance",
    "median_l2_distance",
    "identity_to_reference_mean",
    "amino_acid_jsd",
    "kmer2_jsd",
    "kmer3_jsd",
    "uniqueness_ratio",
    "repeat_ratio_ge4",
    "length_match_ratio",
    "invalid_sequence_count",
)
MANIFEST_FIELDS = frozenset(
    {
        "schema",
        "version",
        "config",
        "environment_lock",
        "records",
        "component_records",
    }
)
MANIFEST_MEMBER_FIELDS = frozenset({"path", "sha256"})
FINAL_RECORD_COMMON_FIELDS = frozenset(
    {
        "schema",
        "version",
        "source_revision",
        "flagquantum_version",
        "kaiwu_pytorch_plugin_revision",
        "source_preflight_sha256",
        "transfer_manifest_sha256",
        "environment_lock_sha256",
        "provider_resource_gate",
        "python_version",
        "torch_version",
        "kaiwu_sdk_version",
        "experiment_config_sha256",
        "execution_host",
        "run_role",
        "requested_cuda_device",
        "observed_tensor_device",
        "observed_gpu_model",
        "transport",
        "qboson_hardware_used",
        "real_provider_evidence",
        "provider_reported_target",
        "qboson_target",
        "qboson_task_ids",
        "sampling_mode",
        "requested_samples",
        "returned_samples",
        "remote_call_budget",
        "remote_call_count",
        "fallback_occurred",
        "retrieval_resubmitted",
        "secrets_redacted",
        "artifacts",
        "precision_policy",
        "training",
        "generation",
        "transfer_accounting",
        "acceptance",
        "limitations",
        "component_bundle_required",
        "system_evidence_sha256",
    }
)
FINAL_PRIMARY_RECORD_FIELDS = FINAL_RECORD_COMMON_FIELDS | {
    "attempted_seeds",
    "baseline_metrics",
    "guided_metrics",
    "application_evidence",
}
FINAL_REPLAY_RECORD_FIELDS = FINAL_RECORD_COMMON_FIELDS | {"portability_evidence"}
FINAL_ARTIFACT_FIELDS = frozenset(
    {
        "dataset_sha256",
        "base_checkpoint_sha256",
        "tokenizer_sha256",
        "evaluation_model_sha256",
        "trained_energy_checkpoint_sha256",
    }
)
FINAL_TRAINING_FIELDS = frozenset(
    {"energy_objective", "gradient_norm", "parameter_delta_max"}
)
FINAL_GENERATION_FIELDS = frozenset(
    {"token_constraints_passed", "invalid_sequence_count"}
)
FINAL_ACCEPTANCE_FIELDS = frozenset({"system", "application"})
APPLICATION_EVIDENCE_FIELDS = frozenset({"aggregation", "records"})
APPLICATION_EVIDENCE_RECORD_FIELDS = frozenset(
    {
        "seed",
        "training_record_sha256",
        "evaluation_record_sha256",
        "trained_energy_checkpoint_sha256",
    }
)
PORTABILITY_EVIDENCE_FIELDS = frozenset(
    {
        "record_sha256",
        "training_seed",
        "training_record_sha256",
        "trained_energy_checkpoint_sha256",
        "acceptance",
    }
)
CONFIG_FIELDS = frozenset(
    {
        "schema",
        "version",
        "preregistered_at",
        "primary_host",
        "replay_host",
        "host_identities",
        "software",
        "kaiwu_sdk",
        "dataset",
        "checkpoint",
        "tokenizer",
        "evaluation_model",
        "training",
        "generation",
        "evaluation",
        "seeds",
        "requested_samples",
        "remote_call_budget",
        "precision_policy",
        "primary_metric",
        "thresholds",
    }
)
CONFIG_SOFTWARE_FIELDS = frozenset(
    {
        "source_revision",
        "flagquantum_version",
        "kaiwu_pytorch_plugin_revision",
        "python_version",
        "torch_version",
        "kaiwu_sdk_version",
        "environment_lock_sha256",
    }
)
CONFIG_ARTIFACT_FIELDS = frozenset(
    {
        "name",
        "revision",
        "source_url",
        "license_id",
        "license_evidence_url",
        "license_reviewed_at",
        "sha256",
    }
)
CONFIG_DATASET_FIELDS = CONFIG_ARTIFACT_FIELDS | {
    "source_archive_sha256",
    "source_archive_bytes",
    "source_archive_format",
    "decompression_policy",
    "split",
    "min_length",
    "max_length",
    "max_records",
    "validation_ratio",
    "test_ratio",
}
ARTIFACT_PREFLIGHT_FIELDS = frozenset(
    {
        "schema",
        "version",
        "offline_preflight_only",
        "acceptance_evidence",
        "config_sha256",
        "artifacts",
        "dataset_source",
    }
)
DATASET_SOURCE_FIELDS = frozenset(
    {
        "source_archive_sha256",
        "source_archive_bytes",
        "source_archive_format",
        "decompression_policy",
        "decompressed_sha256",
        "decompressed_bytes",
    }
)
CONFIG_TRAINING_FIELDS = frozenset(
    {
        "freeze_proposal",
        "epochs",
        "min_epochs",
        "batch_size",
        "num_candidates",
        "learning_rate",
        "weight_decay",
        "grad_clip_norm",
        "validation_steps",
        "scheduler_factor",
        "scheduler_patience",
        "early_stop_patience",
        "remote_call_budget_per_seed",
    }
)
CONFIG_GENERATION_FIELDS = frozenset(
    {
        "sequence_count",
        "max_steps",
        "num_candidates",
        "proposal_temperature",
        "proposal_noise_scale",
        "energy_temperature",
        "disable_resample",
        "resample_ratio",
        "resample_top_p",
        "portability_training_seed",
        "portability_fixture_index",
        "portability_steps",
    }
)
CONFIG_EVALUATION_FIELDS = frozenset({"pair_mode", "pooling", "batch_size"})
CONFIG_PRECISION_FIELDS = frozenset({"name", "target_min", "target_max"})
CONFIG_PRIMARY_METRIC_FIELDS = frozenset({"name", "direction"})
CONFIG_THRESHOLD_FIELDS = frozenset(
    {
        "uniqueness_baseline_fraction_min",
        "repeat_ratio_absolute_increase_max",
        "invalid_sequence_count_max",
    }
)
CONFIG_SECTION_FIELDS = {
    "software": CONFIG_SOFTWARE_FIELDS,
    "dataset": CONFIG_DATASET_FIELDS,
    "checkpoint": CONFIG_ARTIFACT_FIELDS,
    "tokenizer": CONFIG_ARTIFACT_FIELDS,
    "evaluation_model": CONFIG_ARTIFACT_FIELDS,
    "training": CONFIG_TRAINING_FIELDS,
    "generation": CONFIG_GENERATION_FIELDS,
    "evaluation": CONFIG_EVALUATION_FIELDS,
    "precision_policy": CONFIG_PRECISION_FIELDS,
    "primary_metric": CONFIG_PRIMARY_METRIC_FIELDS,
    "thresholds": CONFIG_THRESHOLD_FIELDS,
}

_MAX_MANIFEST_BYTES = 4 * 1024 * 1024
_MAX_CONFIG_BYTES = 1024 * 1024
_MAX_ENVIRONMENT_LOCK_BYTES = 4 * 1024 * 1024
_MAX_EVIDENCE_MEMBER_BYTES = 64 * 1024 * 1024


def _read_private_json(
    path: Path, *, label: str, max_bytes: int
) -> tuple[Any, bytes, str]:
    """Read once through an anchored descriptor, then parse and hash those bytes."""

    encoded = read_private_bytes(path, label=label, max_bytes=max_bytes)
    return loads_json_strict(encoded), encoded, hashlib.sha256(encoded).hexdigest()


def _mapping(value: Any, label: str, errors: list[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        errors.append(f"{label}: expected an object")
        return {}
    return value


def _finite_number(value: Any, label: str, errors: list[str]) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        errors.append(f"{label}: expected a finite number")
        return None
    number = float(value)
    if not math.isfinite(number):
        errors.append(f"{label}: expected a finite number")
        return None
    return number


def _canonical_printable_identifier(value: Any) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and value == value.strip()
        and value.isprintable()
    )


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return timestamp if timestamp.tzinfo is not None else None


def _validate_receipt_submission_floor(
    record: dict[str, Any],
    *,
    earliest: datetime | None,
    label: str,
    prerequisite_label: str,
    errors: list[str],
) -> None:
    """Require every valid UTC receipt to follow one prerequisite boundary."""

    if earliest is None:
        return
    receipts = record.get("task_receipts")
    if not isinstance(receipts, list):
        return
    for index, receipt in enumerate(receipts):
        if not isinstance(receipt, dict):
            continue
        submitted_at = _parse_timestamp(receipt.get("submitted_at"))
        if submitted_at is None or submitted_at.utcoffset() != timedelta(0):
            continue
        if submitted_at < earliest:
            errors.append(
                f"{label}: remote receipt {index} predates the {prerequisite_label}"
            )


def _validate_execution_component_time(
    record: dict[str, Any],
    *,
    label: str,
    config: dict[str, Any],
    provider_smoke_time: datetime | None,
    errors: list[str],
) -> datetime | None:
    """Bind one executable component to its prerequisites and receipts."""

    recorded_at = _parse_timestamp(record.get("recorded_at"))
    if recorded_at is None or recorded_at.utcoffset() != timedelta(0):
        errors.append(f"{label}: recorded_at must be an aware UTC timestamp")
        return None
    prerequisites = (
        ("frozen config", _parse_timestamp(config.get("preregistered_at"))),
        (
            "SDK rights review",
            _parse_timestamp(
                _mapping(config.get("kaiwu_sdk"), "config.kaiwu_sdk", errors).get(
                    "rights_reviewed_at"
                )
            ),
        ),
        (
            "project assignment review",
            _parse_timestamp(
                _mapping(config.get("kaiwu_sdk"), "config.kaiwu_sdk", errors).get(
                    "project_assignment_reviewed_at"
                )
            ),
        ),
        ("provider smoke", provider_smoke_time),
    )
    for prerequisite_label, prerequisite_time in prerequisites:
        if prerequisite_time is not None and recorded_at < prerequisite_time:
            errors.append(f"{label}: predates the {prerequisite_label}")

    receipts = record.get("task_receipts")
    if isinstance(receipts, list):
        for index, receipt in enumerate(receipts):
            if not isinstance(receipt, dict):
                continue
            submitted_at = _parse_timestamp(receipt.get("submitted_at"))
            if submitted_at is None or submitted_at.utcoffset() != timedelta(0):
                continue
            if submitted_at > recorded_at:
                errors.append(
                    f"{label}: remote receipt {index} submission follows its record"
                )
    for prerequisite_label, prerequisite_time in prerequisites:
        _validate_receipt_submission_floor(
            record,
            earliest=prerequisite_time,
            label=label,
            prerequisite_label=prerequisite_label,
            errors=errors,
        )
    return recorded_at


def _validate_provider_resource_gate_binding(
    record: dict[str, Any],
    *,
    label: str,
    provider_resources: dict[str, dict[str, Any]],
    recorded_at: datetime | None,
    errors: list[str],
) -> None:
    """Recompute one retained pre-submission resource-budget decision."""

    try:
        gate = validate_provider_resource_gate(record.get("provider_resource_gate"))
    except ValueError as exc:
        errors.append(f"{label}: invalid provider resource gate: {exc}")
        return
    snapshot_sha256 = gate["snapshot_sha256"]
    snapshot = provider_resources.get(snapshot_sha256)
    if snapshot is None:
        errors.append(f"{label}: provider resource snapshot is unavailable")
        return
    schema = record.get("schema")
    required_calls = (
        record.get("protein_remote_call_budget_per_seed")
        if schema == TRAINING_COMPONENT_SCHEMA
        else record.get("remote_call_budget")
    )
    if gate["mode"] != "sampling" or gate["required_calls"] != required_calls:
        errors.append(f"{label}: provider resource gate differs from the run budget")
        return
    checked_at = _parse_timestamp(gate["checked_at"])
    if checked_at is None or checked_at.utcoffset() != timedelta(0):
        errors.append(f"{label}: provider resource check time is invalid")
        return
    if recorded_at is not None and checked_at > recorded_at:
        errors.append(f"{label}: provider resource check follows its record")
    try:
        ready, reason = assess_provider_budget(
            snapshot,
            mode="sampling",
            required_calls=gate["required_calls"],
            now=checked_at,
        )
    except ValueError as exc:
        errors.append(f"{label}: invalid provider resource snapshot: {exc}")
        return
    if not ready:
        errors.append(f"{label}: provider resource gate failed: {reason}")
    snapshot_captured_at = _parse_timestamp(snapshot.get("captured_at"))
    snapshot_valid_until = _parse_timestamp(snapshot.get("valid_until"))
    receipts = record.get("task_receipts")
    if isinstance(receipts, list):
        for index, receipt in enumerate(receipts):
            if not isinstance(receipt, dict):
                continue
            submitted_at = _parse_timestamp(receipt.get("submitted_at"))
            if (
                submitted_at is not None
                and submitted_at.utcoffset() == timedelta(0)
                and submitted_at < checked_at
            ):
                errors.append(
                    f"{label}: remote receipt {index} predates its resource check"
                )
            if (
                submitted_at is not None
                and submitted_at.utcoffset() == timedelta(0)
                and snapshot_captured_at is not None
                and submitted_at < snapshot_captured_at
            ):
                errors.append(
                    f"{label}: remote receipt {index} predates resource snapshot capture"
                )
            if (
                submitted_at is not None
                and submitted_at.utcoffset() == timedelta(0)
                and snapshot_valid_until is not None
                and submitted_at > snapshot_valid_until
            ):
                errors.append(
                    f"{label}: remote receipt {index} follows resource snapshot expiry"
                )


def _validate_component_field_set(
    record: dict[str, Any], *, label: str, errors: list[str]
) -> None:
    """Reject missing or extended fields for executable evidence components."""

    expected_fields = COMPONENT_FIELDS_BY_SCHEMA.get(record.get("schema"))
    if expected_fields is not None and set(record) != expected_fields:
        errors.append(f"{label}: field set differs from its closed schema")


def _has_exact_fields(
    value: Any, *, expected: frozenset[str], label: str, errors: list[str]
) -> bool:
    if not isinstance(value, dict) or set(value) != expected:
        errors.append(f"{label}: field set differs from its closed schema")
        return False
    return True


def _validate_final_record_field_sets(
    record: dict[str, Any], *, label: str, errors: list[str]
) -> None:
    """Reject extensions at every object boundary in a final host record."""

    role = record.get("run_role")
    expected = (
        FINAL_PRIMARY_RECORD_FIELDS
        if role == "primary"
        else FINAL_REPLAY_RECORD_FIELDS
        if role == "portability_replay"
        else None
    )
    if expected is not None:
        _has_exact_fields(record, expected=expected, label=label, errors=errors)
    for field, fields in (
        ("artifacts", FINAL_ARTIFACT_FIELDS),
        ("precision_policy", PRECISION_POLICY_FIELDS),
        ("training", FINAL_TRAINING_FIELDS),
        ("generation", FINAL_GENERATION_FIELDS),
        ("transfer_accounting", TRANSFER_ACCOUNTING_FIELDS),
        ("acceptance", FINAL_ACCEPTANCE_FIELDS),
    ):
        value = record.get(field)
        if isinstance(value, dict):
            _has_exact_fields(
                value,
                expected=fields,
                label=f"{label}.{field}",
                errors=errors,
            )
    transfers = record.get("transfer_accounting")
    if isinstance(transfers, dict):
        boundaries = transfers.get("sampler_boundaries")
        if isinstance(boundaries, list):
            for index, boundary in enumerate(boundaries):
                if isinstance(boundary, dict):
                    _has_exact_fields(
                        boundary,
                        expected=SAMPLER_BOUNDARY_FIELDS,
                        label=(
                            f"{label}.transfer_accounting.sampler_boundaries[{index}]"
                        ),
                        errors=errors,
                    )
    for field in ("baseline_metrics", "guided_metrics"):
        metrics = record.get(field)
        if isinstance(metrics, dict):
            _has_exact_fields(
                metrics,
                expected=frozenset(METRIC_NAMES),
                label=f"{label}.{field}",
                errors=errors,
            )
    application = record.get("application_evidence")
    if isinstance(application, dict):
        _has_exact_fields(
            application,
            expected=APPLICATION_EVIDENCE_FIELDS,
            label=f"{label}.application_evidence",
            errors=errors,
        )
        evidence_records = application.get("records")
        if isinstance(evidence_records, list):
            for index, evidence_record in enumerate(evidence_records):
                if isinstance(evidence_record, dict):
                    _has_exact_fields(
                        evidence_record,
                        expected=APPLICATION_EVIDENCE_RECORD_FIELDS,
                        label=f"{label}.application_evidence.records[{index}]",
                        errors=errors,
                    )
    portability = record.get("portability_evidence")
    if isinstance(portability, dict):
        _has_exact_fields(
            portability,
            expected=PORTABILITY_EVIDENCE_FIELDS,
            label=f"{label}.portability_evidence",
            errors=errors,
        )


def _validate_provider_value_description(
    value: Any,
    *,
    label: str,
    allow_field_schemas: bool,
    errors: list[str],
) -> None:
    """Validate value-free type metadata emitted by the pinned SDK client."""

    if not isinstance(value, dict):
        errors.append(f"{label}: provider value description is not an object")
        return
    value_type = value.get("type")
    if (
        not isinstance(value_type, str)
        or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]{0,127}", value_type) is None
    ):
        errors.append(f"{label}: provider value type is unsafe")

    fields = set(value)
    scalar_fields = {"type"}
    length_fields = {"type", "length"}
    sequence_fields = {
        "type",
        "length",
        "element_types",
        "element_type_scan_limit",
    }
    array_fields = {"type", "shape", "shape_safe", "dtype", "dtype_safe"}
    mapping_prefix = {"type", "length", "string_keys", "fields_safe"}
    if fields == scalar_fields:
        return
    if fields == length_fields:
        if type(value.get("length")) is not int or value["length"] < 0:
            errors.append(f"{label}: provider value length is invalid")
        return
    if fields == sequence_fields:
        if type(value.get("length")) is not int or value["length"] < 0:
            errors.append(f"{label}: provider sequence length is invalid")
        element_types = value.get("element_types")
        if (
            not isinstance(element_types, list)
            or element_types != sorted(set(element_types))
            or any(
                not isinstance(item, str)
                or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]{0,127}", item) is None
                for item in element_types
            )
        ):
            errors.append(f"{label}: provider sequence element types are invalid")
        if value.get("element_type_scan_limit") != 64:
            errors.append(f"{label}: provider sequence scan limit differs")
        return
    if fields == array_fields:
        shape = value.get("shape")
        if (
            not isinstance(shape, list)
            or len(shape) > 16
            or any(type(size) is not int or size < 0 for size in shape)
        ):
            errors.append(f"{label}: provider array shape is invalid")
        if type(value.get("shape_safe")) is not bool:
            errors.append(f"{label}: provider array shape safety is invalid")
        dtype = value.get("dtype")
        if (
            not isinstance(dtype, str)
            or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]{0,127}", dtype) is None
        ):
            errors.append(f"{label}: provider array dtype is unsafe")
        if type(value.get("dtype_safe")) is not bool:
            errors.append(f"{label}: provider array dtype safety is invalid")
        return
    if mapping_prefix.issubset(fields):
        length = value.get("length")
        if type(length) is not int or length < 0:
            errors.append(f"{label}: provider mapping length is invalid")
        string_keys = value.get("string_keys")
        fields_safe = value.get("fields_safe")
        if string_keys is None:
            expected_fields = mapping_prefix | {"field_limit"}
            valid_flags = fields_safe is False
        elif string_keys is False:
            expected_fields = mapping_prefix
            valid_flags = fields_safe is False
        elif string_keys is True and fields_safe is False:
            expected_fields = mapping_prefix | {"field_limit"}
            valid_flags = True
        elif string_keys is True and fields_safe is True:
            expected_fields = mapping_prefix | {"fields"}
            if allow_field_schemas:
                expected_fields.add("field_schemas")
            valid_flags = True
        else:
            expected_fields = set()
            valid_flags = False
        if fields != expected_fields or not valid_flags:
            errors.append(f"{label}: provider mapping description is inconsistent")
            return
        if "field_limit" in value and value.get("field_limit") != 64:
            errors.append(f"{label}: provider mapping field limit differs")
        if fields_safe is True:
            names = value.get("fields")
            if (
                not isinstance(names, list)
                or names != sorted(set(names))
                or len(names) != length
                or any(
                    not isinstance(name, str)
                    or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]{0,127}", name)
                    is None
                    for name in names
                )
            ):
                errors.append(f"{label}: provider mapping field names are invalid")
                return
            if allow_field_schemas:
                descriptions = value.get("field_schemas")
                if not isinstance(descriptions, dict) or set(descriptions) != set(
                    names
                ):
                    errors.append(
                        f"{label}: provider mapping field descriptions differ"
                    )
                    return
                for name in names:
                    _validate_provider_value_description(
                        descriptions[name],
                        label=f"{label}.{name}",
                        allow_field_schemas=False,
                        errors=errors,
                    )
        return
    errors.append(f"{label}: provider value description has an unknown field set")


def _validate_provider_result_schema(
    value: Any, *, label: str, errors: list[str]
) -> None:
    if not isinstance(value, dict):
        errors.append(f"{label}: provider result schema is not an object")
        return
    if set(value) == {"available", "reason"}:
        reason = value.get("reason")
        if (
            value.get("available") is not False
            or not isinstance(reason, str)
            or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]{0,127}", reason) is None
        ):
            errors.append(f"{label}: unavailable provider result schema is invalid")
        return
    if set(value) != {"available", "result"} or value.get("available") is not True:
        errors.append(f"{label}: provider result schema field set is invalid")
        return
    _validate_provider_value_description(
        value.get("result"),
        label=f"{label}.result",
        allow_field_schemas=True,
        errors=errors,
    )


def _validate_executable_component_nested_fields(
    record: dict[str, Any], *, label: str, errors: list[str]
) -> None:
    """Close nested evidence structures after the top-level schema check."""

    schema = record.get("schema")
    if schema in {
        SYSTEM_COMPONENT_SCHEMA,
        TRAINING_COMPONENT_SCHEMA,
        PORTABILITY_COMPONENT_SCHEMA,
    }:
        _has_exact_fields(
            record.get("precision_policy"),
            expected=PRECISION_POLICY_FIELDS,
            label=f"{label}.precision_policy",
            errors=errors,
        )
        if record.get("failure") is not None:
            errors.append(f"{label}: completed component retains a failure")

    if schema == SYSTEM_COMPONENT_SCHEMA:
        _validate_provider_result_schema(
            record.get("provider_result_schema"),
            label=f"{label}.provider_result_schema",
            errors=errors,
        )
        _has_exact_fields(
            record.get("training"),
            expected=SYSTEM_TRAINING_FIELDS,
            label=f"{label}.training",
            errors=errors,
        )
        generation = record.get("generation")
        if _has_exact_fields(
            generation,
            expected=SYSTEM_GENERATION_FIELDS,
            label=f"{label}.generation",
            errors=errors,
        ):
            tokens = generation["generated_tokens"]
            if (
                not isinstance(tokens, list)
                or not tokens
                or any(
                    not isinstance(row, list)
                    or not row
                    or any(type(token) is not int or token < 0 for token in row)
                    for row in tokens
                )
            ):
                errors.append(f"{label}.generation: generated tokens are invalid")
        transfers = record.get("transfer_accounting")
        if _has_exact_fields(
            transfers,
            expected=TRANSFER_ACCOUNTING_FIELDS,
            label=f"{label}.transfer_accounting",
            errors=errors,
        ):
            boundaries = transfers["sampler_boundaries"]
            if isinstance(boundaries, list):
                for index, boundary in enumerate(boundaries):
                    _has_exact_fields(
                        boundary,
                        expected=SAMPLER_BOUNDARY_FIELDS,
                        label=f"{label}.transfer_accounting.sampler_boundaries[{index}]",
                        errors=errors,
                    )
        if record.get("limitations") != list(SYSTEM_LIMITATIONS):
            errors.append(f"{label}: system claim limitations differ")

    elif schema == TRAINING_COMPONENT_SCHEMA:
        artifacts = record.get("workflow_artifacts")
        if _has_exact_fields(
            artifacts,
            expected=WORKFLOW_ARTIFACT_FIELDS,
            label=f"{label}.workflow_artifacts",
            errors=errors,
        ):
            for name, identity in artifacts.items():
                _has_exact_fields(
                    identity,
                    expected=WORKFLOW_ARTIFACT_IDENTITY_FIELDS,
                    label=f"{label}.workflow_artifacts.{name}",
                    errors=errors,
                )
        if record.get("limitations") != list(TRAINING_LIMITATIONS):
            errors.append(f"{label}: training claim limitations differ")

    elif schema == EVALUATION_COMPONENT_SCHEMA:
        for field in ("baseline_metrics", "guided_metrics"):
            _has_exact_fields(
                record.get(field),
                expected=frozenset(METRIC_NAMES),
                label=f"{label}.{field}",
                errors=errors,
            )

    elif schema == PORTABILITY_COMPONENT_SCHEMA:
        _has_exact_fields(
            record.get("artifacts"),
            expected=PORTABILITY_ARTIFACT_FIELDS,
            label=f"{label}.artifacts",
            errors=errors,
        )
        _has_exact_fields(
            record.get("fixture"),
            expected=PORTABILITY_FIXTURE_FIELDS,
            label=f"{label}.fixture",
            errors=errors,
        )
        if record.get("limitations") != list(PORTABILITY_LIMITATIONS):
            errors.append(f"{label}: portability claim limitations differ")


def _validate_sampling_receipt(
    receipt: dict[str, Any],
    *,
    label: str,
    expected_requested_samples: Any,
    errors: list[str],
) -> tuple[str | None, str | None, str | None]:
    """Validate one complete serialized Remote receipt for final evidence."""

    if set(receipt) != TASK_RECEIPT_FIELDS:
        errors.append(f"{label}: receipt field set is incomplete")
    if receipt.get("schema") != TASK_RECEIPT_SCHEMA:
        errors.append(f"{label}: receipt schema is unsupported")
    task_name = receipt.get("task_name")
    if not _canonical_printable_identifier(task_name):
        errors.append(f"{label}: receipt has no task name")
    matrix_digest = receipt.get("matrix_sha256")
    if (
        not isinstance(matrix_digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", matrix_digest) is None
    ):
        errors.append(f"{label}: receipt has no matrix digest")
        matrix_digest = None
    matrix_size = receipt.get("matrix_size")
    if type(matrix_size) is not int or matrix_size <= 0:
        errors.append(f"{label}: receipt has an invalid matrix size")
    if receipt.get("mode") != "sampling":
        errors.append(f"{label}: receipt is not sampling")
    if receipt.get("requested_samples") != expected_requested_samples:
        errors.append(f"{label}: receipt sample count differs")
    project_no = receipt.get("project_no")
    if not _canonical_printable_identifier(project_no):
        errors.append(f"{label}: receipt has no project number")
    submitted_at = receipt.get("submitted_at")
    try:
        submitted = datetime.fromisoformat(submitted_at)
    except (TypeError, ValueError):
        submitted = None
    if submitted is None or submitted.utcoffset() != timedelta(0):
        errors.append(f"{label}: receipt has no aware UTC submission time")
    task_id = receipt.get("provider_task_id")
    if not _canonical_printable_identifier(task_id):
        errors.append(f"{label}: receipt has no provider_task_id")
        task_id = None
    target = receipt.get("provider_target")
    if not _canonical_printable_identifier(target):
        errors.append(f"{label}: receipt has no provider_target")
        target = None
    return task_id, target, matrix_digest


def _validate_precision_evidence(
    record: dict[str, Any],
    *,
    label: str,
    receipt_matrix_digests: list[str],
    errors: list[str],
) -> None:
    """Bind every original-matrix reduction report to submitted receipts."""

    precision = record.get("precision_policy")
    entries = record.get("precision_evidence")
    if not isinstance(precision, dict):
        errors.append(f"{label}: precision policy is missing")
        return
    if not isinstance(entries, list) or not entries:
        errors.append(f"{label}: per-matrix precision evidence is missing")
        return
    if precision.get("matrix_count") != len(entries):
        errors.append(f"{label}: precision matrix count differs from evidence")
    target_min = precision.get("target_min")
    target_max = precision.get("target_max")
    target_range_valid = (
        type(target_min) is int
        and type(target_max) is int
        and target_min < 0 < target_max
    )
    if not target_range_valid:
        errors.append(f"{label}: precision target range is invalid")

    original_digests: list[str] = []
    submission_digests: list[str] = []
    scales: list[float] = []
    maximum_errors: list[float] = []
    mean_errors: list[float] = []
    for index, entry in enumerate(entries):
        entry_label = f"{label}: precision evidence {index}"
        if not isinstance(entry, dict):
            errors.append(f"{entry_label} is not an object")
            continue
        if set(entry) != PRECISION_EVIDENCE_FIELDS:
            errors.append(f"{entry_label} field set differs from schema")
        original = entry.get("original_matrix_sha256")
        submission = entry.get("submission_matrix_sha256")
        if (
            not isinstance(original, str)
            or re.fullmatch(r"[0-9a-f]{64}", original) is None
        ):
            errors.append(f"{entry_label} has no original matrix digest")
        else:
            original_digests.append(original)
        if (
            not isinstance(submission, str)
            or re.fullmatch(r"[0-9a-f]{64}", submission) is None
        ):
            errors.append(f"{entry_label} has no submitted matrix digest")
        else:
            submission_digests.append(submission)
        if entry.get("source_type") != "numpy.ndarray":
            errors.append(f"{entry_label} source type is not numpy.ndarray")
        if (
            not isinstance(entry.get("source_dtype"), str)
            or not entry["source_dtype"].strip()
        ):
            errors.append(f"{entry_label} source dtype is missing")
        if entry.get("target_min") != precision.get("target_min") or entry.get(
            "target_max"
        ) != precision.get("target_max"):
            errors.append(f"{entry_label} target range differs from policy")
        if entry.get("normalized_dtype") != "torch.float64":
            errors.append(f"{entry_label} normalized dtype is not torch.float64")
        if entry.get("symmetry_normalization") != "arithmetic_mean":
            errors.append(f"{entry_label} symmetry normalization is unsupported")
        if entry.get("rounding_policy") != "round_half_to_even":
            errors.append(f"{entry_label} rounding policy is unsupported")
        normalized_min = _finite_number(
            entry.get("normalized_min"), f"{entry_label}.normalized_min", errors
        )
        normalized_max = _finite_number(
            entry.get("normalized_max"), f"{entry_label}.normalized_max", errors
        )
        if (
            normalized_min is not None
            and normalized_max is not None
            and normalized_min > normalized_max
        ):
            errors.append(f"{entry_label} normalized coefficient range is invalid")
        scale = _finite_number(
            entry.get("scale_factor"), f"{entry_label}.scale_factor", errors
        )
        maximum = _finite_number(
            entry.get("max_abs_error"), f"{entry_label}.max_abs_error", errors
        )
        mean = _finite_number(
            entry.get("mean_abs_error"), f"{entry_label}.mean_abs_error", errors
        )
        if scale is not None:
            scales.append(scale)
            if scale <= 0:
                errors.append(f"{entry_label} scale factor is not positive")
            if (
                target_range_valid
                and normalized_min is not None
                and normalized_max is not None
            ):
                maximum_magnitude = max(abs(normalized_min), abs(normalized_max))
                expected_scale = (
                    1.0
                    if maximum_magnitude == 0.0
                    else min(abs(target_min), target_max) / maximum_magnitude
                )
                if not math.isclose(
                    scale, expected_scale, rel_tol=1e-12, abs_tol=1e-12
                ):
                    errors.append(
                        f"{entry_label} scale factor differs from coefficient range"
                    )
        if maximum is not None:
            maximum_errors.append(maximum)
            if maximum < 0:
                errors.append(f"{entry_label} maximum error is negative")
        if mean is not None:
            mean_errors.append(mean)
            if mean < 0:
                errors.append(f"{entry_label} mean error is negative")
        if maximum is not None and mean is not None and mean > maximum:
            errors.append(f"{entry_label} mean error exceeds maximum")

    if len(original_digests) != len(set(original_digests)):
        errors.append(f"{label}: original precision matrix identities are not unique")
    if set(submission_digests) != set(receipt_matrix_digests):
        errors.append(f"{label}: precision submissions differ from task receipts")
    aggregates = (
        ("scale_factor_min", min(scales) if scales else None),
        ("scale_factor_max", max(scales) if scales else None),
        ("max_abs_error", max(maximum_errors) if maximum_errors else None),
        (
            "mean_of_matrix_mean_abs_error",
            fmean(mean_errors) if mean_errors else None,
        ),
    )
    for field, recomputed in aggregates:
        observed = precision.get(field)
        if (
            recomputed is None
            or isinstance(observed, bool)
            or not isinstance(observed, (int, float))
            or not math.isclose(
                float(observed), recomputed, rel_tol=1e-12, abs_tol=1e-12
            )
        ):
            errors.append(f"{label}: precision aggregate {field} differs from evidence")


def _validate_precision_transfer_origins(
    record: dict[str, Any],
    *,
    label: str,
    boundaries: list[dict[str, Any]],
    errors: list[str],
) -> None:
    """Require precision reports and sampler transfers to name one input set."""

    precision_entries = record.get("precision_evidence")
    if not isinstance(precision_entries, list) or not all(
        isinstance(entry, dict) for entry in precision_entries
    ):
        return
    precision_origins = {
        (
            entry.get("original_matrix_sha256"),
            entry.get("submission_matrix_sha256"),
            entry.get("source_type"),
            entry.get("source_dtype"),
        )
        for entry in precision_entries
    }
    transfer_origins = {
        (
            boundary.get("original_matrix_sha256"),
            boundary.get("submission_matrix_sha256"),
            boundary.get("input_type"),
            boundary.get("input_dtype"),
        )
        for boundary in boundaries
    }
    if precision_origins != transfer_origins:
        errors.append(f"{label}: precision origins differ from sampler transfers")


def _validate_config(config: dict[str, Any], errors: list[str]) -> None:
    _has_exact_fields(
        config, expected=CONFIG_FIELDS, label="config", errors=errors
    )
    for section, expected_fields in CONFIG_SECTION_FIELDS.items():
        _has_exact_fields(
            config.get(section),
            expected=expected_fields,
            label=f"config.{section}",
            errors=errors,
        )
    if config.get("schema") != CONFIG_SCHEMA or config.get("version") != "1.0":
        errors.append("config: unsupported schema or version")
    primary_host = config.get("primary_host")
    replay_host = config.get("replay_host")
    if primary_host not in HOSTS:
        errors.append("config.primary_host: expected one declared A800 host")
    if replay_host not in HOSTS or replay_host == primary_host:
        errors.append("config.replay_host: expected the other declared A800 host")
    host_identities = _mapping(
        config.get("host_identities"), "config.host_identities", errors
    )
    if set(host_identities) != HOSTS:
        errors.append("config.host_identities: expected both validation hosts")
    elif any(
        not _canonical_printable_identifier(hostname) or hostname == "<required>"
        for hostname in host_identities.values()
    ):
        errors.append("config.host_identities: expected frozen machine hostnames")
    elif len(set(host_identities.values())) != len(HOSTS):
        errors.append("config.host_identities: machine hostnames must be distinct")
    preregistered_at = config.get("preregistered_at")
    try:
        timestamp = datetime.fromisoformat(str(preregistered_at).replace("Z", "+00:00"))
        if timestamp.tzinfo is None:
            raise ValueError
    except ValueError:
        errors.append("config.preregistered_at: expected a timezone-aware timestamp")
    seeds = config.get("seeds")
    if (
        not isinstance(seeds, list)
        or len(seeds) < 3
        or any(type(seed) is not int for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        errors.append("config.seeds: expected at least three unique integer seeds")
    for name in (
        "dataset",
        "checkpoint",
        "tokenizer",
        "evaluation_model",
        "training",
        "generation",
        "evaluation",
    ):
        section = _mapping(config.get(name), f"config.{name}", errors)
        if not section or any(
            value is None or value == "" or value == "<required>"
            for value in section.values()
        ):
            errors.append(f"config.{name}: all frozen identity fields are required")
    for name in ("dataset", "checkpoint", "tokenizer", "evaluation_model"):
        artifact = _mapping(config.get(name), f"config.{name}", errors)
        digest = artifact.get("sha256")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            errors.append(f"config.{name}.sha256: expected a SHA-256 digest")
        for field in ("source_url", "license_evidence_url"):
            value = artifact.get(field)
            if not isinstance(value, str):
                errors.append(f"config.{name}.{field}: expected an HTTPS URL")
                continue
            try:
                parsed = urlsplit(value)
                valid_url = (
                    parsed.scheme == "https"
                    and bool(parsed.hostname)
                    and parsed.username is None
                    and parsed.password is None
                    and not parsed.fragment
                )
            except ValueError:
                valid_url = False
            if not valid_url:
                errors.append(f"config.{name}.{field}: expected an HTTPS URL")
        license_id = artifact.get("license_id")
        if (
            not isinstance(license_id, str)
            or LICENSE_ID.fullmatch(license_id) is None
            or license_id.upper() in UNAPPROVED_LICENSE_IDS
        ):
            errors.append(
                f"config.{name}.license_id: expected an approved license identifier"
            )
        reviewed_at = artifact.get("license_reviewed_at")
        try:
            reviewed_timestamp = datetime.fromisoformat(
                str(reviewed_at).replace("Z", "+00:00")
            )
            if reviewed_timestamp.tzinfo is None:
                raise ValueError
        except ValueError:
            errors.append(
                f"config.{name}.license_reviewed_at: expected a timezone-aware timestamp"
            )
    dataset = _mapping(config.get("dataset"), "config.dataset", errors)
    source_digest = dataset.get("source_archive_sha256")
    if (
        not isinstance(source_digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", source_digest) is None
    ):
        errors.append(
            "config.dataset.source_archive_sha256: expected a SHA-256 digest"
        )
    if (
        type(dataset.get("source_archive_bytes")) is not int
        or dataset.get("source_archive_bytes", 0) <= 0
    ):
        errors.append("config.dataset.source_archive_bytes: expected a positive integer")
    if dataset.get("source_archive_format") != "gzip":
        errors.append("config.dataset.source_archive_format: expected gzip")
    if dataset.get("decompression_policy") != "gzip-exact-bytes-v1":
        errors.append(
            "config.dataset.decompression_policy: expected gzip-exact-bytes-v1"
        )
    generation = _mapping(config.get("generation"), "config.generation", errors)
    for field in (
        "sequence_count",
        "max_steps",
        "num_candidates",
        "portability_steps",
    ):
        if type(generation.get(field)) is not int or generation.get(field, 0) <= 0:
            errors.append(f"config.generation.{field}: expected a positive integer")
    portability_seed = generation.get("portability_training_seed")
    if type(portability_seed) is not int or portability_seed not in config.get(
        "seeds", []
    ):
        errors.append(
            "config.generation.portability_training_seed: expected one frozen seed"
        )
    portability_index = generation.get("portability_fixture_index")
    if (
        type(portability_index) is not int
        or portability_index < 0
        or (
            type(generation.get("sequence_count")) is int
            and portability_index >= generation["sequence_count"]
        )
    ):
        errors.append(
            "config.generation.portability_fixture_index: outside the frozen sequence set"
        )
    for field in (
        "proposal_temperature",
        "proposal_noise_scale",
        "energy_temperature",
        "resample_ratio",
        "resample_top_p",
    ):
        value = _finite_number(
            generation.get(field), f"config.generation.{field}", errors
        )
        if value is not None and value < 0:
            errors.append(f"config.generation.{field}: expected a non-negative value")
    if type(generation.get("disable_resample")) is not bool:
        errors.append("config.generation.disable_resample: expected a boolean")
    for field in ("resample_ratio", "resample_top_p"):
        value = generation.get(field)
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and value > 1
        ):
            errors.append(f"config.generation.{field}: expected a value at most one")
    evaluation = _mapping(config.get("evaluation"), "config.evaluation", errors)
    if evaluation.get("pair_mode") != "order":
        errors.append("config.evaluation.pair_mode: expected order")
    if evaluation.get("pooling") != "mean":
        errors.append("config.evaluation.pooling: expected mean")
    if (
        type(evaluation.get("batch_size")) is not int
        or evaluation.get("batch_size", 0) <= 0
    ):
        errors.append("config.evaluation.batch_size: expected a positive integer")
    dataset = _mapping(config.get("dataset"), "config.dataset", errors)
    if dataset.get("split") != "deterministic-shuffle-v1":
        errors.append("config.dataset.split: expected deterministic-shuffle-v1")
    for field in ("min_length", "max_length", "max_records"):
        if type(dataset.get(field)) is not int or dataset.get(field, 0) <= 0:
            errors.append(f"config.dataset.{field}: expected a positive integer")
    if (
        type(dataset.get("min_length")) is int
        and type(dataset.get("max_length")) is int
        and dataset["min_length"] > dataset["max_length"]
    ):
        errors.append("config.dataset.min_length: cannot exceed max_length")
    split_ratios: list[float] = []
    for field in ("validation_ratio", "test_ratio"):
        value = _finite_number(dataset.get(field), f"config.dataset.{field}", errors)
        if value is not None:
            split_ratios.append(value)
            if not 0 < value < 1:
                errors.append(
                    f"config.dataset.{field}: expected a value between zero and one"
                )
    if len(split_ratios) == 2 and sum(split_ratios) >= 1:
        errors.append("config.dataset: validation and test ratios must sum below one")
    if (
        type(dataset.get("max_records")) is int
        and isinstance(dataset.get("test_ratio"), (int, float))
        and not isinstance(dataset.get("test_ratio"), bool)
        and type(generation.get("sequence_count")) is int
    ):
        expected_count = max(1, int(dataset["max_records"] * dataset["test_ratio"]))
        if generation["sequence_count"] != expected_count:
            errors.append(
                "config.generation.sequence_count: differs from the frozen test split"
            )
    training = _mapping(config.get("training"), "config.training", errors)
    if type(training.get("freeze_proposal")) is not bool:
        errors.append("config.training.freeze_proposal: expected a boolean")
    for field in (
        "epochs",
        "min_epochs",
        "batch_size",
        "num_candidates",
        "validation_steps",
        "scheduler_patience",
        "early_stop_patience",
        "remote_call_budget_per_seed",
    ):
        if type(training.get(field)) is not int or training.get(field, 0) <= 0:
            errors.append(f"config.training.{field}: expected a positive integer")
    for field in (
        "learning_rate",
        "weight_decay",
        "grad_clip_norm",
        "scheduler_factor",
    ):
        value = _finite_number(training.get(field), f"config.training.{field}", errors)
        if value is not None and value <= 0:
            errors.append(f"config.training.{field}: expected a positive value")
    if (
        type(training.get("epochs")) is int
        and type(training.get("min_epochs")) is int
        and training["min_epochs"] > training["epochs"]
    ):
        errors.append("config.training.min_epochs: cannot exceed epochs")
    estimated_calls = _estimate_protein_remote_calls(config)
    protein_budget = training.get("remote_call_budget_per_seed")
    if (
        estimated_calls is not None
        and type(protein_budget) is int
        and protein_budget < estimated_calls
    ):
        errors.append(
            "config.training.remote_call_budget_per_seed: below the "
            f"worst-case workflow estimate of {estimated_calls}"
        )
    software = _mapping(config.get("software"), "config.software", errors)
    for field in (
        "source_revision",
        "flagquantum_version",
        "kaiwu_pytorch_plugin_revision",
        "python_version",
        "torch_version",
        "kaiwu_sdk_version",
        "environment_lock_sha256",
    ):
        value = software.get(field)
        if not isinstance(value, str) or value in ("", "<required>"):
            errors.append(f"config.software.{field}: frozen value is required")
    for field in ("source_revision", "kaiwu_pytorch_plugin_revision"):
        revision = software.get(field)
        if not isinstance(revision, str) or FULL_REVISION.fullmatch(revision) is None:
            errors.append(f"config.software.{field}: expected a full Git revision")
    environment_digest = software.get("environment_lock_sha256")
    if (
        not isinstance(environment_digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", environment_digest) is None
    ):
        errors.append(
            "config.software.environment_lock_sha256: expected a SHA-256 digest"
        )
    kaiwu_sdk = config.get("kaiwu_sdk")
    errors.extend(validate_sdk_approval_record(kaiwu_sdk, label="config.kaiwu_sdk"))
    if isinstance(kaiwu_sdk, dict) and kaiwu_sdk.get(
        "sdk_version"
    ) != software.get("kaiwu_sdk_version"):
        errors.append(
            "config.kaiwu_sdk.sdk_version: differs from the frozen software lane"
        )
    precision = _mapping(
        config.get("precision_policy"), "config.precision_policy", errors
    )
    for field in ("name", "target_min", "target_max"):
        if field not in precision:
            errors.append(f"config.precision_policy: missing {field}")
    if config.get("primary_metric") != {
        "name": "mean_cosine_distance",
        "direction": "lower",
    }:
        errors.append(
            "config.primary_metric: expected preregistered lower mean_cosine_distance"
        )
    if (
        type(config.get("remote_call_budget")) is not int
        or config.get("remote_call_budget", 0) <= 0
    ):
        errors.append("config.remote_call_budget: expected a positive integer")
    if (
        type(config.get("requested_samples")) is not int
        or not 10 <= config.get("requested_samples", 0) <= 2000
    ):
        errors.append("config.requested_samples: expected an integer from 10 to 2000")
    portability_calls = _estimate_portability_remote_calls(config)
    if (
        portability_calls is not None
        and type(config.get("remote_call_budget")) is int
        and config["remote_call_budget"] < portability_calls
    ):
        errors.append(
            "config.remote_call_budget: below the portability replay estimate of "
            f"{portability_calls}"
        )
    thresholds = _mapping(config.get("thresholds"), "config.thresholds", errors)
    if thresholds.get("uniqueness_baseline_fraction_min") != 0.95:
        errors.append("config.thresholds: uniqueness floor must remain 0.95")
    if thresholds.get("repeat_ratio_absolute_increase_max") != 0.05:
        errors.append("config.thresholds: repeat-ratio allowance must remain 0.05")
    if thresholds.get("invalid_sequence_count_max") != 0:
        errors.append("config.thresholds: invalid sequence count must remain zero")


def _validate_system_record(
    record: dict[str, Any],
    *,
    config: dict[str, Any],
    config_sha256: str,
    label: str,
    errors: list[str],
) -> None:
    _validate_final_record_field_sets(record, label=label, errors=errors)
    if record.get("schema") != RECORD_SCHEMA or record.get("version") != "1.0":
        errors.append(f"{label}: unsupported schema or version")
    for field in ("source_revision", "kaiwu_pytorch_plugin_revision"):
        value = record.get(field)
        if not isinstance(value, str) or FULL_REVISION.fullmatch(value) is None:
            errors.append(f"{label}.{field}: expected a full Git revision")
    for field in (
        "source_preflight_sha256",
        "transfer_manifest_sha256",
        "environment_lock_sha256",
        "system_evidence_sha256",
    ):
        value = record.get(field)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            errors.append(f"{label}.{field}: expected a SHA-256 digest")
    software = _mapping(config.get("software"), "config.software", errors)
    for field in (
        "source_revision",
        "flagquantum_version",
        "kaiwu_pytorch_plugin_revision",
        "python_version",
        "torch_version",
        "kaiwu_sdk_version",
        "environment_lock_sha256",
    ):
        if record.get(field) != software.get(field):
            errors.append(f"{label}.{field}: differs from the frozen software lane")
    if record.get("experiment_config_sha256") != config_sha256:
        errors.append(f"{label}: frozen experiment config identity mismatch")
    if record.get("execution_host") not in HOSTS:
        errors.append(f"{label}.execution_host: unexpected host")
    if record.get("run_role") not in {"primary", "portability_replay"}:
        errors.append(f"{label}.run_role: unexpected role")
    if record.get("requested_cuda_device") != "cuda:0":
        errors.append(f"{label}: acceptance requires explicit cuda:0")
    if record.get("observed_tensor_device") != "cuda:0":
        errors.append(f"{label}: tensor work was not observed on cuda:0")
    if "A800" not in str(record.get("observed_gpu_model", "")):
        errors.append(f"{label}: observed GPU is not an NVIDIA A800")
    if record.get("transport") != "kaiwu_cim":
        errors.append(f"{label}: transport is not kaiwu_cim")
    if record.get("qboson_hardware_used") is not True:
        errors.append(f"{label}: QBoson hardware use is not proven")
    if record.get("real_provider_evidence") is not True:
        errors.append(f"{label}: real provider evidence is absent")
    if record.get("provider_reported_target") is not True:
        errors.append(f"{label}: target identity is not provider-reported")
    if (
        not isinstance(record.get("qboson_target"), str)
        or not record["qboson_target"].strip()
    ):
        errors.append(f"{label}: provider target is missing")
    task_ids = record.get("qboson_task_ids")
    if (
        not isinstance(task_ids, list)
        or not task_ids
        or any(
            not isinstance(task_id, str) or not task_id.strip() for task_id in task_ids
        )
    ):
        errors.append(f"{label}: provider task identities are missing")
    if record.get("sampling_mode") != "sampling":
        errors.append(f"{label}: expected sampling mode")
    requested = _finite_number(
        record.get("requested_samples"), f"{label}.requested_samples", errors
    )
    returned = _finite_number(
        record.get("returned_samples"), f"{label}.returned_samples", errors
    )
    if requested is not None and returned is not None and requested != returned:
        errors.append(f"{label}: requested and returned sample counts differ")
    if record.get("requested_samples") != config.get("requested_samples"):
        errors.append(f"{label}: requested sample count differs from config")
    call_count = record.get("remote_call_count")
    call_budget = record.get("remote_call_budget")
    if (
        type(call_count) is not int
        or type(call_budget) is not int
        or call_count <= 0
        or call_count > call_budget
    ):
        errors.append(f"{label}: remote-call accounting violates its budget")
    if record.get("fallback_occurred") is not False:
        errors.append(f"{label}: fallback must be explicitly false")
    if record.get("retrieval_resubmitted") is not False:
        errors.append(f"{label}: retrieval must not resubmit")
    if record.get("secrets_redacted") is not True:
        errors.append(f"{label}: secret redaction is not proven")
    if record.get("component_bundle_required") is not True:
        errors.append(f"{label}: component bundle must be explicitly required")
    if record.get("limitations") != list(FINAL_ACCEPTANCE_LIMITATIONS):
        errors.append(f"{label}: final claim limitations differ")
    artifacts = _mapping(record.get("artifacts"), f"{label}.artifacts", errors)
    artifact_config_fields = {
        "dataset_sha256": "dataset",
        "base_checkpoint_sha256": "checkpoint",
        "tokenizer_sha256": "tokenizer",
        "evaluation_model_sha256": "evaluation_model",
    }
    for record_field, config_section in artifact_config_fields.items():
        expected_digest = _mapping(
            config.get(config_section), f"config.{config_section}", errors
        ).get("sha256")
        if artifacts.get(record_field) != expected_digest:
            errors.append(f"{label}.artifacts.{record_field}: differs from config")
    trained_digest = artifacts.get("trained_energy_checkpoint_sha256")
    if (
        not isinstance(trained_digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", trained_digest) is None
    ):
        errors.append(
            f"{label}.artifacts.trained_energy_checkpoint_sha256: expected a SHA-256 digest"
        )
    precision = _mapping(
        record.get("precision_policy"), f"{label}.precision_policy", errors
    )
    expected_precision = _mapping(
        config.get("precision_policy"), "config.precision_policy", errors
    )
    for field in ("name", "target_min", "target_max"):
        if precision.get(field) != expected_precision.get(field):
            errors.append(f"{label}.precision_policy.{field}: differs from config")
    matrix_count = precision.get("matrix_count")
    if type(matrix_count) is not int or matrix_count <= 0:
        errors.append(
            f"{label}.precision_policy.matrix_count: expected a positive integer"
        )
    elif type(call_count) is int and matrix_count < call_count:
        errors.append(f"{label}.precision_policy: fewer reports than remote calls")
    scale_min = _finite_number(
        precision.get("scale_factor_min"),
        f"{label}.precision_policy.scale_factor_min",
        errors,
    )
    scale_max = _finite_number(
        precision.get("scale_factor_max"),
        f"{label}.precision_policy.scale_factor_max",
        errors,
    )
    max_error = _finite_number(
        precision.get("max_abs_error"),
        f"{label}.precision_policy.max_abs_error",
        errors,
    )
    mean_error = _finite_number(
        precision.get("mean_of_matrix_mean_abs_error"),
        f"{label}.precision_policy.mean_of_matrix_mean_abs_error",
        errors,
    )
    if scale_min is not None and scale_min <= 0:
        errors.append(f"{label}.precision_policy.scale_factor_min: expected positive")
    if scale_max is not None and scale_max <= 0:
        errors.append(f"{label}.precision_policy.scale_factor_max: expected positive")
    if scale_min is not None and scale_max is not None and scale_max < scale_min:
        errors.append(f"{label}.precision_policy: invalid scale-factor range")
    if max_error is not None and max_error < 0:
        errors.append(f"{label}.precision_policy.max_abs_error: expected non-negative")
    if mean_error is not None and mean_error < 0:
        errors.append(
            f"{label}.precision_policy.mean_of_matrix_mean_abs_error: expected non-negative"
        )
    if max_error is not None and mean_error is not None and mean_error > max_error:
        errors.append(f"{label}.precision_policy: mean error exceeds maximum error")
    training = _mapping(record.get("training"), f"{label}.training", errors)
    for field in ("energy_objective", "gradient_norm", "parameter_delta_max"):
        value = _finite_number(training.get(field), f"{label}.training.{field}", errors)
        if field != "energy_objective" and value is not None and value <= 0:
            errors.append(f"{label}.training.{field}: expected a positive value")
    generation = _mapping(record.get("generation"), f"{label}.generation", errors)
    if generation.get("token_constraints_passed") is not True:
        errors.append(f"{label}: token constraints did not pass")
    if generation.get("invalid_sequence_count") != 0:
        errors.append(f"{label}: invalid generated sequences were observed")
    transfers = _mapping(
        record.get("transfer_accounting"), f"{label}.transfer_accounting", errors
    )
    requested_device = record.get("requested_cuda_device")
    if transfers.get("matrix_origin_device") != requested_device:
        errors.append(f"{label}: matrix origin was not the requested CUDA device")
    if transfers.get("returned_sample_target_device") != requested_device:
        errors.append(f"{label}: returned samples did not target the CUDA device")
    boundaries = transfers.get("sampler_boundaries")
    if not isinstance(boundaries, list) or not boundaries:
        errors.append(f"{label}: sampler transfer accounting is missing")
    else:
        non_cached = 0
        for index, raw_boundary in enumerate(boundaries):
            boundary = _mapping(
                raw_boundary,
                f"{label}.transfer_accounting.sampler_boundaries[{index}]",
                errors,
            )
            if boundary.get("cache_hit") is False:
                non_cached += 1
            expected_fields = {
                "input_type": "numpy.ndarray",
                "input_device": "cpu",
                "canonical_device": "cpu",
                "canonical_dtype": "torch.float64",
                "submission_storage": "cpu_python_tuple",
                "returned_storage": "cpu_numpy",
                "returned_dtype": "int8",
            }
            for field, expected in expected_fields.items():
                if boundary.get(field) != expected:
                    errors.append(
                        f"{label}.transfer_accounting.sampler_boundaries[{index}]."
                        f"{field}: expected {expected}"
                    )
            matrix_shape = boundary.get("matrix_shape")
            original_digest = boundary.get("original_matrix_sha256")
            submission_digest = boundary.get("submission_matrix_sha256")
            if (
                not isinstance(original_digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", original_digest) is None
            ):
                errors.append(f"{label}: transferred original matrix digest is invalid")
            if (
                not isinstance(submission_digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", submission_digest) is None
            ):
                errors.append(f"{label}: transferred matrix digest is invalid")
            returned_shape = boundary.get("returned_shape")
            if (
                not isinstance(matrix_shape, list)
                or len(matrix_shape) != 2
                or matrix_shape[0] != matrix_shape[1]
                or any(type(size) is not int or size <= 0 for size in matrix_shape)
            ):
                errors.append(f"{label}: invalid transferred matrix shape")
            if (
                not isinstance(returned_shape, list)
                or len(returned_shape) != 2
                or any(type(size) is not int or size <= 0 for size in returned_shape)
                or (
                    isinstance(matrix_shape, list)
                    and len(matrix_shape) == 2
                    and returned_shape[1] != matrix_shape[0]
                )
            ):
                errors.append(f"{label}: invalid returned sample shape")
        if type(call_count) is int and non_cached != call_count:
            errors.append(
                f"{label}: transfer accounting differs from remote-call count"
            )
        _validate_precision_transfer_origins(
            record,
            label=label,
            boundaries=[
                boundary for boundary in boundaries if isinstance(boundary, dict)
            ],
            errors=errors,
        )
    acceptance = _mapping(record.get("acceptance"), f"{label}.acceptance", errors)
    if acceptance.get("system") != "pass":
        errors.append(f"{label}: system acceptance did not pass")


def _metric(
    metrics: dict[str, Any], name: str, label: str, errors: list[str]
) -> float | None:
    return _finite_number(metrics.get(name), f"{label}.{name}", errors)


def _validate_application(
    primary: dict[str, Any], config: dict[str, Any], errors: list[str]
) -> None:
    baseline = _mapping(
        primary.get("baseline_metrics"), "primary.baseline_metrics", errors
    )
    guided = _mapping(primary.get("guided_metrics"), "primary.guided_metrics", errors)
    baseline_values = {
        name: _metric(baseline, name, "primary.baseline_metrics", errors)
        for name in METRIC_NAMES
    }
    guided_values = {
        name: _metric(guided, name, "primary.guided_metrics", errors)
        for name in METRIC_NAMES
    }
    baseline_primary = baseline_values["mean_cosine_distance"]
    guided_primary = guided_values["mean_cosine_distance"]
    if (
        baseline_primary is not None
        and guided_primary is not None
        and guided_primary >= baseline_primary
    ):
        errors.append("primary: guided mean cosine distance did not improve")
    baseline_unique = baseline_values["uniqueness_ratio"]
    guided_unique = guided_values["uniqueness_ratio"]
    thresholds = _mapping(config.get("thresholds"), "config.thresholds", errors)
    uniqueness_floor = _finite_number(
        thresholds.get("uniqueness_baseline_fraction_min"),
        "config.thresholds.uniqueness_baseline_fraction_min",
        errors,
    )
    if (
        baseline_unique is not None
        and guided_unique is not None
        and uniqueness_floor is not None
        and guided_unique < uniqueness_floor * baseline_unique
    ):
        errors.append("primary: guided uniqueness is below its preregistered floor")
    baseline_repeat = baseline_values["repeat_ratio_ge4"]
    guided_repeat = guided_values["repeat_ratio_ge4"]
    repeat_allowance = _finite_number(
        thresholds.get("repeat_ratio_absolute_increase_max"),
        "config.thresholds.repeat_ratio_absolute_increase_max",
        errors,
    )
    if (
        baseline_repeat is not None
        and guided_repeat is not None
        and repeat_allowance is not None
        and guided_repeat > baseline_repeat + repeat_allowance
    ):
        errors.append("primary: guided repeat ratio exceeds its preregistered limit")
    if guided_values["invalid_sequence_count"] != 0:
        errors.append("primary: guided generation contains invalid sequences")
    if primary.get("attempted_seeds") != config.get("seeds"):
        errors.append("primary: attempted seeds differ from the frozen seed set")
    application_evidence = _mapping(
        primary.get("application_evidence"), "primary.application_evidence", errors
    )
    if application_evidence.get("aggregation") != "arithmetic_mean_across_frozen_seeds":
        errors.append("primary: unexpected seed aggregation policy")
    evidence_records = application_evidence.get("records")
    if not isinstance(evidence_records, list) or len(evidence_records) != len(
        config.get("seeds", [])
    ):
        errors.append("primary: application evidence does not cover every seed")
    else:
        evidence_seeds = [
            record.get("seed") if isinstance(record, dict) else None
            for record in evidence_records
        ]
        if evidence_seeds != config.get("seeds"):
            errors.append(
                "primary: application evidence seed order differs from config"
            )
        for index, record in enumerate(evidence_records):
            if not isinstance(record, dict):
                errors.append(
                    f"primary.application_evidence.records[{index}]: expected object"
                )
                continue
            for field in (
                "training_record_sha256",
                "evaluation_record_sha256",
                "trained_energy_checkpoint_sha256",
            ):
                digest = record.get(field)
                if (
                    not isinstance(digest, str)
                    or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                ):
                    errors.append(
                        f"primary.application_evidence.records[{index}].{field}: "
                        "expected a SHA-256 digest"
                    )
    acceptance = _mapping(primary.get("acceptance"), "primary.acceptance", errors)
    if acceptance.get("application") != "pass":
        errors.append("primary: application acceptance did not pass")


def _validate_remote_sampling_component_evidence(
    record: dict[str, Any], label: str, errors: list[str]
) -> None:
    if record.get("run_completed") is not True:
        errors.append(f"{label}: remote component did not complete")
    for field in (
        "pinned_sdk_client",
        "provider_identity_complete",
        "provider_reported_target",
        "real_provider_evidence",
        "qboson_hardware_used",
        "precision_evidence_complete",
        "secrets_redacted",
    ):
        if record.get(field) is not True:
            errors.append(f"{label}: remote component {field} is not proven")
    if record.get("transport") != "kaiwu_cim":
        errors.append(f"{label}: remote component transport is not kaiwu_cim")
    if record.get("fallback_occurred") is not False:
        errors.append(f"{label}: remote component fallback is not explicitly false")
    if record.get("retrieval_resubmitted") is not False:
        errors.append(f"{label}: remote component retrieval resubmitted")

    remote_calls = record.get("remote_call_count")
    receipts = record.get("task_receipts")
    if type(remote_calls) is not int or remote_calls <= 0:
        errors.append(f"{label}: remote component calls must be positive")
    if not isinstance(receipts, list) or not receipts:
        errors.append(f"{label}: remote component task receipts are missing")
        return
    if type(remote_calls) is int and len(receipts) != remote_calls:
        errors.append(f"{label}: remote receipt count differs from calls")

    receipt_task_ids: list[str] = []
    receipt_targets: set[str] = set()
    receipt_matrix_digests: list[str] = []
    requested_samples = record.get("requested_samples")
    for index, receipt in enumerate(receipts):
        if not isinstance(receipt, dict):
            errors.append(f"{label}: remote receipt {index} is not an object")
            continue
        task_id, target, matrix_digest = _validate_sampling_receipt(
            receipt,
            label=f"{label}: remote receipt {index}",
            expected_requested_samples=requested_samples,
            errors=errors,
        )
        if task_id is not None:
            receipt_task_ids.append(task_id)
        if target is not None:
            receipt_targets.add(target)
        if matrix_digest is not None:
            receipt_matrix_digests.append(matrix_digest)

    if record.get("qboson_task_ids") != receipt_task_ids:
        errors.append(f"{label}: task IDs differ from remote receipts")
    if len(receipt_targets) != 1 or record.get("qboson_target") not in receipt_targets:
        errors.append(f"{label}: target differs from remote receipts")
    if len(receipt_task_ids) != len(set(receipt_task_ids)):
        errors.append(f"{label}: provider task IDs are not unique")
    if len(receipt_matrix_digests) != len(set(receipt_matrix_digests)):
        errors.append(f"{label}: receipt matrix identities are not unique")
    _validate_precision_evidence(
        record,
        label=label,
        receipt_matrix_digests=receipt_matrix_digests,
        errors=errors,
    )

    if record.get("schema") == SYSTEM_COMPONENT_SCHEMA:
        transfers = record.get("transfer_accounting")
        boundaries = (
            transfers.get("sampler_boundaries") if isinstance(transfers, dict) else None
        )
        if isinstance(boundaries, list) and all(
            isinstance(boundary, dict) for boundary in boundaries
        ):
            non_cached = [
                boundary
                for boundary in boundaries
                if boundary.get("cache_hit") is False
            ]
            if len(non_cached) != len(receipts):
                errors.append(
                    f"{label}: non-cached transfers differ from remote receipts"
                )
            else:
                for index, (receipt, boundary) in enumerate(
                    zip(receipts, non_cached, strict=True)
                ):
                    if not isinstance(receipt, dict):
                        continue
                    matrix_size = receipt.get("matrix_size")
                    requested_count = receipt.get("requested_samples")
                    matrix_shape = boundary.get("matrix_shape")
                    returned_shape = boundary.get("returned_shape")
                    if boundary.get("submission_matrix_sha256") != receipt.get(
                        "matrix_sha256"
                    ):
                        errors.append(
                            f"{label}: transfer {index} matrix identity differs "
                            "from its receipt"
                        )
                    if matrix_shape != [matrix_size, matrix_size]:
                        errors.append(
                            f"{label}: transfer {index} matrix size differs "
                            "from its receipt"
                        )
                    if returned_shape != [requested_count, matrix_size]:
                        errors.append(
                            f"{label}: transfer {index} returned shape differs "
                            "from its receipt"
                        )
            _validate_precision_transfer_origins(
                record,
                label=label,
                boundaries=boundaries,
                errors=errors,
            )

    returned_samples = record.get("returned_samples")
    if (
        type(requested_samples) is not int
        or type(returned_samples) is not int
        or requested_samples != returned_samples
    ):
        errors.append(f"{label}: remote requested and returned samples differ")
    budget = record.get("remote_call_budget")
    if type(budget) is not int or budget <= 0:
        errors.append(f"{label}: remote call budget must be positive")
    elif type(remote_calls) is int and remote_calls > budget:
        errors.append(f"{label}: remote calls exceed their budget")
    precision = record.get("precision_policy")
    matrix_count = (
        precision.get("matrix_count") if isinstance(precision, dict) else None
    )
    if (
        type(matrix_count) is not int
        or matrix_count <= 0
        or (type(remote_calls) is int and matrix_count < remote_calls)
    ):
        errors.append(f"{label}: remote precision evidence is incomplete")


def _validate_portability_component_evidence(
    record: dict[str, Any],
    label: str,
    errors: list[str],
    *,
    expected_requested_samples: int | None = None,
    config: dict[str, Any] | None = None,
) -> None:
    _validate_remote_sampling_component_evidence(record, label, errors)
    if record.get("artifact_inputs_unchanged") is not True:
        errors.append(f"{label}: portability frozen inputs changed during execution")
    if (
        expected_requested_samples is not None
        and record.get("requested_samples") != expected_requested_samples
    ):
        errors.append(f"{label}: portability sample count differs from config")
    if record.get("requested_cuda_device") != "cuda:0":
        errors.append(f"{label}: portability did not request cuda:0")
    if record.get("observed_tensor_device") != "cuda:0":
        errors.append(f"{label}: portability tensor work was not on cuda:0")
    if "A800" not in str(record.get("observed_gpu_model", "")):
        errors.append(f"{label}: portability GPU is not an NVIDIA A800")
    fixture = record.get("fixture")
    if (
        not isinstance(fixture, dict)
        or fixture.get("token_constraints_passed") is not True
    ):
        errors.append(f"{label}: portability token constraints did not pass")
    if config is None:
        return

    software = _mapping(config.get("software"), "config.software", errors)
    for field in (
        "source_revision",
        "flagquantum_version",
        "kaiwu_pytorch_plugin_revision",
        "python_version",
        "torch_version",
        "kaiwu_sdk_version",
        "environment_lock_sha256",
    ):
        if record.get(field) != software.get(field):
            errors.append(f"{label}: portability {field} differs from config")

    artifacts = _mapping(record.get("artifacts"), f"{label}.artifacts", errors)
    expected_artifacts = {
        "dataset_sha256": _mapping(
            config.get("dataset"), "config.dataset", errors
        ).get("sha256"),
        "base_checkpoint_sha256": _mapping(
            config.get("checkpoint"), "config.checkpoint", errors
        ).get("sha256"),
        "tokenizer_sha256": _mapping(
            config.get("tokenizer"), "config.tokenizer", errors
        ).get("sha256"),
        "evaluation_model_sha256": _mapping(
            config.get("evaluation_model"), "config.evaluation_model", errors
        ).get("sha256"),
        "trained_energy_checkpoint_sha256": record.get(
            "trained_energy_checkpoint_sha256"
        ),
    }
    if artifacts != expected_artifacts:
        errors.append(f"{label}: portability artifact identities differ from config")

    generation = _mapping(config.get("generation"), "config.generation", errors)
    fixture = _mapping(record.get("fixture"), f"{label}.fixture", errors)
    expected_fixture = {
        "training_seed": generation.get("portability_training_seed"),
        "index": generation.get("portability_fixture_index"),
        "steps": generation.get("portability_steps"),
    }
    for field, expected in expected_fixture.items():
        if fixture.get(field) != expected:
            errors.append(f"{label}: portability fixture {field} differs from config")
    _finite_number(
        fixture.get("energy_objective"),
        f"{label}.fixture.energy_objective",
        errors,
    )
    generated_length = fixture.get("generated_length")
    if type(generated_length) is not int or generated_length <= 0:
        errors.append(f"{label}: portability generated length is invalid")
    generated_digest = fixture.get("generated_sha256")
    if (
        not isinstance(generated_digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", generated_digest) is None
    ):
        errors.append(f"{label}: portability generated identity is invalid")


def _validate_training_provider_evidence(
    record: dict[str, Any], label: str, errors: list[str]
) -> None:
    if record.get("transport") != "kaiwu_cim":
        errors.append(f"{label}: training transport is not kaiwu_cim")
    for field in (
        "pinned_sdk_client",
        "real_provider_evidence",
        "qboson_hardware_used",
        "provider_identity_complete",
        "provider_reported_target",
        "precision_evidence_complete",
        "secrets_redacted",
    ):
        if record.get(field) is not True:
            errors.append(f"{label}: training {field} is not proven")
    if record.get("fallback_occurred") is not False:
        errors.append(f"{label}: training fallback is not explicitly false")

    remote_calls = record.get("remote_call_count")
    budget = record.get("protein_remote_call_budget_per_seed")
    if type(remote_calls) is not int or remote_calls <= 0:
        errors.append(f"{label}: training remote_call_count must be positive")
    if type(budget) is not int or budget <= 0:
        errors.append(f"{label}: training per-seed call budget must be positive")
    elif type(remote_calls) is int and remote_calls > budget:
        errors.append(f"{label}: training remote calls exceed the per-seed budget")

    receipts = record.get("task_receipts")
    if not isinstance(receipts, list) or not receipts:
        errors.append(f"{label}: training task receipts are missing")
    else:
        if type(remote_calls) is int and len(receipts) != remote_calls:
            errors.append(f"{label}: training receipt count differs from remote calls")
        receipt_task_ids: list[str] = []
        receipt_targets: set[str] = set()
        receipt_matrix_digests: list[str] = []
        requested_samples = record.get("requested_samples")
        for index, receipt in enumerate(receipts):
            if not isinstance(receipt, dict):
                errors.append(f"{label}: training receipt {index} is not an object")
                continue
            task_id, target, matrix_digest = _validate_sampling_receipt(
                receipt,
                label=f"{label}: training receipt {index}",
                expected_requested_samples=requested_samples,
                errors=errors,
            )
            if task_id is not None:
                receipt_task_ids.append(task_id)
            if target is not None:
                receipt_targets.add(target)
            if matrix_digest is not None:
                receipt_matrix_digests.append(matrix_digest)
        if record.get("qboson_task_ids") != receipt_task_ids:
            errors.append(f"{label}: training task IDs differ from receipts")
        if (
            len(receipt_targets) != 1
            or record.get("qboson_target") not in receipt_targets
        ):
            errors.append(f"{label}: training target differs from receipts")
        if len(receipt_task_ids) != len(set(receipt_task_ids)):
            errors.append(f"{label}: training provider task IDs are not unique")
        if len(receipt_matrix_digests) != len(set(receipt_matrix_digests)):
            errors.append(f"{label}: training receipt matrix identities are not unique")
        _validate_precision_evidence(
            record,
            label=label,
            receipt_matrix_digests=receipt_matrix_digests,
            errors=errors,
        )

    if record.get("sampling_mode") != "sampling":
        errors.append(f"{label}: training sampling mode is not sampling")
    requested_samples = record.get("requested_samples")
    if type(requested_samples) is not int or not 10 <= requested_samples <= 2000:
        errors.append(f"{label}: training requested sample count is invalid")

    precision_count = record.get("precision_report_count")
    if (
        type(precision_count) is not int
        or precision_count <= 0
        or (type(remote_calls) is int and precision_count < remote_calls)
    ):
        errors.append(f"{label}: training precision evidence is incomplete")
    precision = record.get("precision_policy")
    matrix_count = (
        precision.get("matrix_count") if isinstance(precision, dict) else None
    )
    if matrix_count != precision_count:
        errors.append(f"{label}: training precision summary count differs")


def _validate_training_component(
    record: dict[str, Any],
    *,
    config: dict[str, Any],
    config_sha256: str,
    label: str,
    errors: list[str],
) -> None:
    _validate_training_provider_evidence(record, label, errors)
    if record.get("artifact_inputs_unchanged") is not True:
        errors.append(f"{label}: training frozen inputs changed during execution")
    if record.get("experiment_config_sha256") != config_sha256:
        errors.append(f"{label}: training uses another frozen config")
    software = _mapping(config.get("software"), "config.software", errors)
    for field in (
        "source_revision",
        "flagquantum_version",
        "kaiwu_pytorch_plugin_revision",
        "python_version",
        "torch_version",
        "kaiwu_sdk_version",
        "environment_lock_sha256",
    ):
        if record.get(field) != software.get(field):
            errors.append(f"{label}: training {field} differs from config")
    if record.get("execution_host") != config.get("primary_host"):
        errors.append(f"{label}: training is not from primary_host")
    if record.get("requested_cuda_device") != "cuda:0":
        errors.append(f"{label}: training did not request cuda:0")
    if "A800" not in str(record.get("observed_gpu_model", "")):
        errors.append(f"{label}: training GPU is not an NVIDIA A800")
    if record.get("seed") not in config.get("seeds", []):
        errors.append(f"{label}: training seed is not preregistered")
    if record.get("remote_call_budget") != config.get("remote_call_budget"):
        errors.append(f"{label}: training system-call budget differs from config")
    if record.get("requested_samples") != config.get("requested_samples"):
        errors.append(f"{label}: training sample count differs from config")
    training_config = _mapping(config.get("training"), "config.training", errors)
    protein_budget = record.get("protein_remote_call_budget_per_seed")
    if protein_budget != training_config.get("remote_call_budget_per_seed"):
        errors.append(f"{label}: training per-seed budget differs from config")
    estimated_calls = record.get("estimated_worst_case_remote_calls")
    if (
        type(estimated_calls) is not int
        or estimated_calls <= 0
        or (type(protein_budget) is int and estimated_calls > protein_budget)
    ):
        errors.append(f"{label}: training worst-case call bound is invalid")
    for field in ("artifact_preflight_sha256", "trained_energy_checkpoint_sha256"):
        value = record.get(field)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            errors.append(f"{label}: training {field} is not a SHA-256 digest")
    for field in ("run_directory_name", "trained_energy_checkpoint_name"):
        value = record.get(field)
        if (
            not isinstance(value, str)
            or not value
            or PurePosixPath(value).name != value
            or value in {".", ".."}
        ):
            errors.append(f"{label}: training {field} is not a safe basename")
    if record.get("acceptance") != {
        "system": "not_evaluated",
        "application": "not_evaluated",
    }:
        errors.append(f"{label}: training overstates standalone acceptance")

    precision = _mapping(
        record.get("precision_policy"), f"{label}.training_precision_policy", errors
    )
    expected_precision = _mapping(
        config.get("precision_policy"), "config.precision_policy", errors
    )
    for field in ("name", "target_min", "target_max"):
        if precision.get(field) != expected_precision.get(field):
            errors.append(f"{label}: training precision {field} differs from config")
    scale_min = _finite_number(
        precision.get("scale_factor_min"),
        f"{label}.training_precision_policy.scale_factor_min",
        errors,
    )
    scale_max = _finite_number(
        precision.get("scale_factor_max"),
        f"{label}.training_precision_policy.scale_factor_max",
        errors,
    )
    max_error = _finite_number(
        precision.get("max_abs_error"),
        f"{label}.training_precision_policy.max_abs_error",
        errors,
    )
    mean_error = _finite_number(
        precision.get("mean_of_matrix_mean_abs_error"),
        f"{label}.training_precision_policy.mean_of_matrix_mean_abs_error",
        errors,
    )
    if scale_min is not None and scale_min <= 0:
        errors.append(f"{label}: training precision minimum scale is not positive")
    if scale_max is not None and scale_max <= 0:
        errors.append(f"{label}: training precision maximum scale is not positive")
    if scale_min is not None and scale_max is not None and scale_max < scale_min:
        errors.append(f"{label}: training precision scale range is invalid")
    if max_error is not None and max_error < 0:
        errors.append(f"{label}: training maximum precision error is negative")
    if mean_error is not None and mean_error < 0:
        errors.append(f"{label}: training mean precision error is negative")
    if max_error is not None and mean_error is not None and mean_error > max_error:
        errors.append(f"{label}: training mean precision error exceeds maximum")

    expected_artifacts = {
        "test_fasta": "data_splits/test.fasta",
        "baseline_fasta": "baseline/proposal_only_generated_sequences.fasta",
        "guided_fasta": "guided/energy_guided_generated_sequences.fasta",
        "training_history": "history.json",
        "sequence_metrics": "baseline_vs_guided.json",
        "baseline_quality": "baseline_eval/quality_summary.json",
        "guided_quality": "guided_eval/quality_summary.json",
    }
    artifacts = record.get("workflow_artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(expected_artifacts):
        errors.append(f"{label}: training workflow artifact set is incomplete")
    else:
        for name, relative_path in expected_artifacts.items():
            identity = artifacts.get(name)
            if not isinstance(identity, dict):
                errors.append(f"{label}: training artifact {name} is not an object")
                continue
            if identity.get("relative_path") != relative_path:
                errors.append(f"{label}: training artifact {name} path differs")
            digest = identity.get("sha256")
            if (
                not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            ):
                errors.append(f"{label}: training artifact {name} has no digest")


def _validate_evaluation_component(
    record: dict[str, Any],
    *,
    config: dict[str, Any],
    config_sha256: str,
    label: str,
    errors: list[str],
) -> None:
    if record.get("artifact_inputs_unchanged") is not True:
        errors.append(f"{label}: evaluation frozen inputs changed during execution")
    if record.get("experiment_config_sha256") != config_sha256:
        errors.append(f"{label}: evaluation uses another frozen config")
    software = _mapping(config.get("software"), "config.software", errors)
    for field in (
        "source_revision",
        "flagquantum_version",
        "kaiwu_pytorch_plugin_revision",
        "python_version",
        "torch_version",
        "kaiwu_sdk_version",
        "environment_lock_sha256",
    ):
        if record.get(field) != software.get(field):
            errors.append(f"{label}: evaluation {field} differs from config")
    if record.get("execution_host") != config.get("primary_host"):
        errors.append(f"{label}: evaluation is not from primary_host")
    if record.get("observed_tensor_device") != "cuda:0":
        errors.append(f"{label}: evaluation tensor work was not on cuda:0")
    if "A800" not in str(record.get("observed_gpu_model", "")):
        errors.append(f"{label}: evaluation GPU is not an NVIDIA A800")
    evaluation_model = _mapping(
        config.get("evaluation_model"), "config.evaluation_model", errors
    )
    if record.get("evaluation_model_sha256") != evaluation_model.get("sha256"):
        errors.append(f"{label}: evaluation model differs from config")
    if record.get("provider_quota_consumed") is not False:
        errors.append(f"{label}: evaluation must not consume provider quota")
    if record.get("secrets_redacted") is not True:
        errors.append(f"{label}: evaluation secret redaction is not proven")
    if record.get("acceptance") != "candidate_evidence_only":
        errors.append(f"{label}: evaluation overstates its standalone acceptance")

    nonnegative = {
        "mean_cosine_distance",
        "median_cosine_distance",
        "mean_l2_distance",
        "median_l2_distance",
        "amino_acid_jsd",
        "kmer2_jsd",
        "kmer3_jsd",
    }
    ratios = {
        "identity_to_reference_mean",
        "uniqueness_ratio",
        "repeat_ratio_ge4",
        "length_match_ratio",
    }
    for field in ("baseline_metrics", "guided_metrics"):
        metrics = _mapping(record.get(field), f"{label}.{field}", errors)
        if set(metrics) != set(METRIC_NAMES):
            errors.append(f"{label}.{field}: metric field set is not closed")
        for metric in METRIC_NAMES:
            value = _finite_number(
                metrics.get(metric), f"{label}.{field}.{metric}", errors
            )
            if value is None:
                continue
            if metric in nonnegative and value < 0:
                errors.append(f"{label}.{field}.{metric}: expected non-negative")
            if metric in ratios and not 0 <= value <= 1:
                errors.append(f"{label}.{field}.{metric}: expected a ratio in [0, 1]")
            if metric == "invalid_sequence_count" and value != 0:
                errors.append(f"{label}.{field}: invalid sequences were observed")


def _validate_provider_smoke_component(
    record: dict[str, Any],
    *,
    config: dict[str, Any],
    sdk_approval_sha256: str,
    provider_resources: tuple[dict[str, Any], str],
    errors: list[str],
) -> None:
    """Prove the retained Phase 2 optimization and sampling smoke passed."""

    label = "provider smoke"
    if set(record) != SMOKE_RECORD_FIELDS:
        errors.append(f"{label}: field set is incomplete or contains extensions")
    if record.get("limitations") != list(SMOKE_LIMITATIONS):
        errors.append(f"{label}: claim limitations differ from the fixed boundary")
    software = _mapping(config.get("software"), "config.software", errors)
    if record.get("environment_lock_sha256") != software.get(
        "environment_lock_sha256"
    ):
        errors.append(f"{label}: environment lock identity mismatch")
    if record.get("sdk_approval_sha256") != sdk_approval_sha256:
        errors.append(f"{label}: SDK approval identity mismatch")
    provider_resource_record, provider_resources_sha256 = provider_resources
    if record.get("provider_resources_sha256") != provider_resources_sha256:
        errors.append(f"{label}: provider resource snapshot identity mismatch")
    try:
        validate_provider_resources(provider_resource_record)
    except ValueError as exc:
        errors.append(f"{label}: provider resource snapshot is invalid: {exc}")
    snapshot_captured_at = _parse_timestamp(
        provider_resource_record.get("captured_at")
    )
    snapshot_valid_until = _parse_timestamp(
        provider_resource_record.get("valid_until")
    )
    required_true = (
        "real_provider_evidence",
        "qboson_hardware_used",
        "run_completed",
        "live_provider_smoke_passed",
        "provider_identity_complete",
        "hardware_acceptance",
        "secrets_redacted",
    )
    for field in required_true:
        if record.get(field) is not True:
            errors.append(f"{label}: {field} is not proven")
    if record.get("transport") != "kaiwu_cim":
        errors.append(f"{label}: transport is not kaiwu_cim")
    if record.get("fallback_occurred") is not False:
        errors.append(f"{label}: fallback must be explicitly false")
    if record.get("failure") is not None:
        errors.append(f"{label}: retained failure is not empty")
    if not _canonical_printable_identifier(record.get("project_no")):
        errors.append(f"{label}: project number is invalid")
    approved_project = _mapping(
        config.get("kaiwu_sdk"), "config.kaiwu_sdk", errors
    ).get("project_no")
    if record.get("project_no") != approved_project:
        errors.append(f"{label}: project number differs from reviewed assignment")
    if not _canonical_printable_identifier(record.get("qboson_target")):
        errors.append(f"{label}: top-level provider target is invalid")
    smoke_time: datetime | None = None
    try:
        smoke_time = datetime.fromisoformat(
            str(record.get("recorded_at")).replace("Z", "+00:00")
        )
        if smoke_time.utcoffset() != timedelta(0):
            raise ValueError
    except (TypeError, ValueError):
        errors.append(f"{label}: recorded_at must be an aware UTC timestamp")
    smoke_prerequisites = (
        ("frozen config", config.get("preregistered_at")),
        (
            "SDK rights review",
            _mapping(config.get("kaiwu_sdk"), "config.kaiwu_sdk", errors).get(
                "rights_reviewed_at"
            ),
        ),
        (
            "project assignment review",
            _mapping(config.get("kaiwu_sdk"), "config.kaiwu_sdk", errors).get(
                "project_assignment_reviewed_at"
            ),
        ),
    )
    if smoke_time is not None:
        try:
            resources_ready, resource_reason = assess_provider_resources(
                provider_resource_record, now=smoke_time
            )
        except ValueError as exc:
            errors.append(f"{label}: provider resource snapshot is invalid: {exc}")
        else:
            if not resources_ready:
                errors.append(
                    f"{label}: provider resource snapshot was not ready: "
                    f"{resource_reason}"
                )
        for timestamp_label, timestamp in smoke_prerequisites:
            try:
                prerequisite = datetime.fromisoformat(
                    str(timestamp).replace("Z", "+00:00")
                )
            except (TypeError, ValueError):
                continue
            if prerequisite.utcoffset() is not None and smoke_time < prerequisite:
                errors.append(f"{label}: predates the {timestamp_label}")

    tasks = record.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 2:
        errors.append(f"{label}: exactly two task results are required")
        return
    expected_modes = ("optimization", "sampling")
    task_names: list[str] = []
    task_ids: list[str] = []
    targets: set[str] = set()
    matrix_digests: set[str] = set()
    for index, (task, expected_mode) in enumerate(
        zip(tasks, expected_modes, strict=True)
    ):
        task_record = _mapping(task, f"{label}.tasks[{index}]", errors)
        if set(task_record) != SMOKE_TASK_FIELDS:
            errors.append(
                f"{label}: task {index} field set is incomplete or contains extensions"
            )
        if task_record.get("receipt_schema") != TASK_RECEIPT_SCHEMA:
            errors.append(f"{label}: task {index} receipt schema is unsupported")
        if task_record.get("task_mode") != expected_mode:
            errors.append(f"{label}: task {index} mode differs")
        for field in ("task_name", "provider_task_id", "provider_target"):
            if not _canonical_printable_identifier(task_record.get(field)):
                errors.append(f"{label}: task {index} {field} is invalid")
        task_name = task_record.get("task_name")
        task_id = task_record.get("provider_task_id")
        target = task_record.get("provider_target")
        if isinstance(task_name, str) and _canonical_printable_identifier(
            task_name
        ):
            task_names.append(task_name)
        if isinstance(task_id, str):
            task_ids.append(task_id)
        if isinstance(target, str):
            targets.add(target)
        if task_record.get("provider_task_id_available") is not True:
            errors.append(f"{label}: task {index} provider task ID is unavailable")
        if task_record.get("provider_target_available") is not True:
            errors.append(f"{label}: task {index} provider target is unavailable")
        _validate_provider_result_schema(
            task_record.get("provider_result_schema"),
            label=f"{label}.tasks[{index}].provider_result_schema",
            errors=errors,
        )
        if task_record.get("fallback_occurred") is not False:
            errors.append(f"{label}: task {index} fallback is not false")
        matrix_digest = task_record.get("matrix_sha256")
        if matrix_digest != SMOKE_MATRIX_SHA256:
            errors.append(f"{label}: task {index} matrix identity differs")
        else:
            matrix_digests.add(matrix_digest)
        if task_record.get("matrix_size") != 2:
            errors.append(f"{label}: task {index} matrix size differs")
        if task_record.get("requested_samples") != config.get("requested_samples"):
            errors.append(f"{label}: task {index} sample count differs from config")
        if task_record.get("project_no") != record.get("project_no"):
            errors.append(f"{label}: task {index} project number differs")
        try:
            submitted_at = datetime.fromisoformat(
                str(task_record.get("submitted_at")).replace("Z", "+00:00")
            )
            if submitted_at.utcoffset() != timedelta(0):
                raise ValueError
            if smoke_time is not None and submitted_at > smoke_time:
                errors.append(f"{label}: task {index} submission follows its record")
            for prerequisite_label, prerequisite_value in smoke_prerequisites:
                prerequisite_time = _parse_timestamp(prerequisite_value)
                if (
                    prerequisite_time is not None
                    and submitted_at < prerequisite_time
                ):
                    errors.append(
                        f"{label}: task {index} submission predates the "
                        f"{prerequisite_label}"
                    )
            if (
                snapshot_captured_at is not None
                and submitted_at < snapshot_captured_at
            ):
                errors.append(
                    f"{label}: task {index} submission predates its resource snapshot"
                )
            if (
                snapshot_valid_until is not None
                and submitted_at > snapshot_valid_until
            ):
                errors.append(
                    f"{label}: task {index} submission follows resource snapshot expiry"
                )
        except (TypeError, ValueError):
            errors.append(
                f"{label}: task {index} submitted_at must be an aware UTC timestamp"
            )
        returned = task_record.get("returned_samples")
        if (
            expected_mode == "sampling"
            and returned != task_record.get("requested_samples")
        ) or (expected_mode == "optimization" and (type(returned) is not int or returned <= 0)):
            errors.append(f"{label}: task {index} returned sample count is invalid")
        samples = task_record.get("samples")
        energies = task_record.get("energies")
        if (
            not isinstance(samples, list)
            or not isinstance(energies, list)
            or len(samples) != returned
            or len(energies) != returned
        ):
            errors.append(f"{label}: task {index} result vectors are incomplete")
        else:
            observed_energies: list[float] = []
            for sample_index, (sample, energy) in enumerate(
                zip(samples, energies, strict=True)
            ):
                if (
                    not isinstance(sample, list)
                    or len(sample) != len(SMOKE_MATRIX)
                    or any(type(spin) is not int or spin not in {-1, 1} for spin in sample)
                ):
                    errors.append(
                        f"{label}: task {index} sample {sample_index} is invalid"
                    )
                    continue
                observed_energy = _finite_number(
                    energy,
                    f"{label}.tasks[{index}].energies[{sample_index}]",
                    errors,
                )
                if observed_energy is not None:
                    observed_energies.append(observed_energy)
                expected_energy = -sum(
                    sample[row] * SMOKE_MATRIX[row][column] * sample[column]
                    for row in range(len(SMOKE_MATRIX))
                    for column in range(len(SMOKE_MATRIX))
                )
                if observed_energy is not None and not math.isclose(
                    observed_energy,
                    expected_energy,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ):
                    errors.append(
                        f"{label}: task {index} sample {sample_index} energy differs"
                    )
            minimum = _finite_number(
                task_record.get("minimum_energy"),
                f"{label}.tasks[{index}].minimum_energy",
                errors,
            )
            maximum = _finite_number(
                task_record.get("maximum_energy"),
                f"{label}.tasks[{index}].maximum_energy",
                errors,
            )
            if observed_energies and len(observed_energies) == len(energies):
                if minimum != min(observed_energies):
                    errors.append(f"{label}: task {index} minimum energy differs")
                if maximum != max(observed_energies):
                    errors.append(f"{label}: task {index} maximum energy differs")
        raw_status = task_record.get("raw_status")
        if not isinstance(raw_status, str) or raw_status.strip().lower() not in {
            "finished",
            "completed",
            "done",
            "success",
            "succeed",
            "succeeded",
        }:
            errors.append(f"{label}: task {index} did not reach success")
    if len(task_names) != len(set(task_names)):
        errors.append(f"{label}: SDK task names are not unique")
    if len(task_ids) != len(set(task_ids)):
        errors.append(f"{label}: provider task IDs are not unique")
    if len(targets) != 1:
        errors.append(f"{label}: provider targets are inconsistent")
    elif record.get("qboson_target") != next(iter(targets)):
        errors.append(f"{label}: top-level provider target differs from tasks")
    if len(matrix_digests) != 1:
        errors.append(f"{label}: task matrix identities are inconsistent")


def _validate_component_bundle(
    component_payloads: dict[str, dict[str, Any]],
    *,
    config: dict[str, Any],
    config_sha256: str,
    primary: dict[str, Any],
    replay: dict[str, Any],
    errors: list[str],
) -> None:
    """Independently prove that final records are derived from their components."""

    seeds = config.get("seeds", [])
    remote_component_schemas = {
        SYSTEM_COMPONENT_SCHEMA,
        PORTABILITY_COMPONENT_SCHEMA,
        TRAINING_COMPONENT_SCHEMA,
    }
    referenced_provider_resources: set[str] = set()
    for payload in component_payloads.values():
        schema = payload.get("schema")
        if schema == PROVIDER_SMOKE_COMPONENT_SCHEMA:
            digest = payload.get("provider_resources_sha256")
        elif schema in remote_component_schemas:
            gate = payload.get("provider_resource_gate")
            digest = gate.get("snapshot_sha256") if isinstance(gate, dict) else None
        else:
            continue
        if isinstance(digest, str):
            referenced_provider_resources.add(digest)
    provider_resources_by_digest = {
        digest: payload
        for digest, payload in component_payloads.items()
        if payload.get("schema") == PROVIDER_RESOURCES_COMPONENT_SCHEMA
    }
    if set(provider_resources_by_digest) != referenced_provider_resources:
        errors.append(
            "manifest: provider resource snapshots differ from retained gate references"
        )
        return
    expected_schema_counts = {
        ARTIFACT_PREFLIGHT_COMPONENT_SCHEMA: 1,
        SDK_APPROVAL_COMPONENT_SCHEMA: 1,
        PROVIDER_RESOURCES_COMPONENT_SCHEMA: len(referenced_provider_resources),
        PROVIDER_SMOKE_COMPONENT_SCHEMA: 1,
        SOURCE_PREFLIGHT_COMPONENT_SCHEMA: 2,
        TRANSFER_MANIFEST_SCHEMA: 1,
        SYSTEM_COMPONENT_SCHEMA: 2,
        PORTABILITY_COMPONENT_SCHEMA: 1,
        TRAINING_COMPONENT_SCHEMA: len(seeds),
        EVALUATION_COMPONENT_SCHEMA: len(seeds),
    }
    observed_schema_counts = {
        schema: sum(
            payload.get("schema") == schema for payload in component_payloads.values()
        )
        for schema in expected_schema_counts
    }
    if observed_schema_counts != expected_schema_counts:
        errors.append("manifest: component record schema counts are incomplete")
        return

    for digest, payload in component_payloads.items():
        expected_version = (
            SDK_APPROVAL_COMPONENT_VERSION
            if payload.get("schema") == SDK_APPROVAL_COMPONENT_SCHEMA
            else "1.0"
        )
        if payload.get("version") != expected_version:
            errors.append(f"component {digest}: unsupported version")
        _validate_component_field_set(
            payload, label=f"component {digest}", errors=errors
        )
        _validate_executable_component_nested_fields(
            payload, label=f"component {digest}", errors=errors
        )
        if payload.get("schema") in COMPONENT_FIELDS_BY_SCHEMA:
            execution_host = payload.get("execution_host")
            expected_hostname = (
                config.get("host_identities", {}).get(execution_host)
                if isinstance(config.get("host_identities"), dict)
                and isinstance(execution_host, str)
                else None
            )
            if payload.get("observed_hostname") != expected_hostname:
                errors.append(
                    f"component {digest}: observed hostname differs from frozen identity"
                )
        if (
            payload.get("schema")
            not in (
                ARTIFACT_PREFLIGHT_COMPONENT_SCHEMA,
                SDK_APPROVAL_COMPONENT_SCHEMA,
                PROVIDER_RESOURCES_COMPONENT_SCHEMA,
                PROVIDER_SMOKE_COMPONENT_SCHEMA,
                SOURCE_PREFLIGHT_COMPONENT_SCHEMA,
                TRANSFER_MANIFEST_SCHEMA,
            )
            and payload.get("experiment_config_sha256") != config_sha256
        ):
            errors.append(
                f"component {digest}: frozen experiment config identity mismatch"
            )

    sdk_approvals = [
        (digest, payload)
        for digest, payload in component_payloads.items()
        if payload.get("schema") == SDK_APPROVAL_COMPONENT_SCHEMA
    ]
    sdk_approval_digest, sdk_approval = sdk_approvals[0]
    approval_errors = validate_sdk_approval_record(
        sdk_approval, label="retained sdk approval"
    )
    errors.extend(approval_errors)
    if sdk_approval != config.get("kaiwu_sdk"):
        errors.append("retained sdk approval differs from frozen configuration")
    approved_project = sdk_approval.get("project_no")
    for digest, payload in component_payloads.items():
        if payload.get("schema") not in remote_component_schemas:
            continue
        receipts = payload.get("task_receipts")
        if not isinstance(receipts, list):
            continue
        for index, receipt in enumerate(receipts):
            if isinstance(receipt, dict) and receipt.get("project_no") != approved_project:
                errors.append(
                    f"component {digest}: remote receipt {index} project number "
                    "differs from reviewed assignment"
                )

    provider_smokes = [
        payload
        for payload in component_payloads.values()
        if payload.get("schema") == PROVIDER_SMOKE_COMPONENT_SCHEMA
    ]
    provider_smoke_resource_digest = provider_smokes[0].get(
        "provider_resources_sha256"
    )
    if (
        not isinstance(provider_smoke_resource_digest, str)
        or provider_smoke_resource_digest not in provider_resources_by_digest
    ):
        errors.append("provider smoke: provider resource snapshot is unavailable")
        return
    _validate_provider_smoke_component(
        provider_smokes[0],
        config=config,
        sdk_approval_sha256=sdk_approval_digest,
        provider_resources=(
            provider_resources_by_digest[provider_smoke_resource_digest],
            provider_smoke_resource_digest,
        ),
        errors=errors,
    )
    provider_smoke_time = _parse_timestamp(provider_smokes[0].get("recorded_at"))
    component_times: dict[str, datetime] = {}
    executable_schemas = {
        SYSTEM_COMPONENT_SCHEMA,
        PORTABILITY_COMPONENT_SCHEMA,
        TRAINING_COMPONENT_SCHEMA,
        EVALUATION_COMPONENT_SCHEMA,
    }
    for digest, payload in component_payloads.items():
        if payload.get("schema") not in executable_schemas:
            continue
        recorded_at = _validate_execution_component_time(
            payload,
            label=f"component {digest}",
            config=config,
            provider_smoke_time=provider_smoke_time,
            errors=errors,
        )
        if recorded_at is not None:
            component_times[digest] = recorded_at
        if payload.get("schema") in remote_component_schemas:
            _validate_provider_resource_gate_binding(
                payload,
                label=f"component {digest}",
                provider_resources=provider_resources_by_digest,
                recorded_at=recorded_at,
                errors=errors,
            )

    provider_task_owners: dict[str, list[str]] = {}
    smoke_tasks = provider_smokes[0].get("tasks")
    if isinstance(smoke_tasks, list):
        for index, task in enumerate(smoke_tasks):
            if isinstance(task, dict) and isinstance(
                task.get("provider_task_id"), str
            ):
                provider_task_owners.setdefault(task["provider_task_id"], []).append(
                    f"provider smoke task {index}"
                )
    for digest, payload in component_payloads.items():
        if payload.get("schema") not in remote_component_schemas:
            continue
        task_ids = payload.get("qboson_task_ids")
        if not isinstance(task_ids, list):
            continue
        for task_id in task_ids:
            if isinstance(task_id, str):
                provider_task_owners.setdefault(task_id, []).append(
                    f"component {digest}"
                )
    duplicate_provider_tasks = {
        task_id: owners
        for task_id, owners in provider_task_owners.items()
        if len(owners) > 1
    }
    if duplicate_provider_tasks:
        errors.append(
            "manifest: provider task identities are reused across remote components"
        )

    artifact_preflights = [
        (digest, payload)
        for digest, payload in component_payloads.items()
        if payload.get("schema") == ARTIFACT_PREFLIGHT_COMPONENT_SCHEMA
    ]
    artifact_preflight_digest, artifact_preflight = artifact_preflights[0]
    if set(artifact_preflight) != ARTIFACT_PREFLIGHT_FIELDS:
        errors.append("artifact preflight: field set differs from its closed schema")
    if artifact_preflight.get("offline_preflight_only") is not True:
        errors.append("artifact preflight: offline-only flag is not proven")
    if artifact_preflight.get("acceptance_evidence") is not False:
        errors.append("artifact preflight: record overstates acceptance")
    if artifact_preflight.get("config_sha256") != config_sha256:
        errors.append("artifact preflight: frozen config identity mismatch")
    preflight_artifacts = _mapping(
        artifact_preflight.get("artifacts"), "artifact preflight.artifacts", errors
    )
    if set(preflight_artifacts) != set(ARTIFACT_FIELDS):
        errors.append("artifact preflight: artifact set is incomplete")
    else:
        for artifact_name, config_name in ARTIFACT_FIELDS.items():
            identity = _mapping(
                preflight_artifacts.get(artifact_name),
                f"artifact preflight.{artifact_name}",
                errors,
            )
            frozen = _mapping(
                config.get(config_name), f"config.{config_name}", errors
            )
            if identity.get("sha256") != frozen.get("sha256"):
                errors.append(
                    f"artifact preflight: {artifact_name} identity differs from config"
                )
            if identity.get("algorithm") not in {
                "file-sha256-v1",
                "tree-sha256-v1",
            }:
                errors.append(
                    f"artifact preflight: {artifact_name} algorithm is unsupported"
                )
            if type(identity.get("file_count")) is not int or identity["file_count"] <= 0:
                errors.append(
                    f"artifact preflight: {artifact_name} file count is invalid"
                )
    dataset_source = _mapping(
        artifact_preflight.get("dataset_source"),
        "artifact preflight.dataset_source",
        errors,
    )
    if set(dataset_source) != DATASET_SOURCE_FIELDS:
        errors.append("artifact preflight: dataset source schema differs")
    frozen_dataset = _mapping(config.get("dataset"), "config.dataset", errors)
    source_bindings = {
        "source_archive_sha256": frozen_dataset.get("source_archive_sha256"),
        "source_archive_bytes": frozen_dataset.get("source_archive_bytes"),
        "source_archive_format": frozen_dataset.get("source_archive_format"),
        "decompression_policy": frozen_dataset.get("decompression_policy"),
        "decompressed_sha256": frozen_dataset.get("sha256"),
    }
    for field, expected in source_bindings.items():
        if dataset_source.get(field) != expected:
            errors.append(f"artifact preflight: dataset source {field} differs")
    if (
        type(dataset_source.get("decompressed_bytes")) is not int
        or dataset_source.get("decompressed_bytes", 0) <= 0
    ):
        errors.append("artifact preflight: decompressed dataset size is invalid")

    software = _mapping(config.get("software"), "config.software", errors)
    source_preflights: dict[str, tuple[str, dict[str, Any]]] = {}
    for digest, payload in component_payloads.items():
        if payload.get("schema") != SOURCE_PREFLIGHT_COMPONENT_SCHEMA:
            continue
        host = payload.get("verified_for_target_host")
        if not isinstance(host, str) or host in source_preflights:
            errors.append("manifest: source preflights contain an invalid host")
            continue
        try:
            validate_source_preflight_record(
                payload,
                execution_host=host,
                source_revision=software.get("source_revision", ""),
                plugin_revision=software.get("kaiwu_pytorch_plugin_revision", ""),
            )
        except ValueError as exc:
            errors.append(f"source preflight {host}: {exc}")
        source_preflights[host] = (digest, payload)
    if set(source_preflights) != HOSTS:
        errors.append("manifest: source preflights must cover both validation hosts")
    else:
        try:
            validate_common_transfer_manifest(
                (
                    source_preflights["jp-a800-171"][1],
                    source_preflights["jp-a800-172"][1],
                )
            )
        except ValueError as exc:
            errors.append(f"manifest: {exc}")

    transfer_manifests = [
        (digest, payload)
        for digest, payload in component_payloads.items()
        if payload.get("schema") == TRANSFER_MANIFEST_SCHEMA
    ]
    if len(transfer_manifests) == 1:
        transfer_manifest_digest, transfer_manifest = transfer_manifests[0]
        try:
            validate_transfer_manifest_record(
                transfer_manifest,
                source_revision=software.get("source_revision", ""),
                plugin_revision=software.get("kaiwu_pytorch_plugin_revision", ""),
            )
        except ValueError as exc:
            errors.append(f"transfer manifest: {exc}")
        if source_preflights and any(
            preflight[1].get("manifest_sha256") != transfer_manifest_digest
            for preflight in source_preflights.values()
        ):
            errors.append("manifest: copied transfer manifest identity mismatch")

    for digest, payload in component_payloads.items():
        if payload.get("schema") in (
            ARTIFACT_PREFLIGHT_COMPONENT_SCHEMA,
            SDK_APPROVAL_COMPONENT_SCHEMA,
            PROVIDER_RESOURCES_COMPONENT_SCHEMA,
            PROVIDER_SMOKE_COMPONENT_SCHEMA,
            SOURCE_PREFLIGHT_COMPONENT_SCHEMA,
            TRANSFER_MANIFEST_SCHEMA,
        ):
            continue
        host = payload.get("execution_host")
        source_entry = source_preflights.get(host) if isinstance(host, str) else None
        if source_entry is None:
            errors.append(f"component {digest}: source preflight host is unavailable")
            continue
        source_digest, source_record = source_entry
        if payload.get("source_preflight_sha256") != source_digest:
            errors.append(f"component {digest}: source preflight identity mismatch")
        if payload.get("transfer_manifest_sha256") != source_record.get(
            "manifest_sha256"
        ):
            errors.append(f"component {digest}: transfer manifest identity mismatch")
        if payload.get("environment_lock_sha256") != software.get(
            "environment_lock_sha256"
        ):
            errors.append(f"component {digest}: environment lock identity mismatch")

    for final, label in ((primary, "primary"), (replay, "replay")):
        system_digest = final.get("system_evidence_sha256")
        component = (
            component_payloads.get(system_digest)
            if isinstance(system_digest, str)
            else None
        )
        if component is None or component.get("schema") != SYSTEM_COMPONENT_SCHEMA:
            errors.append(
                f"{label}: system evidence does not identify a system component"
            )
            continue
        _validate_remote_sampling_component_evidence(component, label, errors)
        copied_fields = (
            "source_revision",
            "kaiwu_pytorch_plugin_revision",
            "source_preflight_sha256",
            "transfer_manifest_sha256",
            "environment_lock_sha256",
            "provider_resource_gate",
            "python_version",
            "torch_version",
            "kaiwu_sdk_version",
            "execution_host",
            "run_role",
            "requested_cuda_device",
            "observed_tensor_device",
            "observed_gpu_model",
            "transport",
            "qboson_hardware_used",
            "real_provider_evidence",
            "provider_reported_target",
            "qboson_target",
            "qboson_task_ids",
            "sampling_mode",
            "requested_samples",
            "returned_samples",
            "remote_call_budget",
            "remote_call_count",
            "fallback_occurred",
            "retrieval_resubmitted",
            "secrets_redacted",
            "precision_policy",
            "transfer_accounting",
        )
        for field in copied_fields:
            if component.get(field) != final.get(field):
                errors.append(
                    f"{label}: system component {field} differs from final record"
                )
        component_training = _mapping(
            component.get("training"), f"{label}.system_component.training", errors
        )
        final_training = _mapping(final.get("training"), f"{label}.training", errors)
        training_fields = {
            "objective": "energy_objective",
            "gradient_norm": "gradient_norm",
            "parameter_delta_max": "parameter_delta_max",
        }
        for component_field, final_field in training_fields.items():
            if component_training.get(component_field) != final_training.get(
                final_field
            ):
                errors.append(
                    f"{label}: system component training.{component_field} "
                    "differs from final record"
                )
        component_generation = _mapping(
            component.get("generation"),
            f"{label}.system_component.generation",
            errors,
        )
        final_generation = _mapping(
            final.get("generation"), f"{label}.generation", errors
        )
        if component_generation.get("token_constraints_passed") != final_generation.get(
            "token_constraints_passed"
        ):
            errors.append(
                f"{label}: system component generation result differs from final record"
            )
        if component.get("acceptance") != {"system": "pass", "application": "not_run"}:
            errors.append(f"{label}: system component did not pass its isolated gate")

    application = _mapping(
        primary.get("application_evidence"), "primary.application_evidence", errors
    )
    evidence_records = application.get("records")
    if not isinstance(evidence_records, list):
        return
    training_by_seed: dict[int, tuple[str, dict[str, Any]]] = {}
    evaluation_by_seed: dict[int, tuple[str, dict[str, Any]]] = {}
    for digest, payload in component_payloads.items():
        seed = payload.get("seed")
        if payload.get("schema") == TRAINING_COMPONENT_SCHEMA:
            if type(seed) is not int or seed in training_by_seed:
                errors.append(
                    "manifest: training components contain an invalid or duplicate seed"
                )
            else:
                training_by_seed[seed] = (digest, payload)
        elif payload.get("schema") == EVALUATION_COMPONENT_SCHEMA:
            if type(seed) is not int or seed in evaluation_by_seed:
                errors.append(
                    "manifest: evaluation components contain an invalid or duplicate seed"
                )
            else:
                evaluation_by_seed[seed] = (digest, payload)
    if list(training_by_seed) != seeds or list(evaluation_by_seed) != seeds:
        errors.append("manifest: component seed order differs from the frozen seed set")

    primary_system_digest = primary.get("system_evidence_sha256")
    primary_system_time = (
        component_times.get(primary_system_digest)
        if isinstance(primary_system_digest, str)
        else None
    )
    for seed, (training_digest, training) in training_by_seed.items():
        _validate_training_component(
            training,
            config=config,
            config_sha256=config_sha256,
            label=f"seed {seed}",
            errors=errors,
        )
        if training.get("artifact_preflight_sha256") != artifact_preflight_digest:
            errors.append(
                f"seed {seed}: training references another artifact preflight"
            )
        training_time = component_times.get(training_digest)
        if (
            primary_system_time is not None
            and training_time is not None
            and training_time < primary_system_time
        ):
            errors.append(f"seed {seed}: training predates primary system evidence")
        _validate_receipt_submission_floor(
            training,
            earliest=primary_system_time,
            label=f"seed {seed}",
            prerequisite_label="primary system evidence",
            errors=errors,
        )
    for seed, (_, evaluation) in evaluation_by_seed.items():
        _validate_evaluation_component(
            evaluation,
            config=config,
            config_sha256=config_sha256,
            label=f"seed {seed}",
            errors=errors,
        )

    for evidence in evidence_records:
        if not isinstance(evidence, dict) or type(evidence.get("seed")) is not int:
            continue
        seed = evidence["seed"]
        training_entry = training_by_seed.get(seed)
        evaluation_entry = evaluation_by_seed.get(seed)
        if training_entry is None or evaluation_entry is None:
            continue
        training_digest, training = training_entry
        evaluation_digest, evaluation = evaluation_entry
        if evidence.get("training_record_sha256") != training_digest:
            errors.append(
                f"seed {seed}: final record references another training component"
            )
        if evidence.get("evaluation_record_sha256") != evaluation_digest:
            errors.append(
                f"seed {seed}: final record references another evaluation component"
            )
        if training.get("run_completed") is not True:
            errors.append(f"seed {seed}: training component did not complete")
        if training.get("execution_host") != config.get("primary_host"):
            errors.append(f"seed {seed}: training component is from the wrong host")
        checkpoint = training.get("trained_energy_checkpoint_sha256")
        if evidence.get("trained_energy_checkpoint_sha256") != checkpoint:
            errors.append(
                f"seed {seed}: final checkpoint differs from training component"
            )
        if evaluation.get("training_record_sha256") != training_digest:
            errors.append(
                f"seed {seed}: evaluation is linked to another training component"
            )
        if evaluation.get("execution_host") != config.get("primary_host"):
            errors.append(f"seed {seed}: evaluation component is from the wrong host")
        training_time = component_times.get(training_digest)
        evaluation_time = component_times.get(evaluation_digest)
        if (
            training_time is not None
            and evaluation_time is not None
            and evaluation_time < training_time
        ):
            errors.append(f"seed {seed}: evaluation predates its training component")

    for field in ("baseline_metrics", "guided_metrics"):
        final_metrics = _mapping(primary.get(field), f"primary.{field}", errors)
        for metric in METRIC_NAMES:
            values: list[float] = []
            for seed in seeds:
                entry = evaluation_by_seed.get(seed)
                if entry is None:
                    continue
                metrics = _mapping(entry[1].get(field), f"seed {seed}.{field}", errors)
                value = _finite_number(
                    metrics.get(metric), f"seed {seed}.{field}.{metric}", errors
                )
                if value is not None:
                    values.append(value)
            if len(values) != len(seeds):
                continue
            expected = (
                sum(values) if metric == "invalid_sequence_count" else fmean(values)
            )
            if final_metrics.get(metric) != expected:
                errors.append(
                    f"primary.{field}.{metric}: differs from component aggregation"
                )

    portability_evidence = _mapping(
        replay.get("portability_evidence"), "replay.portability_evidence", errors
    )
    portability_digest = portability_evidence.get("record_sha256")
    portability = (
        component_payloads.get(portability_digest)
        if isinstance(portability_digest, str)
        else None
    )
    if portability is None or portability.get("schema") != PORTABILITY_COMPONENT_SCHEMA:
        errors.append(
            "replay: portability evidence does not identify a portability component"
        )
        return
    if portability.get("execution_host") != config.get("replay_host"):
        errors.append("replay: portability component is from the wrong host")
    _validate_portability_component_evidence(
        portability,
        "replay",
        errors,
        expected_requested_samples=config.get("requested_samples"),
        config=config,
    )
    if portability.get("acceptance") != {"portability": "pass"}:
        errors.append("replay: portability component did not pass")
    if portability.get("artifact_preflight_sha256") != artifact_preflight_digest:
        errors.append("replay: portability references another artifact preflight")
    for field in ("training_record_sha256", "trained_energy_checkpoint_sha256"):
        if portability.get(field) != portability_evidence.get(field):
            errors.append(
                f"replay: portability component {field} differs from final record"
            )
    portability_time = component_times.get(portability_digest)
    replay_system_digest = replay.get("system_evidence_sha256")
    replay_system_time = (
        component_times.get(replay_system_digest)
        if isinstance(replay_system_digest, str)
        else None
    )
    portability_training_digest = portability.get("training_record_sha256")
    training_time = (
        component_times.get(portability_training_digest)
        if isinstance(portability_training_digest, str)
        else None
    )
    if (
        replay_system_time is not None
        and portability_time is not None
        and portability_time < replay_system_time
    ):
        errors.append("replay: portability predates replay system evidence")
    if (
        training_time is not None
        and portability_time is not None
        and portability_time < training_time
    ):
        errors.append("replay: portability predates its selected training component")
    _validate_receipt_submission_floor(
        portability,
        earliest=replay_system_time,
        label="replay",
        prerequisite_label="replay system evidence",
        errors=errors,
    )
    _validate_receipt_submission_floor(
        portability,
        earliest=training_time,
        label="replay",
        prerequisite_label="selected training component",
        errors=errors,
    )


def validate_acceptance(manifest_path: Path) -> list[str]:
    """Return all validation errors; an empty list means the evidence passes."""

    errors: list[str] = []
    try:
        effective_uid = _effective_uid()
    except ValueError as exc:
        return [f"manifest: cannot validate ownership: {exc}"]
    if not manifest_path.is_absolute():
        return ["manifest: path must be absolute"]
    if manifest_path.is_symlink() or not manifest_path.is_file():
        return ["manifest: must be a regular, non-symlink file"]
    manifest_metadata = manifest_path.stat()
    if manifest_metadata.st_mode & 0o077:
        return ["manifest: must not be accessible by group or others"]
    if manifest_metadata.st_uid != effective_uid:
        return ["manifest: must be owned by the current effective user"]
    root = manifest_path.resolve().parent
    try:
        manifest_value, _, _ = _read_private_json(
            manifest_path,
            label="acceptance manifest",
            max_bytes=_MAX_MANIFEST_BYTES,
        )
    except (OSError, ValueError) as exc:
        return [f"manifest: cannot be read safely: {exc}"]
    manifest = _mapping(manifest_value, "manifest", errors)
    _has_exact_fields(
        manifest, expected=MANIFEST_FIELDS, label="manifest", errors=errors
    )
    if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("version") != "1.0":
        errors.append("manifest: unsupported schema or version")

    declared_member_paths = {manifest_path.name}

    def resolve_member(value: Any, label: str) -> Path | None:
        if not isinstance(value, str) or not value:
            errors.append(f"{label}: expected a relative path")
            return None
        posix_path = PurePosixPath(value)
        if (
            "\\" in value
            or posix_path.is_absolute()
            or any(part in ("", ".", "..") for part in posix_path.parts)
            or str(posix_path) != value
        ):
            errors.append(f"{label}: path is not a normalized relative POSIX path")
            return None
        if value in declared_member_paths:
            errors.append(f"{label}: path is declared more than once")
            return None
        declared_member_paths.add(value)
        relative = Path(value)
        candidate = root / relative
        cursor = root
        for part in relative.parts:
            cursor /= part
            if cursor.is_symlink():
                errors.append(f"{label}: path contains a symlink")
                return None
        member = candidate.resolve()
        if not member.is_relative_to(root):
            errors.append(f"{label}: path escapes the evidence directory")
            return None
        if not member.is_file():
            errors.append(f"{label}: file does not exist")
            return None
        member_metadata = member.stat()
        if member_metadata.st_mode & 0o077:
            errors.append(f"{label}: file is accessible by group or others")
            return None
        if member_metadata.st_uid != effective_uid:
            errors.append(f"{label}: file is not owned by the current effective user")
            return None
        return member

    def validate_exact_tree() -> None:
        actual_files: set[str] = set()
        actual_directories: set[str] = set()
        for current_root, directory_names, file_names in os.walk(
            root, followlinks=False
        ):
            current = Path(current_root)
            for name in directory_names:
                directory = current / name
                relative_name = directory.relative_to(root).as_posix()
                if directory.is_symlink():
                    errors.append(
                        "manifest: evidence directory contains an unlisted symlink"
                    )
                elif not directory.is_dir():
                    errors.append(
                        "manifest: evidence tree contains a special directory entry: "
                        f"{relative_name}"
                    )
                else:
                    actual_directories.add(relative_name)
                    directory_metadata = directory.stat()
                    if directory_metadata.st_mode & 0o077:
                        errors.append(
                            "manifest: evidence directory is accessible by group or "
                            f"others: {relative_name}"
                        )
                    if directory_metadata.st_uid != effective_uid:
                        errors.append(
                            "manifest: evidence directory is not owned by the current "
                            f"effective user: {relative_name}"
                        )
            for name in file_names:
                member = current / name
                relative_name = member.relative_to(root).as_posix()
                if member.is_symlink():
                    errors.append(
                        f"manifest: evidence tree contains a symlink: {relative_name}"
                    )
                elif not member.is_file():
                    errors.append(
                        f"manifest: evidence tree contains a special file: {relative_name}"
                    )
                else:
                    actual_files.add(relative_name)
                    if member.stat().st_uid != effective_uid:
                        errors.append(
                            "manifest: evidence file is not owned by the current "
                            f"effective user: {relative_name}"
                        )
        if actual_files != declared_member_paths:
            errors.append(
                "manifest: evidence files differ from the exact declared member set"
            )
        expected_directories = {
            parent.as_posix()
            for member in declared_member_paths
            for parent in PurePosixPath(member).parents
            if parent != PurePosixPath(".")
        }
        if actual_directories != expected_directories:
            errors.append(
                "manifest: evidence directories differ from the exact declared set"
            )

    config_entry = _mapping(manifest.get("config"), "manifest.config", errors)
    _has_exact_fields(
        config_entry,
        expected=MANIFEST_MEMBER_FIELDS,
        label="manifest.config",
        errors=errors,
    )
    config_path = resolve_member(config_entry.get("path"), "manifest.config.path")
    if config_path is None:
        return errors
    try:
        config_value, _, config_hash = _read_private_json(
            config_path,
            label="acceptance configuration",
            max_bytes=_MAX_CONFIG_BYTES,
        )
    except (OSError, ValueError) as exc:
        errors.append(f"manifest.config: cannot be read safely: {exc}")
        return errors
    if config_entry.get("sha256") != config_hash:
        errors.append("manifest.config: SHA-256 mismatch")
    config = _mapping(config_value, "config", errors)
    _validate_config(config, errors)

    environment_entry = _mapping(
        manifest.get("environment_lock"), "manifest.environment_lock", errors
    )
    _has_exact_fields(
        environment_entry,
        expected=MANIFEST_MEMBER_FIELDS,
        label="manifest.environment_lock",
        errors=errors,
    )
    environment_path = resolve_member(
        environment_entry.get("path"), "manifest.environment_lock.path"
    )
    if environment_path is None:
        return errors
    try:
        environment_bytes = read_private_bytes(
            environment_path,
            label="acceptance environment lock",
            max_bytes=_MAX_ENVIRONMENT_LOCK_BYTES,
        )
    except (OSError, ValueError) as exc:
        errors.append(
            f"manifest.environment_lock: cannot be read safely: {exc}"
        )
        return errors
    environment_hash = hashlib.sha256(environment_bytes).hexdigest()
    if environment_entry.get("sha256") != environment_hash:
        errors.append("manifest.environment_lock: SHA-256 mismatch")
    environment_record: dict[str, Any] | None = None
    try:
        environment_record, parsed_environment_hash = parse_environment_lock_bytes(
            environment_bytes
        )
        if parsed_environment_hash != environment_hash:
            errors.append("manifest.environment_lock: internal digest mismatch")
    except ValueError as exc:
        errors.append(f"manifest.environment_lock: {exc}")
    software = _mapping(config.get("software"), "config.software", errors)
    if software.get("environment_lock_sha256") != environment_hash:
        errors.append("manifest.environment_lock: differs from frozen configuration")
    if environment_record is not None:
        frozen_kaiwu = config.get("kaiwu_sdk")
        if isinstance(frozen_kaiwu, dict):
            try:
                verify_approved_kaiwu_distribution(environment_record, frozen_kaiwu)
            except ValueError as exc:
                errors.append(f"manifest.environment_lock: {exc}")

    component_payloads: dict[str, dict[str, Any]] = {}
    raw_components = manifest.get("component_records")
    if not isinstance(raw_components, list):
        errors.append("manifest.component_records: expected a list")
    else:
        for index, raw_entry in enumerate(raw_components):
            entry = _mapping(
                raw_entry, f"manifest.component_records[{index}]", errors
            )
            _has_exact_fields(
                entry,
                expected=MANIFEST_MEMBER_FIELDS,
                label=f"manifest.component_records[{index}]",
                errors=errors,
            )
            component_path = resolve_member(
                entry.get("path"),
                f"manifest.component_records[{index}].path",
            )
            if component_path is None:
                continue
            try:
                component_value, _, digest = _read_private_json(
                    component_path,
                    label=f"acceptance component record {index}",
                    max_bytes=_MAX_EVIDENCE_MEMBER_BYTES,
                )
            except (OSError, ValueError) as exc:
                errors.append(
                    f"manifest.component_records[{index}]: "
                    f"cannot be read safely: {exc}"
                )
                continue
            if entry.get("sha256") != digest:
                errors.append(
                    f"manifest.component_records[{index}]: SHA-256 mismatch"
                )
                continue
            component_payloads[digest] = _mapping(
                component_value,
                f"component_record[{index}]",
                errors,
            )

    entries = manifest.get("records")
    if not isinstance(entries, list) or len(entries) != 2:
        errors.append("manifest.records: expected exactly two host records")
        return errors
    records: list[dict[str, Any]] = []
    for index, raw_entry in enumerate(entries):
        entry = _mapping(raw_entry, f"manifest.records[{index}]", errors)
        _has_exact_fields(
            entry,
            expected=MANIFEST_MEMBER_FIELDS,
            label=f"manifest.records[{index}]",
            errors=errors,
        )
        record_path = resolve_member(
            entry.get("path"), f"manifest.records[{index}].path"
        )
        if record_path is None:
            continue
        try:
            record_value, _, record_digest = _read_private_json(
                record_path,
                label=f"acceptance host record {index}",
                max_bytes=_MAX_EVIDENCE_MEMBER_BYTES,
            )
        except (OSError, ValueError) as exc:
            errors.append(
                f"manifest.records[{index}]: cannot be read safely: {exc}"
            )
            continue
        if entry.get("sha256") != record_digest:
            errors.append(f"manifest.records[{index}]: SHA-256 mismatch")
        record = _mapping(record_value, f"record[{index}]", errors)
        _validate_system_record(
            record,
            config=config,
            config_sha256=config_hash,
            label=f"record[{index}]",
            errors=errors,
        )
        records.append(record)
    if len(records) != 2:
        return errors
    validate_exact_tree()

    by_host = {str(record.get("execution_host")): record for record in records}
    if set(by_host) != HOSTS:
        errors.append("manifest: records must cover jp-a800-171 and jp-a800-172")
        return errors
    primary_host = config.get("primary_host")
    replay_host = config.get("replay_host")
    if primary_host not in by_host or replay_host not in by_host:
        errors.append("manifest: configured host roles do not match the records")
        return errors
    primary = by_host[primary_host]
    replay = by_host[replay_host]
    for label, record in (("primary", primary), ("replay", replay)):
        if record.get("component_bundle_required") is not True:
            errors.append(f"{label}: component bundle must be explicitly required")
    provider_resource_count = sum(
        payload.get("schema") == PROVIDER_RESOURCES_COMPONENT_SCHEMA
        for payload in component_payloads.values()
    )
    expected_component_count = (
        9 + provider_resource_count + 2 * len(config.get("seeds", []))
    )
    if len(component_payloads) != expected_component_count:
        errors.append(
            "manifest: component bundle does not contain every source record"
        )
    if primary.get("run_role") != "primary":
        errors.append("manifest: configured primary host lacks the primary role")
    if replay.get("run_role") != "portability_replay":
        errors.append("manifest: configured replay host lacks the portability role")
    if (
        _mapping(replay.get("acceptance"), "replay.acceptance", errors).get(
            "application"
        )
        != "not_run"
    ):
        errors.append("replay: application acceptance must be explicitly not_run")
    portability_evidence = _mapping(
        replay.get("portability_evidence"), "replay.portability_evidence", errors
    )
    if portability_evidence.get("acceptance") != "pass":
        errors.append("replay: portability evidence did not pass")
    expected_portability_seed = _mapping(
        config.get("generation"), "config.generation", errors
    ).get("portability_training_seed")
    if portability_evidence.get("training_seed") != expected_portability_seed:
        errors.append("replay: portability evidence uses another training seed")
    for field in ("record_sha256", "training_record_sha256"):
        portability_digest = portability_evidence.get(field)
        if (
            not isinstance(portability_digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", portability_digest) is None
        ):
            errors.append(f"replay.portability_evidence.{field}: expected SHA-256")
    revisions = {record.get("source_revision") for record in records}
    plugin_revisions = {
        record.get("kaiwu_pytorch_plugin_revision") for record in records
    }
    if len(revisions) != 1 or len(plugin_revisions) != 1:
        errors.append("manifest: both hosts must use identical source revisions")
    trained_checkpoint_digests = {
        _mapping(record.get("artifacts"), "record.artifacts", errors).get(
            "trained_energy_checkpoint_sha256"
        )
        for record in records
    }
    if len(trained_checkpoint_digests) != 1:
        errors.append(
            "manifest: both hosts must use the same trained energy checkpoint"
        )
    shared_checkpoint_digest = next(iter(trained_checkpoint_digests), None)
    if (
        portability_evidence.get("trained_energy_checkpoint_sha256")
        != shared_checkpoint_digest
    ):
        errors.append("replay: portability checkpoint differs from host records")
    application_evidence = _mapping(
        primary.get("application_evidence"), "primary.application_evidence", errors
    )
    evidence_records = application_evidence.get("records")
    if isinstance(evidence_records, list):
        selected = [
            record
            for record in evidence_records
            if isinstance(record, dict)
            and record.get("seed") == expected_portability_seed
        ]
        if (
            len(selected) != 1
            or selected[0].get("trained_energy_checkpoint_sha256")
            != shared_checkpoint_digest
            or selected[0].get("training_record_sha256")
            != portability_evidence.get("training_record_sha256")
        ):
            errors.append(
                "manifest: portability evidence is not linked to the selected seed checkpoint"
            )
    referenced_component_hashes = {
            primary.get("source_preflight_sha256"),
            replay.get("source_preflight_sha256"),
            primary.get("transfer_manifest_sha256"),
            primary.get("system_evidence_sha256"),
            replay.get("system_evidence_sha256"),
            portability_evidence.get("record_sha256"),
    }
    if isinstance(evidence_records, list):
        for record in evidence_records:
            if isinstance(record, dict):
                training_digest = record.get("training_record_sha256")
                referenced_component_hashes.add(training_digest)
                referenced_component_hashes.add(record.get("evaluation_record_sha256"))
                training_component = component_payloads.get(training_digest)
                if isinstance(training_component, dict):
                    referenced_component_hashes.add(
                        training_component.get("artifact_preflight_sha256")
                    )
    portability_component = component_payloads.get(
        portability_evidence.get("record_sha256")
    )
    if isinstance(portability_component, dict):
        referenced_component_hashes.add(
            portability_component.get("artifact_preflight_sha256")
        )
    referenced_component_hashes.update(
        digest
        for digest, component in component_payloads.items()
        if component.get("schema")
        in (
            SDK_APPROVAL_COMPONENT_SCHEMA,
            PROVIDER_RESOURCES_COMPONENT_SCHEMA,
            PROVIDER_SMOKE_COMPONENT_SCHEMA,
        )
    )
    if referenced_component_hashes != set(component_payloads):
        errors.append(
            "manifest: final records do not reference the exact component bundle"
        )
    _validate_component_bundle(
        component_payloads,
        config=config,
        config_sha256=config_hash,
        primary=primary,
        replay=replay,
        errors=errors,
    )
    task_sets = []
    for record in records:
        raw_task_ids = record.get("qboson_task_ids")
        task_sets.append(
            {task_id for task_id in raw_task_ids if isinstance(task_id, str)}
            if isinstance(raw_task_ids, list)
            else set()
        )
    if task_sets[0] & task_sets[1]:
        errors.append(
            "manifest: host runs must have independent QBoson task identities"
        )
    if any(
        record.get("remote_call_budget") != config.get("remote_call_budget")
        for record in records
    ):
        errors.append("manifest: host call budgets differ from the frozen config")
    _validate_application(primary, config, errors)
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    arguments = parser.parse_args()
    errors = validate_acceptance(arguments.manifest)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print("QBoson QDiffusion acceptance evidence passed")


if __name__ == "__main__":
    main()
