from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any

import pytest

from examples.qdiffusion_kaiwu import assemble_acceptance as assembler_module
from examples.qdiffusion_kaiwu.assemble_acceptance import (
    _load_component,
    _open_output_parent,
    _publish_acceptance_bundle,
    assemble_records,
)
from examples.qdiffusion_kaiwu.validate_acceptance import (
    _validate_remote_sampling_component_evidence,
    validate_acceptance,
)
from tests.team.ecosystem.test_qdiffusion_acceptance_validator import (
    _config as _full_config,
)

pytestmark = pytest.mark.unit

PRIMARY_SOURCE_PREFLIGHT_SHA = "d" * 64
REPLAY_SOURCE_PREFLIGHT_SHA = "e" * 64
TRANSFER_MANIFEST_SHA = "f" * 64
ENVIRONMENT_LOCK_SHA = "6" * 64
OBSERVED_HOSTNAMES = {
    "jp-a800-171": "node-a800-171",
    "jp-a800-172": "node-a800-172",
}


def _config() -> dict[str, Any]:
    return _full_config()


def test_assembler_accepts_repeated_provider_resource_snapshots() -> None:
    source = (
        Path(__file__).parents[3]
        / "examples"
        / "qdiffusion_kaiwu"
        / "assemble_acceptance.py"
    ).read_text(encoding="utf-8")

    assert '"--provider-resources", action="append", required=True' in source
    assert "provider_resources_by_digest.setdefault(digest" in source
    assert '"--provider-reconciliation", action="append"' in source
    assert "apply_provider_reconciliations(" in source


def _provider_resources() -> dict[str, Any]:
    return {
        "schema": "flagquantum.qboson_provider_resources",
        "version": "1.0",
        "source": "authenticated_resource_bill",
        "captured_at": "2026-10-05T12:00:00+00:00",
        "valid_until": "2026-10-06T12:00:00+00:00",
        "resources": [
            {
                "target": target,
                "mode": mode,
                "available": 100000 if mode == "sampling" else 1,
                "used": 0,
            }
            for target in ("SPQC-1", "SPQC-550", "SPQC-1000")
            for mode in ("optimization", "sampling")
        ],
        "claim_boundary": (
            "Account-resource observation only; it is not spend approval, project "
            "assignment, provider evidence, execution evidence, or acceptance evidence."
        ),
    }


def _artifact_preflight(config: dict[str, Any], config_sha256: str) -> dict[str, Any]:
    return {
        "schema": "flagquantum.qboson_qdiffusion_artifact_preflight",
        "version": "1.0",
        "offline_preflight_only": True,
        "acceptance_evidence": False,
        "config_sha256": config_sha256,
        "dataset_source": {
            "source_archive_sha256": config["dataset"]["source_archive_sha256"],
            "source_archive_bytes": config["dataset"]["source_archive_bytes"],
            "source_archive_format": config["dataset"]["source_archive_format"],
            "decompression_policy": config["dataset"]["decompression_policy"],
            "decompressed_sha256": config["dataset"]["sha256"],
            "decompressed_bytes": 456,
        },
        "artifacts": {
            artifact_name: {
                "sha256": config[config_name]["sha256"],
                "algorithm": "file-sha256-v1",
                "file_count": 1,
            }
            for artifact_name, config_name in {
                "dataset": "dataset",
                "base_checkpoint": "checkpoint",
                "tokenizer": "tokenizer",
                "evaluation_model": "evaluation_model",
            }.items()
        },
    }


def _provider_smoke(
    *,
    environment_lock_sha256: str,
    sdk_approval_sha256: str,
    provider_resources_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": "flagquantum.qboson_kaiwu_live_smoke",
        "version": "1.0",
        "recorded_at": "2026-10-06T00:00:00+00:00",
        "transport": "kaiwu_cim",
        "real_provider_evidence": True,
        "qboson_hardware_used": True,
        "qboson_target": "SPQC-provider",
        "project_no": "CPQC-test",
        "environment_lock_sha256": environment_lock_sha256,
        "sdk_approval_sha256": sdk_approval_sha256,
        "provider_resources_sha256": provider_resources_sha256,
        "tasks": [
            {
                "receipt_schema": "flagquantum.kaiwu-task.v1",
                "task_name": f"smoke-{mode}",
                "task_mode": mode,
                "matrix_sha256": (
                    "0352923b6964d8a65fc742c5a5b251ab967d43e8c4db9e3ee3a0f2f2fa5b0487"
                ),
                "matrix_size": 2,
                "requested_samples": 10,
                "project_no": "CPQC-test",
                "submitted_at": "2026-10-06T00:00:00+00:00",
                "returned_samples": 10,
                "samples": [[1, -1] for _ in range(10)],
                "energies": [2.0 for _ in range(10)],
                "provider_task_id": f"smoke-{mode}-task",
                "provider_target": "SPQC-provider",
                "raw_status": "completed",
                "fallback_occurred": False,
                "minimum_energy": 2.0,
                "maximum_energy": 2.0,
                "provider_task_id_available": True,
                "provider_target_available": True,
                "provider_result_schema": {
                    "available": False,
                    "reason": "test_fixture",
                },
            }
            for mode in ("optimization", "sampling")
        ],
        "run_completed": True,
        "failure": None,
        "live_provider_smoke_passed": True,
        "provider_identity_complete": True,
        "hardware_acceptance": True,
        "fallback_occurred": False,
        "secrets_redacted": True,
        "limitations": [
            "This smoke test does not execute QDiffusion or A800 tensor work.",
            "Hardware acceptance remains false without provider-reported task and target identities.",
            "This record does not establish performance, quantum advantage, or production maturity.",
        ],
    }


def _system(
    host: str,
    role: str,
    task_id: str,
    transfer_manifest_sha256: str = TRANSFER_MANIFEST_SHA,
    provider_resources_sha256: str = "c" * 64,
) -> dict[str, Any]:
    return {
        "schema": "flagquantum.qboson_qdiffusion_system_live_probe",
        "version": "1.0",
        "recorded_at": "2026-10-06T00:00:00+00:00",
        "source_revision": "a" * 40,
        "flagquantum_version": "0.2.0",
        "kaiwu_pytorch_plugin_revision": "b" * 40,
        "source_preflight_sha256": (
            PRIMARY_SOURCE_PREFLIGHT_SHA
            if host == "jp-a800-171"
            else REPLAY_SOURCE_PREFLIGHT_SHA
        ),
        "transfer_manifest_sha256": transfer_manifest_sha256,
        "environment_lock_sha256": ENVIRONMENT_LOCK_SHA,
        "provider_resource_gate": {
            "snapshot_sha256": provider_resources_sha256,
            "checked_at": "2026-10-05T12:00:00+00:00",
            "mode": "sampling",
            "required_calls": 128,
        },
        "python_version": "3.10.18",
        "torch_version": "2.7.0",
        "kaiwu_sdk_version": "1.3.1",
        "execution_host": host,
        "observed_hostname": OBSERVED_HOSTNAMES[host],
        "run_role": role,
        "requested_cuda_device": "cuda:0",
        "observed_tensor_device": "cuda:0",
        "observed_gpu_model": "NVIDIA A800-SXM4-80GB",
        "run_completed": True,
        "transport": "kaiwu_cim",
        "pinned_sdk_client": True,
        "qboson_hardware_used": True,
        "real_provider_evidence": True,
        "provider_identity_complete": True,
        "provider_reported_target": True,
        "provider_result_schema": {
            "available": False,
            "reason": "test_fixture",
        },
        "qboson_target": "SPQC-provider",
        "qboson_task_ids": [task_id],
        "task_receipts": [
            {
                "schema": "flagquantum.kaiwu-task.v1",
                "task_name": f"system-{task_id}",
                "matrix_sha256": "7" * 64,
                "matrix_size": 3,
                "mode": "sampling",
                "requested_samples": 10,
                "project_no": "CPQC-test",
                "submitted_at": "2026-10-06T00:00:00+00:00",
                "provider_task_id": task_id,
                "provider_target": "SPQC-provider",
            }
        ],
        "sampling_mode": "sampling",
        "requested_samples": 10,
        "returned_samples": 10,
        "remote_call_budget": 128,
        "remote_call_count": 1,
        "fallback_occurred": False,
        "retrieval_resubmitted": False,
        "secrets_redacted": True,
        "precision_policy": {
            "name": "explicit-int8",
            "target_min": -127,
            "target_max": 127,
            "matrix_count": 1,
            "scale_factor_min": 1.0,
            "scale_factor_max": 1.0,
            "max_abs_error": 0.0,
            "mean_of_matrix_mean_abs_error": 0.0,
        },
        "precision_evidence": [
            {
                "original_matrix_sha256": "6" * 64,
                "submission_matrix_sha256": "7" * 64,
                "source_type": "numpy.ndarray",
                "source_dtype": "float32",
                "normalized_dtype": "torch.float64",
                "normalized_min": -127.0,
                "normalized_max": 127.0,
                "symmetry_normalization": "arithmetic_mean",
                "rounding_policy": "round_half_to_even",
                "scale_factor": 1.0,
                "target_min": -127,
                "target_max": 127,
                "max_abs_error": 0.0,
                "mean_abs_error": 0.0,
            }
        ],
        "precision_evidence_complete": True,
        "failure": None,
        "training": {
            "objective": -0.5,
            "gradient_norm": 1.0,
            "parameter_delta_max": 0.01,
        },
        "generation": {
            "generated_tokens": [[1, 2, 3]],
            "token_constraints_passed": True,
        },
        "transfer_accounting": {
            "matrix_origin_device": "cuda:0",
            "sampler_boundaries": [
                {
                    "input_type": "numpy.ndarray",
                    "input_device": "cpu",
                    "input_dtype": "float32",
                    "matrix_shape": [3, 3],
                    "original_matrix_sha256": "6" * 64,
                    "submission_matrix_sha256": "7" * 64,
                    "canonical_device": "cpu",
                    "canonical_dtype": "torch.float64",
                    "submission_storage": "cpu_python_tuple",
                    "returned_storage": "cpu_numpy",
                    "returned_dtype": "int8",
                    "returned_shape": [10, 3],
                    "cache_hit": False,
                }
            ],
            "returned_sample_target_device": "cuda:0",
        },
        "acceptance": {"system": "pass", "application": "not_run"},
        "limitations": [
            "This bounded system probe does not run the frozen protein effectiveness experiment.",
            "System acceptance remains failed without provider-reported task and target identities.",
            "Two independent passing host records are required; this is one single-device run.",
            "No performance, distributed, domestic-accelerator, or quantum-advantage claim is made.",
        ],
    }


def _source_preflight(
    host: str, manifest_sha256: str = TRANSFER_MANIFEST_SHA
) -> dict[str, Any]:
    revisions = (
        ("flagquantum-qboson-", "FlagQuantum-", "a" * 40),
        ("kaiwu-plugin-", "kaiwu-pytorch-plugin-", "b" * 40),
        (
            "kaiwu-community-",
            "kaiwu-community-",
            "b648b531c034bd6ae9b7a34fed994c717967cc72",
        ),
    )
    return {
        "schema": "flagquantum.qboson_a800_extracted_bundle_verification",
        "version": "1.0",
        "evidence_class": "extraction_preflight_only",
        "verification_hostname": f"hostname-{host}",
        "verified_for_target_host": host,
        "manifest_sha256": manifest_sha256,
        "extracted_content_verified": True,
        "artifacts": [
            {
                "filename": f"{filename_prefix}{revision[:10]}.tar.gz",
                "revision": revision,
                "extracted_root": f"{root_prefix}{revision[:10]}",
                "file_count": 10,
                "content_set_sha256": "9" * 64,
            }
            for filename_prefix, root_prefix, revision in revisions
        ],
        "qboson_hardware_used": False,
        "a800_execution_verified": False,
        "acceptance_evidence": False,
    }


def _transfer_manifest() -> dict[str, Any]:
    revisions = (
        ("flagquantum-qboson-", "a" * 40),
        ("kaiwu-plugin-", "b" * 40),
        ("kaiwu-community-", "b648b531c034bd6ae9b7a34fed994c717967cc72"),
    )
    return {
        "schema": "flagquantum.qboson_a800_transfer_bundle",
        "version": "1.0",
        "created_for_hosts": ["jp-a800-171", "jp-a800-172"],
        "classification": "local_preparation_only_not_execution_evidence",
        "artifacts": [
            {
                "filename": f"{prefix}{revision[:10]}.tar.gz",
                "revision": revision,
                "sha256": "8" * 64,
            }
            for prefix, revision in revisions
        ],
    }


def _metrics(cosine: float) -> dict[str, float]:
    return {
        "mean_cosine_distance": cosine,
        "median_cosine_distance": cosine,
        "mean_l2_distance": cosine + 1,
        "median_l2_distance": cosine + 1,
        "identity_to_reference_mean": 0.5,
        "amino_acid_jsd": 0.1,
        "kmer2_jsd": 0.2,
        "kmer3_jsd": 0.3,
        "uniqueness_ratio": 1.0,
        "repeat_ratio_ge4": 0.0,
        "length_match_ratio": 1.0,
        "invalid_sequence_count": 0.0,
    }


def _components(
    config_sha256: str = "9" * 64,
    source_preflight_sha256: str = PRIMARY_SOURCE_PREFLIGHT_SHA,
    transfer_manifest_sha256: str = TRANSFER_MANIFEST_SHA,
    provider_resources_sha256: str = "c" * 64,
) -> tuple[
    list[tuple[dict[str, Any], str]],
    list[tuple[dict[str, Any], str]],
]:
    training = []
    evaluations = []
    for index, seed in enumerate((1701, 1702, 1703), start=1):
        training_sha = str(index) * 64
        training.append(
            (
                {
                    "schema": "flagquantum.qboson_qdiffusion_protein_training",
                    "version": "1.0",
                    "recorded_at": "2026-10-06T00:00:00+00:00",
                    "source_revision": "a" * 40,
                    "flagquantum_version": "0.2.0",
                    "kaiwu_pytorch_plugin_revision": "b" * 40,
                    "python_version": "3.10.18",
                    "torch_version": "2.7.0",
                    "kaiwu_sdk_version": "1.3.1",
                    "provider_resource_gate": {
                        "snapshot_sha256": provider_resources_sha256,
                        "checked_at": "2026-10-05T12:00:00+00:00",
                        "mode": "sampling",
                        "required_calls": 71269,
                    },
                    "seed": seed,
                    "run_completed": True,
                    "failure": None,
                    "artifact_inputs_unchanged": True,
                    "execution_host": "jp-a800-171",
                    "observed_hostname": OBSERVED_HOSTNAMES["jp-a800-171"],
                    "requested_cuda_device": "cuda:0",
                    "observed_gpu_model": "NVIDIA A800-SXM4-80GB",
                    "transport": "kaiwu_cim",
                    "pinned_sdk_client": True,
                    "real_provider_evidence": True,
                    "qboson_hardware_used": True,
                    "provider_identity_complete": True,
                    "provider_reported_target": True,
                    "qboson_target": "SPQC-provider",
                    "qboson_task_ids": [f"protein-task-{seed}"],
                    "sampling_mode": "sampling",
                    "requested_samples": 10,
                    "fallback_occurred": False,
                    "secrets_redacted": True,
                    "remote_call_budget": 128,
                    "remote_call_count": 1,
                    "protein_remote_call_budget_per_seed": 71269,
                    "estimated_worst_case_remote_calls": 100,
                    "precision_report_count": 1,
                    "precision_policy": {
                        "name": "explicit-int8",
                        "target_min": -127,
                        "target_max": 127,
                        "matrix_count": 1,
                        "scale_factor_min": 1.0,
                        "scale_factor_max": 1.0,
                        "max_abs_error": 0.0,
                        "mean_of_matrix_mean_abs_error": 0.0,
                    },
                    "precision_evidence": [
                        {
                            "original_matrix_sha256": str(index + 5) * 64,
                            "submission_matrix_sha256": str(index) * 64,
                            "source_type": "numpy.ndarray",
                            "source_dtype": "float32",
                            "normalized_dtype": "torch.float64",
                            "normalized_min": -127.0,
                            "normalized_max": 127.0,
                            "symmetry_normalization": "arithmetic_mean",
                            "rounding_policy": "round_half_to_even",
                            "scale_factor": 1.0,
                            "target_min": -127,
                            "target_max": 127,
                            "max_abs_error": 0.0,
                            "mean_abs_error": 0.0,
                        }
                    ],
                    "precision_evidence_complete": True,
                    "task_receipts": [
                        {
                            "schema": "flagquantum.kaiwu-task.v1",
                            "task_name": f"protein-seed-{seed}",
                            "matrix_sha256": str(index) * 64,
                            "matrix_size": 3,
                            "mode": "sampling",
                            "requested_samples": 10,
                            "project_no": "CPQC-test",
                            "submitted_at": "2026-10-06T00:00:00+00:00",
                            "provider_task_id": f"protein-task-{seed}",
                            "provider_target": "SPQC-provider",
                        }
                    ],
                    "source_preflight_sha256": source_preflight_sha256,
                    "transfer_manifest_sha256": transfer_manifest_sha256,
                    "environment_lock_sha256": ENVIRONMENT_LOCK_SHA,
                    "experiment_config_sha256": config_sha256,
                    "artifact_preflight_sha256": "c" * 64,
                    "run_directory_name": f"seed-{seed}-run",
                    "trained_energy_checkpoint_name": "best_epoch_1.pt",
                    "trained_energy_checkpoint_sha256": str(index + 3) * 64,
                    "acceptance": {
                        "system": "not_evaluated",
                        "application": "not_evaluated",
                    },
                    "workflow_artifacts": {
                        "test_fasta": {
                            "relative_path": "data_splits/test.fasta",
                            "sha256": "1" * 64,
                        },
                        "baseline_fasta": {
                            "relative_path": "baseline/proposal_only_generated_sequences.fasta",
                            "sha256": "2" * 64,
                        },
                        "guided_fasta": {
                            "relative_path": "guided/energy_guided_generated_sequences.fasta",
                            "sha256": "3" * 64,
                        },
                        "training_history": {
                            "relative_path": "history.json",
                            "sha256": "4" * 64,
                        },
                        "sequence_metrics": {
                            "relative_path": "baseline_vs_guided.json",
                            "sha256": "5" * 64,
                        },
                        "baseline_quality": {
                            "relative_path": "baseline_eval/quality_summary.json",
                            "sha256": "6" * 64,
                        },
                        "guided_quality": {
                            "relative_path": "guided_eval/quality_summary.json",
                            "sha256": "7" * 64,
                        },
                    },
                    "limitations": [
                        "This record covers one protein-training seed only.",
                        "ESM2 evaluation and two-host system acceptance are separate gates.",
                        "No performance, distributed, domestic-accelerator, or quantum-advantage claim is made.",
                    ],
                },
                training_sha,
            )
        )
        evaluations.append(
            (
                {
                    "schema": "flagquantum.qboson_qdiffusion_protein_evaluation",
                    "version": "1.0",
                    "recorded_at": "2026-10-06T00:00:00+00:00",
                    "seed": seed,
                    "training_record_sha256": training_sha,
                    "experiment_config_sha256": config_sha256,
                    "source_revision": "a" * 40,
                    "flagquantum_version": "0.2.0",
                    "kaiwu_pytorch_plugin_revision": "b" * 40,
                    "python_version": "3.10.18",
                    "torch_version": "2.7.0",
                    "kaiwu_sdk_version": "1.3.1",
                    "execution_host": "jp-a800-171",
                    "observed_hostname": OBSERVED_HOSTNAMES["jp-a800-171"],
                    "observed_gpu_model": "NVIDIA A800-SXM4-80GB",
                    "observed_tensor_device": "cuda:0",
                    "evaluation_model_sha256": "f" * 64,
                    "artifact_inputs_unchanged": True,
                    "provider_quota_consumed": False,
                    "secrets_redacted": True,
                    "acceptance": "candidate_evidence_only",
                    "source_preflight_sha256": source_preflight_sha256,
                    "transfer_manifest_sha256": transfer_manifest_sha256,
                    "environment_lock_sha256": ENVIRONMENT_LOCK_SHA,
                    "baseline_metrics": _metrics(0.6 + index * 0.01),
                    "guided_metrics": _metrics(0.4 + index * 0.01),
                },
                str(index + 6) * 64,
            )
        )
    return training, evaluations


def _portability(
    training: tuple[dict[str, Any], str],
    *,
    config_sha256: str = "9" * 64,
    source_preflight_sha256: str = REPLAY_SOURCE_PREFLIGHT_SHA,
    transfer_manifest_sha256: str = TRANSFER_MANIFEST_SHA,
    environment_lock_sha256: str = ENVIRONMENT_LOCK_SHA,
    provider_resources_sha256: str = "c" * 64,
) -> dict[str, Any]:
    return {
        "schema": "flagquantum.qboson_qdiffusion_portability_replay",
        "version": "1.0",
        "recorded_at": "2026-10-06T00:00:00+00:00",
        "experiment_config_sha256": config_sha256,
        "execution_host": "jp-a800-172",
        "observed_hostname": OBSERVED_HOSTNAMES["jp-a800-172"],
        "run_completed": True,
        "failure": None,
        "artifact_inputs_unchanged": True,
        "requested_cuda_device": "cuda:0",
        "observed_tensor_device": "cuda:0",
        "observed_gpu_model": "NVIDIA A800-SXM4-80GB",
        "source_revision": "a" * 40,
        "flagquantum_version": "0.2.0",
        "kaiwu_pytorch_plugin_revision": "b" * 40,
        "python_version": "3.10.18",
        "torch_version": "2.7.0",
        "kaiwu_sdk_version": "1.3.1",
        "provider_resource_gate": {
            "snapshot_sha256": provider_resources_sha256,
            "checked_at": "2026-10-05T12:00:00+00:00",
            "mode": "sampling",
            "required_calls": 128,
        },
        "transport": "kaiwu_cim",
        "pinned_sdk_client": True,
        "qboson_hardware_used": True,
        "real_provider_evidence": True,
        "provider_identity_complete": True,
        "provider_reported_target": True,
        "qboson_target": "SPQC-provider",
        "qboson_task_ids": ["portability-task"],
        "task_receipts": [
            {
                "schema": "flagquantum.kaiwu-task.v1",
                "task_name": "portability-seed-1701",
                "matrix_sha256": "8" * 64,
                "matrix_size": 3,
                "mode": "sampling",
                "requested_samples": 10,
                "project_no": "CPQC-test",
                "submitted_at": "2026-10-06T00:00:00+00:00",
                "provider_task_id": "portability-task",
                "provider_target": "SPQC-provider",
            }
        ],
        "sampling_mode": "sampling",
        "requested_samples": 10,
        "returned_samples": 10,
        "remote_call_budget": 128,
        "remote_call_count": 1,
        "precision_policy": {
            "name": "explicit-int8",
            "target_min": -127,
            "target_max": 127,
            "matrix_count": 1,
            "scale_factor_min": 1.0,
            "scale_factor_max": 1.0,
            "max_abs_error": 0.0,
            "mean_of_matrix_mean_abs_error": 0.0,
        },
        "precision_evidence": [
            {
                "original_matrix_sha256": "9" * 64,
                "submission_matrix_sha256": "8" * 64,
                "source_type": "numpy.ndarray",
                "source_dtype": "float32",
                "normalized_dtype": "torch.float64",
                "normalized_min": -127.0,
                "normalized_max": 127.0,
                "symmetry_normalization": "arithmetic_mean",
                "rounding_policy": "round_half_to_even",
                "scale_factor": 1.0,
                "target_min": -127,
                "target_max": 127,
                "max_abs_error": 0.0,
                "mean_abs_error": 0.0,
            }
        ],
        "precision_evidence_complete": True,
        "fallback_occurred": False,
        "retrieval_resubmitted": False,
        "secrets_redacted": True,
        "artifacts": {
            "dataset_sha256": "c" * 64,
            "base_checkpoint_sha256": "d" * 64,
            "tokenizer_sha256": "e" * 64,
            "evaluation_model_sha256": "f" * 64,
            "trained_energy_checkpoint_sha256": training[0][
                "trained_energy_checkpoint_sha256"
            ],
        },
        "fixture": {
            "training_seed": 1701,
            "index": 0,
            "steps": 3,
            "energy_objective": -0.5,
            "generated_length": 8,
            "generated_sha256": "a" * 64,
            "token_constraints_passed": True,
        },
        "acceptance": {"portability": "pass"},
        "training_record_sha256": training[1],
        "trained_energy_checkpoint_sha256": training[0][
            "trained_energy_checkpoint_sha256"
        ],
        "source_preflight_sha256": source_preflight_sha256,
        "transfer_manifest_sha256": transfer_manifest_sha256,
        "environment_lock_sha256": environment_lock_sha256,
        "limitations": [
            "This is one fixed replay fixture, not a second training run.",
            "Final acceptance also requires both system gates and all primary-host seeds.",
            "No performance, distributed, domestic-accelerator, or quantum-advantage claim is made.",
        ],
    }


def _write_json(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    path.chmod(0o600)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_component_loader_requires_private_regular_file(tmp_path: Path) -> None:
    path = tmp_path / "component.json"
    _write_json(
        path,
        {"schema": "test.schema", "version": "1.0"},
    )
    path.chmod(0o644)
    with pytest.raises(ValueError, match="group or others"):
        _load_component(path, "test.schema")

    path.chmod(0o600)
    link = tmp_path / "component-link.json"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="non-symlink"):
        _load_component(link, "test.schema")


@pytest.mark.parametrize("unsafe_kind", ("public", "symlink"))
def test_component_loader_requires_private_real_parent(
    tmp_path: Path, unsafe_kind: str
) -> None:
    private_parent = tmp_path / "private-parent"
    private_parent.mkdir(mode=0o700)
    component = private_parent / "component.json"
    _write_json(component, {"schema": "test.schema", "version": "1.0"})
    if unsafe_kind == "public":
        private_parent.chmod(0o755)
        path = component
    else:
        linked_parent = tmp_path / "linked-parent"
        linked_parent.symlink_to(private_parent, target_is_directory=True)
        path = linked_parent / component.name

    with pytest.raises(ValueError, match="parent must be an existing private"):
        _load_component(path, "test.schema")


def test_bundle_is_published_only_after_final_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config.json"
    component_path = tmp_path / "component.json"
    environment_lock_path = tmp_path / "environment-lock.json"
    _write_json(config_path, {"schema": "test.config", "version": "1.0"})
    _write_json(component_path, {"schema": "test.component", "version": "1.0"})
    _write_json(environment_lock_path, {"schema": "test.environment"})
    destination = tmp_path / "acceptance"
    arguments = {
        "config_path": config_path,
        "environment_lock_path": environment_lock_path,
        "config": {
            "primary_host": "jp-a800-171",
            "replay_host": "jp-a800-172",
        },
        "primary": {},
        "replay": {},
        "component_sources": {"component.json": component_path},
    }
    monkeypatch.setattr(
        "examples.qdiffusion_kaiwu.assemble_acceptance.validate_acceptance",
        lambda _: ["forced validation failure"],
    )

    with pytest.raises(RuntimeError, match="forced validation failure"):
        _publish_acceptance_bundle(destination, **arguments)

    assert not destination.exists()
    assert list(tmp_path.glob(".acceptance.staging-*")) == []

    monkeypatch.setattr(
        "examples.qdiffusion_kaiwu.assemble_acceptance.validate_acceptance",
        lambda _: [],
    )
    _publish_acceptance_bundle(destination, **arguments)

    assert (destination / "manifest.json").is_file()
    assert destination.stat().st_mode & 0o077 == 0


def test_bundle_publisher_rejects_foreign_owned_output_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    effective_uid = os.geteuid()
    monkeypatch.setattr(assembler_module.os, "geteuid", lambda: effective_uid + 1)

    with pytest.raises(ValueError, match="output parent must be an existing private"):
        _open_output_parent(tmp_path)


def test_bundle_publisher_does_not_replace_dangling_destination_symlink(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.json"
    environment_lock_path = tmp_path / "environment-lock.json"
    component_path = tmp_path / "component.json"
    _write_json(config_path, {"schema": "test.config", "version": "1.0"})
    _write_json(environment_lock_path, {"schema": "test.environment"})
    _write_json(component_path, {"schema": "test.component", "version": "1.0"})
    destination = tmp_path / "acceptance"
    destination.symlink_to(tmp_path / "missing-target", target_is_directory=True)

    with pytest.raises(RuntimeError, match="evidence-dir already exists"):
        _publish_acceptance_bundle(
            destination,
            config_path=config_path,
            environment_lock_path=environment_lock_path,
            config={
                "primary_host": "jp-a800-171",
                "replay_host": "jp-a800-172",
            },
            primary={},
            replay={},
            component_sources={"component.json": component_path},
        )

    assert destination.is_symlink()


def test_bundle_publisher_syncs_files_staging_and_output_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config.json"
    environment_lock_path = tmp_path / "environment-lock.json"
    component_path = tmp_path / "component.json"
    _write_json(config_path, {"schema": "test.config", "version": "1.0"})
    _write_json(environment_lock_path, {"schema": "test.environment"})
    _write_json(component_path, {"schema": "test.component", "version": "1.0"})
    destination = tmp_path / "acceptance"
    synced_types: list[int] = []
    real_fsync = os.fsync

    def record_fsync(descriptor: int) -> None:
        synced_types.append(os.fstat(descriptor).st_mode)
        real_fsync(descriptor)

    monkeypatch.setattr(
        "examples.qdiffusion_kaiwu.assemble_acceptance.validate_acceptance",
        lambda _: [],
    )
    monkeypatch.setattr(
        "examples.qdiffusion_kaiwu.assemble_acceptance.os.fsync", record_fsync
    )

    _publish_acceptance_bundle(
        destination,
        config_path=config_path,
        environment_lock_path=environment_lock_path,
        config={
            "primary_host": "jp-a800-171",
            "replay_host": "jp-a800-172",
        },
        primary={},
        replay={},
        component_sources={"component.json": component_path},
    )

    assert any(stat.S_ISREG(mode) for mode in synced_types)
    assert sum(stat.S_ISDIR(mode) for mode in synced_types) >= 3
    assert (destination / "manifest.json").is_file()


def test_bundle_publisher_detects_dangling_symlink_created_during_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config.json"
    environment_lock_path = tmp_path / "environment-lock.json"
    component_path = tmp_path / "component.json"
    _write_json(config_path, {"schema": "test.config", "version": "1.0"})
    _write_json(environment_lock_path, {"schema": "test.environment"})
    _write_json(component_path, {"schema": "test.component", "version": "1.0"})
    destination = tmp_path / "acceptance"

    def create_destination(_: Path) -> list[str]:
        destination.symlink_to(tmp_path / "missing-target", target_is_directory=True)
        return []

    monkeypatch.setattr(
        "examples.qdiffusion_kaiwu.assemble_acceptance.validate_acceptance",
        create_destination,
    )

    with pytest.raises(RuntimeError, match="appeared during final validation"):
        _publish_acceptance_bundle(
            destination,
            config_path=config_path,
            environment_lock_path=environment_lock_path,
            config={
                "primary_host": "jp-a800-171",
                "replay_host": "jp-a800-172",
            },
            primary={},
            replay={},
            component_sources={"component.json": component_path},
        )

    assert destination.is_symlink()
    assert list(tmp_path.glob(".acceptance.staging-*")) == []


def test_bundle_publisher_rejects_output_parent_replacement_during_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config.json"
    environment_lock_path = tmp_path / "environment-lock.json"
    component_path = tmp_path / "component.json"
    _write_json(config_path, {"schema": "test.config", "version": "1.0"})
    _write_json(environment_lock_path, {"schema": "test.environment"})
    _write_json(component_path, {"schema": "test.component", "version": "1.0"})
    output_parent = tmp_path / "private-output"
    output_parent.mkdir(mode=0o700)
    moved_parent = tmp_path / "moved-output"
    destination = output_parent / "acceptance"

    def replace_parent(manifest_path: Path) -> list[str]:
        staging_name = manifest_path.parent.name
        output_parent.rename(moved_parent)
        output_parent.mkdir(mode=0o700)
        attacker_staging = output_parent / staging_name
        attacker_staging.mkdir(mode=0o700)
        (attacker_staging / "attacker.json").write_text("{}", encoding="utf-8")
        return []

    monkeypatch.setattr(
        "examples.qdiffusion_kaiwu.assemble_acceptance.validate_acceptance",
        replace_parent,
    )

    with pytest.raises(ValueError, match="output parent changed"):
        _publish_acceptance_bundle(
            destination,
            config_path=config_path,
            environment_lock_path=environment_lock_path,
            config={
                "primary_host": "jp-a800-171",
                "replay_host": "jp-a800-172",
            },
            primary={},
            replay={},
            component_sources={"component.json": component_path},
        )

    assert not destination.exists()
    assert not (output_parent / "acceptance").exists()


def test_bundle_publisher_anchors_final_rename_to_open_output_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config.json"
    environment_lock_path = tmp_path / "environment-lock.json"
    component_path = tmp_path / "component.json"
    _write_json(config_path, {"schema": "test.config", "version": "1.0"})
    _write_json(environment_lock_path, {"schema": "test.environment"})
    _write_json(component_path, {"schema": "test.component", "version": "1.0"})
    output_parent = tmp_path / "private-output"
    output_parent.mkdir(mode=0o700)
    moved_parent = tmp_path / "moved-output"
    destination = output_parent / "acceptance"
    real_replace = os.replace

    def swap_then_replace(
        source: str,
        target: str,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        assert src_dir_fd is not None
        assert dst_dir_fd == src_dir_fd
        output_parent.rename(moved_parent)
        output_parent.mkdir(mode=0o700)
        real_replace(
            source,
            target,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    monkeypatch.setattr(
        "examples.qdiffusion_kaiwu.assemble_acceptance.validate_acceptance",
        lambda _: [],
    )
    monkeypatch.setattr(
        "examples.qdiffusion_kaiwu.assemble_acceptance.os.replace",
        swap_then_replace,
    )

    with pytest.raises(ValueError, match="output parent changed"):
        _publish_acceptance_bundle(
            destination,
            config_path=config_path,
            environment_lock_path=environment_lock_path,
            config={
                "primary_host": "jp-a800-171",
                "replay_host": "jp-a800-172",
            },
            primary={},
            replay={},
            component_sources={"component.json": component_path},
        )

    assert not destination.exists()
    assert not (output_parent / "acceptance").exists()
    assert (moved_parent / "acceptance" / "manifest.json").is_file()


def test_bundle_publisher_revalidates_component_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config.json"
    environment_lock_path = tmp_path / "environment-lock.json"
    _write_json(config_path, {"schema": "test.config", "version": "1.0"})
    _write_json(environment_lock_path, {"schema": "test.environment"})
    public_parent = tmp_path / "public-components"
    public_parent.mkdir(mode=0o755)
    public_parent.chmod(0o755)
    component_path = public_parent / "component.json"
    _write_json(component_path, {"schema": "test.component", "version": "1.0"})
    destination = tmp_path / "acceptance"
    monkeypatch.setattr(
        "examples.qdiffusion_kaiwu.assemble_acceptance.validate_acceptance",
        lambda _: [],
    )

    with pytest.raises(ValueError, match="parent must be an existing private"):
        _publish_acceptance_bundle(
            destination,
            config_path=config_path,
            environment_lock_path=environment_lock_path,
            config={
                "primary_host": "jp-a800-171",
                "replay_host": "jp-a800-172",
            },
            primary={},
            replay={},
            component_sources={"component.json": component_path},
        )

    assert not destination.exists()


def test_bundle_publisher_rejects_inputs_changed_after_assembly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config.json"
    environment_lock_path = tmp_path / "environment-lock.json"
    component_path = tmp_path / "component.json"
    config_sha = _write_json(config_path, {"schema": "test.config"})
    _write_json(environment_lock_path, {"schema": "test.environment"})
    component_sha = _write_json(component_path, {"schema": "test.component"})
    monkeypatch.setattr(
        "examples.qdiffusion_kaiwu.assemble_acceptance.validate_acceptance",
        lambda _: [],
    )

    _write_json(component_path, {"schema": "changed.component"})
    with pytest.raises(ValueError, match="component record changed after assembly"):
        _publish_acceptance_bundle(
            tmp_path / "acceptance",
            config_path=config_path,
            environment_lock_path=environment_lock_path,
            config={
                "primary_host": "jp-a800-171",
                "replay_host": "jp-a800-172",
            },
            primary={},
            replay={},
            component_sources={"component.json": component_path},
            expected_config_sha256=config_sha,
            expected_component_sha256={"component.json": component_sha},
        )

    assert not (tmp_path / "acceptance").exists()


@pytest.mark.parametrize("unsafe_kind", ("public", "symlink"))
def test_bundle_publisher_requires_private_real_output_parent(
    tmp_path: Path, unsafe_kind: str
) -> None:
    config_path = tmp_path / "config.json"
    environment_lock_path = tmp_path / "environment-lock.json"
    component_path = tmp_path / "component.json"
    _write_json(config_path, {"schema": "test.config", "version": "1.0"})
    _write_json(environment_lock_path, {"schema": "test.environment"})
    _write_json(component_path, {"schema": "test.component", "version": "1.0"})
    target = tmp_path / "target"
    target.mkdir(mode=0o700)
    if unsafe_kind == "public":
        target.chmod(0o755)
        output_parent = target
    else:
        output_parent = tmp_path / "linked-target"
        output_parent.symlink_to(target, target_is_directory=True)
    destination = output_parent / "acceptance"

    with pytest.raises(ValueError, match="output parent must be an existing private"):
        _publish_acceptance_bundle(
            destination,
            config_path=config_path,
            environment_lock_path=environment_lock_path,
            config={
                "primary_host": "jp-a800-171",
                "replay_host": "jp-a800-172",
            },
            primary={},
            replay={},
            component_sources={"component.json": component_path},
        )

    assert not destination.exists()


def test_assembler_links_all_seeds_and_recomputes_metric_means() -> None:
    config = _config()
    training, evaluations = _components()
    selected_checkpoint = training[0][0]["trained_energy_checkpoint_sha256"]
    portability = _portability(training[0])

    primary, replay = assemble_records(
        config=config,
        config_sha256="9" * 64,
        primary_system=(_system("jp-a800-171", "primary", "primary-task"), "a" * 64),
        replay_system=(
            _system("jp-a800-172", "portability_replay", "replay-task"),
            "b" * 64,
        ),
        primary_source_preflight=(
            _source_preflight("jp-a800-171"),
            PRIMARY_SOURCE_PREFLIGHT_SHA,
        ),
        replay_source_preflight=(
            _source_preflight("jp-a800-172"),
            REPLAY_SOURCE_PREFLIGHT_SHA,
        ),
        transfer_manifest=(_transfer_manifest(), TRANSFER_MANIFEST_SHA),
        portability=(portability, "c" * 64),
        training_records=training,
        evaluation_records=evaluations,
    )

    assert primary["attempted_seeds"] == [1701, 1702, 1703]
    assert primary["flagquantum_version"] == "0.2.0"
    assert replay["flagquantum_version"] == "0.2.0"
    assert primary["baseline_metrics"]["mean_cosine_distance"] == pytest.approx(0.62)
    assert primary["guided_metrics"]["mean_cosine_distance"] == pytest.approx(0.42)
    assert len(primary["application_evidence"]["records"]) == 3
    assert replay["portability_evidence"]["trained_energy_checkpoint_sha256"] == (
        selected_checkpoint
    )
    assert replay["artifacts"]["trained_energy_checkpoint_sha256"] == (
        selected_checkpoint
    )


def test_system_component_requires_its_precision_completeness_flag() -> None:
    record = _system("jp-a800-171", "primary", "primary-task")
    record["precision_evidence_complete"] = False
    errors: list[str] = []

    _validate_remote_sampling_component_evidence(record, "system", errors)

    assert (
        "system: remote component precision_evidence_complete is not proven" in errors
    )


def test_assembler_rejects_evaluation_linked_to_another_training_record() -> None:
    config = _config()
    training, evaluations = _components()
    evaluations[1][0]["training_record_sha256"] = "0" * 64
    portability = _portability(training[0])

    with pytest.raises(ValueError, match="not linked to its training record"):
        assemble_records(
            config=config,
            config_sha256="9" * 64,
            primary_system=(
                _system("jp-a800-171", "primary", "primary-task"),
                "a" * 64,
            ),
            replay_system=(
                _system("jp-a800-172", "portability_replay", "replay-task"),
                "b" * 64,
            ),
            primary_source_preflight=(
                _source_preflight("jp-a800-171"),
                PRIMARY_SOURCE_PREFLIGHT_SHA,
            ),
            replay_source_preflight=(
                _source_preflight("jp-a800-172"),
                REPLAY_SOURCE_PREFLIGHT_SHA,
            ),
            transfer_manifest=(_transfer_manifest(), TRANSFER_MANIFEST_SHA),
            portability=(portability, "c" * 64),
            training_records=training,
            evaluation_records=evaluations,
        )


def test_assembler_rejects_training_without_real_provider_evidence() -> None:
    config = _config()
    training, evaluations = _components()
    training[0][0]["real_provider_evidence"] = False
    portability = _portability(training[0])

    with pytest.raises(ValueError, match="real_provider_evidence is not proven"):
        assemble_records(
            config=config,
            config_sha256="9" * 64,
            primary_system=(
                _system("jp-a800-171", "primary", "primary-task"),
                "a" * 64,
            ),
            replay_system=(
                _system("jp-a800-172", "portability_replay", "replay-task"),
                "b" * 64,
            ),
            primary_source_preflight=(
                _source_preflight("jp-a800-171"),
                PRIMARY_SOURCE_PREFLIGHT_SHA,
            ),
            replay_source_preflight=(
                _source_preflight("jp-a800-172"),
                REPLAY_SOURCE_PREFLIGHT_SHA,
            ),
            transfer_manifest=(_transfer_manifest(), TRANSFER_MANIFEST_SHA),
            portability=(portability, "c" * 64),
            training_records=training,
            evaluation_records=evaluations,
        )


def test_assembler_rejects_different_host_transfer_manifests() -> None:
    config = _config()
    training, evaluations = _components()
    portability = _portability(training[0], transfer_manifest_sha256="0" * 64)
    replay_system = _system("jp-a800-172", "portability_replay", "replay-task")
    replay_system["transfer_manifest_sha256"] = "0" * 64
    replay_preflight = _source_preflight("jp-a800-172")
    replay_preflight["manifest_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="share one transfer manifest"):
        assemble_records(
            config=config,
            config_sha256="9" * 64,
            primary_system=(
                _system("jp-a800-171", "primary", "primary-task"),
                "a" * 64,
            ),
            replay_system=(replay_system, "b" * 64),
            primary_source_preflight=(
                _source_preflight("jp-a800-171"),
                PRIMARY_SOURCE_PREFLIGHT_SHA,
            ),
            replay_source_preflight=(
                replay_preflight,
                REPLAY_SOURCE_PREFLIGHT_SHA,
            ),
            transfer_manifest=(_transfer_manifest(), TRANSFER_MANIFEST_SHA),
            portability=(portability, "c" * 64),
            training_records=training,
            evaluation_records=evaluations,
        )


def test_assembled_component_bundle_passes_final_validator(tmp_path: Path) -> None:
    environment_lock_path = tmp_path / "environment_lock.json"
    environment_lock_sha = _write_json(
        environment_lock_path,
        {
            "schema": "flagquantum.qboson_qdiffusion_environment_lock",
            "version": "1.0",
            "inventory_policy": "exact",
            "python_version": "3.10.18",
            "distributions": [
                {
                    "name": "kaiwu",
                    "version": "1.3.1",
                    "approved_artifact_sha256": "7334cabd4ff0ae02e042d1c38ed292211573e83e2ed8e92fdf41af52e8991455",
                    "installed_content_sha256": "c" * 64,
                },
                {
                    "name": "torch",
                    "version": "2.7.0",
                    "approved_artifact_sha256": "a" * 64,
                    "installed_content_sha256": "b" * 64,
                },
            ],
        },
    )
    config = _config()
    config["software"]["environment_lock_sha256"] = environment_lock_sha
    config_path = tmp_path / "acceptance_config.json"
    config_sha = _write_json(config_path, config)

    artifact_preflight_path = tmp_path / "components" / "artifact-preflight.json"
    artifact_preflight_record = _artifact_preflight(config, config_sha)
    artifact_preflight_sha = _write_json(
        artifact_preflight_path, artifact_preflight_record
    )

    components = tmp_path / "components"
    components.mkdir(mode=0o700, exist_ok=True)
    components.chmod(0o700)
    sdk_approval_path = components / "sdk-approval.json"
    sdk_approval_record = config["kaiwu_sdk"]
    sdk_approval_sha = _write_json(sdk_approval_path, sdk_approval_record)
    provider_resources_path = components / "provider-resources.json"
    provider_resources_record = _provider_resources()
    provider_resources_sha = _write_json(
        provider_resources_path, provider_resources_record
    )
    provider_smoke_path = components / "provider-smoke.json"
    provider_smoke_record = _provider_smoke(
        environment_lock_sha256=environment_lock_sha,
        sdk_approval_sha256=sdk_approval_sha,
        provider_resources_sha256=provider_resources_sha,
    )
    _write_json(provider_smoke_path, provider_smoke_record)
    transfer_manifest_path = components / "transfer-manifest.json"
    transfer_manifest_record = _transfer_manifest()
    transfer_manifest_sha = _write_json(
        transfer_manifest_path, transfer_manifest_record
    )

    primary_preflight_path = tmp_path / "components" / "primary-source-preflight.json"
    replay_preflight_path = tmp_path / "components" / "replay-source-preflight.json"
    primary_preflight_record = _source_preflight("jp-a800-171", transfer_manifest_sha)
    replay_preflight_record = _source_preflight("jp-a800-172", transfer_manifest_sha)
    primary_preflight_sha = _write_json(
        primary_preflight_path, primary_preflight_record
    )
    replay_preflight_sha = _write_json(replay_preflight_path, replay_preflight_record)

    primary_system_path = tmp_path / "components" / "primary-system.json"
    replay_system_path = tmp_path / "components" / "replay-system.json"
    primary_system_record = _system(
        "jp-a800-171",
        "primary",
        "primary-task",
        transfer_manifest_sha,
        provider_resources_sha,
    )
    replay_system_record = _system(
        "jp-a800-172",
        "portability_replay",
        "replay-task",
        transfer_manifest_sha,
        provider_resources_sha,
    )
    primary_system_record["source_preflight_sha256"] = primary_preflight_sha
    replay_system_record["source_preflight_sha256"] = replay_preflight_sha
    primary_system_record["environment_lock_sha256"] = environment_lock_sha
    replay_system_record["environment_lock_sha256"] = environment_lock_sha
    primary_system_record["experiment_config_sha256"] = config_sha
    replay_system_record["experiment_config_sha256"] = config_sha
    primary_system_sha = _write_json(primary_system_path, primary_system_record)
    replay_system_sha = _write_json(replay_system_path, replay_system_record)

    training_templates, evaluation_templates = _components(
        config_sha,
        primary_preflight_sha,
        transfer_manifest_sha,
        provider_resources_sha,
    )
    training_entries: list[tuple[dict[str, Any], str]] = []
    evaluation_entries: list[tuple[dict[str, Any], str]] = []
    component_paths = [
        primary_system_path,
        replay_system_path,
        primary_preflight_path,
        replay_preflight_path,
        transfer_manifest_path,
    ]
    for index, (training_record, _) in enumerate(training_templates):
        training_record["environment_lock_sha256"] = environment_lock_sha
        training_record["artifact_preflight_sha256"] = artifact_preflight_sha
        path = tmp_path / "components" / f"training-{index}.json"
        digest = _write_json(path, training_record)
        training_entries.append((training_record, digest))
        component_paths.append(path)
    for index, (evaluation_record, _) in enumerate(evaluation_templates):
        evaluation_record["environment_lock_sha256"] = environment_lock_sha
        evaluation_record["training_record_sha256"] = training_entries[index][1]
        path = tmp_path / "components" / f"evaluation-{index}.json"
        digest = _write_json(path, evaluation_record)
        evaluation_entries.append((evaluation_record, digest))
        component_paths.append(path)

    portability_record = _portability(
        training_entries[0],
        config_sha256=config_sha,
        source_preflight_sha256=replay_preflight_sha,
        transfer_manifest_sha256=transfer_manifest_sha,
        environment_lock_sha256=environment_lock_sha,
        provider_resources_sha256=provider_resources_sha,
    )
    portability_record["artifact_preflight_sha256"] = artifact_preflight_sha
    portability_path = tmp_path / "components" / "portability.json"
    portability_sha = _write_json(portability_path, portability_record)
    component_paths.append(portability_path)
    component_paths.append(artifact_preflight_path)
    component_paths.append(sdk_approval_path)
    component_paths.append(provider_smoke_path)
    component_paths.append(provider_resources_path)

    primary, replay = assemble_records(
        config=config,
        config_sha256=config_sha,
        primary_system=(primary_system_record, primary_system_sha),
        replay_system=(replay_system_record, replay_system_sha),
        primary_source_preflight=(
            primary_preflight_record,
            primary_preflight_sha,
        ),
        replay_source_preflight=(replay_preflight_record, replay_preflight_sha),
        transfer_manifest=(transfer_manifest_record, transfer_manifest_sha),
        portability=(portability_record, portability_sha),
        training_records=training_entries,
        evaluation_records=evaluation_entries,
    )
    primary_path = tmp_path / "jp-a800-171.json"
    replay_path = tmp_path / "jp-a800-172.json"
    primary_sha = _write_json(primary_path, primary)
    replay_sha = _write_json(replay_path, replay)
    manifest = {
        "schema": "flagquantum.qboson_qdiffusion_manifest",
        "version": "1.0",
        "config": {"path": config_path.name, "sha256": config_sha},
        "environment_lock": {
            "path": environment_lock_path.name,
            "sha256": environment_lock_sha,
        },
        "records": [
            {"path": primary_path.name, "sha256": primary_sha},
            {"path": replay_path.name, "sha256": replay_sha},
        ],
        "component_records": [
            {
                "path": str(path.relative_to(tmp_path)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in component_paths
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    _write_json(manifest_path, manifest)

    assert validate_acceptance(manifest_path) == []

    tampered_smoke_project = json.loads(json.dumps(provider_smoke_record))
    tampered_smoke_project["project_no"] = "CPQC-other"
    tampered_smoke_project_sha = _write_json(
        component_paths[14], tampered_smoke_project
    )
    manifest["component_records"][14]["sha256"] = tampered_smoke_project_sha
    _write_json(manifest_path, manifest)
    assert any(
        "provider smoke: project number differs from reviewed assignment" in error
        for error in validate_acceptance(manifest_path)
    )
    restored_smoke_sha = _write_json(component_paths[14], provider_smoke_record)
    manifest["component_records"][14]["sha256"] = restored_smoke_sha
    _write_json(manifest_path, manifest)
    assert validate_acceptance(manifest_path) == []

    tampered_system_project = json.loads(json.dumps(primary_system_record))
    tampered_system_project["task_receipts"][0]["project_no"] = "CPQC-other"
    tampered_system_project_sha = _write_json(
        component_paths[0], tampered_system_project
    )
    manifest["component_records"][0]["sha256"] = tampered_system_project_sha
    _write_json(manifest_path, manifest)
    assert any(
        "project number differs from reviewed assignment" in error
        for error in validate_acceptance(manifest_path)
    )
    restored_system_sha = _write_json(component_paths[0], primary_system_record)
    manifest["component_records"][0]["sha256"] = restored_system_sha
    _write_json(manifest_path, manifest)
    assert validate_acceptance(manifest_path) == []

    tampered_artifact_preflight = json.loads(
        component_paths[12].read_text(encoding="utf-8")
    )
    tampered_artifact_preflight["artifacts"]["dataset"]["sha256"] = "0" * 64
    tampered_artifact_preflight_sha = _write_json(
        component_paths[12], tampered_artifact_preflight
    )
    manifest["component_records"][12]["sha256"] = tampered_artifact_preflight_sha
    _write_json(manifest_path, manifest)
    assert any(
        "artifact preflight: dataset identity differs from config" in error
        for error in validate_acceptance(manifest_path)
    )

    restored_artifact_preflight_sha = _write_json(
        component_paths[12], artifact_preflight_record
    )
    manifest["component_records"][12]["sha256"] = restored_artifact_preflight_sha
    _write_json(manifest_path, manifest)
    assert validate_acceptance(manifest_path) == []

    tampered_smoke = dict(provider_smoke_record)
    tampered_smoke["hardware_acceptance"] = False
    tampered_smoke_sha = _write_json(component_paths[14], tampered_smoke)
    manifest["component_records"][14]["sha256"] = tampered_smoke_sha
    _write_json(manifest_path, manifest)
    assert any(
        "provider smoke: hardware_acceptance is not proven" in error
        for error in validate_acceptance(manifest_path)
    )

    tampered_smoke = json.loads(json.dumps(provider_smoke_record))
    tampered_smoke["tasks"][0]["matrix_sha256"] = "0" * 64
    tampered_smoke_sha = _write_json(component_paths[14], tampered_smoke)
    manifest["component_records"][14]["sha256"] = tampered_smoke_sha
    _write_json(manifest_path, manifest)
    assert any(
        "provider smoke: task 0 matrix identity differs" in error
        for error in validate_acceptance(manifest_path)
    )

    tampered_smoke = json.loads(json.dumps(provider_smoke_record))
    tampered_smoke["tasks"][0]["energies"][0] = 99.0
    tampered_smoke["tasks"][0]["maximum_energy"] = 99.0
    tampered_smoke_sha = _write_json(component_paths[14], tampered_smoke)
    manifest["component_records"][14]["sha256"] = tampered_smoke_sha
    _write_json(manifest_path, manifest)
    assert any(
        "provider smoke: task 0 sample 0 energy differs" in error
        for error in validate_acceptance(manifest_path)
    )

    tampered_smoke = json.loads(json.dumps(provider_smoke_record))
    tampered_smoke["unexpected_secret_field"] = "not-accepted"
    tampered_smoke_sha = _write_json(component_paths[14], tampered_smoke)
    manifest["component_records"][14]["sha256"] = tampered_smoke_sha
    _write_json(manifest_path, manifest)
    assert any(
        "provider smoke: field set is incomplete or contains extensions" in error
        for error in validate_acceptance(manifest_path)
    )

    tampered_smoke = json.loads(json.dumps(provider_smoke_record))
    tampered_smoke["qboson_target"] = "SPQC-other"
    tampered_smoke_sha = _write_json(component_paths[14], tampered_smoke)
    manifest["component_records"][14]["sha256"] = tampered_smoke_sha
    _write_json(manifest_path, manifest)
    assert any(
        "provider smoke: top-level provider target differs from tasks" in error
        for error in validate_acceptance(manifest_path)
    )

    tampered_smoke = json.loads(json.dumps(provider_smoke_record))
    tampered_smoke["tasks"][0]["provider_task_id"] = "primary-task"
    tampered_smoke_sha = _write_json(component_paths[14], tampered_smoke)
    manifest["component_records"][14]["sha256"] = tampered_smoke_sha
    _write_json(manifest_path, manifest)
    assert any(
        "provider task identities are reused across remote components" in error
        for error in validate_acceptance(manifest_path)
    )

    restored_smoke_sha = _write_json(component_paths[14], provider_smoke_record)
    manifest["component_records"][14]["sha256"] = restored_smoke_sha
    tampered_approval = dict(sdk_approval_record)
    tampered_approval["approval_reference"] = "LEGAL-OTHER"
    tampered_approval_sha = _write_json(component_paths[13], tampered_approval)
    manifest["component_records"][13]["sha256"] = tampered_approval_sha
    _write_json(manifest_path, manifest)
    errors = validate_acceptance(manifest_path)
    assert "retained sdk approval differs from frozen configuration" in errors
    assert "provider smoke: SDK approval identity mismatch" in errors

    restored_approval_sha = _write_json(component_paths[13], sdk_approval_record)
    manifest["component_records"][13]["sha256"] = restored_approval_sha
    _write_json(manifest_path, manifest)
    assert validate_acceptance(manifest_path) == []

    tampered_resources = json.loads(json.dumps(provider_resources_record))
    for resource in tampered_resources["resources"]:
        if resource["mode"] == "sampling":
            resource["available"] = 0
    tampered_resources_sha = _write_json(component_paths[15], tampered_resources)
    manifest["component_records"][15]["sha256"] = tampered_resources_sha
    _write_json(manifest_path, manifest)
    assert any(
        "provider resource snapshots differ from retained gate references" in error
        for error in validate_acceptance(manifest_path)
    )

    tampered_smoke = json.loads(json.dumps(provider_smoke_record))
    tampered_smoke["provider_resources_sha256"] = tampered_resources_sha
    tampered_smoke_sha = _write_json(component_paths[14], tampered_smoke)
    manifest["component_records"][14]["sha256"] = tampered_smoke_sha
    _write_json(manifest_path, manifest)
    assert any(
        "provider resource snapshots differ from retained gate references" in error
        for error in validate_acceptance(manifest_path)
    )

    provider_resources_sha = _write_json(component_paths[15], provider_resources_record)
    manifest["component_records"][15]["sha256"] = provider_resources_sha
    restored_smoke_sha = _write_json(component_paths[14], provider_smoke_record)
    manifest["component_records"][14]["sha256"] = restored_smoke_sha
    _write_json(manifest_path, manifest)
    assert validate_acceptance(manifest_path) == []

    loaded_training, loaded_sha = _load_component(
        component_paths[5], "flagquantum.qboson_qdiffusion_protein_training"
    )
    assert loaded_training["seed"] == 1701
    assert loaded_sha == training_entries[0][1]

    tampered_system = json.loads(component_paths[0].read_text(encoding="utf-8"))
    tampered_system["transport"] = "in_memory_fake"
    tampered_system_sha = _write_json(component_paths[0], tampered_system)
    primary["system_evidence_sha256"] = tampered_system_sha
    primary_sha = _write_json(primary_path, primary)
    manifest["records"][0]["sha256"] = primary_sha
    manifest["component_records"][0]["sha256"] = tampered_system_sha
    _write_json(manifest_path, manifest)
    assert any(
        "system component transport differs" in error
        for error in validate_acceptance(manifest_path)
    )

    primary_system_sha = _write_json(component_paths[0], primary_system_record)
    primary["system_evidence_sha256"] = primary_system_sha
    primary_sha = _write_json(primary_path, primary)
    manifest["records"][0]["sha256"] = primary_sha
    manifest["component_records"][0]["sha256"] = primary_system_sha

    tampered_system = json.loads(component_paths[0].read_text(encoding="utf-8"))
    tampered_system["observed_hostname"] = "unrelated-host"
    tampered_system_sha = _write_json(component_paths[0], tampered_system)
    primary["system_evidence_sha256"] = tampered_system_sha
    primary_sha = _write_json(primary_path, primary)
    manifest["records"][0]["sha256"] = primary_sha
    manifest["component_records"][0]["sha256"] = tampered_system_sha
    _write_json(manifest_path, manifest)
    assert any(
        "observed hostname differs from frozen identity" in error
        for error in validate_acceptance(manifest_path)
    )

    primary_system_sha = _write_json(component_paths[0], primary_system_record)
    primary["system_evidence_sha256"] = primary_system_sha
    primary_sha = _write_json(primary_path, primary)
    manifest["records"][0]["sha256"] = primary_sha
    manifest["component_records"][0]["sha256"] = primary_system_sha

    tampered_system = json.loads(component_paths[0].read_text(encoding="utf-8"))
    tampered_system["task_receipts"] = []
    tampered_system_sha = _write_json(component_paths[0], tampered_system)
    primary["system_evidence_sha256"] = tampered_system_sha
    primary_sha = _write_json(primary_path, primary)
    manifest["records"][0]["sha256"] = primary_sha
    manifest["component_records"][0]["sha256"] = tampered_system_sha
    _write_json(manifest_path, manifest)
    assert any(
        "remote component task receipts are missing" in error
        for error in validate_acceptance(manifest_path)
    )

    primary_system_sha = _write_json(component_paths[0], primary_system_record)
    primary["system_evidence_sha256"] = primary_system_sha
    primary_sha = _write_json(primary_path, primary)
    manifest["records"][0]["sha256"] = primary_sha
    manifest["component_records"][0]["sha256"] = primary_system_sha

    tampered_system = json.loads(component_paths[0].read_text(encoding="utf-8"))
    tampered_system["transfer_accounting"]["sampler_boundaries"][0][
        "submission_matrix_sha256"
    ] = ("8" * 64)
    tampered_system_sha = _write_json(component_paths[0], tampered_system)
    primary["system_evidence_sha256"] = tampered_system_sha
    primary["transfer_accounting"] = tampered_system["transfer_accounting"]
    primary_sha = _write_json(primary_path, primary)
    manifest["records"][0]["sha256"] = primary_sha
    manifest["component_records"][0]["sha256"] = tampered_system_sha
    _write_json(manifest_path, manifest)
    assert any(
        "transfer 0 matrix identity differs from its receipt" in error
        for error in validate_acceptance(manifest_path)
    )

    primary_system_sha = _write_json(component_paths[0], primary_system_record)
    primary["system_evidence_sha256"] = primary_system_sha
    primary["transfer_accounting"] = primary_system_record["transfer_accounting"]
    primary_sha = _write_json(primary_path, primary)
    manifest["records"][0]["sha256"] = primary_sha
    manifest["component_records"][0]["sha256"] = primary_system_sha

    tampered_system = json.loads(component_paths[0].read_text(encoding="utf-8"))
    tampered_system["transfer_accounting"]["sampler_boundaries"][0][
        "original_matrix_sha256"
    ] = ("a" * 64)
    tampered_system_sha = _write_json(component_paths[0], tampered_system)
    primary["system_evidence_sha256"] = tampered_system_sha
    primary["transfer_accounting"] = tampered_system["transfer_accounting"]
    primary_sha = _write_json(primary_path, primary)
    manifest["records"][0]["sha256"] = primary_sha
    manifest["component_records"][0]["sha256"] = tampered_system_sha
    _write_json(manifest_path, manifest)
    assert any(
        "precision origins differ from sampler transfers" in error
        for error in validate_acceptance(manifest_path)
    )

    primary_system_sha = _write_json(component_paths[0], primary_system_record)
    primary["system_evidence_sha256"] = primary_system_sha
    primary["transfer_accounting"] = primary_system_record["transfer_accounting"]
    primary_sha = _write_json(primary_path, primary)
    manifest["records"][0]["sha256"] = primary_sha
    manifest["component_records"][0]["sha256"] = primary_system_sha

    tampered_training = json.loads(component_paths[5].read_text(encoding="utf-8"))
    tampered_training["artifact_inputs_unchanged"] = False
    tampered_training_sha = _write_json(component_paths[5], tampered_training)
    manifest["component_records"][5]["sha256"] = tampered_training_sha
    _write_json(manifest_path, manifest)
    assert any(
        "training frozen inputs changed" in error
        for error in validate_acceptance(manifest_path)
    )

    restored_training_sha = _write_json(component_paths[5], training_entries[0][0])
    manifest["component_records"][5]["sha256"] = restored_training_sha

    tampered_training = json.loads(component_paths[5].read_text(encoding="utf-8"))
    tampered_training["real_provider_evidence"] = False
    tampered_training_sha = _write_json(component_paths[5], tampered_training)
    manifest["component_records"][5]["sha256"] = tampered_training_sha
    _write_json(manifest_path, manifest)
    assert any(
        "training real_provider_evidence is not proven" in error
        for error in validate_acceptance(manifest_path)
    )

    restored_training_sha = _write_json(component_paths[5], training_entries[0][0])
    manifest["component_records"][5]["sha256"] = restored_training_sha

    tampered_training = json.loads(component_paths[5].read_text(encoding="utf-8"))
    tampered_training["task_receipts"][0]["requested_samples"] = 11
    tampered_training_sha = _write_json(component_paths[5], tampered_training)
    manifest["component_records"][5]["sha256"] = tampered_training_sha
    _write_json(manifest_path, manifest)
    assert any(
        "training receipt 0: receipt sample count differs" in error
        for error in validate_acceptance(manifest_path)
    )

    restored_training_sha = _write_json(component_paths[5], training_entries[0][0])
    manifest["component_records"][5]["sha256"] = restored_training_sha

    tampered_training = json.loads(component_paths[5].read_text(encoding="utf-8"))
    tampered_training["precision_policy"]["scale_factor_max"] = 0.5
    tampered_training_sha = _write_json(component_paths[5], tampered_training)
    manifest["component_records"][5]["sha256"] = tampered_training_sha
    _write_json(manifest_path, manifest)
    assert any(
        "training precision scale range is invalid" in error
        for error in validate_acceptance(manifest_path)
    )

    restored_training_sha = _write_json(component_paths[5], training_entries[0][0])
    manifest["component_records"][5]["sha256"] = restored_training_sha

    tampered_training = json.loads(component_paths[5].read_text(encoding="utf-8"))
    tampered_training["workflow_artifacts"]["guided_fasta"]["sha256"] = "bad"
    tampered_training_sha = _write_json(component_paths[5], tampered_training)
    manifest["component_records"][5]["sha256"] = tampered_training_sha
    _write_json(manifest_path, manifest)
    assert any(
        "training artifact guided_fasta has no digest" in error
        for error in validate_acceptance(manifest_path)
    )

    restored_training_sha = _write_json(component_paths[5], training_entries[0][0])
    manifest["component_records"][5]["sha256"] = restored_training_sha

    tampered_portability = json.loads(component_paths[11].read_text(encoding="utf-8"))
    tampered_portability["artifact_inputs_unchanged"] = False
    tampered_portability_sha = _write_json(component_paths[11], tampered_portability)
    replay["portability_evidence"]["record_sha256"] = tampered_portability_sha
    replay_sha = _write_json(replay_path, replay)
    manifest["records"][1]["sha256"] = replay_sha
    manifest["component_records"][11]["sha256"] = tampered_portability_sha
    _write_json(manifest_path, manifest)
    assert any(
        "portability frozen inputs changed" in error
        for error in validate_acceptance(manifest_path)
    )

    portability_sha = _write_json(component_paths[11], portability_record)
    replay["portability_evidence"]["record_sha256"] = portability_sha
    replay_sha = _write_json(replay_path, replay)
    manifest["records"][1]["sha256"] = replay_sha
    manifest["component_records"][11]["sha256"] = portability_sha

    tampered_portability = json.loads(component_paths[11].read_text(encoding="utf-8"))
    tampered_portability["task_receipts"] = []
    tampered_portability_sha = _write_json(component_paths[11], tampered_portability)
    replay["portability_evidence"]["record_sha256"] = tampered_portability_sha
    replay_sha = _write_json(replay_path, replay)
    manifest["records"][1]["sha256"] = replay_sha
    manifest["component_records"][11]["sha256"] = tampered_portability_sha
    _write_json(manifest_path, manifest)
    assert any(
        "remote component task receipts are missing" in error
        for error in validate_acceptance(manifest_path)
    )

    portability_sha = _write_json(component_paths[11], portability_record)
    replay["portability_evidence"]["record_sha256"] = portability_sha
    replay_sha = _write_json(replay_path, replay)
    manifest["records"][1]["sha256"] = replay_sha
    manifest["component_records"][11]["sha256"] = portability_sha

    tampered_portability = json.loads(component_paths[11].read_text(encoding="utf-8"))
    tampered_portability["flagquantum_version"] = "999.0.0"
    tampered_portability_sha = _write_json(component_paths[11], tampered_portability)
    replay["portability_evidence"]["record_sha256"] = tampered_portability_sha
    replay_sha = _write_json(replay_path, replay)
    manifest["records"][1]["sha256"] = replay_sha
    manifest["component_records"][11]["sha256"] = tampered_portability_sha
    _write_json(manifest_path, manifest)
    assert any(
        "portability flagquantum_version differs from config" in error
        for error in validate_acceptance(manifest_path)
    )

    portability_sha = _write_json(component_paths[11], portability_record)
    replay["portability_evidence"]["record_sha256"] = portability_sha
    replay_sha = _write_json(replay_path, replay)
    manifest["records"][1]["sha256"] = replay_sha
    manifest["component_records"][11]["sha256"] = portability_sha

    tampered_portability = json.loads(component_paths[11].read_text(encoding="utf-8"))
    tampered_portability["artifacts"]["dataset_sha256"] = "0" * 64
    tampered_portability_sha = _write_json(component_paths[11], tampered_portability)
    replay["portability_evidence"]["record_sha256"] = tampered_portability_sha
    replay_sha = _write_json(replay_path, replay)
    manifest["records"][1]["sha256"] = replay_sha
    manifest["component_records"][11]["sha256"] = tampered_portability_sha
    _write_json(manifest_path, manifest)
    assert any(
        "portability artifact identities differ from config" in error
        for error in validate_acceptance(manifest_path)
    )

    portability_sha = _write_json(component_paths[11], portability_record)
    replay["portability_evidence"]["record_sha256"] = portability_sha
    replay_sha = _write_json(replay_path, replay)
    manifest["records"][1]["sha256"] = replay_sha
    manifest["component_records"][11]["sha256"] = portability_sha

    tampered_portability = json.loads(component_paths[11].read_text(encoding="utf-8"))
    tampered_portability["fixture"]["steps"] = 2
    tampered_portability_sha = _write_json(component_paths[11], tampered_portability)
    replay["portability_evidence"]["record_sha256"] = tampered_portability_sha
    replay_sha = _write_json(replay_path, replay)
    manifest["records"][1]["sha256"] = replay_sha
    manifest["component_records"][11]["sha256"] = tampered_portability_sha
    _write_json(manifest_path, manifest)
    assert any(
        "portability fixture steps differs from config" in error
        for error in validate_acceptance(manifest_path)
    )

    portability_sha = _write_json(component_paths[11], portability_record)
    replay["portability_evidence"]["record_sha256"] = portability_sha
    replay_sha = _write_json(replay_path, replay)
    manifest["records"][1]["sha256"] = replay_sha
    manifest["component_records"][11]["sha256"] = portability_sha

    tampered_portability = json.loads(component_paths[11].read_text(encoding="utf-8"))
    tampered_portability["recorded_at"] = "2026-10-05T00:00:00+00:00"
    tampered_portability_sha = _write_json(component_paths[11], tampered_portability)
    replay["portability_evidence"]["record_sha256"] = tampered_portability_sha
    replay_sha = _write_json(replay_path, replay)
    manifest["records"][1]["sha256"] = replay_sha
    manifest["component_records"][11]["sha256"] = tampered_portability_sha
    _write_json(manifest_path, manifest)
    assert any(
        "portability predates its selected training component" in error
        for error in validate_acceptance(manifest_path)
    )

    portability_sha = _write_json(component_paths[11], portability_record)
    replay["portability_evidence"]["record_sha256"] = portability_sha
    replay_sha = _write_json(replay_path, replay)
    manifest["records"][1]["sha256"] = replay_sha
    manifest["component_records"][11]["sha256"] = portability_sha

    tampered_evaluation = json.loads(component_paths[8].read_text(encoding="utf-8"))
    tampered_evaluation["artifact_inputs_unchanged"] = False
    tampered_evaluation_sha = _write_json(component_paths[8], tampered_evaluation)
    primary["application_evidence"]["records"][0][
        "evaluation_record_sha256"
    ] = tampered_evaluation_sha
    primary_sha = _write_json(primary_path, primary)
    manifest["records"][0]["sha256"] = primary_sha
    manifest["component_records"][8]["sha256"] = tampered_evaluation_sha
    _write_json(manifest_path, manifest)
    assert any(
        "evaluation frozen inputs changed" in error
        for error in validate_acceptance(manifest_path)
    )

    restored_evaluation_sha = _write_json(component_paths[8], evaluation_entries[0][0])
    primary["application_evidence"]["records"][0][
        "evaluation_record_sha256"
    ] = restored_evaluation_sha
    primary_sha = _write_json(primary_path, primary)
    manifest["records"][0]["sha256"] = primary_sha
    manifest["component_records"][8]["sha256"] = restored_evaluation_sha

    tampered_evaluation = json.loads(component_paths[8].read_text(encoding="utf-8"))
    tampered_evaluation["recorded_at"] = "2026-10-05T00:00:00+00:00"
    tampered_evaluation_sha = _write_json(component_paths[8], tampered_evaluation)
    primary["application_evidence"]["records"][0][
        "evaluation_record_sha256"
    ] = tampered_evaluation_sha
    primary_sha = _write_json(primary_path, primary)
    manifest["records"][0]["sha256"] = primary_sha
    manifest["component_records"][8]["sha256"] = tampered_evaluation_sha
    _write_json(manifest_path, manifest)
    assert any(
        "evaluation predates its training component" in error
        for error in validate_acceptance(manifest_path)
    )

    restored_evaluation_sha = _write_json(component_paths[8], evaluation_entries[0][0])
    primary["application_evidence"]["records"][0][
        "evaluation_record_sha256"
    ] = restored_evaluation_sha
    primary_sha = _write_json(primary_path, primary)
    manifest["records"][0]["sha256"] = primary_sha
    manifest["component_records"][8]["sha256"] = restored_evaluation_sha

    tampered_evaluation = json.loads(component_paths[8].read_text(encoding="utf-8"))
    tampered_evaluation["observed_gpu_model"] = "NVIDIA H100"
    tampered_evaluation_sha = _write_json(component_paths[8], tampered_evaluation)
    primary["application_evidence"]["records"][0][
        "evaluation_record_sha256"
    ] = tampered_evaluation_sha
    primary_sha = _write_json(primary_path, primary)
    manifest["records"][0]["sha256"] = primary_sha
    manifest["component_records"][8]["sha256"] = tampered_evaluation_sha
    _write_json(manifest_path, manifest)
    assert any(
        "evaluation GPU is not an NVIDIA A800" in error
        for error in validate_acceptance(manifest_path)
    )

    restored_evaluation_sha = _write_json(component_paths[8], evaluation_entries[0][0])
    primary["application_evidence"]["records"][0][
        "evaluation_record_sha256"
    ] = restored_evaluation_sha
    primary_sha = _write_json(primary_path, primary)
    manifest["records"][0]["sha256"] = primary_sha
    manifest["component_records"][8]["sha256"] = restored_evaluation_sha

    tampered_evaluation = json.loads(component_paths[8].read_text(encoding="utf-8"))
    tampered_evaluation["guided_metrics"]["mean_cosine_distance"] = 0.99
    tampered_evaluation_sha = _write_json(component_paths[8], tampered_evaluation)
    primary["application_evidence"]["records"][0][
        "evaluation_record_sha256"
    ] = tampered_evaluation_sha
    primary_sha = _write_json(primary_path, primary)
    manifest["records"][0]["sha256"] = primary_sha
    manifest["component_records"][8]["sha256"] = tampered_evaluation_sha
    _write_json(manifest_path, manifest)

    assert any(
        "differs from component aggregation" in error
        for error in validate_acceptance(manifest_path)
    )

    tampered_training = dict(training_entries[0][0])
    tampered_training["artifact_preflight_sha256"] = "0" * 64
    tampered_training_sha = _write_json(component_paths[5], tampered_training)
    manifest["component_records"][5]["sha256"] = tampered_training_sha
    _write_json(manifest_path, manifest)
    assert any(
        "training references another artifact preflight" in error
        for error in validate_acceptance(manifest_path)
    )
