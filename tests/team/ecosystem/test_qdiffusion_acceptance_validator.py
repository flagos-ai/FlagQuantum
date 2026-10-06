from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from examples.qdiffusion_kaiwu.validate_acceptance import (
    _validate_component_bundle,
    _validate_config,
    _validate_precision_evidence,
    _validate_sampling_receipt,
    validate_acceptance,
)

pytestmark = pytest.mark.unit

_REVISION = "a" * 40
_PLUGIN_REVISION = "b" * 40


def _complete_task_receipt() -> dict[str, Any]:
    return {
        "schema": "flagquantum.kaiwu-task.v1",
        "task_name": "system-task",
        "matrix_sha256": "7" * 64,
        "matrix_size": 3,
        "mode": "sampling",
        "requested_samples": 10,
        "project_no": "CPQC-test",
        "submitted_at": "2026-10-05T00:00:00+00:00",
        "provider_task_id": "provider-task",
        "provider_target": "SPQC-provider",
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("schema", "wrong", "schema is unsupported"),
        ("task_name", "", "no task name"),
        ("task_name", "system\ntask", "no task name"),
        ("task_name", " system-task", "no task name"),
        ("matrix_sha256", "bad", "no matrix digest"),
        ("matrix_size", True, "invalid matrix size"),
        ("mode", "optimization", "not sampling"),
        ("requested_samples", 11, "sample count differs"),
        ("project_no", " ", "no project number"),
        ("project_no", "CPQC\ttest", "no project number"),
        ("submitted_at", "2026-10-05T08:00:00+08:00", "aware UTC"),
        ("provider_task_id", "", "no provider_task_id"),
        ("provider_task_id", "provider\ntask", "no provider_task_id"),
        ("provider_target", "", "no provider_target"),
        ("provider_target", "SPQC\u200bprovider", "no provider_target"),
    ),
)
def test_complete_sampling_receipt_rejects_tampered_identity_fields(
    field: str, value: object, message: str
) -> None:
    receipt = _complete_task_receipt()
    receipt[field] = value
    errors: list[str] = []

    _validate_sampling_receipt(
        receipt,
        label="receipt",
        expected_requested_samples=10,
        errors=errors,
    )

    assert any(message in error for error in errors)


def test_complete_sampling_receipt_rejects_missing_or_extra_fields() -> None:
    for receipt in (
        {
            key: value
            for key, value in _complete_task_receipt().items()
            if key != "schema"
        },
        {**_complete_task_receipt(), "unexpected": True},
    ):
        errors: list[str] = []
        _validate_sampling_receipt(
            receipt,
            label="receipt",
            expected_requested_samples=10,
            errors=errors,
        )
        assert any("field set is incomplete" in error for error in errors)


def _complete_precision_record() -> dict[str, Any]:
    return {
        "precision_policy": {
            "target_min": -127,
            "target_max": 127,
            "matrix_count": 1,
            "scale_factor_min": 2.0,
            "scale_factor_max": 2.0,
            "max_abs_error": 0.25,
            "mean_of_matrix_mean_abs_error": 0.125,
        },
        "precision_evidence": [
            {
                "original_matrix_sha256": "6" * 64,
                "submission_matrix_sha256": "7" * 64,
                "source_type": "numpy.ndarray",
                "source_dtype": "float32",
                "normalized_dtype": "torch.float64",
                "normalized_min": -63.5,
                "normalized_max": 63.5,
                "symmetry_normalization": "arithmetic_mean",
                "rounding_policy": "round_half_to_even",
                "scale_factor": 2.0,
                "target_min": -127,
                "target_max": 127,
                "max_abs_error": 0.25,
                "mean_abs_error": 0.125,
            }
        ],
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda record: record["precision_evidence"].clear(), "evidence is missing"),
        (
            lambda record: record["precision_evidence"][0].__setitem__(
                "submission_matrix_sha256", "8" * 64
            ),
            "submissions differ from task receipts",
        ),
        (
            lambda record: record["precision_evidence"][0].__setitem__(
                "rounding_policy", "truncate"
            ),
            "rounding policy is unsupported",
        ),
        (
            lambda record: record["precision_evidence"][0].__setitem__(
                "source_type", "torch.Tensor"
            ),
            "source type is not numpy.ndarray",
        ),
        (
            lambda record: record["precision_evidence"][0].__setitem__(
                "normalized_min", 64.0
            ),
            "normalized coefficient range is invalid",
        ),
        (
            lambda record: record["precision_evidence"][0].__setitem__(
                "normalized_max", 100.0
            ),
            "scale factor differs from coefficient range",
        ),
        (
            lambda record: record["precision_policy"].__setitem__("max_abs_error", 0.5),
            "aggregate max_abs_error differs",
        ),
    ),
)
def test_per_matrix_precision_evidence_fails_closed(mutation, message: str) -> None:
    record = _complete_precision_record()
    mutation(record)
    errors: list[str] = []

    _validate_precision_evidence(
        record,
        label="precision",
        receipt_matrix_digests=["7" * 64],
        errors=errors,
    )

    assert any(message in error for error in errors)


def _write_json(path: Path, payload: object) -> str:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    path.chmod(0o600)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config() -> dict[str, Any]:
    return {
        "schema": "flagquantum.qboson_qdiffusion_config",
        "version": "1.0",
        "preregistered_at": "2026-10-05T00:00:00Z",
        "primary_host": "jp-a800-171",
        "replay_host": "jp-a800-172",
        "software": {
            "source_revision": _REVISION,
            "kaiwu_pytorch_plugin_revision": _PLUGIN_REVISION,
            "python_version": "3.10.18",
            "torch_version": "2.7.0",
            "kaiwu_sdk_version": "1.3.1",
            "environment_lock_sha256": "6" * 64,
        },
        "dataset": {
            "name": "frozen",
            "revision": "v1",
            "source_url": "https://example.test/dataset.fasta",
            "license_id": "CC-BY-4.0",
            "license_evidence_url": "https://example.test/dataset-license",
            "license_reviewed_at": "2026-10-05T00:00:00Z",
            "split": "deterministic-shuffle-v1",
            "sha256": "c" * 64,
            "min_length": 50,
            "max_length": 256,
            "max_records": 640,
            "validation_ratio": 0.05,
            "test_ratio": 0.05,
        },
        "checkpoint": {
            "name": "dplm",
            "revision": "v1",
            "source_url": "https://example.test/dplm",
            "license_id": "Apache-2.0",
            "license_evidence_url": "https://example.test/dplm-license",
            "license_reviewed_at": "2026-10-05T00:00:00Z",
            "sha256": "d" * 64,
        },
        "tokenizer": {
            "name": "dplm",
            "revision": "v1",
            "source_url": "https://example.test/dplm",
            "license_id": "Apache-2.0",
            "license_evidence_url": "https://example.test/dplm-license",
            "license_reviewed_at": "2026-10-05T00:00:00Z",
            "sha256": "e" * 64,
        },
        "evaluation_model": {
            "name": "esm2_t33_650M_UR50D",
            "revision": "v1",
            "source_url": "https://example.test/esm2.pt",
            "license_id": "MIT",
            "license_evidence_url": "https://example.test/esm2-license",
            "license_reviewed_at": "2026-10-05T00:00:00Z",
            "sha256": "f" * 64,
        },
        "training": {
            "freeze_proposal": True,
            "epochs": 20,
            "min_epochs": 3,
            "batch_size": 4,
            "num_candidates": 4,
            "learning_rate": 0.00005,
            "weight_decay": 0.01,
            "grad_clip_norm": 1.0,
            "validation_steps": 3,
            "scheduler_factor": 0.5,
            "scheduler_patience": 1,
            "early_stop_patience": 4,
            "remote_call_budget_per_seed": 71269,
        },
        "generation": {
            "sequence_count": 32,
            "max_steps": 64,
            "num_candidates": 4,
            "proposal_temperature": 0.3,
            "proposal_noise_scale": 1.0,
            "energy_temperature": 1.25,
            "disable_resample": False,
            "resample_ratio": 0.2,
            "resample_top_p": 0.9,
            "portability_training_seed": 1701,
            "portability_fixture_index": 0,
            "portability_steps": 3,
        },
        "evaluation": {"pair_mode": "order", "pooling": "mean", "batch_size": 1},
        "seeds": [1701, 1702, 1703],
        "requested_samples": 10,
        "remote_call_budget": 128,
        "precision_policy": {
            "name": "explicit-int8",
            "target_min": -127,
            "target_max": 127,
        },
        "primary_metric": {"name": "mean_cosine_distance", "direction": "lower"},
        "thresholds": {
            "uniqueness_baseline_fraction_min": 0.95,
            "repeat_ratio_absolute_increase_max": 0.05,
            "invalid_sequence_count_max": 0,
        },
    }


def _metrics(*, cosine: float, uniqueness: float, repeat: float) -> dict[str, float]:
    return {
        "mean_cosine_distance": cosine,
        "median_cosine_distance": cosine,
        "mean_l2_distance": 1.0,
        "median_l2_distance": 1.0,
        "identity_to_reference_mean": 0.4,
        "amino_acid_jsd": 0.1,
        "kmer2_jsd": 0.2,
        "kmer3_jsd": 0.3,
        "uniqueness_ratio": uniqueness,
        "repeat_ratio_ge4": repeat,
        "length_match_ratio": 1.0,
        "invalid_sequence_count": 0.0,
    }


def _record(
    host: str, role: str, config_sha256: str, environment_lock_sha256: str
) -> dict[str, Any]:
    transfer_boundary = {
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
    record = {
        "schema": "flagquantum.qboson_qdiffusion_acceptance",
        "version": "1.0",
        "source_revision": _REVISION,
        "kaiwu_pytorch_plugin_revision": _PLUGIN_REVISION,
        "source_preflight_sha256": ("8" * 64 if host == "jp-a800-171" else "9" * 64),
        "transfer_manifest_sha256": "7" * 64,
        "environment_lock_sha256": environment_lock_sha256,
        "python_version": "3.10.18",
        "torch_version": "2.7.0",
        "kaiwu_sdk_version": "1.3.1",
        "experiment_config_sha256": config_sha256,
        "execution_host": host,
        "run_role": role,
        "requested_cuda_device": "cuda:0",
        "observed_tensor_device": "cuda:0",
        "observed_gpu_model": "NVIDIA A800-SXM4-80GB",
        "transport": "kaiwu_cim",
        "qboson_hardware_used": True,
        "real_provider_evidence": True,
        "provider_reported_target": True,
        "qboson_target": "SPQC-provider-label",
        "qboson_task_ids": [f"task-{host}"],
        "sampling_mode": "sampling",
        "requested_samples": 10,
        "returned_samples": 10,
        "precision_policy": {
            "name": "explicit-int8",
            "target_min": -127,
            "target_max": 127,
            "matrix_count": 2,
            "scale_factor_min": 1.0,
            "scale_factor_max": 2.0,
            "max_abs_error": 0.0,
            "mean_of_matrix_mean_abs_error": 0.0,
        },
        "remote_call_budget": 128,
        "remote_call_count": 2,
        "fallback_occurred": False,
        "retrieval_resubmitted": False,
        "secrets_redacted": True,
        "artifacts": {
            "dataset_sha256": "c" * 64,
            "base_checkpoint_sha256": "d" * 64,
            "tokenizer_sha256": "e" * 64,
            "evaluation_model_sha256": "f" * 64,
            "trained_energy_checkpoint_sha256": "1" * 64,
        },
        "training": {
            "energy_objective": -0.5,
            "gradient_norm": 1.0,
            "parameter_delta_max": 0.01,
        },
        "generation": {"token_constraints_passed": True, "invalid_sequence_count": 0},
        "transfer_accounting": {
            "matrix_origin_device": "cuda:0",
            "sampler_boundaries": [transfer_boundary, dict(transfer_boundary)],
            "returned_sample_target_device": "cuda:0",
        },
        "attempted_seeds": [1701, 1702, 1703],
        "baseline_metrics": _metrics(cosine=0.5, uniqueness=1.0, repeat=0.0),
        "guided_metrics": _metrics(cosine=0.4, uniqueness=0.96, repeat=0.04),
        "acceptance": {
            "system": "pass",
            "application": "pass" if role == "primary" else "not_run",
        },
    }
    if role == "primary":
        record["application_evidence"] = {
            "aggregation": "arithmetic_mean_across_frozen_seeds",
            "records": [
                {
                    "seed": seed,
                    "training_record_sha256": str(index) * 64,
                    "evaluation_record_sha256": str(index + 3) * 64,
                    "trained_energy_checkpoint_sha256": "1" * 64,
                }
                for index, seed in enumerate((1701, 1702, 1703), start=1)
            ],
        }
    else:
        record["portability_evidence"] = {
            "record_sha256": "7" * 64,
            "training_seed": 1701,
            "training_record_sha256": "1" * 64,
            "trained_energy_checkpoint_sha256": "1" * 64,
            "acceptance": "pass",
        }
    return record


def _bundle(tmp_path: Path) -> tuple[Path, list[dict[str, Any]]]:
    environment_lock = {
        "schema": "flagquantum.qboson_qdiffusion_environment_lock",
        "version": "1.0",
        "inventory_policy": "exact",
        "python_version": "3.10.18",
        "distributions": [
            {
                "name": "torch",
                "version": "2.7.0",
                "approved_artifact_sha256": "a" * 64,
                "installed_content_sha256": "b" * 64,
            }
        ],
    }
    environment_path = tmp_path / "environment-lock.json"
    environment_hash = _write_json(environment_path, environment_lock)
    config = _config()
    config["software"]["environment_lock_sha256"] = environment_hash
    config_path = tmp_path / "config.json"
    config_hash = _write_json(config_path, config)
    records = [
        _record("jp-a800-171", "primary", config_hash, environment_hash),
        _record("jp-a800-172", "portability_replay", config_hash, environment_hash),
    ]
    entries = []
    for record in records:
        path = tmp_path / f"{record['execution_host']}.json"
        entries.append({"path": path.name, "sha256": _write_json(path, record)})
    manifest = {
        "schema": "flagquantum.qboson_qdiffusion_manifest",
        "version": "1.0",
        "config": {"path": config_path.name, "sha256": config_hash},
        "environment_lock": {
            "path": environment_path.name,
            "sha256": environment_hash,
        },
        "records": entries,
    }
    manifest_path = tmp_path / "manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path, records


def _replace_record(manifest_path: Path, index: int, record: dict[str, Any]) -> None:
    manifest = json.loads(manifest_path.read_text())
    path = manifest_path.parent / manifest["records"][index]["path"]
    manifest["records"][index]["sha256"] = _write_json(path, record)
    _write_json(manifest_path, manifest)


def _source_preflight(host: str, manifest_sha256: str) -> dict[str, Any]:
    revisions = (
        ("flagquantum-qboson-", "FlagQuantum-", _REVISION),
        ("kaiwu-plugin-", "kaiwu-pytorch-plugin-", _PLUGIN_REVISION),
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
                "file_count": 1,
                "content_set_sha256": "f" * 64,
            }
            for filename_prefix, root_prefix, revision in revisions
        ],
        "qboson_hardware_used": False,
        "a800_execution_verified": False,
        "acceptance_evidence": False,
    }


def _transfer_manifest() -> dict[str, Any]:
    revisions = (
        ("flagquantum-qboson-", _REVISION),
        ("kaiwu-plugin-", _PLUGIN_REVISION),
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
                "sha256": "f" * 64,
            }
            for prefix, revision in revisions
        ],
    }


def test_complete_two_host_real_provider_bundle_passes(tmp_path: Path) -> None:
    manifest_path, _ = _bundle(tmp_path)

    assert validate_acceptance(manifest_path) == []


def test_validator_rejects_symlinked_or_public_bundle_members(tmp_path: Path) -> None:
    manifest_path, _ = _bundle(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    original = tmp_path / manifest["records"][0]["path"]
    link = tmp_path / "linked-primary.json"
    link.symlink_to(original)
    manifest["records"][0]["path"] = link.name
    _write_json(manifest_path, manifest)

    assert any(
        "path contains a symlink" in error
        for error in validate_acceptance(manifest_path)
    )

    manifest["records"][0]["path"] = original.name
    _write_json(manifest_path, manifest)
    original.chmod(0o644)
    assert any(
        "accessible by group or others" in error
        for error in validate_acceptance(manifest_path)
    )


def test_validator_rejects_extra_or_duplicate_declared_members(tmp_path: Path) -> None:
    manifest_path, _ = _bundle(tmp_path)
    _write_json(tmp_path / "unlisted.json", {"not": "evidence"})
    assert any(
        "exact declared member set" in error
        for error in validate_acceptance(manifest_path)
    )

    (tmp_path / "unlisted.json").unlink()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["records"][1]["path"] = manifest["records"][0]["path"]
    _write_json(manifest_path, manifest)
    assert any(
        "path is declared more than once" in error
        for error in validate_acceptance(manifest_path)
    )


def test_component_validator_rejects_different_host_transfer_manifests() -> None:
    config = _config()
    component_payloads = {
        "1" * 64: _source_preflight("jp-a800-171", "a" * 64),
        "2" * 64: _source_preflight("jp-a800-172", "b" * 64),
        "a" * 64: _transfer_manifest(),
    }
    schemas = (
        "flagquantum.qboson_qdiffusion_system_live_probe",
        "flagquantum.qboson_qdiffusion_system_live_probe",
        "flagquantum.qboson_qdiffusion_portability_replay",
        *("flagquantum.qboson_qdiffusion_protein_training",) * 3,
        *("flagquantum.qboson_qdiffusion_protein_evaluation",) * 3,
    )
    for index, schema in enumerate(schemas, start=3):
        component_payloads[str(index) * 64] = {
            "schema": schema,
            "version": "1.0",
            "experiment_config_sha256": "c" * 64,
        }
    errors: list[str] = []

    _validate_component_bundle(
        component_payloads,
        config=config,
        config_sha256="c" * 64,
        primary={},
        replay={},
        errors=errors,
    )

    assert "manifest: source preflights do not share one transfer manifest" in errors


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("transport", "in_memory_fake", "transport is not kaiwu_cim"),
        ("qboson_hardware_used", False, "hardware use is not proven"),
        ("fallback_occurred", True, "fallback must be explicitly false"),
        ("observed_tensor_device", "cpu", "tensor work was not observed"),
        ("real_provider_evidence", False, "real provider evidence is absent"),
    ),
)
def test_system_gate_rejects_false_claims(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    manifest_path, records = _bundle(tmp_path)
    changed = copy.deepcopy(records[0])
    changed[field] = value
    _replace_record(manifest_path, 0, changed)

    assert any(message in error for error in validate_acceptance(manifest_path))


def test_application_gate_is_recomputed_from_metrics(tmp_path: Path) -> None:
    manifest_path, records = _bundle(tmp_path)
    changed = copy.deepcopy(records[0])
    changed["guided_metrics"]["mean_cosine_distance"] = 0.6
    changed["guided_metrics"]["uniqueness_ratio"] = 0.8
    changed["guided_metrics"]["repeat_ratio_ge4"] = 0.1
    _replace_record(manifest_path, 0, changed)

    errors = validate_acceptance(manifest_path)

    assert any("mean cosine distance did not improve" in error for error in errors)
    assert any("uniqueness is below" in error for error in errors)
    assert any("repeat ratio exceeds" in error for error in errors)


def test_manifest_requires_independent_host_task_ids(tmp_path: Path) -> None:
    manifest_path, records = _bundle(tmp_path)
    changed = copy.deepcopy(records[1])
    changed["qboson_task_ids"] = records[0]["qboson_task_ids"]
    _replace_record(manifest_path, 1, changed)

    assert any(
        "independent QBoson task identities" in error
        for error in validate_acceptance(manifest_path)
    )


def test_tampered_record_hash_is_rejected(tmp_path: Path) -> None:
    manifest_path, _ = _bundle(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    record_path = tmp_path / manifest["records"][0]["path"]
    record_path.write_text(record_path.read_text() + " ")

    assert any(
        "SHA-256 mismatch" in error for error in validate_acceptance(manifest_path)
    )


def test_system_gate_requires_aggregate_precision_evidence(tmp_path: Path) -> None:
    manifest_path, records = _bundle(tmp_path)
    changed = copy.deepcopy(records[0])
    changed["precision_policy"]["matrix_count"] = 1
    changed["precision_policy"]["scale_factor_min"] = 3.0
    changed["precision_policy"]["scale_factor_max"] = 2.0
    _replace_record(manifest_path, 0, changed)

    errors = validate_acceptance(manifest_path)

    assert any("fewer reports than remote calls" in error for error in errors)
    assert any("invalid scale-factor range" in error for error in errors)


def test_system_gate_requires_source_preflight_identity(tmp_path: Path) -> None:
    manifest_path, records = _bundle(tmp_path)
    changed = copy.deepcopy(records[0])
    changed["source_preflight_sha256"] = "self-reported"
    _replace_record(manifest_path, 0, changed)

    assert any(
        "source_preflight_sha256: expected a SHA-256" in error
        for error in validate_acceptance(manifest_path)
    )


def test_system_gate_rejects_inconsistent_transfer_accounting(tmp_path: Path) -> None:
    manifest_path, records = _bundle(tmp_path)
    changed = copy.deepcopy(records[0])
    boundary = changed["transfer_accounting"]["sampler_boundaries"][0]
    boundary["input_device"] = "cuda:0"
    boundary["cache_hit"] = True
    _replace_record(manifest_path, 0, changed)

    errors = validate_acceptance(manifest_path)

    assert any("input_device: expected cpu" in error for error in errors)
    assert any(
        "transfer accounting differs from remote-call count" in error
        for error in errors
    )


def test_manifest_requires_same_trained_energy_checkpoint(tmp_path: Path) -> None:
    manifest_path, records = _bundle(tmp_path)
    changed = copy.deepcopy(records[1])
    changed["artifacts"]["trained_energy_checkpoint_sha256"] = "2" * 64
    _replace_record(manifest_path, 1, changed)

    assert any(
        "same trained energy checkpoint" in error
        for error in validate_acceptance(manifest_path)
    )


def test_environment_lock_is_frozen_and_retained(tmp_path: Path) -> None:
    manifest_path, records = _bundle(tmp_path)
    changed = copy.deepcopy(records[0])
    changed["environment_lock_sha256"] = "0" * 64
    _replace_record(manifest_path, 0, changed)

    assert any(
        "environment_lock_sha256: differs from the frozen software lane" in error
        for error in validate_acceptance(manifest_path)
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    lock_path = tmp_path / manifest["environment_lock"]["path"]
    lock_path.write_text("{}", encoding="utf-8")
    lock_path.chmod(0o600)
    assert any(
        "manifest.environment_lock: SHA-256 mismatch" in error
        for error in validate_acceptance(manifest_path)
    )


def test_config_requires_environment_lock_digest() -> None:
    config = _config()
    config["software"]["environment_lock_sha256"] = "not-a-digest"
    errors: list[str] = []

    _validate_config(config, errors)

    assert any("environment_lock_sha256" in error for error in errors)


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    (
        ("dataset", "source_url", "http://example.test/data", "expected an HTTPS URL"),
        ("dataset", "source_url", "https://[broken", "expected an HTTPS URL"),
        ("checkpoint", "license_id", "NOASSERTION", "approved license identifier"),
        ("tokenizer", "license_evidence_url", "<required>", "expected an HTTPS URL"),
        (
            "evaluation_model",
            "license_reviewed_at",
            "2026-10-05",
            "timezone-aware timestamp",
        ),
    ),
)
def test_config_rejects_unapproved_artifact_provenance(
    section: str, field: str, value: object, message: str
) -> None:
    config = _config()
    config[section][field] = value
    errors: list[str] = []

    _validate_config(config, errors)

    assert any(message in error for error in errors)


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    (
        (
            "dataset",
            "split",
            "ad-hoc",
            "expected deterministic-shuffle-v1",
        ),
        (
            "generation",
            "sequence_count",
            31,
            "differs from the frozen test split",
        ),
        (
            "generation",
            "disable_resample",
            "false",
            "expected a boolean",
        ),
        ("evaluation", "pair_mode", "nearest", "expected order"),
    ),
)
def test_config_rejects_workflow_parameter_drift(
    section: str, field: str, value: object, message: str
) -> None:
    config = _config()
    config[section][field] = value
    errors: list[str] = []

    _validate_config(config, errors)

    assert any(message in error for error in errors)


def test_config_rejects_protein_budget_below_worst_case_estimate() -> None:
    config = _config()
    config["training"]["remote_call_budget_per_seed"] = 71268
    errors: list[str] = []

    _validate_config(config, errors)

    assert any("worst-case workflow estimate of 71269" in error for error in errors)


def test_config_rejects_portability_fixture_outside_frozen_lane() -> None:
    config = _config()
    config["generation"]["portability_training_seed"] = 9999
    config["generation"]["portability_fixture_index"] = 32
    errors: list[str] = []

    _validate_config(config, errors)

    assert any("expected one frozen seed" in error for error in errors)
    assert any("outside the frozen sequence set" in error for error in errors)


def test_config_rejects_system_budget_below_portability_estimate() -> None:
    config = _config()
    config["remote_call_budget"] = 16
    errors: list[str] = []

    _validate_config(config, errors)

    assert any("portability replay estimate of 17" in error for error in errors)


@pytest.mark.parametrize("value", (None, 9, 2001, 10.0))
def test_config_rejects_unfrozen_provider_sample_count(value: object) -> None:
    config = _config()
    config["requested_samples"] = value
    errors: list[str] = []

    _validate_config(config, errors)

    assert any("config.requested_samples" in error for error in errors)


def test_component_bundle_is_required_for_assembled_records(tmp_path: Path) -> None:
    manifest_path, records = _bundle(tmp_path)
    changed = copy.deepcopy(records[0])
    changed["component_bundle_required"] = True
    _replace_record(manifest_path, 0, changed)

    errors = validate_acceptance(manifest_path)

    assert any("does not contain every source record" in error for error in errors)
    assert any(
        "do not reference the exact component bundle" in error for error in errors
    )


def test_application_evidence_must_cover_frozen_seed_order(tmp_path: Path) -> None:
    manifest_path, records = _bundle(tmp_path)
    changed = copy.deepcopy(records[0])
    changed["application_evidence"]["records"].reverse()
    _replace_record(manifest_path, 0, changed)

    assert any(
        "seed order differs from config" in error
        for error in validate_acceptance(manifest_path)
    )
