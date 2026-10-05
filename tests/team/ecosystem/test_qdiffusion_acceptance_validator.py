from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from examples.qdiffusion_kaiwu.validate_acceptance import validate_acceptance

pytestmark = pytest.mark.unit

_REVISION = "a" * 40
_PLUGIN_REVISION = "b" * 40


def _write_json(path: Path, payload: object) -> str:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
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
        },
        "dataset": {
            "name": "frozen",
            "revision": "v1",
            "split": "test",
            "sha256": "c" * 64,
        },
        "checkpoint": {"name": "dplm", "revision": "v1", "sha256": "d" * 64},
        "tokenizer": {"name": "dplm", "revision": "v1", "sha256": "e" * 64},
        "evaluation_model": {
            "name": "esm2_t33_650M_UR50D",
            "revision": "v1",
            "sha256": "f" * 64,
        },
        "training": {
            "epochs": 20,
            "min_epochs": 3,
            "batch_size": 4,
            "learning_rate": 0.00005,
            "weight_decay": 0.01,
            "grad_clip_norm": 1.0,
            "validation_steps": 3,
            "scheduler_factor": 0.5,
            "scheduler_patience": 1,
            "early_stop_patience": 4,
        },
        "generation": {"sequence_count": 32, "max_steps": 64, "num_candidates": 4},
        "seeds": [1701, 1702, 1703],
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


def _record(host: str, role: str, config_sha256: str) -> dict[str, Any]:
    return {
        "schema": "flagquantum.qboson_qdiffusion_acceptance",
        "version": "1.0",
        "source_revision": _REVISION,
        "kaiwu_pytorch_plugin_revision": _PLUGIN_REVISION,
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
        "attempted_seeds": [1701, 1702, 1703],
        "baseline_metrics": _metrics(cosine=0.5, uniqueness=1.0, repeat=0.0),
        "guided_metrics": _metrics(cosine=0.4, uniqueness=0.96, repeat=0.04),
        "acceptance": {
            "system": "pass",
            "application": "pass" if role == "primary" else "not_run",
        },
    }


def _bundle(tmp_path: Path) -> tuple[Path, list[dict[str, Any]]]:
    config = _config()
    config_path = tmp_path / "config.json"
    config_hash = _write_json(config_path, config)
    records = [
        _record("jp-a800-171", "primary", config_hash),
        _record("jp-a800-172", "portability_replay", config_hash),
    ]
    entries = []
    for record in records:
        path = tmp_path / f"{record['execution_host']}.json"
        entries.append({"path": path.name, "sha256": _write_json(path, record)})
    manifest = {
        "schema": "flagquantum.qboson_qdiffusion_manifest",
        "version": "1.0",
        "config": {"path": config_path.name, "sha256": config_hash},
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


def test_complete_two_host_real_provider_bundle_passes(tmp_path: Path) -> None:
    manifest_path, _ = _bundle(tmp_path)

    assert validate_acceptance(manifest_path) == []


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


def test_manifest_requires_same_trained_energy_checkpoint(tmp_path: Path) -> None:
    manifest_path, records = _bundle(tmp_path)
    changed = copy.deepcopy(records[1])
    changed["artifacts"]["trained_energy_checkpoint_sha256"] = "2" * 64
    _replace_record(manifest_path, 1, changed)

    assert any(
        "same trained energy checkpoint" in error
        for error in validate_acceptance(manifest_path)
    )
