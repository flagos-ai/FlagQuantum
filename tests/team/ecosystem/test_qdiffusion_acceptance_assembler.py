from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from examples.qdiffusion_kaiwu.assemble_acceptance import (
    _load_component,
    assemble_records,
)
from examples.qdiffusion_kaiwu.validate_acceptance import validate_acceptance
from tests.team.ecosystem.test_qdiffusion_acceptance_validator import (
    _config as _full_config,
)

pytestmark = pytest.mark.unit

PRIMARY_SOURCE_PREFLIGHT_SHA = "d" * 64
REPLAY_SOURCE_PREFLIGHT_SHA = "e" * 64
TRANSFER_MANIFEST_SHA = "f" * 64


def _config() -> dict[str, Any]:
    return _full_config()


def _system(host: str, role: str, task_id: str) -> dict[str, Any]:
    return {
        "schema": "flagquantum.qboson_qdiffusion_system_live_probe",
        "version": "1.0",
        "source_revision": "a" * 40,
        "kaiwu_pytorch_plugin_revision": "b" * 40,
        "source_preflight_sha256": (
            PRIMARY_SOURCE_PREFLIGHT_SHA
            if host == "jp-a800-171"
            else REPLAY_SOURCE_PREFLIGHT_SHA
        ),
        "transfer_manifest_sha256": TRANSFER_MANIFEST_SHA,
        "python_version": "3.10.18",
        "torch_version": "2.7.0",
        "kaiwu_sdk_version": "1.3.1",
        "execution_host": host,
        "run_role": role,
        "requested_cuda_device": "cuda:0",
        "observed_tensor_device": "cuda:0",
        "observed_gpu_model": "NVIDIA A800-SXM4-80GB",
        "transport": "kaiwu_cim",
        "qboson_hardware_used": True,
        "real_provider_evidence": True,
        "provider_reported_target": True,
        "qboson_target": "SPQC-provider",
        "qboson_task_ids": [task_id],
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
        "training": {
            "objective": -0.5,
            "gradient_norm": 1.0,
            "parameter_delta_max": 0.01,
        },
        "generation": {"token_constraints_passed": True},
        "transfer_accounting": {
            "matrix_origin_device": "cuda:0",
            "sampler_boundaries": [
                {
                    "input_type": "numpy.ndarray",
                    "input_device": "cpu",
                    "input_dtype": "float32",
                    "matrix_shape": [3, 3],
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
    }


def _source_preflight(host: str) -> dict[str, Any]:
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
        "manifest_sha256": TRANSFER_MANIFEST_SHA,
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
                    "seed": seed,
                    "run_completed": True,
                    "execution_host": "jp-a800-171",
                    "source_preflight_sha256": source_preflight_sha256,
                    "transfer_manifest_sha256": TRANSFER_MANIFEST_SHA,
                    "experiment_config_sha256": config_sha256,
                    "trained_energy_checkpoint_sha256": str(index + 3) * 64,
                },
                training_sha,
            )
        )
        evaluations.append(
            (
                {
                    "schema": "flagquantum.qboson_qdiffusion_protein_evaluation",
                    "version": "1.0",
                    "seed": seed,
                    "training_record_sha256": training_sha,
                    "experiment_config_sha256": config_sha256,
                    "execution_host": "jp-a800-171",
                    "source_preflight_sha256": source_preflight_sha256,
                    "transfer_manifest_sha256": TRANSFER_MANIFEST_SHA,
                    "baseline_metrics": _metrics(0.6 + index * 0.01),
                    "guided_metrics": _metrics(0.4 + index * 0.01),
                },
                str(index + 6) * 64,
            )
        )
    return training, evaluations


def _write_json(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_assembler_links_all_seeds_and_recomputes_metric_means() -> None:
    config = _config()
    training, evaluations = _components()
    selected_checkpoint = training[0][0]["trained_energy_checkpoint_sha256"]
    portability = {
        "execution_host": "jp-a800-172",
        "acceptance": {"portability": "pass"},
        "training_record_sha256": training[0][1],
        "trained_energy_checkpoint_sha256": selected_checkpoint,
        "source_preflight_sha256": REPLAY_SOURCE_PREFLIGHT_SHA,
        "transfer_manifest_sha256": TRANSFER_MANIFEST_SHA,
    }

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
        portability=(portability, "c" * 64),
        training_records=training,
        evaluation_records=evaluations,
    )

    assert primary["attempted_seeds"] == [1701, 1702, 1703]
    assert primary["baseline_metrics"]["mean_cosine_distance"] == pytest.approx(0.62)
    assert primary["guided_metrics"]["mean_cosine_distance"] == pytest.approx(0.42)
    assert len(primary["application_evidence"]["records"]) == 3
    assert replay["portability_evidence"]["trained_energy_checkpoint_sha256"] == (
        selected_checkpoint
    )
    assert replay["artifacts"]["trained_energy_checkpoint_sha256"] == (
        selected_checkpoint
    )


def test_assembler_rejects_evaluation_linked_to_another_training_record() -> None:
    config = _config()
    training, evaluations = _components()
    evaluations[1][0]["training_record_sha256"] = "0" * 64
    portability = {
        "execution_host": "jp-a800-172",
        "acceptance": {"portability": "pass"},
        "training_record_sha256": training[0][1],
        "trained_energy_checkpoint_sha256": training[0][0][
            "trained_energy_checkpoint_sha256"
        ],
        "source_preflight_sha256": REPLAY_SOURCE_PREFLIGHT_SHA,
        "transfer_manifest_sha256": TRANSFER_MANIFEST_SHA,
    }

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
            portability=(portability, "c" * 64),
            training_records=training,
            evaluation_records=evaluations,
        )


def test_assembled_component_bundle_passes_final_validator(tmp_path: Path) -> None:
    config = _config()
    config_path = tmp_path / "acceptance_config.json"
    config_sha = _write_json(config_path, config)

    primary_preflight_path = tmp_path / "components" / "primary-source-preflight.json"
    replay_preflight_path = tmp_path / "components" / "replay-source-preflight.json"
    primary_preflight_record = _source_preflight("jp-a800-171")
    replay_preflight_record = _source_preflight("jp-a800-172")
    primary_preflight_sha = _write_json(
        primary_preflight_path, primary_preflight_record
    )
    replay_preflight_sha = _write_json(replay_preflight_path, replay_preflight_record)

    primary_system_path = tmp_path / "components" / "primary-system.json"
    replay_system_path = tmp_path / "components" / "replay-system.json"
    primary_system_record = _system("jp-a800-171", "primary", "primary-task")
    replay_system_record = _system("jp-a800-172", "portability_replay", "replay-task")
    primary_system_record["source_preflight_sha256"] = primary_preflight_sha
    replay_system_record["source_preflight_sha256"] = replay_preflight_sha
    primary_system_record["experiment_config_sha256"] = config_sha
    replay_system_record["experiment_config_sha256"] = config_sha
    primary_system_sha = _write_json(primary_system_path, primary_system_record)
    replay_system_sha = _write_json(replay_system_path, replay_system_record)

    training_templates, evaluation_templates = _components(
        config_sha, primary_preflight_sha
    )
    training_entries: list[tuple[dict[str, Any], str]] = []
    evaluation_entries: list[tuple[dict[str, Any], str]] = []
    component_paths = [
        primary_system_path,
        replay_system_path,
        primary_preflight_path,
        replay_preflight_path,
    ]
    for index, (training_record, _) in enumerate(training_templates):
        path = tmp_path / "components" / f"training-{index}.json"
        digest = _write_json(path, training_record)
        training_entries.append((training_record, digest))
        component_paths.append(path)
    for index, (evaluation_record, _) in enumerate(evaluation_templates):
        evaluation_record["training_record_sha256"] = training_entries[index][1]
        path = tmp_path / "components" / f"evaluation-{index}.json"
        digest = _write_json(path, evaluation_record)
        evaluation_entries.append((evaluation_record, digest))
        component_paths.append(path)

    selected_checkpoint = training_entries[0][0]["trained_energy_checkpoint_sha256"]
    portability_record = {
        "schema": "flagquantum.qboson_qdiffusion_portability_replay",
        "version": "1.0",
        "experiment_config_sha256": config_sha,
        "execution_host": "jp-a800-172",
        "acceptance": {"portability": "pass"},
        "training_record_sha256": training_entries[0][1],
        "trained_energy_checkpoint_sha256": selected_checkpoint,
        "source_preflight_sha256": replay_preflight_sha,
        "transfer_manifest_sha256": TRANSFER_MANIFEST_SHA,
    }
    portability_path = tmp_path / "components" / "portability.json"
    portability_sha = _write_json(portability_path, portability_record)
    component_paths.append(portability_path)

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

    loaded_training, loaded_sha = _load_component(
        component_paths[4], "flagquantum.qboson_qdiffusion_protein_training"
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

    tampered_evaluation = json.loads(component_paths[7].read_text(encoding="utf-8"))
    tampered_evaluation["guided_metrics"]["mean_cosine_distance"] = 0.99
    tampered_evaluation_sha = _write_json(component_paths[7], tampered_evaluation)
    primary["application_evidence"]["records"][0][
        "evaluation_record_sha256"
    ] = tampered_evaluation_sha
    primary_sha = _write_json(primary_path, primary)
    manifest["records"][0]["sha256"] = primary_sha
    manifest["component_records"][7]["sha256"] = tampered_evaluation_sha
    _write_json(manifest_path, manifest)

    assert any(
        "differs from component aggregation" in error
        for error in validate_acceptance(manifest_path)
    )
