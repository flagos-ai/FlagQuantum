from __future__ import annotations

from typing import Any

import pytest

from examples.qdiffusion_kaiwu.assemble_acceptance import assemble_records

pytestmark = pytest.mark.unit


def _config() -> dict[str, Any]:
    return {
        "primary_host": "jp-a800-171",
        "replay_host": "jp-a800-172",
        "seeds": [1701, 1702, 1703],
        "remote_call_budget": 128,
        "software": {
            "source_revision": "a" * 40,
            "kaiwu_pytorch_plugin_revision": "b" * 40,
            "python_version": "3.10.18",
            "torch_version": "2.7.0",
            "kaiwu_sdk_version": "1.3.1",
        },
        "dataset": {"sha256": "c" * 64},
        "checkpoint": {"sha256": "d" * 64},
        "tokenizer": {"sha256": "e" * 64},
        "evaluation_model": {"sha256": "f" * 64},
        "precision_policy": {
            "name": "explicit-int8",
            "target_min": -127,
            "target_max": 127,
        },
        "generation": {"portability_training_seed": 1701},
        "thresholds": {
            "uniqueness_baseline_fraction_min": 0.95,
            "repeat_ratio_absolute_increase_max": 0.05,
            "invalid_sequence_count_max": 0,
        },
    }


def _system(host: str, role: str, task_id: str) -> dict[str, Any]:
    return {
        "schema": "flagquantum.qboson_qdiffusion_system_live_probe",
        "version": "1.0",
        "source_revision": "a" * 40,
        "kaiwu_pytorch_plugin_revision": "b" * 40,
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
        "acceptance": {"system": "pass", "application": "not_run"},
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


def _components() -> tuple[
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
                    "seed": seed,
                    "run_completed": True,
                    "execution_host": "jp-a800-171",
                    "trained_energy_checkpoint_sha256": str(index + 3) * 64,
                },
                training_sha,
            )
        )
        evaluations.append(
            (
                {
                    "seed": seed,
                    "training_record_sha256": training_sha,
                    "experiment_config_sha256": "9" * 64,
                    "execution_host": "jp-a800-171",
                    "baseline_metrics": _metrics(0.6 + index * 0.01),
                    "guided_metrics": _metrics(0.4 + index * 0.01),
                },
                str(index + 6) * 64,
            )
        )
    return training, evaluations


def test_assembler_links_all_seeds_and_recomputes_metric_means() -> None:
    config = _config()
    training, evaluations = _components()
    selected_checkpoint = training[0][0]["trained_energy_checkpoint_sha256"]
    portability = {
        "execution_host": "jp-a800-172",
        "acceptance": {"portability": "pass"},
        "training_record_sha256": training[0][1],
        "trained_energy_checkpoint_sha256": selected_checkpoint,
    }

    primary, replay = assemble_records(
        config=config,
        config_sha256="9" * 64,
        primary_system=(_system("jp-a800-171", "primary", "primary-task"), "a" * 64),
        replay_system=(
            _system("jp-a800-172", "portability_replay", "replay-task"),
            "b" * 64,
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
            portability=(portability, "c" * 64),
            training_records=training,
            evaluation_records=evaluations,
        )
