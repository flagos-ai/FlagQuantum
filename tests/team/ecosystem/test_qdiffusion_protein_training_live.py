from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from examples.qdiffusion_kaiwu.qdiffusion_protein_training_live import (
    _build_workflow_config,
    _checkpoint_identity,
    _run_workflow,
    _workflow_artifact_identities,
    run_training_seed,
)
from flagquantum.ecosystem.kaiwu import KaiwuSampler

pytestmark = pytest.mark.unit


class _ConfigObject:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


def _frozen_config() -> dict[str, Any]:
    return {
        "requested_samples": 10,
        "remote_call_budget": 128,
        "precision_policy": {
            "name": "explicit-int8",
            "target_min": -127,
            "target_max": 127,
        },
        "dataset": {
            "min_length": 50,
            "max_length": 256,
            "max_records": 640,
            "validation_ratio": 0.05,
            "test_ratio": 0.05,
        },
        "training": {
            "freeze_proposal": True,
            "epochs": 20,
            "min_epochs": 3,
            "batch_size": 4,
            "learning_rate": 0.00005,
            "weight_decay": 0.01,
            "grad_clip_norm": 1.0,
            "num_candidates": 4,
            "validation_steps": 3,
            "scheduler_factor": 0.5,
            "scheduler_patience": 1,
            "early_stop_patience": 4,
            "remote_call_budget_per_seed": 71269,
        },
        "generation": {
            "num_candidates": 4,
            "energy_temperature": 1.25,
            "proposal_temperature": 0.3,
            "proposal_noise_scale": 1.0,
            "disable_resample": False,
            "resample_ratio": 0.2,
            "resample_top_p": 0.9,
            "max_steps": 64,
            "sequence_count": 32,
            "portability_training_seed": 1701,
            "portability_fixture_index": 0,
            "portability_steps": 3,
        },
    }


def _workflow_types() -> SimpleNamespace:
    return SimpleNamespace(
        WorkflowConfig=_ConfigObject,
        DataConfig=_ConfigObject,
        ModelConfig=_ConfigObject,
        SamplerConfig=_ConfigObject,
        TrainConfig=_ConfigObject,
        GenerateConfig=_ConfigObject,
    )


def test_build_workflow_config_maps_every_frozen_knob(tmp_path: Path) -> None:
    dataset = tmp_path / "proteins.fasta"
    checkpoint = tmp_path / "dplm"

    result = _build_workflow_config(
        _workflow_types(),
        _frozen_config(),
        dataset_path=dataset,
        checkpoint_path=checkpoint,
        seed=1701,
    )

    assert result.data.__dict__ == {
        "fasta_path": str(dataset),
        "min_length": 50,
        "max_length": 256,
        "max_records": 640,
        "val_ratio": 0.05,
        "test_ratio": 0.05,
        "seed": 1701,
    }
    assert result.model.proposal_ckpt == str(checkpoint)
    assert result.model.energy_ckpt == str(checkpoint)
    assert result.sampler.__dict__ == {
        "sampler_type": "flagquantum-kaiwu",
        "sampler_kwargs": {},
    }
    assert result.train.__dict__["require_cuda"] is True
    assert result.train.__dict__["epochs"] == 20
    assert result.generate.__dict__["steps"] == 64
    assert result.generate.__dict__["resample_top_p"] == 0.9


def test_run_workflow_binds_sampler_and_restores_factories(tmp_path: Path) -> None:
    sampler = cast(KaiwuSampler, object())
    observed: list[object] = []

    def original_builder(**kwargs: Any) -> SimpleNamespace:
        observed.append(kwargs["bm_sampler"])
        return SimpleNamespace(
            energy_model=SimpleNamespace(sampler=kwargs["bm_sampler"])
        )

    def original_config() -> str:
        return "original-config"

    def original_outputs() -> Path:
        return tmp_path / "original"

    workflow = SimpleNamespace(
        build_qdiffusion=original_builder,
        build_default_workflow_config=original_config,
        default_outputs_root=original_outputs,
    )

    def main() -> None:
        assert workflow.build_default_workflow_config() == "frozen-config"
        workflow.build_qdiffusion(bm_sampler_type="sa")
        (workflow.default_outputs_root() / "real_full_workflow_test").mkdir()

    workflow.main = main
    result = _run_workflow(workflow, "frozen-config", sampler, tmp_path / "runs")

    assert result.name == "real_full_workflow_test"
    assert observed == [sampler]
    assert workflow.build_qdiffusion is original_builder
    assert workflow.build_default_workflow_config is original_config
    assert workflow.default_outputs_root is original_outputs


def test_checkpoint_identity_selects_latest_best_epoch(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "best_epoch_2.pt").write_bytes(b"older")
    (checkpoint_dir / "best_epoch_10.pt").write_bytes(b"newer")

    name, digest = _checkpoint_identity(tmp_path)

    assert name == "best_epoch_10.pt"
    assert digest == "804f51f71254c4081e37e7c887073560f4a6fa6cdad202e9ac67e032c43ed1e1"


def test_workflow_artifact_identities_freeze_evaluation_inputs(
    tmp_path: Path,
) -> None:
    paths = (
        "data_splits/test.fasta",
        "baseline/proposal_only_generated_sequences.fasta",
        "guided/energy_guided_generated_sequences.fasta",
        "history.json",
        "baseline_vs_guided.json",
        "baseline_eval/quality_summary.json",
        "guided_eval/quality_summary.json",
    )
    for relative in paths:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")

    identities = _workflow_artifact_identities(tmp_path)

    assert set(identities) == {
        "test_fasta",
        "baseline_fasta",
        "guided_fasta",
        "training_history",
        "sequence_metrics",
        "baseline_quality",
        "guided_quality",
    }
    assert identities["test_fasta"]["relative_path"] == "data_splits/test.fasta"
    assert len(identities["guided_fasta"]["sha256"]) == 64


def test_live_training_source_guards_cost_and_preflights_before_credentials() -> None:
    source = (
        Path(__file__).parents[3]
        / "examples"
        / "qdiffusion_kaiwu"
        / "qdiffusion_protein_training_live.py"
    ).read_text(encoding="utf-8")

    assert "ACKNOWLEDGEMENT" in source
    assert source.index("preflight_artifacts(") < source.index(
        "resolve_kaiwu_credentials()"
    )
    assert source.index("load_source_preflight(") < source.index(
        "resolve_kaiwu_credentials()"
    )
    assert source.index("verify_frozen_environment_lock(") < source.index(
        "resolve_kaiwu_credentials()"
    )
    assert '"application": "not_evaluated"' in source
    assert "HF_HUB_OFFLINE" in source


def test_training_seed_records_interruption_without_claiming_acceptance(
    tmp_path: Path,
) -> None:
    workflow = _workflow_types()
    workflow.build_qdiffusion = lambda **kwargs: SimpleNamespace(
        energy_model=SimpleNamespace(sampler=kwargs["bm_sampler"])
    )
    workflow.build_default_workflow_config = lambda: None
    workflow.default_outputs_root = lambda: tmp_path

    def interrupted() -> None:
        raise KeyboardInterrupt

    workflow.main = interrupted
    receipt = SimpleNamespace(
        task_name="protein-task",
        matrix_sha256="1" * 64,
        mode="sampling",
        requested_samples=10,
        project_no="CPQC-test",
        submitted_at="2026-10-05T00:00:00+00:00",
        provider_task_id="provider-task",
        provider_target="SPQC-provider",
    )
    precision_report = SimpleNamespace(
        scale_factor=2.0,
        max_abs_error=0.25,
        mean_abs_error=0.125,
    )
    sampler = cast(
        KaiwuSampler,
        SimpleNamespace(
            client=object(),
            remote_call_count=1,
            receipts=(receipt,),
            precision_reports=(precision_report,),
        ),
    )

    record = run_training_seed(
        workflow=workflow,
        config=_frozen_config(),
        config_sha256="a" * 64,
        sampler=sampler,
        dataset_path=tmp_path / "dataset.fasta",
        checkpoint_path=tmp_path / "dplm",
        output_root=tmp_path / "runs",
        seed=1701,
        execution_host="jp-a800-171",
        observed_hostname="host-171",
        observed_gpu="NVIDIA A800-SXM4-80GB",
        source_revision="b" * 40,
        plugin_revision="c" * 40,
        source_preflight_sha256="e" * 64,
        transfer_manifest_sha256="f" * 64,
        environment_lock_sha256="0" * 64,
        sdk_version="1.3.1",
        preflight_sha256="d" * 64,
    )

    assert record["run_completed"] is False
    assert record["source_preflight_sha256"] == "e" * 64
    assert record["transfer_manifest_sha256"] == "f" * 64
    assert record["environment_lock_sha256"] == "0" * 64
    assert record["failure"]["type"] == "KeyboardInterrupt"
    assert record["transport"] == "injected_test"
    assert record["pinned_sdk_client"] is False
    assert record["real_provider_evidence"] is False
    assert record["qboson_hardware_used"] is False
    assert record["qboson_target"] == "SPQC-provider"
    assert record["qboson_task_ids"] == ["provider-task"]
    assert record["requested_samples"] == 10
    assert record["precision_evidence_complete"] is True
    assert record["precision_policy"] == {
        "name": "explicit-int8",
        "target_min": -127,
        "target_max": 127,
        "matrix_count": 1,
        "scale_factor_min": 2.0,
        "scale_factor_max": 2.0,
        "max_abs_error": 0.25,
        "mean_of_matrix_mean_abs_error": 0.125,
    }
    assert record["acceptance"] == {
        "system": "not_evaluated",
        "application": "not_evaluated",
    }
