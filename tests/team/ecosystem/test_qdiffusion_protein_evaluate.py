from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from examples.qdiffusion_kaiwu import provider_reconciliation as reconciliation
from examples.qdiffusion_kaiwu import qdiffusion_protein_evaluate as evaluation_module
from examples.qdiffusion_kaiwu.qdiffusion_protein_evaluate import (
    _invalid_sequence_count,
    _load_training_record,
    _read_aligned_records,
    _source_preflight_identity,
    _validate_evaluation_candidate,
    _verified_evaluation_source,
    _verified_training_paths,
    evaluate_outputs,
)
from examples.qdiffusion_kaiwu.validate_acceptance import (
    EVALUATION_COMPONENT_FIELDS,
)

pytestmark = pytest.mark.unit


def test_evaluation_preflights_private_output_before_workflow() -> None:
    source = (
        Path(__file__).parents[3]
        / "examples"
        / "qdiffusion_kaiwu"
        / "qdiffusion_protein_evaluate.py"
    ).read_text(encoding="utf-8")

    assert source.index(
        "validate_private_json_output_path(args.output)"
    ) < source.index("config, config_sha256 = _load_frozen_config(args.config)")
    assert source.index(
        "paths = _verified_training_paths(args.run_directory, record)"
    ) < source.index('device = torch.device("cuda:0")')
    assert source.rindex(
        "_artifact_identity_snapshot(args.evaluation_model)"
    ) < source.index('device = torch.device("cuda:0")')
    write_index = source.index("_write_private_redacted_json(args.output")
    assert (
        source.rindex(
            'revalidate_regular_file(evaluation_snapshot, label="ESM2 checkpoint")'
        )
        < write_index
    )
    assert source.rindex("_revalidate_training_paths(paths)") < write_index


def _provider_training_record() -> dict[str, Any]:
    return {
        "schema": "flagquantum.qboson_qdiffusion_protein_training",
        "version": "1.0",
        "run_completed": True,
        "artifact_inputs_unchanged": True,
        "transport": "kaiwu_cim",
        "pinned_sdk_client": True,
        "real_provider_evidence": True,
        "qboson_hardware_used": True,
        "provider_identity_complete": True,
        "provider_reported_target": True,
        "qboson_target": "SPQC-provider",
        "qboson_task_ids": ["provider-task"],
        "sampling_mode": "sampling",
        "requested_samples": 10,
        "fallback_occurred": False,
        "secrets_redacted": True,
        "remote_call_count": 1,
        "protein_remote_call_budget_per_seed": 10,
        "precision_report_count": 1,
        "precision_policy": {
            "matrix_count": 1,
            "target_min": -127,
            "target_max": 127,
            "scale_factor_min": 1.0,
            "scale_factor_max": 1.0,
            "max_abs_error": 0.0,
            "mean_of_matrix_mean_abs_error": 0.0,
        },
        "precision_evidence": [
            {
                "original_matrix_sha256": "b" * 64,
                "submission_matrix_sha256": "a" * 64,
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
                "task_name": "protein-task",
                "matrix_sha256": "a" * 64,
                "matrix_size": 3,
                "mode": "sampling",
                "requested_samples": 10,
                "project_no": "CPQC-test",
                "submitted_at": "2026-10-05T00:00:00+00:00",
                "provider_task_id": "provider-task",
                "provider_target": "SPQC-provider",
            }
        ],
    }


def _evaluation_metrics() -> dict[str, float | int]:
    return {
        "mean_cosine_distance": 0.2,
        "median_cosine_distance": 0.2,
        "mean_l2_distance": 1.0,
        "median_l2_distance": 1.0,
        "identity_to_reference_mean": 0.8,
        "amino_acid_jsd": 0.1,
        "kmer2_jsd": 0.1,
        "kmer3_jsd": 0.1,
        "uniqueness_ratio": 1.0,
        "repeat_ratio_ge4": 0.0,
        "length_match_ratio": 1.0,
        "invalid_sequence_count": 0,
    }


def _evaluation_config_and_record() -> tuple[dict[str, Any], dict[str, Any]]:
    software = {
        "source_revision": "a" * 40,
        "flagquantum_version": "0.2.0",
        "kaiwu_pytorch_plugin_revision": "b" * 40,
        "python_version": "3.10.18",
        "torch_version": "2.7.0",
        "kaiwu_sdk_version": "1.3.1",
        "environment_lock_sha256": "c" * 64,
    }
    config = {
        "software": software,
        "primary_host": "jp-a800-171",
        "evaluation_model": {"sha256": "d" * 64},
    }
    record = {
        "schema": "flagquantum.qboson_qdiffusion_protein_evaluation",
        "version": "1.0",
        "recorded_at": "2026-10-06T00:00:00+00:00",
        "experiment_config_sha256": "e" * 64,
        "training_record_sha256": "f" * 64,
        **software,
        "source_preflight_sha256": "1" * 64,
        "transfer_manifest_sha256": "2" * 64,
        "execution_host": "jp-a800-171",
        "observed_hostname": "node-171",
        "observed_gpu_model": "NVIDIA A800-SXM4-80GB",
        "observed_tensor_device": "cuda:0",
        "seed": 1701,
        "evaluation_model_sha256": "d" * 64,
        "artifact_inputs_unchanged": True,
        "baseline_metrics": _evaluation_metrics(),
        "guided_metrics": _evaluation_metrics(),
        "secrets_redacted": True,
        "provider_quota_consumed": False,
        "acceptance": "candidate_evidence_only",
    }
    assert set(record) == EVALUATION_COMPONENT_FIELDS
    return config, record


def test_evaluation_candidate_is_validated_before_publication() -> None:
    config, record = _evaluation_config_and_record()

    assert (
        _validate_evaluation_candidate(record, config=config, config_sha256="e" * 64)
        == []
    )

    record["guided_metrics"]["uniqueness_ratio"] = 1.1
    errors = _validate_evaluation_candidate(
        record, config=config, config_sha256="e" * 64
    )
    assert any("expected a ratio in [0, 1]" in error for error in errors)


def test_evaluation_metrics_reject_undeclared_extensions() -> None:
    config, record = _evaluation_config_and_record()
    record["baseline_metrics"]["provider_note"] = 1.0

    errors = _validate_evaluation_candidate(
        record, config=config, config_sha256="e" * 64
    )

    assert any("metric field set is not closed" in error for error in errors)


def _write(path: Path, content: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(0o600)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _training_artifacts(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    contents = {
        "test_fasta": ("data_splits/test.fasta", ">p1\nACDE\n"),
        "baseline_fasta": (
            "baseline/proposal_only_generated_sequences.fasta",
            ">p1\nACDF\n",
        ),
        "guided_fasta": (
            "guided/energy_guided_generated_sequences.fasta",
            ">p1\nACDG\n",
        ),
        "baseline_quality": (
            "baseline_eval/quality_summary.json",
            json.dumps(
                {
                    "identity_to_reference_mean": 0.75,
                    "amino_acid_jsd": 0.2,
                    "kmer2_jsd": 0.3,
                    "kmer3_jsd": 0.4,
                    "uniqueness_ratio": 1.0,
                    "repeat_ratio_ge4": 0.0,
                    "length_match_ratio": 1.0,
                }
            ),
        ),
        "guided_quality": (
            "guided_eval/quality_summary.json",
            json.dumps(
                {
                    "identity_to_reference_mean": 0.8,
                    "amino_acid_jsd": 0.1,
                    "kmer2_jsd": 0.2,
                    "kmer3_jsd": 0.3,
                    "uniqueness_ratio": 1.0,
                    "repeat_ratio_ge4": 0.0,
                    "length_match_ratio": 1.0,
                }
            ),
        ),
    }
    identities: dict[str, dict[str, str]] = {}
    paths: dict[str, Path] = {}
    for name, (relative, content) in contents.items():
        path = tmp_path / relative
        identities[name] = {
            "relative_path": relative,
            "sha256": _write(path, content),
        }
        paths[name] = path
    record = {
        "run_directory_name": tmp_path.name,
        "workflow_artifacts": identities,
    }
    return record, paths


def test_verified_training_paths_recomputes_every_digest(tmp_path: Path) -> None:
    record, expected = _training_artifacts(tmp_path)

    assert _verified_training_paths(tmp_path, record) == expected

    expected["guided_fasta"].write_text(">p1\nXXXX\n", encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch: guided_fasta"):
        _verified_training_paths(tmp_path, record)


def test_verified_training_paths_requires_private_run_directory(
    tmp_path: Path,
) -> None:
    record, _ = _training_artifacts(tmp_path)
    tmp_path.chmod(0o755)

    with pytest.raises(ValueError, match="private, non-symlink directory"):
        _verified_training_paths(tmp_path, record)


def test_verified_training_paths_rejects_symlinked_run_directory(
    tmp_path: Path,
) -> None:
    real_run = tmp_path / "real-run"
    real_run.mkdir(mode=0o700)
    record, _ = _training_artifacts(real_run)
    linked_run = tmp_path / "linked-run"
    linked_run.symlink_to(real_run, target_is_directory=True)

    with pytest.raises(ValueError, match="private, non-symlink directory"):
        _verified_training_paths(linked_run, record)


def test_verified_training_paths_rejects_public_artifact(tmp_path: Path) -> None:
    record, paths = _training_artifacts(tmp_path)
    paths["guided_fasta"].chmod(0o644)

    with pytest.raises(ValueError, match="must be owner-only: guided_fasta"):
        _verified_training_paths(tmp_path, record)


def test_verified_training_paths_rejects_foreign_owned_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record, paths = _training_artifacts(tmp_path)
    monkeypatch.setattr(
        evaluation_module,
        "_effective_uid",
        lambda: paths["guided_fasta"].stat().st_uid + 1,
    )

    with pytest.raises(ValueError, match="owned by the current effective user"):
        _verified_training_paths(tmp_path, record)


def test_training_record_loader_rejects_injected_provider_evidence(
    tmp_path: Path,
) -> None:
    path = tmp_path / "training.json"
    record = _provider_training_record()
    path.write_text(json.dumps(record), encoding="utf-8")
    path.chmod(0o600)

    loaded, digest = _load_training_record(path)

    assert loaded == record
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()

    record["transport"] = "injected_test"
    record["real_provider_evidence"] = False
    path.write_text(json.dumps(record), encoding="utf-8")
    path.chmod(0o600)
    with pytest.raises(ValueError, match="provider evidence is invalid"):
        _load_training_record(path)


def test_training_record_loader_applies_bill_reconciliation_without_rehashing(
    tmp_path: Path,
) -> None:
    record = _provider_training_record()
    record.update(
        real_provider_evidence=False,
        qboson_hardware_used=False,
        provider_identity_complete=False,
        provider_reported_target=False,
        qboson_target=None,
        qboson_task_ids=[],
    )
    record["task_receipts"][0]["provider_task_id"] = None
    record["task_receipts"][0]["provider_target"] = None
    path = tmp_path / "training.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    path.chmod(0o600)
    original_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    reconciliation_record = {
        "schema": reconciliation.SCHEMA,
        "version": reconciliation.VERSION,
        "captured_at": "2026-10-05T00:01:00+00:00",
        "source": reconciliation.SOURCE,
        "component_record_sha256": original_digest,
        "local_task_name": "protein-task",
        "local_submitted_at": "2026-10-05T00:00:00+00:00",
        "matrix_sha256": "a" * 64,
        "mode": "sampling",
        "requested_samples": 10,
        "provider_batch_id": "provider-batch",
        "associated_task_id": "provider-task",
        "provider_target": "SPQC-provider",
        "transaction_channel": reconciliation.TRANSACTION_CHANNEL,
        "transaction_type": reconciliation.TRANSACTION_TYPE,
        "resource_delta": -10,
        "unique_account_match": True,
        "hardware_acceptance": False,
        "qdiffusion_acceptance": False,
        "claim_boundary": reconciliation.CLAIM_BOUNDARY,
    }
    reconciliation_path = tmp_path / "reconciliation.json"
    reconciliation_path.write_text(
        json.dumps(reconciliation_record), encoding="utf-8"
    )
    reconciliation_path.chmod(0o600)

    loaded, digest = _load_training_record(
        path, reconciliation_paths=(reconciliation_path,)
    )

    assert digest == original_digest
    assert loaded["provider_identity_complete"] is True
    assert loaded["real_provider_evidence"] is True
    assert loaded["qboson_hardware_used"] is True
    assert loaded["qboson_task_ids"] == ["provider-task"]
    assert loaded["task_receipts"][0]["provider_task_id"] == "provider-task"
    assert loaded["task_receipts"][0]["provider_target"] == "SPQC-provider"

    reconciliation_record["component_record_sha256"] = "f" * 64
    reconciliation_path.write_text(
        json.dumps(reconciliation_record), encoding="utf-8"
    )
    reconciliation_path.chmod(0o600)
    with pytest.raises(ValueError, match="another training record"):
        _load_training_record(path, reconciliation_paths=(reconciliation_path,))


@pytest.mark.parametrize("unsafe_kind", ("public-file", "public-parent", "symlink"))
def test_training_record_loader_requires_private_anchored_input(
    tmp_path: Path, unsafe_kind: str
) -> None:
    private_parent = tmp_path / "private"
    private_parent.mkdir(mode=0o700)
    path = private_parent / "training.json"
    path.write_text(json.dumps(_provider_training_record()), encoding="utf-8")
    path.chmod(0o600)
    candidate = path
    if unsafe_kind == "public-file":
        path.chmod(0o644)
    elif unsafe_kind == "public-parent":
        private_parent.chmod(0o755)
    else:
        candidate = private_parent / "training-link.json"
        candidate.symlink_to(path)

    with pytest.raises(ValueError, match="protein-training record"):
        _load_training_record(candidate)


def test_aligned_fasta_reader_rejects_header_drift(tmp_path: Path) -> None:
    record, paths = _training_artifacts(tmp_path)
    del record
    paths["guided_fasta"].write_text(">another\nACDG\n", encoding="utf-8")
    workflow = SimpleNamespace(
        read_fasta_records=lambda path: [
            tuple(line.split("\n"))
            for line in path.read_text(encoding="utf-8").strip().split("\n>")
        ],
        normalize_sequence=lambda sequence: sequence,
    )

    with pytest.raises(ValueError, match="headers are not aligned"):
        _read_aligned_records(workflow, paths, expected_count=1)


def test_aligned_fasta_reader_rejects_change_during_plugin_consumption(
    tmp_path: Path,
) -> None:
    record, _ = _training_artifacts(tmp_path)
    paths = _verified_training_paths(tmp_path, record)
    calls = 0

    def read_then_mutate(path: Path) -> list[tuple[str, str]]:
        nonlocal calls
        calls += 1
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        if calls == 1:
            path.write_text(">p1\nXXXX\n", encoding="utf-8")
        return [(lines[0][1:], lines[1])]

    workflow = SimpleNamespace(
        read_fasta_records=read_then_mutate,
        normalize_sequence=lambda sequence: sequence,
    )

    with pytest.raises(ValueError, match="changed after identity capture"):
        _read_aligned_records(workflow, paths, expected_count=1)


def test_invalid_sequence_count_is_fail_closed() -> None:
    records = [("ok", "ACDE"), ("empty", ""), ("bad", "ACD-")]

    assert _invalid_sequence_count(records) == 2


def test_evaluation_requires_training_source_preflight_identity() -> None:
    record = {
        "source_preflight_sha256": "a" * 64,
        "transfer_manifest_sha256": "b" * 64,
    }

    assert _source_preflight_identity(record) == ("a" * 64, "b" * 64)

    record["source_preflight_sha256"] = "missing"
    with pytest.raises(ValueError, match="source_preflight_sha256"):
        _source_preflight_identity(record)


def test_evaluation_revalidates_local_source_preflight() -> None:
    source = (
        Path(__file__).parents[3]
        / "examples"
        / "qdiffusion_kaiwu"
        / "qdiffusion_protein_evaluate.py"
    ).read_text(encoding="utf-8")

    assert 'parser.add_argument("--source-preflight"' in source
    assert source.index(
        "record, record_sha256 = _load_training_record("
    ) < source.index("torch.cuda.set_device(device)")
    assert source.index("_load_pinned_qdiffusion_api(root)") < source.index(
        'importlib.import_module("dplm.workflows.esm2_eval")'
    )
    assert source.index("load_source_preflight(") < source.index(
        "workflow, helpers = _load_pinned_eval_workflow("
    )
    assert source.index("verify_frozen_environment_lock(") < source.index(
        "workflow, helpers = _load_pinned_eval_workflow("
    )
    assert source.index("verify_approved_kaiwu_distribution(") < source.index(
        "workflow, helpers = _load_pinned_eval_workflow("
    )
    assert "evaluation source preflight differs from training record" in source
    assert "evaluation transfer manifest differs from training record" in source


def test_evaluation_source_must_match_training_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    training_record = {
        "source_preflight_sha256": "a" * 64,
        "transfer_manifest_sha256": "b" * 64,
    }
    monkeypatch.setattr(
        "examples.qdiffusion_kaiwu.qdiffusion_protein_evaluate.load_source_preflight",
        lambda *args, **kwargs: ({"manifest_sha256": "b" * 64}, "a" * 64),
    )
    arguments = {
        "execution_host": "jp-a800-171",
        "source_revision": "c" * 40,
        "plugin_revision": "d" * 40,
        "source_root": tmp_path / "FlagQuantum-cccccccccc",
        "plugin_root": tmp_path / "kaiwu-pytorch-plugin-dddddddddd",
    }

    assert _verified_evaluation_source(
        tmp_path / "preflight.json", training_record, **arguments
    ) == ("a" * 64, "b" * 64)

    monkeypatch.setattr(
        "examples.qdiffusion_kaiwu.qdiffusion_protein_evaluate.load_source_preflight",
        lambda *args, **kwargs: ({"manifest_sha256": "b" * 64}, "e" * 64),
    )
    with pytest.raises(ValueError, match="source preflight differs"):
        _verified_evaluation_source(
            tmp_path / "preflight.json", training_record, **arguments
        )


@dataclass
class _DistanceSummary:
    label: str
    paired_count: int
    mean_cosine_distance: float
    median_cosine_distance: float
    mean_l2_distance: float
    median_l2_distance: float


class _Model:
    def eval(self) -> _Model:
        return self

    def to(self, device: torch.device) -> _Model:
        assert str(device) == "cpu"
        return self


def test_evaluate_outputs_uses_local_model_and_merges_sequence_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, paths = _training_artifacts(tmp_path)
    loader_paths: list[str] = []

    def load_checkpoint(
        path: str, *, map_location: str, weights_only: bool
    ) -> dict[str, object]:
        assert argparse.Namespace in torch.serialization.get_safe_globals()
        assert map_location == "cpu"
        assert weights_only is True
        loader_paths.append(path)
        return {"cfg": {}, "model": {}}

    def load_core(
        model_name: str,
        model_data: dict[str, object],
        regression_data: None,
    ) -> tuple[_Model, object]:
        assert model_name == "esm2"
        assert model_data == {"cfg": {}, "model": {}}
        assert regression_data is None
        return _Model(), object()

    monkeypatch.setattr(evaluation_module.torch, "load", load_checkpoint)
    helpers = SimpleNamespace(
        esm=SimpleNamespace(
            pretrained=SimpleNamespace(load_model_and_alphabet_core=load_core)
        )
    )

    def read_fasta(path: Path) -> list[tuple[str, str]]:
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        return [(lines[0][1:], lines[1])]

    def evaluate(**kwargs: Any) -> tuple[list[object], _DistanceSummary]:
        label = kwargs["label"]
        value = 0.5 if label == "baseline" else 0.4
        return [], _DistanceSummary(label, 1, value, value, 1.0, 1.0)

    workflow = SimpleNamespace(
        read_fasta_records=read_fasta,
        normalize_sequence=lambda sequence: sequence,
        embed_sequences=lambda records, **kwargs: {
            header: sequence for header, sequence in records
        },
        evaluate_candidate_set=evaluate,
    )
    checkpoint = tmp_path / "esm2.pt"
    checkpoint.write_bytes(b"esm2")
    config = {
        "generation": {"sequence_count": 1},
        "evaluation": {"batch_size": 1, "pooling": "mean", "pair_mode": "order"},
    }

    baseline, guided = evaluate_outputs(
        workflow=workflow,
        helpers=helpers,
        paths=paths,
        evaluation_checkpoint=checkpoint,
        config=config,
        device=torch.device("cpu"),
    )

    assert loader_paths == [str(checkpoint)]
    assert baseline["mean_cosine_distance"] == 0.5
    assert guided["mean_cosine_distance"] == 0.4
    assert guided["identity_to_reference_mean"] == 0.8
    assert baseline["invalid_sequence_count"] == 0
    assert set(baseline) == set(evaluation_module.METRIC_NAMES)
    assert set(guided) == set(evaluation_module.METRIC_NAMES)


def test_evaluate_outputs_rejects_esm2_change_during_local_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, paths = _training_artifacts(tmp_path)
    checkpoint = tmp_path / "esm2.pt"
    checkpoint.write_bytes(b"esm2")

    def load_then_mutate(
        path: str, *, map_location: str, weights_only: bool
    ) -> dict[str, object]:
        assert path == str(checkpoint)
        assert map_location == "cpu"
        assert weights_only is True
        checkpoint.write_bytes(b"changed")
        return {"cfg": {}, "model": {}}

    monkeypatch.setattr(evaluation_module.torch, "load", load_then_mutate)
    helpers = SimpleNamespace(
        esm=SimpleNamespace(
            pretrained=SimpleNamespace(
                load_model_and_alphabet_core=lambda *args: (_Model(), object())
            )
        )
    )

    def read_fasta(path: Path) -> list[tuple[str, str]]:
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        return [(lines[0][1:], lines[1])]

    workflow = SimpleNamespace(
        read_fasta_records=read_fasta,
        normalize_sequence=lambda sequence: sequence,
    )
    config = {
        "generation": {"sequence_count": 1},
        "evaluation": {"batch_size": 1, "pooling": "mean", "pair_mode": "order"},
    }

    with pytest.raises(ValueError, match="changed after identity capture"):
        evaluate_outputs(
            workflow=workflow,
            helpers=helpers,
            paths=paths,
            evaluation_checkpoint=checkpoint,
            config=config,
            device=torch.device("cpu"),
        )
