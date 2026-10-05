from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from examples.qdiffusion_kaiwu.qdiffusion_protein_evaluate import (
    _invalid_sequence_count,
    _read_aligned_records,
    _source_preflight_identity,
    _verified_evaluation_source,
    _verified_training_paths,
    evaluate_outputs,
)

pytestmark = pytest.mark.unit


def _write(path: Path, content: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
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
    assert source.index("load_source_preflight(") < source.index(
        "workflow, helpers = _load_pinned_eval_workflow("
    )
    assert source.index("verify_frozen_environment_lock(") < source.index(
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
) -> None:
    _, paths = _training_artifacts(tmp_path)
    loader_paths: list[str] = []
    helpers = SimpleNamespace(
        esm=SimpleNamespace(
            pretrained=SimpleNamespace(
                load_model_and_alphabet_local=lambda path: (
                    loader_paths.append(path) or _Model(),
                    object(),
                )
            )
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
