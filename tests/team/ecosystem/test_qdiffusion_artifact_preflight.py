from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from examples.qdiffusion_kaiwu.preflight_protein_artifacts import (
    _artifact_identity,
    preflight_artifacts,
)

pytestmark = pytest.mark.unit


def _write(path: Path, content: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    paths = {
        "dataset": tmp_path / "dataset.fasta",
        "base_checkpoint": tmp_path / "checkpoint.bin",
        "tokenizer": tmp_path / "tokenizer",
        "evaluation_model": tmp_path / "evaluation-model",
    }
    dataset_hash = _write(paths["dataset"], b">p1\nACDE\n")
    checkpoint_hash = _write(paths["base_checkpoint"], b"weights")
    _write(paths["tokenizer"] / "tokenizer.json", b"tokens")
    _write(paths["evaluation_model"] / "weights.pt", b"esm2")
    tokenizer_hash, _, _ = _artifact_identity(paths["tokenizer"])
    evaluation_hash, _, _ = _artifact_identity(paths["evaluation_model"])
    config = {
        "schema": "flagquantum.qboson_qdiffusion_config",
        "version": "1.0",
        "dataset": {"sha256": dataset_hash},
        "checkpoint": {"sha256": checkpoint_hash},
        "tokenizer": {"sha256": tokenizer_hash},
        "evaluation_model": {"sha256": evaluation_hash},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config) + "\n", encoding="utf-8")
    return config_path, paths


def test_preflight_verifies_files_and_trees_without_recording_paths(
    tmp_path: Path,
) -> None:
    config_path, paths = _fixture(tmp_path)
    output = tmp_path / "preflight.json"

    record = preflight_artifacts(config_path, paths, output)

    assert record["offline_preflight_only"] is True
    assert record["acceptance_evidence"] is False
    assert record["artifacts"]["dataset"]["algorithm"] == "file-sha256-v1"
    assert record["artifacts"]["tokenizer"]["algorithm"] == "tree-sha256-v1"
    assert str(tmp_path) not in output.read_text(encoding="utf-8")
    assert output.stat().st_mode & 0o777 == 0o600


def test_preflight_rejects_digest_mismatch_without_writing_record(
    tmp_path: Path,
) -> None:
    config_path, paths = _fixture(tmp_path)
    paths["dataset"].write_bytes(b"changed")
    output = tmp_path / "preflight.json"

    with pytest.raises(ValueError, match="artifact preflight failed"):
        preflight_artifacts(config_path, paths, output)

    assert not output.exists()


def test_preflight_refuses_to_overwrite_evidence(tmp_path: Path) -> None:
    config_path, paths = _fixture(tmp_path)
    output = tmp_path / "preflight.json"
    output.write_text("preserve me", encoding="utf-8")

    with pytest.raises(FileExistsError):
        preflight_artifacts(config_path, paths, output)

    assert output.read_text(encoding="utf-8") == "preserve me"


def test_tree_identity_rejects_symlinks(tmp_path: Path) -> None:
    tree = tmp_path / "model"
    target = tmp_path / "outside.bin"
    _write(target, b"outside")
    tree.mkdir()
    (tree / "link.bin").symlink_to(target)

    with pytest.raises(ValueError, match="contains a symlink"):
        _artifact_identity(tree)


def test_artifact_path_must_be_absolute(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be absolute"):
        _artifact_identity(Path("relative.bin"))
