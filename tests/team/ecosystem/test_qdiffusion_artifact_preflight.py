from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from examples.qdiffusion_kaiwu import preflight_protein_artifacts as preflight_module
from examples.qdiffusion_kaiwu.preflight_protein_artifacts import (
    _artifact_identity,
    capture_artifact_snapshots,
    revalidate_artifact_snapshots,
)
from examples.qdiffusion_kaiwu.preflight_protein_artifacts import (
    assert_artifacts_unchanged as _assert_artifacts_unchanged,
)
from examples.qdiffusion_kaiwu.preflight_protein_artifacts import (
    preflight_artifacts as _preflight_artifacts,
)
from examples.qdiffusion_kaiwu.preflight_protein_artifacts import (
    preflight_artifacts_with_snapshots as _preflight_artifacts_with_snapshots,
)

pytestmark = pytest.mark.unit


def _write(path: Path, content: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def _dataset_source_archive(paths: dict[str, Path]) -> Path:
    return paths["dataset"].with_suffix(".fasta.gz")


def preflight_artifacts(
    config_path: Path, artifact_paths: dict[str, Path], output_path: Path
) -> dict[str, object]:
    return _preflight_artifacts(
        config_path,
        artifact_paths,
        output_path,
        dataset_source_archive=_dataset_source_archive(artifact_paths),
    )


def preflight_artifacts_with_snapshots(
    config_path: Path, artifact_paths: dict[str, Path], output_path: Path
):
    return _preflight_artifacts_with_snapshots(
        config_path,
        artifact_paths,
        output_path,
        dataset_source_archive=_dataset_source_archive(artifact_paths),
    )


def assert_artifacts_unchanged(
    config_path: Path,
    artifact_paths: dict[str, Path],
    preflight_record: dict[str, object],
) -> None:
    _assert_artifacts_unchanged(
        config_path,
        artifact_paths,
        preflight_record,
        dataset_source_archive=_dataset_source_archive(artifact_paths),
    )


def _fixture(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    paths = {
        "dataset": tmp_path / "dataset.fasta",
        "base_checkpoint": tmp_path / "checkpoint.bin",
        "tokenizer": tmp_path / "tokenizer",
        "evaluation_model": tmp_path / "evaluation-model",
    }
    dataset_bytes = b">p1\nACDE\n>p2\nFGHI\n>p3\nKLMN\n>p4\nPQRS\n"
    dataset_hash = _write(paths["dataset"], dataset_bytes)
    source_bytes = gzip.compress(dataset_bytes, mtime=0)
    source_hash = _write(_dataset_source_archive(paths), source_bytes)
    checkpoint_hash = _write(paths["base_checkpoint"], b"weights")
    _write(paths["tokenizer"] / "tokenizer.json", b"tokens")
    _write(paths["evaluation_model"] / "weights.pt", b"esm2")
    tokenizer_hash, _, _ = _artifact_identity(paths["tokenizer"])
    evaluation_hash, _, _ = _artifact_identity(paths["evaluation_model"])
    config = {
        "schema": "flagquantum.qboson_qdiffusion_config",
        "version": "1.0",
        "dataset": {
            "sha256": dataset_hash,
            "source_archive_sha256": source_hash,
            "source_archive_bytes": len(source_bytes),
            "source_archive_format": "gzip",
            "decompression_policy": "gzip-exact-bytes-v1",
            "min_length": 4,
            "max_length": 4,
            "max_records": 4,
            "validation_ratio": 0.25,
            "test_ratio": 0.25,
        },
        "checkpoint": {"sha256": checkpoint_hash},
        "tokenizer": {"sha256": tokenizer_hash},
        "evaluation_model": {"sha256": evaluation_hash},
        "generation": {"sequence_count": 1},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config) + "\n", encoding="utf-8")
    config_path.chmod(0o600)
    return config_path, paths


@pytest.mark.parametrize("unsafe_kind", ("public_file", "public_parent"))
def test_preflight_rejects_unsafe_config_control_file(
    tmp_path: Path, unsafe_kind: str
) -> None:
    config_path, paths = _fixture(tmp_path)
    if unsafe_kind == "public_file":
        config_path.chmod(0o644)
    else:
        tmp_path.chmod(0o755)

    with pytest.raises(ValueError):
        preflight_artifacts(config_path, paths, tmp_path / "preflight.json")


def test_preflight_verifies_files_and_trees_without_recording_paths(
    tmp_path: Path,
) -> None:
    config_path, paths = _fixture(tmp_path)
    output = tmp_path / "preflight.json"

    record = preflight_artifacts(config_path, paths, output)

    assert record["offline_preflight_only"] is True
    assert record["acceptance_evidence"] is False
    assert record["artifacts"]["dataset"]["algorithm"] == "file-sha256-v1"
    assert record["artifacts"]["dataset"]["profile"] == {
        "record_count": 4,
        "eligible_count": 4,
        "selected_count": 4,
        "train_count": 2,
        "validation_count": 1,
        "test_count": 1,
    }
    assert record["artifacts"]["tokenizer"]["algorithm"] == "tree-sha256-v1"
    assert record["dataset_source"] == {
        "source_archive_sha256": hashlib.sha256(
            _dataset_source_archive(paths).read_bytes()
        ).hexdigest(),
        "source_archive_bytes": _dataset_source_archive(paths).stat().st_size,
        "source_archive_format": "gzip",
        "decompression_policy": "gzip-exact-bytes-v1",
        "decompressed_sha256": record["artifacts"]["dataset"]["sha256"],
        "decompressed_bytes": paths["dataset"].stat().st_size,
    }
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


def test_preflight_rejects_dataset_source_that_does_not_produce_frozen_fasta(
    tmp_path: Path,
) -> None:
    config_path, paths = _fixture(tmp_path)
    source = _dataset_source_archive(paths)
    replacement = gzip.compress(b">other\nAAAA\n", mtime=0)
    source.write_bytes(replacement)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["dataset"]["source_archive_sha256"] = hashlib.sha256(
        replacement
    ).hexdigest()
    config["dataset"]["source_archive_bytes"] = len(replacement)
    config_path.write_text(json.dumps(config) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="decompressed dataset .* differs"):
        preflight_artifacts(config_path, paths, tmp_path / "preflight.json")


def test_preflight_refuses_to_overwrite_evidence(tmp_path: Path) -> None:
    config_path, paths = _fixture(tmp_path)
    output = tmp_path / "preflight.json"
    output.write_text("preserve me", encoding="utf-8")

    with pytest.raises(FileExistsError):
        preflight_artifacts(config_path, paths, output)

    assert output.read_text(encoding="utf-8") == "preserve me"


@pytest.mark.parametrize("unsafe_kind", ("public", "symlink"))
def test_preflight_requires_private_real_output_parent(
    tmp_path: Path, unsafe_kind: str
) -> None:
    config_path, paths = _fixture(tmp_path)
    private_parent = tmp_path / "private"
    private_parent.mkdir(mode=0o700)
    if unsafe_kind == "public":
        private_parent.chmod(0o755)
        output = private_parent / "preflight.json"
    else:
        linked_parent = tmp_path / "linked"
        linked_parent.symlink_to(private_parent, target_is_directory=True)
        output = linked_parent / "preflight.json"

    with pytest.raises(ValueError, match="existing private, non-symlink"):
        preflight_artifacts(config_path, paths, output)

    assert not output.exists()


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


def test_preflight_rejects_insufficient_eligible_dataset_records(
    tmp_path: Path,
) -> None:
    config_path, paths = _fixture(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["dataset"]["min_length"] = 5
    config["dataset"]["sha256"] = hashlib.sha256(
        paths["dataset"].read_bytes()
    ).hexdigest()
    config_path.write_text(json.dumps(config) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="fewer eligible records"):
        preflight_artifacts(config_path, paths, tmp_path / "preflight.json")


def test_preflight_rejects_test_split_count_drift(tmp_path: Path) -> None:
    config_path, paths = _fixture(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["generation"]["sequence_count"] = 2
    config_path.write_text(json.dumps(config) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="differs from computed test split"):
        preflight_artifacts(config_path, paths, tmp_path / "preflight.json")


def test_preflight_binds_dataset_profile_to_hashed_file_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, paths = _fixture(tmp_path)
    real_capture = preflight_module._artifact_identity_snapshot

    def capture_then_mutate(path: Path):
        identity = real_capture(path)
        if path == paths["dataset"]:
            path.write_bytes(b">changed\nAAAA\n")
        return identity

    monkeypatch.setattr(
        preflight_module, "_artifact_identity_snapshot", capture_then_mutate
    )

    with pytest.raises(ValueError, match="changed after identity capture"):
        preflight_artifacts(config_path, paths, tmp_path / "preflight.json")


def test_artifact_snapshot_rejects_same_digest_rewrite_after_preflight(
    tmp_path: Path,
) -> None:
    config_path, paths = _fixture(tmp_path)
    preflight = preflight_artifacts(config_path, paths, tmp_path / "preflight.json")
    snapshots = capture_artifact_snapshots(paths, preflight)
    original = paths["dataset"].read_bytes()

    paths["dataset"].write_bytes(original)

    with pytest.raises(ValueError, match="changed after identity capture"):
        revalidate_artifact_snapshots(snapshots)


def test_preflight_returns_the_exact_snapshots_used_for_published_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, paths = _fixture(tmp_path)
    real_write = preflight_module.write_private_json_exclusive
    original = paths["dataset"].read_bytes()

    def write_then_replace(path: Path, value: object) -> None:
        real_write(path, value)
        paths["dataset"].write_bytes(original)

    monkeypatch.setattr(
        preflight_module, "write_private_json_exclusive", write_then_replace
    )

    with pytest.raises(ValueError, match="changed after identity capture"):
        preflight_artifacts_with_snapshots(
            config_path, paths, tmp_path / "preflight.json"
        )


def test_postflight_rejects_artifact_drift(tmp_path: Path) -> None:
    config_path, paths = _fixture(tmp_path)
    preflight = preflight_artifacts(config_path, paths, tmp_path / "preflight.json")
    paths["base_checkpoint"].write_bytes(b"changed weights")

    with pytest.raises(ValueError, match="artifact preflight failed|changed after"):
        assert_artifacts_unchanged(config_path, paths, preflight)


def test_postflight_rejects_config_drift(tmp_path: Path) -> None:
    config_path, paths = _fixture(tmp_path)
    preflight = preflight_artifacts(config_path, paths, tmp_path / "preflight.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["dataset"]["name"] = "changed-after-preflight"
    config_path.write_text(json.dumps(config) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="config changed after"):
        assert_artifacts_unchanged(config_path, paths, preflight)
