"""Verify frozen QDiffusion protein artifacts without network access."""

from __future__ import annotations

import argparse
import hashlib
import io
import stat
from pathlib import Path
from typing import Any

from examples.qdiffusion_kaiwu.private_io import (
    read_private_bytes,
    write_private_json_exclusive,
)
from examples.qdiffusion_kaiwu.stable_source_tree import (
    RegularFileSnapshot,
    RegularTreeSnapshot,
    capture_regular_file,
    capture_regular_tree,
    open_captured_regular_file,
    revalidate_regular_file,
    revalidate_regular_tree,
)
from examples.qdiffusion_kaiwu.strict_json import loads_json_strict

CONFIG_SCHEMA = "flagquantum.qboson_qdiffusion_config"
_MAX_CONFIG_BYTES = 1024 * 1024
PREFLIGHT_SCHEMA = "flagquantum.qboson_qdiffusion_artifact_preflight"
ARTIFACT_FIELDS = {
    "dataset": "dataset",
    "base_checkpoint": "checkpoint",
    "tokenizer": "tokenizer",
    "evaluation_model": "evaluation_model",
}
AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWYBXZJUO")
ArtifactSnapshot = RegularFileSnapshot | RegularTreeSnapshot


def _artifact_identity_snapshot(
    path: Path,
) -> tuple[str, str, int, RegularFileSnapshot | RegularTreeSnapshot]:
    """Return one stable artifact identity and its re-openable snapshot."""

    if not path.is_absolute():
        raise ValueError(f"artifact path must be absolute: {path}")
    try:
        metadata = path.lstat()
    except OSError:
        raise ValueError(f"artifact does not exist: {path}") from None
    if path.is_symlink():
        raise ValueError(f"artifact root must not be a symlink: {path}")
    if stat.S_ISREG(metadata.st_mode):
        snapshot = capture_regular_file(path, label="artifact")
        return snapshot.sha256, "file-sha256-v1", 1, snapshot
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"artifact must be a regular file or directory: {path}")
    snapshot = capture_regular_tree(path, label="artifact tree")
    if not snapshot.verified_files:
        raise ValueError(f"artifact directory contains no regular files: {path}")

    digest = hashlib.sha256(b"flagquantum-tree-sha256-v1\0")
    for entry in snapshot.verified_files:
        relative = entry["path"].encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(entry["sha256"]))
    return (
        digest.hexdigest(),
        "tree-sha256-v1",
        len(snapshot.verified_files),
        snapshot,
    )


def _artifact_identity(path: Path) -> tuple[str, str, int]:
    """Return digest, algorithm, and regular-file count for one local artifact."""

    digest, algorithm, file_count, _ = _artifact_identity_snapshot(path)
    return digest, algorithm, file_count


def capture_artifact_snapshots(
    artifact_paths: dict[str, Path], preflight_record: dict[str, Any]
) -> dict[str, ArtifactSnapshot]:
    """Capture the exact artifact identities accepted by one preflight record."""

    if set(artifact_paths) != set(ARTIFACT_FIELDS):
        raise ValueError("all four named artifact paths are required")
    expected_artifacts = preflight_record.get("artifacts")
    if not isinstance(expected_artifacts, dict):
        raise ValueError("artifact preflight has no artifact identities")
    snapshots: dict[str, ArtifactSnapshot] = {}
    for name, path in artifact_paths.items():
        digest, algorithm, file_count, snapshot = _artifact_identity_snapshot(path)
        expected = expected_artifacts.get(name)
        if not isinstance(expected, dict) or any(
            expected.get(field) != value
            for field, value in (
                ("sha256", digest),
                ("algorithm", algorithm),
                ("file_count", file_count),
            )
        ):
            raise ValueError(f"{name} differs from artifact preflight")
        snapshots[name] = snapshot
    return snapshots


def revalidate_artifact_snapshots(
    snapshots: dict[str, ArtifactSnapshot],
) -> None:
    """Require every frozen artifact to retain its original captured identity."""

    if set(snapshots) != set(ARTIFACT_FIELDS):
        raise ValueError("all four artifact snapshots are required")
    for name, snapshot in snapshots.items():
        label = f"frozen {name} artifact"
        if isinstance(snapshot, RegularFileSnapshot):
            revalidate_regular_file(snapshot, label=label)
        else:
            revalidate_regular_tree(snapshot, label=label)


def _read_config(path: Path) -> tuple[dict[str, Any], str]:
    raw = read_private_bytes(
        path, label="frozen configuration", max_bytes=_MAX_CONFIG_BYTES
    )
    value = loads_json_strict(raw)
    if not isinstance(value, dict):
        raise ValueError("config must be a JSON object")
    if value.get("schema") != CONFIG_SCHEMA or value.get("version") != "1.0":
        raise ValueError("config has an unsupported schema or version")
    return value, hashlib.sha256(raw).hexdigest()


def _dataset_profile(
    snapshot: RegularFileSnapshot | RegularTreeSnapshot,
    config: dict[str, Any],
) -> dict[str, int]:
    if not isinstance(snapshot, RegularFileSnapshot):
        raise ValueError("dataset artifact must be one FASTA file")
    section = config.get("dataset")
    generation = config.get("generation")
    if not isinstance(section, dict) or not isinstance(generation, dict):
        raise ValueError("config must contain dataset and generation objects")
    integers: dict[str, int] = {}
    for field in ("min_length", "max_length", "max_records"):
        value = section.get(field)
        if type(value) is not int or value <= 0:
            raise ValueError(f"config.dataset.{field} must be a positive integer")
        integers[field] = value
    ratios: dict[str, float] = {}
    for field in ("validation_ratio", "test_ratio"):
        value = section.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0 < float(value) < 1
        ):
            raise ValueError(f"config.dataset.{field} must be between zero and one")
        ratios[field] = float(value)

    records: list[tuple[str, str]] = []
    header: str | None = None
    sequence: list[str] = []

    def finish_record() -> None:
        if header is None:
            return
        joined = "".join(sequence).upper()
        if not joined:
            raise ValueError(f"FASTA record has no sequence: {header}")
        unsupported = sorted(set(joined) - AMINO_ACIDS)
        if unsupported:
            raise ValueError(
                f"FASTA record {header} contains unsupported residues: "
                + "".join(unsupported)
            )
        records.append((header, joined))

    with (
        open_captured_regular_file(snapshot, label="dataset artifact") as binary,
        io.TextIOWrapper(binary, encoding="utf-8") as stream,
    ):
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                finish_record()
                header = line[1:].strip()
                sequence = []
                if not header:
                    raise ValueError(
                        f"FASTA header is empty at line {line_number}"
                    )
            else:
                if header is None:
                    raise ValueError(
                        "FASTA sequence appears before a header at line "
                        f"{line_number}"
                    )
                sequence.append(line)
    finish_record()
    if not records:
        raise ValueError("dataset FASTA contains no records")
    headers = [item[0] for item in records]
    if len(set(headers)) != len(headers):
        raise ValueError("dataset FASTA contains duplicate headers")

    eligible = sum(
        integers["min_length"] <= len(item[1]) <= integers["max_length"]
        for item in records
    )
    selected = min(eligible, integers["max_records"])
    if selected < integers["max_records"]:
        raise ValueError(
            "dataset has fewer eligible records than config.dataset.max_records"
        )
    validation_count = max(1, int(selected * ratios["validation_ratio"]))
    test_count = max(1, int(selected * ratios["test_ratio"]))
    train_count = selected - validation_count - test_count
    if train_count <= 0:
        raise ValueError("frozen dataset split leaves no training records")
    if generation.get("sequence_count") != test_count:
        raise ValueError(
            "frozen generation sequence_count differs from computed test split"
        )
    return {
        "record_count": len(records),
        "eligible_count": eligible,
        "selected_count": selected,
        "train_count": train_count,
        "validation_count": validation_count,
        "test_count": test_count,
    }


def _inspect_artifacts(
    config_path: Path, artifact_paths: dict[str, Path]
) -> tuple[
    str,
    dict[str, dict[str, Any]],
    dict[str, ArtifactSnapshot],
]:
    """Recompute the complete frozen input set without publishing evidence."""

    config, config_sha256 = _read_config(config_path)
    if set(artifact_paths) != set(ARTIFACT_FIELDS):
        raise ValueError("all four named artifact paths are required")

    artifacts: dict[str, dict[str, Any]] = {}
    snapshots: dict[str, ArtifactSnapshot] = {}
    errors: list[str] = []
    for artifact_name, config_name in ARTIFACT_FIELDS.items():
        section = config.get(config_name)
        if not isinstance(section, dict):
            errors.append(f"config.{config_name}: expected an object")
            continue
        expected = section.get("sha256")
        if not isinstance(expected, str):
            errors.append(f"config.{config_name}.sha256: expected a digest")
            continue
        digest, algorithm, file_count, snapshot = _artifact_identity_snapshot(
            artifact_paths[artifact_name]
        )
        if digest != expected:
            errors.append(
                f"{artifact_name}: computed {digest}, expected config digest {expected}"
            )
        artifacts[artifact_name] = {
            "sha256": digest,
            "algorithm": algorithm,
            "file_count": file_count,
        }
        snapshots[artifact_name] = snapshot
        if artifact_name == "dataset" and digest == expected:
            artifacts[artifact_name]["profile"] = _dataset_profile(
                snapshot, config
            )
    if errors:
        raise ValueError("artifact preflight failed:\n- " + "\n- ".join(errors))
    return config_sha256, artifacts, snapshots


def assert_artifacts_unchanged(
    config_path: Path,
    artifact_paths: dict[str, Path],
    preflight_record: dict[str, Any],
) -> None:
    """Fail if any frozen input or the config changed after preflight."""

    config_sha256, artifacts, _ = _inspect_artifacts(config_path, artifact_paths)
    if preflight_record.get("config_sha256") != config_sha256:
        raise ValueError("frozen experiment config changed after artifact preflight")
    if preflight_record.get("artifacts") != artifacts:
        raise ValueError("frozen protein artifacts changed after artifact preflight")


def preflight_artifacts(
    config_path: Path,
    artifact_paths: dict[str, Path],
    output_path: Path,
) -> dict[str, Any]:
    """Hash all required local artifacts and fail if config identities differ."""

    record, _ = preflight_artifacts_with_snapshots(
        config_path, artifact_paths, output_path
    )
    return record


def preflight_artifacts_with_snapshots(
    config_path: Path,
    artifact_paths: dict[str, Path],
    output_path: Path,
) -> tuple[dict[str, Any], dict[str, ArtifactSnapshot]]:
    """Publish preflight and retain the exact artifact snapshots it verified."""

    config_sha256, artifacts, snapshots = _inspect_artifacts(
        config_path, artifact_paths
    )

    record: dict[str, Any] = {
        "schema": PREFLIGHT_SCHEMA,
        "version": "1.0",
        "offline_preflight_only": True,
        "acceptance_evidence": False,
        "config_sha256": config_sha256,
        "artifacts": artifacts,
    }
    write_private_json_exclusive(output_path, record)
    revalidate_artifact_snapshots(snapshots)
    return record, snapshots


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--evaluation-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    preflight_artifacts(
        args.config,
        {
            "dataset": args.dataset,
            "base_checkpoint": args.base_checkpoint,
            "tokenizer": args.tokenizer,
            "evaluation_model": args.evaluation_model,
        },
        args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
