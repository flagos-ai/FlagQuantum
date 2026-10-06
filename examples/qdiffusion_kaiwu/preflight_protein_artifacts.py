"""Verify frozen QDiffusion protein artifacts without network access."""

from __future__ import annotations

import argparse
import hashlib
import stat
from pathlib import Path
from typing import Any

from examples.qdiffusion_kaiwu.private_io import write_private_json_exclusive
from examples.qdiffusion_kaiwu.strict_json import loads_json_strict

CONFIG_SCHEMA = "flagquantum.qboson_qdiffusion_config"
PREFLIGHT_SCHEMA = "flagquantum.qboson_qdiffusion_artifact_preflight"
ARTIFACT_FIELDS = {
    "dataset": "dataset",
    "base_checkpoint": "checkpoint",
    "tokenizer": "tokenizer",
    "evaluation_model": "evaluation_model",
}
AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWYBXZJUO")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_identity(path: Path) -> tuple[str, str, int]:
    """Return digest, algorithm, and regular-file count for one local artifact."""
    if not path.is_absolute():
        raise ValueError(f"artifact path must be absolute: {path}")
    if path.is_symlink():
        raise ValueError(f"artifact root must not be a symlink: {path}")
    if not path.exists():
        raise ValueError(f"artifact does not exist: {path}")
    if path.is_file():
        return _file_sha256(path), "file-sha256-v1", 1
    if not path.is_dir():
        raise ValueError(f"artifact must be a regular file or directory: {path}")

    files: list[Path] = []
    for child in path.rglob("*"):
        if child.is_symlink():
            raise ValueError(f"artifact tree contains a symlink: {child}")
        mode = child.stat().st_mode
        if stat.S_ISREG(mode):
            files.append(child)
        elif not stat.S_ISDIR(mode):
            raise ValueError(f"artifact tree contains a special file: {child}")
    if not files:
        raise ValueError(f"artifact directory contains no regular files: {path}")

    digest = hashlib.sha256(b"flagquantum-tree-sha256-v1\0")
    for child in sorted(files, key=lambda item: item.relative_to(path).as_posix()):
        relative = child.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_file_sha256(child)))
    return digest.hexdigest(), "tree-sha256-v1", len(files)


def _read_config(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    value = loads_json_strict(raw)
    if not isinstance(value, dict):
        raise ValueError("config must be a JSON object")
    if value.get("schema") != CONFIG_SCHEMA or value.get("version") != "1.0":
        raise ValueError("config has an unsupported schema or version")
    return value, hashlib.sha256(raw).hexdigest()


def _dataset_profile(path: Path, config: dict[str, Any]) -> dict[str, int]:
    if not path.is_file():
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

    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                finish_record()
                header = line[1:].strip()
                sequence = []
                if not header:
                    raise ValueError(f"FASTA header is empty at line {line_number}")
            else:
                if header is None:
                    raise ValueError(
                        f"FASTA sequence appears before a header at line {line_number}"
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
) -> tuple[str, dict[str, dict[str, Any]]]:
    """Recompute the complete frozen input set without publishing evidence."""

    config, config_sha256 = _read_config(config_path)
    if set(artifact_paths) != set(ARTIFACT_FIELDS):
        raise ValueError("all four named artifact paths are required")

    artifacts: dict[str, dict[str, Any]] = {}
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
        digest, algorithm, file_count = _artifact_identity(
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
        if artifact_name == "dataset" and digest == expected:
            artifacts[artifact_name]["profile"] = _dataset_profile(
                artifact_paths[artifact_name], config
            )
    if errors:
        raise ValueError("artifact preflight failed:\n- " + "\n- ".join(errors))
    return config_sha256, artifacts


def assert_artifacts_unchanged(
    config_path: Path,
    artifact_paths: dict[str, Path],
    preflight_record: dict[str, Any],
) -> None:
    """Fail if any frozen input or the config changed after preflight."""

    config_sha256, artifacts = _inspect_artifacts(config_path, artifact_paths)
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

    config_sha256, artifacts = _inspect_artifacts(config_path, artifact_paths)

    record: dict[str, Any] = {
        "schema": PREFLIGHT_SCHEMA,
        "version": "1.0",
        "offline_preflight_only": True,
        "acceptance_evidence": False,
        "config_sha256": config_sha256,
        "artifacts": artifacts,
    }
    write_private_json_exclusive(output_path, record)
    return record


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
