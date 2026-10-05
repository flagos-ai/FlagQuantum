"""Verify frozen QDiffusion protein artifacts without network access."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any

CONFIG_SCHEMA = "flagquantum.qboson_qdiffusion_config"
PREFLIGHT_SCHEMA = "flagquantum.qboson_qdiffusion_artifact_preflight"
ARTIFACT_FIELDS = {
    "dataset": "dataset",
    "base_checkpoint": "checkpoint",
    "tokenizer": "tokenizer",
    "evaluation_model": "evaluation_model",
}


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
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("config must be a JSON object")
    if value.get("schema") != CONFIG_SCHEMA or value.get("version") != "1.0":
        raise ValueError("config has an unsupported schema or version")
    return value, hashlib.sha256(raw).hexdigest()


def _write_private_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def preflight_artifacts(
    config_path: Path,
    artifact_paths: dict[str, Path],
    output_path: Path,
) -> dict[str, Any]:
    """Hash all required local artifacts and fail if config identities differ."""
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
    if errors:
        raise ValueError("artifact preflight failed:\n- " + "\n- ".join(errors))

    record: dict[str, Any] = {
        "schema": PREFLIGHT_SCHEMA,
        "version": "1.0",
        "offline_preflight_only": True,
        "acceptance_evidence": False,
        "config_sha256": config_sha256,
        "artifacts": artifacts,
    }
    _write_private_json(output_path, record)
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
