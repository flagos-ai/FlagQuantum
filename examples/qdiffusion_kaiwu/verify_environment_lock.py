"""Verify one exact, approved QDiffusion Python environment lock offline."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
from pathlib import Path
from typing import Any

from examples.qdiffusion_kaiwu.private_io import read_private_bytes
from examples.qdiffusion_kaiwu.strict_json import loads_json_strict

SCHEMA = "flagquantum.qboson_qdiffusion_environment_lock"
SHA256 = re.compile(r"[0-9a-f]{64}")
PLACEHOLDER = re.compile(r"<[^>]+>")
_MAX_ENVIRONMENT_LOCK_BYTES = 4 * 1024 * 1024


def _canonical_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _stream_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _distribution_content_sha256(
    distribution: importlib.metadata.Distribution, *, name: str
) -> str:
    files = distribution.files
    if not files:
        raise ValueError(f"installed distribution lacks a RECORD file set: {name}")
    records: list[dict[str, Any]] = []
    labels: set[str] = set()
    for entry in sorted(files, key=str):
        label = str(entry).replace("\\", "/")
        if not label or label in labels:
            raise ValueError(
                f"installed distribution has invalid file identity: {name}"
            )
        labels.add(label)
        path = Path(distribution.locate_file(entry))
        if path.is_symlink() or not path.is_file():
            raise ValueError(
                f"installed distribution file is missing or unsafe: {name}:{label}"
            )
        records.append(
            {
                "path": label,
                "bytes": path.stat().st_size,
                "sha256": _stream_sha256(path),
            }
        )
    return hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _installed_distribution_inventory() -> dict[str, tuple[str, str]]:
    installed: dict[str, tuple[str, str]] = {}
    for distribution in importlib.metadata.distributions():
        raw_name = distribution.metadata.get("Name")
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ValueError("installed distribution without a usable name")
        name = _canonical_distribution_name(raw_name)
        version = distribution.version
        if name in installed:
            raise ValueError(f"duplicate installed distribution identity: {name}")
        installed[name] = (
            version,
            _distribution_content_sha256(distribution, name=name),
        )
    return installed


def load_environment_lock(path: Path) -> tuple[dict[str, Any], str]:
    encoded = read_private_bytes(
        path,
        label="environment lock",
        max_bytes=_MAX_ENVIRONMENT_LOCK_BYTES,
    )
    try:
        record = loads_json_strict(encoded)
    except json.JSONDecodeError as exc:
        raise ValueError("environment lock is not valid JSON") from exc
    if not isinstance(record, dict):
        raise ValueError("environment lock must be a JSON object")
    if record.get("schema") != SCHEMA or record.get("version") != "1.0":
        raise ValueError("unsupported environment lock schema or version")
    if record.get("inventory_policy") != "exact":
        raise ValueError("environment lock inventory policy must be exact")
    python_version = record.get("python_version")
    if (
        not isinstance(python_version, str)
        or not python_version
        or PLACEHOLDER.search(python_version)
    ):
        raise ValueError("environment lock requires an exact Python version")
    distributions = record.get("distributions")
    if not isinstance(distributions, list) or not distributions:
        raise ValueError("environment lock requires a non-empty distribution list")
    names: list[str] = []
    for index, distribution in enumerate(distributions):
        if not isinstance(distribution, dict):
            raise ValueError(f"environment lock distribution {index} is not an object")
        name = distribution.get("name")
        version = distribution.get("version")
        artifact_sha256 = distribution.get("approved_artifact_sha256")
        installed_content_sha256 = distribution.get("installed_content_sha256")
        if (
            not isinstance(name, str)
            or not name
            or name != _canonical_distribution_name(name)
        ):
            raise ValueError(
                f"environment lock distribution {index} name is not canonical"
            )
        if not isinstance(version, str) or not version or PLACEHOLDER.search(version):
            raise ValueError(
                f"environment lock distribution {name!r} requires an exact version"
            )
        if (
            not isinstance(artifact_sha256, str)
            or SHA256.fullmatch(artifact_sha256) is None
        ):
            raise ValueError(
                f"environment lock distribution {name!r} artifact digest is invalid"
            )
        if (
            not isinstance(installed_content_sha256, str)
            or SHA256.fullmatch(installed_content_sha256) is None
        ):
            raise ValueError(
                f"environment lock distribution {name!r} installed content digest is invalid"
            )
        names.append(name)
    if names != sorted(names) or len(names) != len(set(names)):
        raise ValueError(
            "environment lock distributions must be unique and sorted by name"
        )
    return record, hashlib.sha256(encoded).hexdigest()


def verify_environment_lock(path: Path) -> tuple[dict[str, Any], str]:
    record, digest = load_environment_lock(path)
    if record["python_version"] != platform.python_version():
        raise ValueError("observed Python version differs from environment lock")
    expected = {
        distribution["name"]: (
            distribution["version"],
            distribution["installed_content_sha256"],
        )
        for distribution in record["distributions"]
    }
    observed = _installed_distribution_inventory()
    missing = sorted(expected.keys() - observed.keys())
    extra = sorted(observed.keys() - expected.keys())
    mismatched = sorted(
        name
        for name in expected.keys() & observed.keys()
        if expected[name][0] != observed[name][0]
    )
    content_mismatched = sorted(
        name
        for name in expected.keys() & observed.keys()
        if expected[name][1] != observed[name][1]
    )
    if missing:
        raise ValueError(f"environment lock distributions are missing: {missing}")
    if extra:
        raise ValueError(f"environment contains unlisted distributions: {extra}")
    if mismatched:
        raise ValueError(f"environment distribution versions differ: {mismatched}")
    if content_mismatched:
        raise ValueError(
            f"environment distribution installed contents differ: {content_mismatched}"
        )
    return record, digest


def verify_frozen_environment_lock(
    path: Path, *, expected_sha256: str
) -> tuple[dict[str, Any], str]:
    if SHA256.fullmatch(expected_sha256) is None:
        raise ValueError("frozen environment lock digest is invalid")
    record, digest = verify_environment_lock(path)
    if digest != expected_sha256:
        raise ValueError("environment lock differs from frozen configuration")
    return record, digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", required=True, type=Path)
    arguments = parser.parse_args()
    _, digest = verify_environment_lock(arguments.lock)
    print(f"Environment lock verified: sha256:{digest}")


if __name__ == "__main__":
    main()
