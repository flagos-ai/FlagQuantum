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

SCHEMA = "flagquantum.qboson_qdiffusion_environment_lock"
SHA256 = re.compile(r"[0-9a-f]{64}")
PLACEHOLDER = re.compile(r"<[^>]+>")


def _canonical_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _installed_distribution_versions() -> dict[str, str]:
    installed: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        raw_name = distribution.metadata.get("Name")
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ValueError("installed distribution without a usable name")
        name = _canonical_distribution_name(raw_name)
        version = distribution.version
        if name in installed:
            raise ValueError(f"duplicate installed distribution identity: {name}")
        installed[name] = version
    return installed


def load_environment_lock(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_absolute():
        raise ValueError("environment lock path must be absolute")
    if path.is_symlink() or not path.is_file():
        raise ValueError("environment lock must be a regular, non-symlink file")
    if path.stat().st_mode & 0o077:
        raise ValueError("environment lock must not be accessible by group or others")
    encoded = path.read_bytes()
    try:
        record = json.loads(encoded)
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
        distribution["name"]: distribution["version"]
        for distribution in record["distributions"]
    }
    observed = _installed_distribution_versions()
    missing = sorted(expected.keys() - observed.keys())
    extra = sorted(observed.keys() - expected.keys())
    mismatched = sorted(
        name
        for name in expected.keys() & observed.keys()
        if expected[name] != observed[name]
    )
    if missing:
        raise ValueError(f"environment lock distributions are missing: {missing}")
    if extra:
        raise ValueError(f"environment contains unlisted distributions: {extra}")
    if mismatched:
        raise ValueError(f"environment distribution versions differ: {mismatched}")
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
