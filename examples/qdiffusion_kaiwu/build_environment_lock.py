"""Build an exact QDiffusion environment lock from reviewed wheel artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import zipfile
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from typing import Any

from examples.qdiffusion_kaiwu.verify_environment_lock import (
    SCHEMA,
    _canonical_distribution_name,
    _installed_distribution_versions,
    load_environment_lock,
)

_MAX_METADATA_BYTES = 1024 * 1024


def _wheel_identity(path: Path) -> tuple[str, str]:
    if not path.is_absolute():
        raise ValueError("wheel artifact paths must be absolute")
    if path.is_symlink() or not path.is_file():
        raise ValueError("wheel artifacts must be regular, non-symlink files")
    if path.suffix != ".whl":
        raise ValueError("environment lock artifacts must be wheel files")
    try:
        with zipfile.ZipFile(path) as wheel:
            metadata_members = [
                member
                for member in wheel.infolist()
                if not member.is_dir()
                and len(PurePosixPath(member.filename).parts) == 2
                and PurePosixPath(member.filename).parts[0].endswith(".dist-info")
                and PurePosixPath(member.filename).name == "METADATA"
            ]
            if len(metadata_members) != 1:
                raise ValueError(
                    "wheel artifact must contain exactly one dist-info/METADATA"
                )
            metadata_member = metadata_members[0]
            if metadata_member.file_size > _MAX_METADATA_BYTES:
                raise ValueError("wheel METADATA exceeds the bounded size")
            metadata = BytesParser().parsebytes(wheel.read(metadata_member))
    except zipfile.BadZipFile:
        raise ValueError("wheel artifact is not a valid ZIP archive") from None
    raw_name = metadata.get("Name")
    version = metadata.get("Version")
    if not isinstance(raw_name, str) or not raw_name.strip():
        raise ValueError("wheel METADATA does not contain a usable Name")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("wheel METADATA does not contain a usable Version")
    name = _canonical_distribution_name(raw_name)
    dist_info_name = PurePosixPath(metadata_member.filename).parts[0][
        : -len(".dist-info")
    ]
    if "-" not in dist_info_name:
        raise ValueError("wheel dist-info identity does not contain a version")
    dist_info_distribution = dist_info_name.rsplit("-", 1)[0]
    if _canonical_distribution_name(dist_info_distribution) != name:
        raise ValueError("wheel dist-info identity differs from METADATA Name")
    return name, version.strip()


def _write_private_json(path: Path, payload: dict[str, Any]) -> None:
    if not path.is_absolute():
        raise ValueError("environment lock output path must be absolute")
    if not path.parent.is_dir():
        raise ValueError("environment lock output parent must already exist")
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(encoded)


def build_environment_lock(*, artifacts: list[Path], output: Path) -> str:
    """Bind the exact installed inventory to one reviewed wheel per distribution."""

    if not artifacts:
        raise ValueError("at least one reviewed wheel artifact is required")
    installed = _installed_distribution_versions()
    reviewed: dict[str, dict[str, str]] = {}
    for artifact in artifacts:
        name, version = _wheel_identity(artifact)
        if name in reviewed:
            raise ValueError(f"duplicate reviewed wheel identity: {name}")
        reviewed[name] = {
            "name": name,
            "version": version,
            "approved_artifact_sha256": hashlib.sha256(
                artifact.read_bytes()
            ).hexdigest(),
        }
    missing = sorted(installed.keys() - reviewed.keys())
    extra = sorted(reviewed.keys() - installed.keys())
    mismatched = sorted(
        name
        for name in installed.keys() & reviewed.keys()
        if installed[name] != reviewed[name]["version"]
    )
    if missing:
        raise ValueError(f"installed distributions lack reviewed wheels: {missing}")
    if extra:
        raise ValueError(f"reviewed wheels are not installed: {extra}")
    if mismatched:
        raise ValueError(f"reviewed wheel versions differ from runtime: {mismatched}")
    record: dict[str, Any] = {
        "schema": SCHEMA,
        "version": "1.0",
        "inventory_policy": "exact",
        "python_version": platform.python_version(),
        "distributions": [reviewed[name] for name in sorted(reviewed)],
    }
    _write_private_json(output, record)
    _, digest = load_environment_lock(output)
    return digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact", action="extend", nargs="+", required=True, type=Path
    )
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    try:
        digest = build_environment_lock(
            artifacts=arguments.artifact,
            output=arguments.output,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Private environment lock written: sha256:{digest}")


if __name__ == "__main__":
    main()
