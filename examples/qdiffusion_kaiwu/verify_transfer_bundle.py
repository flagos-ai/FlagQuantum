"""Fail-closed verification for a reviewed QBoson A800 transfer bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import tarfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA = "flagquantum.qboson_a800_transfer_bundle"
CLASSIFICATION = "local_preparation_only_not_execution_evidence"
HOSTS = {"jp-a800-171", "jp-a800-172"}
FULL_REVISION = re.compile(r"[0-9a-f]{40}")
ARTIFACT_PREFIXES = (
    "flagquantum-qboson-",
    "kaiwu-plugin-",
    "kaiwu-community-",
)
ARCHIVE_ROOT_PREFIXES = {
    "flagquantum-qboson-": "FlagQuantum-",
    "kaiwu-plugin-": "kaiwu-pytorch-plugin-",
    "kaiwu-community-": "kaiwu-community-",
}
MAX_COMPRESSED_BYTES = 1024 * 1024 * 1024
MAX_UNPACKED_BYTES = 5 * 1024 * 1024 * 1024
MAX_MEMBERS = 250_000


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_archive_summary(path: Path, *, expected_root: str) -> dict[str, int]:
    names: set[str] = set()
    file_count = 0
    directory_count = 0
    unpacked_bytes = 0
    try:
        with tarfile.open(path, mode="r:gz") as archive:
            for member in archive:
                if len(names) >= MAX_MEMBERS:
                    raise ValueError(f"archive has too many members: {path.name}")
                if "\\" in member.name:
                    raise ValueError(
                        f"archive member uses a non-POSIX separator: {member.name!r}"
                    )
                member_path = PurePosixPath(member.name)
                if (
                    not member.name
                    or member_path.is_absolute()
                    or ".." in member_path.parts
                ):
                    raise ValueError(
                        f"archive member escapes its extraction root: {member.name!r}"
                    )
                if not member_path.parts or member_path.parts[0] != expected_root:
                    raise ValueError(
                        "archive member is outside its revision-bound root: "
                        f"{member.name!r}"
                    )
                if len(member_path.parts) == 1 and not member.isdir():
                    raise ValueError(
                        "archive revision-bound root is not a directory: "
                        f"{member.name!r}"
                    )
                normalized = str(member_path)
                if normalized in names:
                    raise ValueError(
                        f"archive contains a duplicate member: {member.name!r}"
                    )
                names.add(normalized)
                if member.isfile():
                    if member.size < 0:
                        raise ValueError(
                            f"archive member has a negative size: {member.name!r}"
                        )
                    file_count += 1
                    unpacked_bytes += member.size
                    if unpacked_bytes > MAX_UNPACKED_BYTES:
                        raise ValueError(
                            f"archive exceeds the unpacked-size limit: {path.name}"
                        )
                elif member.isdir():
                    directory_count += 1
                else:
                    raise ValueError(
                        "archive contains a link or special member: " f"{member.name!r}"
                    )
            if not names:
                raise ValueError(f"archive is empty: {path.name}")
    except tarfile.TarError as exc:
        raise ValueError(f"invalid gzip tar archive: {path.name}") from exc
    return {
        "member_count": len(names),
        "file_count": file_count,
        "directory_count": directory_count,
        "unpacked_bytes": unpacked_bytes,
    }


def verify_transfer_bundle(manifest_path: Path, *, target_host: str) -> dict[str, Any]:
    """Verify identities and archive safety without extracting any content."""

    if not manifest_path.is_absolute():
        raise ValueError("manifest path must be absolute")
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("manifest must be a regular, non-symlink file")
    if target_host not in HOSTS:
        raise ValueError("target host is outside the reviewed A800 pair")
    encoded_manifest = manifest_path.read_bytes()
    try:
        manifest = json.loads(encoded_manifest)
    except json.JSONDecodeError as exc:
        raise ValueError("manifest is not valid JSON") from exc
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a JSON object")
    if manifest.get("schema") != SCHEMA or manifest.get("version") != "1.0":
        raise ValueError("unsupported transfer-manifest schema or version")
    if manifest.get("classification") != CLASSIFICATION:
        raise ValueError("transfer manifest has an unsafe evidence classification")
    if manifest.get("created_for_hosts") != ["jp-a800-171", "jp-a800-172"]:
        raise ValueError("transfer manifest does not cover the reviewed host pair")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != len(ARTIFACT_PREFIXES):
        raise ValueError("transfer manifest must contain exactly three artifacts")

    root = manifest_path.resolve().parent
    listed_filenames = {
        entry.get("filename") for entry in artifacts if isinstance(entry, dict)
    }
    colocated_archives = {path.name for path in root.glob("*.tar.gz")}
    if colocated_archives != listed_filenames:
        raise ValueError(
            "transfer directory archives differ from the exact manifest artifact set"
        )
    observed_prefixes: set[str] = set()
    verified_artifacts: list[dict[str, Any]] = []
    for index, raw_entry in enumerate(artifacts):
        if not isinstance(raw_entry, dict):
            raise ValueError(f"artifact {index} must be an object")
        filename = raw_entry.get("filename")
        revision = raw_entry.get("revision")
        digest = raw_entry.get("sha256")
        if (
            not isinstance(filename, str)
            or not filename
            or Path(filename).name != filename
            or not filename.endswith(".tar.gz")
        ):
            raise ValueError(f"artifact {index} has an unsafe filename")
        if not isinstance(revision, str) or FULL_REVISION.fullmatch(revision) is None:
            raise ValueError(f"artifact {filename} has an invalid revision")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"artifact {filename} has an invalid SHA-256 digest")
        matching_prefixes = [
            prefix for prefix in ARTIFACT_PREFIXES if filename.startswith(prefix)
        ]
        if len(matching_prefixes) != 1:
            raise ValueError(f"artifact {filename} has an unexpected role")
        prefix = matching_prefixes[0]
        if prefix in observed_prefixes:
            raise ValueError(f"transfer manifest repeats the {prefix} artifact")
        observed_prefixes.add(prefix)
        abbreviated_revision = filename[len(prefix) : -len(".tar.gz")]
        if not 7 <= len(abbreviated_revision) <= 12 or not revision.startswith(
            abbreviated_revision
        ):
            raise ValueError(f"artifact filename does not match revision: {filename}")

        artifact_path = (root / filename).resolve()
        if (
            not artifact_path.is_relative_to(root)
            or artifact_path.is_symlink()
            or not artifact_path.is_file()
        ):
            raise ValueError(f"artifact must be a regular, colocated file: {filename}")
        if artifact_path.stat().st_size > MAX_COMPRESSED_BYTES:
            raise ValueError(f"artifact exceeds the compressed-size limit: {filename}")
        observed_digest = _sha256(artifact_path)
        if observed_digest != digest:
            raise ValueError(f"artifact SHA-256 mismatch: {filename}")
        verified_artifacts.append(
            {
                "filename": filename,
                "revision": revision,
                "sha256": observed_digest,
                "compressed_bytes": artifact_path.stat().st_size,
                **_safe_archive_summary(
                    artifact_path,
                    expected_root=(f"{ARCHIVE_ROOT_PREFIXES[prefix]}{revision[:10]}"),
                ),
            }
        )

    if observed_prefixes != set(ARTIFACT_PREFIXES):
        raise ValueError("transfer manifest does not contain every required artifact")
    return {
        "schema": "flagquantum.qboson_a800_transfer_verification",
        "version": "1.0",
        "evidence_class": "transfer_preflight_only",
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "verification_hostname": socket.gethostname(),
        "verified_for_target_host": target_host,
        "manifest_filename": manifest_path.name,
        "manifest_sha256": hashlib.sha256(encoded_manifest).hexdigest(),
        "safe_to_extract": True,
        "artifacts": verified_artifacts,
        "qboson_hardware_used": False,
        "a800_execution_verified": False,
        "acceptance_evidence": False,
    }


def _write_private_json(path: Path, payload: dict[str, Any]) -> None:
    if not path.is_absolute():
        raise ValueError("output path must be absolute")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--target-host", choices=sorted(HOSTS), required=True)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    try:
        record = verify_transfer_bundle(
            arguments.manifest, target_host=arguments.target_host
        )
        _write_private_json(arguments.output, record)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Transfer bundle verified without extraction: {arguments.output}")


if __name__ == "__main__":
    main()
