"""Verify extracted QBoson transfer sources against their reviewed archives."""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import stat
import tarfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

from examples.qdiffusion_kaiwu.private_io import (
    open_private_binary,
    read_private_bytes,
)
from examples.qdiffusion_kaiwu.strict_json import loads_json_strict
from examples.qdiffusion_kaiwu.verify_transfer_bundle import (
    ARCHIVE_ROOT_PREFIXES,
    ARTIFACT_PREFIXES,
    MAX_COMPRESSED_BYTES,
    MAX_TRANSFER_MANIFEST_BYTES,
    _write_private_json,
    verify_transfer_bundle,
)


def _stream_sha256(stream: BinaryIO) -> str:
    digest = hashlib.sha256()
    while chunk := stream.read(1024 * 1024):
        digest.update(chunk)
    return digest.hexdigest()


def _regular_tree(root: Path) -> tuple[set[str], set[str]]:
    files: set[str] = set()
    directories: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"extracted tree contains a symlink: {relative}")
        if stat.S_ISDIR(metadata.st_mode):
            directories.add(relative)
        elif stat.S_ISREG(metadata.st_mode):
            files.add(relative)
        else:
            raise ValueError(
                f"extracted tree contains a special filesystem entry: {relative}"
            )
    return files, directories


def _expected_tree(
    archive_stream: BinaryIO, *, extraction_root: Path
) -> tuple[set[str], set[str], list[dict[str, Any]]]:
    files: set[str] = set()
    directories: set[str] = set()
    verified_files: list[dict[str, Any]] = []
    with tarfile.open(fileobj=archive_stream, mode="r:gz") as archive:
        for member in archive:
            relative = PurePosixPath(member.name)
            parents = list(relative.parents)
            directories.update(
                str(parent) for parent in parents if str(parent) not in {".", ""}
            )
            target = extraction_root.joinpath(*relative.parts)
            if member.isdir():
                directories.add(str(relative))
                if target.is_symlink() or not target.is_dir():
                    raise ValueError(
                        f"extracted directory differs from archive: {relative}"
                    )
                continue
            if not member.isfile():
                raise ValueError(f"archive member is not a regular file: {relative}")
            files.add(str(relative))
            if target.is_symlink() or not target.is_file():
                raise ValueError(f"extracted file is missing or unsafe: {relative}")
            if target.stat().st_size != member.size:
                raise ValueError(f"extracted file size mismatch: {relative}")
            archived = archive.extractfile(member)
            if archived is None:
                raise ValueError(f"archive file cannot be read: {relative}")
            archive_digest = _stream_sha256(archived)
            with target.open("rb") as stream:
                extracted_digest = _stream_sha256(stream)
            if extracted_digest != archive_digest:
                raise ValueError(f"extracted file digest mismatch: {relative}")
            verified_files.append(
                {
                    "path": str(relative),
                    "bytes": member.size,
                    "sha256": extracted_digest,
                }
            )
    return files, directories, verified_files


def verify_extracted_bundle(
    manifest_path: Path, *, extraction_root: Path, target_host: str
) -> dict[str, Any]:
    """Bind an exact extracted tree to a verified transfer manifest."""

    preflight = verify_transfer_bundle(manifest_path, target_host=target_host)
    if not extraction_root.is_absolute():
        raise ValueError("extraction root must be absolute")
    if extraction_root.is_symlink() or not extraction_root.is_dir():
        raise ValueError("extraction root must be a regular directory")
    observed_files, observed_directories = _regular_tree(extraction_root)
    expected_files: set[str] = set()
    expected_directories: set[str] = set()
    artifacts: list[dict[str, Any]] = []
    encoded_manifest = read_private_bytes(
        manifest_path,
        label="transfer manifest",
        max_bytes=MAX_TRANSFER_MANIFEST_BYTES,
    )
    if hashlib.sha256(encoded_manifest).hexdigest() != preflight["manifest_sha256"]:
        raise ValueError("transfer manifest changed after bundle verification")
    manifest = loads_json_strict(encoded_manifest)
    manifest_root = manifest_path.parent
    listed_filenames = {entry["filename"] for entry in manifest["artifacts"]}
    for entry in manifest["artifacts"]:
        filename = entry["filename"]
        revision = entry["revision"]
        role = next(
            prefix for prefix in ARTIFACT_PREFIXES if filename.startswith(prefix)
        )
        expected_archive_root = f"{ARCHIVE_ROOT_PREFIXES[role]}{revision[:10]}"
        with open_private_binary(
            manifest_root / filename,
            label=f"transfer artifact {filename}",
            max_bytes=MAX_COMPRESSED_BYTES,
        ) as archive_stream:
            archive_sha256 = _stream_sha256(archive_stream)
            if archive_sha256 != entry["sha256"]:
                raise ValueError(
                    f"transfer artifact changed after bundle verification: {filename}"
                )
            archive_stream.seek(0)
            archive_files, archive_directories, verified_files = _expected_tree(
                archive_stream,
                extraction_root=extraction_root,
            )
        if not all(
            PurePosixPath(path).parts[0] == expected_archive_root
            for path in archive_files | archive_directories
        ):
            raise ValueError(f"archive content escaped expected root: {filename}")
        expected_files.update(archive_files)
        expected_directories.update(archive_directories)
        artifacts.append(
            {
                "filename": filename,
                "revision": revision,
                "extracted_root": expected_archive_root,
                "file_count": len(verified_files),
                "content_set_sha256": hashlib.sha256(
                    json.dumps(
                        sorted(verified_files, key=lambda entry: entry["path"]),
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest(),
            }
        )
    if observed_files != expected_files:
        raise ValueError("extracted file set differs from the reviewed archives")
    if observed_directories != expected_directories:
        raise ValueError("extracted directory set differs from the reviewed archives")
    if {path.name for path in manifest_root.glob("*.tar.gz")} != listed_filenames:
        raise ValueError(
            "transfer directory archives changed during extraction verification"
        )
    return {
        "schema": "flagquantum.qboson_a800_extracted_bundle_verification",
        "version": "1.0",
        "evidence_class": "extraction_preflight_only",
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "verification_hostname": socket.gethostname(),
        "verified_for_target_host": target_host,
        "manifest_filename": manifest_path.name,
        "manifest_sha256": preflight["manifest_sha256"],
        "extracted_content_verified": True,
        "artifacts": artifacts,
        "qboson_hardware_used": False,
        "a800_execution_verified": False,
        "acceptance_evidence": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--extraction-root", required=True, type=Path)
    parser.add_argument(
        "--target-host", choices=("jp-a800-171", "jp-a800-172"), required=True
    )
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    try:
        record = verify_extracted_bundle(
            arguments.manifest,
            extraction_root=arguments.extraction_root,
            target_host=arguments.target_host,
        )
        _write_private_json(arguments.output, record)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Extracted transfer bundle verified: {arguments.output}")


if __name__ == "__main__":
    main()
