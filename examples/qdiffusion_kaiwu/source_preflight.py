"""Validate the reviewed source identity used by a QDiffusion execution."""

from __future__ import annotations

import hashlib
import json
import re
import stat
from pathlib import Path
from typing import Any

SCHEMA = "flagquantum.qboson_a800_extracted_bundle_verification"
TRANSFER_MANIFEST_SCHEMA = "flagquantum.qboson_a800_transfer_bundle"
EVIDENCE_CLASS = "extraction_preflight_only"
TRANSFER_MANIFEST_CLASSIFICATION = "local_preparation_only_not_execution_evidence"
HOSTS = {"jp-a800-171", "jp-a800-172"}
COMMUNITY_REVISION = "b648b531c034bd6ae9b7a34fed994c717967cc72"
FULL_REVISION = re.compile(r"[0-9a-f]{40}")
SHA256 = re.compile(r"[0-9a-f]{64}")
ROLES = {
    "flagquantum-qboson-": "FlagQuantum-",
    "kaiwu-plugin-": "kaiwu-pytorch-plugin-",
    "kaiwu-community-": "kaiwu-community-",
}


def _artifact_for(artifacts: list[object], *, filename_prefix: str) -> dict[str, Any]:
    matches = [
        artifact
        for artifact in artifacts
        if isinstance(artifact, dict)
        and isinstance(artifact.get("filename"), str)
        and artifact["filename"].startswith(filename_prefix)
    ]
    if len(matches) != 1:
        raise ValueError(f"source preflight must contain one {filename_prefix} role")
    return matches[0]


def _validate_artifact(
    artifact: dict[str, Any], *, filename_prefix: str, expected_revision: str
) -> None:
    filename = artifact.get("filename")
    revision = artifact.get("revision")
    extracted_root = artifact.get("extracted_root")
    file_count = artifact.get("file_count")
    content_digest = artifact.get("content_set_sha256")
    if revision != expected_revision or FULL_REVISION.fullmatch(str(revision)) is None:
        raise ValueError(f"source preflight {filename_prefix} revision mismatch")
    abbreviated = expected_revision[:10]
    if filename != f"{filename_prefix}{abbreviated}.tar.gz":
        raise ValueError(f"source preflight {filename_prefix} filename mismatch")
    if extracted_root != f"{ROLES[filename_prefix]}{abbreviated}":
        raise ValueError(f"source preflight {filename_prefix} root mismatch")
    if (
        not isinstance(file_count, int)
        or isinstance(file_count, bool)
        or file_count <= 0
    ):
        raise ValueError(f"source preflight {filename_prefix} file count is invalid")
    if not isinstance(content_digest, str) or SHA256.fullmatch(content_digest) is None:
        raise ValueError(f"source preflight {filename_prefix} digest is invalid")


def load_source_preflight(
    path: Path,
    *,
    execution_host: str,
    source_revision: str,
    plugin_revision: str,
    source_root: Path | None = None,
    plugin_root: Path | None = None,
) -> tuple[dict[str, Any], str]:
    """Load and validate one private post-extraction source record."""

    if not path.is_absolute():
        raise ValueError("source preflight path must be absolute")
    if path.is_symlink() or not path.is_file():
        raise ValueError("source preflight must be a regular, non-symlink file")
    if path.stat().st_mode & 0o077:
        raise ValueError("source preflight must not be accessible by group or others")
    if execution_host not in HOSTS:
        raise ValueError("source preflight execution host is outside the reviewed pair")
    encoded = path.read_bytes()
    try:
        record = json.loads(encoded)
    except json.JSONDecodeError as exc:
        raise ValueError("source preflight is not valid JSON") from exc
    if not isinstance(record, dict):
        raise ValueError("source preflight must be a JSON object")
    validate_source_preflight_record(
        record,
        execution_host=execution_host,
        source_revision=source_revision,
        plugin_revision=plugin_revision,
    )
    if source_root is not None:
        validate_runtime_source_root(
            record,
            filename_prefix="flagquantum-qboson-",
            root=source_root,
        )
    if plugin_root is not None:
        validate_runtime_source_root(
            record,
            filename_prefix="kaiwu-plugin-",
            root=plugin_root,
        )
    return record, hashlib.sha256(encoded).hexdigest()


def validate_source_preflight_record(
    record: dict[str, Any],
    *,
    execution_host: str,
    source_revision: str,
    plugin_revision: str,
) -> None:
    """Validate source-preflight semantics after an outer hash check."""

    if execution_host not in HOSTS:
        raise ValueError("source preflight execution host is outside the reviewed pair")
    if record.get("schema") != SCHEMA or record.get("version") != "1.0":
        raise ValueError("unsupported source preflight schema or version")
    if record.get("evidence_class") != EVIDENCE_CLASS:
        raise ValueError("source preflight has an unsafe evidence class")
    if record.get("verified_for_target_host") != execution_host:
        raise ValueError("source preflight target host mismatch")
    if (
        not isinstance(record.get("verification_hostname"), str)
        or not record["verification_hostname"].strip()
    ):
        raise ValueError("source preflight verification hostname is missing")
    if record.get("extracted_content_verified") is not True:
        raise ValueError("source preflight did not verify extracted content")
    if any(
        record.get(field) is not False
        for field in (
            "qboson_hardware_used",
            "a800_execution_verified",
            "acceptance_evidence",
        )
    ):
        raise ValueError("source preflight overstates execution or acceptance evidence")
    manifest_digest = record.get("manifest_sha256")
    if (
        not isinstance(manifest_digest, str)
        or SHA256.fullmatch(manifest_digest) is None
    ):
        raise ValueError("source preflight manifest digest is invalid")
    artifacts = record.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 3:
        raise ValueError("source preflight must contain exactly three artifacts")
    expected_revisions = {
        "flagquantum-qboson-": source_revision,
        "kaiwu-plugin-": plugin_revision,
        "kaiwu-community-": COMMUNITY_REVISION,
    }
    for filename_prefix, expected_revision in expected_revisions.items():
        if FULL_REVISION.fullmatch(expected_revision) is None:
            raise ValueError(
                "expected source revisions must be full lowercase revisions"
            )
        _validate_artifact(
            _artifact_for(artifacts, filename_prefix=filename_prefix),
            filename_prefix=filename_prefix,
            expected_revision=expected_revision,
        )


def validate_common_transfer_manifest(
    records: tuple[dict[str, Any], dict[str, Any]],
) -> str:
    """Require both host preflights to bind the same reviewed manifest."""

    digests = tuple(record.get("manifest_sha256") for record in records)
    if any(
        not isinstance(digest, str) or SHA256.fullmatch(digest) is None
        for digest in digests
    ):
        raise ValueError(
            "source preflights contain an invalid transfer manifest digest"
        )
    if digests[0] != digests[1]:
        raise ValueError("source preflights do not share one transfer manifest")
    return digests[0]


def validate_transfer_manifest_record(
    record: dict[str, Any],
    *,
    source_revision: str,
    plugin_revision: str,
) -> None:
    """Validate the reviewed transfer manifest copied into final evidence."""

    if (
        record.get("schema") != TRANSFER_MANIFEST_SCHEMA
        or record.get("version") != "1.0"
    ):
        raise ValueError("unsupported transfer manifest schema or version")
    if record.get("classification") != TRANSFER_MANIFEST_CLASSIFICATION:
        raise ValueError("transfer manifest has an unsafe classification")
    if record.get("created_for_hosts") != ["jp-a800-171", "jp-a800-172"]:
        raise ValueError("transfer manifest does not cover both validation hosts")
    artifacts = record.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 3:
        raise ValueError("transfer manifest must contain exactly three artifacts")
    expected_revisions = {
        "flagquantum-qboson-": source_revision,
        "kaiwu-plugin-": plugin_revision,
        "kaiwu-community-": COMMUNITY_REVISION,
    }
    for filename_prefix, expected_revision in expected_revisions.items():
        if FULL_REVISION.fullmatch(expected_revision) is None:
            raise ValueError(
                "expected source revisions must be full lowercase revisions"
            )
        artifact = _artifact_for(artifacts, filename_prefix=filename_prefix)
        filename = artifact.get("filename")
        if artifact.get("revision") != expected_revision:
            raise ValueError(f"transfer manifest {filename_prefix} revision mismatch")
        if filename != f"{filename_prefix}{expected_revision[:10]}.tar.gz":
            raise ValueError(f"transfer manifest {filename_prefix} filename mismatch")
        digest = artifact.get("sha256")
        if not isinstance(digest, str) or SHA256.fullmatch(digest) is None:
            raise ValueError(f"transfer manifest {filename_prefix} digest is invalid")


def validate_runtime_source_root(
    record: dict[str, Any], *, filename_prefix: str, root: Path
) -> None:
    """Recompute one execution-time source tree against its preflight identity."""

    if not root.is_absolute():
        raise ValueError("runtime source root must be absolute")
    if root.is_symlink() or not root.is_dir():
        raise ValueError("runtime source root must be a regular, non-symlink directory")
    artifacts = record.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("source preflight artifacts are unavailable")
    artifact = _artifact_for(artifacts, filename_prefix=filename_prefix)
    if root.name != artifact.get("extracted_root"):
        raise ValueError(f"runtime {filename_prefix} root name differs from preflight")
    verified_files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(root).as_posix()
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"runtime source tree contains a symlink: {relative}")
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"runtime source tree contains a special file: {relative}")
        verified_files.append(
            {
                "path": f"{root.name}/{relative}",
                "bytes": metadata.st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    if len(verified_files) != artifact.get("file_count"):
        raise ValueError(f"runtime {filename_prefix} file count differs from preflight")
    content_set_sha256 = hashlib.sha256(
        json.dumps(
            verified_files,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    if content_set_sha256 != artifact.get("content_set_sha256"):
        raise ValueError(f"runtime {filename_prefix} content differs from preflight")
