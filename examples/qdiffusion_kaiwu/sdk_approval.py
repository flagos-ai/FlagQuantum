"""Validate the reviewed Kaiwu SDK artifact and permitted-use boundary."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from datetime import datetime
from pathlib import Path
from typing import Any

from examples.qdiffusion_kaiwu.strict_json import loads_json_strict

SCHEMA = "flagquantum.qboson_kaiwu_sdk_approval"
VERSION = "1.0"
SDK_APPROVAL_FIELDS = frozenset(
    {
        "schema",
        "version",
        "distribution",
        "sdk_version",
        "wheel_filename",
        "source_url",
        "sha256",
        "service_terms_url",
        "service_terms_effective_date",
        "rights_reviewed_at",
        "approval_reference",
        "organizational_use_approved",
        "isolated_container_use_approved",
        "host_staging_approved",
        "adapter_distribution_approved",
        "sdk_redistribution_policy",
    }
)
EXPECTED_IDENTITY = {
    "schema": SCHEMA,
    "version": VERSION,
    "distribution": "kaiwu",
    "sdk_version": "1.3.1",
    "wheel_filename": "kaiwu-1.3.1-cp310-none-manylinux1_x86_64.whl",
    "source_url": "https://pypi.org/pypi/kaiwu/1.3.1/json",
    "service_terms_url": (
        "https://platform.qboson.com/agreement?"
        "type=QBoson-SPQC-Platform-Users-Agreement"
    ),
    "service_terms_effective_date": "2026-07-09",
    "sdk_redistribution_policy": "no-sdk-redistribution",
}
SHA256 = re.compile(r"[0-9a-f]{64}")
_MAX_APPROVAL_BYTES = 64 * 1024


def _canonical_printable_identifier(value: Any) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and value == value.strip()
        and value.isprintable()
    )


def validate_sdk_approval_record(
    record: Any, *, label: str = "sdk_approval"
) -> list[str]:
    """Return every reason one SDK approval record cannot authorize a run."""

    errors: list[str] = []
    if not isinstance(record, dict):
        return [f"{label}: expected an object"]
    if set(record) != SDK_APPROVAL_FIELDS:
        errors.append(f"{label}: field set differs from schema")
    for field, expected in EXPECTED_IDENTITY.items():
        if record.get(field) != expected:
            errors.append(f"{label}.{field}: differs from the reviewed 1.3.1 lane")
    digest = record.get("sha256")
    if not isinstance(digest, str) or SHA256.fullmatch(digest) is None:
        errors.append(f"{label}.sha256: expected a SHA-256 digest")
    reviewed_at = record.get("rights_reviewed_at")
    try:
        reviewed_timestamp = datetime.fromisoformat(
            str(reviewed_at).replace("Z", "+00:00")
        )
        if reviewed_timestamp.tzinfo is None:
            raise ValueError
    except ValueError:
        errors.append(f"{label}.rights_reviewed_at: expected a timezone-aware timestamp")
    approval_reference = record.get("approval_reference")
    if (
        not _canonical_printable_identifier(approval_reference)
        or approval_reference == "<required>"
    ):
        errors.append(f"{label}.approval_reference: frozen value is required")
    for field in (
        "organizational_use_approved",
        "isolated_container_use_approved",
        "host_staging_approved",
        "adapter_distribution_approved",
    ):
        if record.get(field) is not True:
            errors.append(f"{label}.{field}: explicit approval is required")
    return errors


def load_sdk_approval(path: Path) -> tuple[dict[str, Any], str]:
    """Load one private approval record without following a final symlink."""

    if not path.is_absolute():
        raise ValueError("SDK approval path must be absolute")
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise ValueError("platform cannot safely open the SDK approval")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | no_follow
    try:
        descriptor = os.open(path, flags)
    except OSError:
        raise ValueError("SDK approval must be a regular, non-symlink file") from None
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError("SDK approval must be a regular, non-symlink file")
        if opened.st_mode & 0o077:
            raise ValueError("SDK approval must not be accessible by group or others")
        if opened.st_size > _MAX_APPROVAL_BYTES:
            raise ValueError("SDK approval exceeds the bounded size")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            encoded = stream.read(_MAX_APPROVAL_BYTES + 1)
        if len(encoded) > _MAX_APPROVAL_BYTES:
            raise ValueError("SDK approval exceeds the bounded size")
        visible = path.lstat()
        if (
            not stat.S_ISREG(visible.st_mode)
            or (visible.st_dev, visible.st_ino) != (opened.st_dev, opened.st_ino)
        ):
            raise ValueError("SDK approval binding changed during validation")
    except OSError:
        raise ValueError("SDK approval binding changed during validation") from None
    finally:
        os.close(descriptor)
    try:
        raw = loads_json_strict(encoded)
    except json.JSONDecodeError as exc:
        raise ValueError("SDK approval is not valid JSON") from exc
    errors = validate_sdk_approval_record(raw)
    if errors:
        raise ValueError("; ".join(errors))
    return raw, hashlib.sha256(encoded).hexdigest()


def verify_approved_kaiwu_distribution(
    environment_record: dict[str, Any], approval: dict[str, Any]
) -> None:
    """Bind the exact installed-lane artifact to the reviewed SDK approval."""

    errors = validate_sdk_approval_record(approval)
    if errors:
        raise ValueError("; ".join(errors))
    distributions = environment_record.get("distributions")
    if not isinstance(distributions, list):
        raise ValueError("environment lock has no distribution inventory")
    kaiwu = [
        distribution
        for distribution in distributions
        if isinstance(distribution, dict) and distribution.get("name") == "kaiwu"
    ]
    if len(kaiwu) != 1:
        raise ValueError("environment lock requires exactly one Kaiwu distribution")
    locked = kaiwu[0]
    if locked.get("version") != approval.get("sdk_version"):
        raise ValueError("environment lock Kaiwu version differs from SDK approval")
    if locked.get("approved_artifact_sha256") != approval.get("sha256"):
        raise ValueError("environment lock Kaiwu artifact differs from SDK approval")
