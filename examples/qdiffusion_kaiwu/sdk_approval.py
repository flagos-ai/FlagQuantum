"""Validate the reviewed Kaiwu SDK artifact and permitted-use boundary."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from examples.qdiffusion_kaiwu.private_io import read_private_bytes
from examples.qdiffusion_kaiwu.strict_json import loads_json_strict

SCHEMA = "flagquantum.qboson_kaiwu_sdk_approval"
VERSION = "1.1"
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
        "project_no",
        "project_assignment_reviewed_at",
        "project_assignment_reference",
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
    reviewed_timestamp: datetime | None = None
    try:
        reviewed_timestamp = datetime.fromisoformat(
            str(reviewed_at).replace("Z", "+00:00")
        )
        if reviewed_timestamp.tzinfo is None:
            raise ValueError
    except ValueError:
        errors.append(f"{label}.rights_reviewed_at: expected a timezone-aware timestamp")
    service_terms_date = datetime.fromisoformat(
        EXPECTED_IDENTITY["service_terms_effective_date"]
    ).date()
    if (
        reviewed_timestamp is not None
        and reviewed_timestamp.date() < service_terms_date
    ):
        errors.append(
            f"{label}.rights_reviewed_at: predates the reviewed service terms"
        )
    approval_reference = record.get("approval_reference")
    if (
        not _canonical_printable_identifier(approval_reference)
        or approval_reference == "<required>"
    ):
        errors.append(f"{label}.approval_reference: frozen value is required")
    project_no = record.get("project_no")
    if not _canonical_printable_identifier(project_no) or project_no == "<required>":
        errors.append(f"{label}.project_no: assigned project is required")
    project_reviewed_at = record.get("project_assignment_reviewed_at")
    try:
        project_reviewed_timestamp = datetime.fromisoformat(
            str(project_reviewed_at).replace("Z", "+00:00")
        )
        if project_reviewed_timestamp.tzinfo is None:
            raise ValueError
    except ValueError:
        errors.append(
            f"{label}.project_assignment_reviewed_at: expected a timezone-aware "
            "timestamp"
        )
    project_reference = record.get("project_assignment_reference")
    if (
        not _canonical_printable_identifier(project_reference)
        or project_reference == "<required>"
    ):
        errors.append(
            f"{label}.project_assignment_reference: frozen value is required"
        )
    for field in (
        "organizational_use_approved",
        "isolated_container_use_approved",
        "host_staging_approved",
        "adapter_distribution_approved",
    ):
        if record.get(field) is not True:
            errors.append(f"{label}.{field}: explicit approval is required")
    return errors


def verify_approved_project_assignment(
    project_no: str, approval: dict[str, Any]
) -> None:
    """Require a runtime project to match the reviewed account assignment."""

    errors = validate_sdk_approval_record(approval)
    if errors:
        raise ValueError("; ".join(errors))
    if not _canonical_printable_identifier(project_no):
        raise ValueError("project_no must be a canonical printable identifier")
    if project_no != approval["project_no"]:
        raise ValueError("project_no differs from the reviewed project assignment")


def load_sdk_approval(path: Path) -> tuple[dict[str, Any], str]:
    """Load one private approval record without following a final symlink."""

    if not path.is_absolute():
        raise ValueError("SDK approval path must be absolute")
    encoded = read_private_bytes(
        path, label="SDK approval", max_bytes=_MAX_APPROVAL_BYTES
    )
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
