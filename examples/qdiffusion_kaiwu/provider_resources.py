"""Validate a credential-free snapshot of QBoson account resources."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from examples.qdiffusion_kaiwu.private_io import read_private_bytes
from examples.qdiffusion_kaiwu.strict_json import loads_json_strict

SCHEMA = "flagquantum.qboson_provider_resources"
VERSION = "1.0"
SOURCE = "authenticated_resource_bill"
TARGETS = ("SPQC-1", "SPQC-550", "SPQC-1000")
MODES = ("optimization", "sampling")
MAX_VALIDITY = timedelta(hours=24)
_MAX_BYTES = 64 * 1024
_TOP_LEVEL_FIELDS = {
    "schema",
    "version",
    "source",
    "captured_at",
    "valid_until",
    "resources",
    "claim_boundary",
}
_RESOURCE_FIELDS = {"target", "mode", "available", "used"}
_CLAIM_BOUNDARY = (
    "Account-resource observation only; it is not spend approval, project "
    "assignment, provider evidence, execution evidence, or acceptance evidence."
)
GATE_FIELDS = frozenset({"snapshot_sha256", "checked_at", "mode", "required_calls"})
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _parse_utc_timestamp(value: object, *, label: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an aware UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError(f"{label} must be an aware UTC timestamp") from None
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError(f"{label} must be an aware UTC timestamp")
    return parsed


def validate_provider_resources(record: object) -> dict[str, Any]:
    """Validate the closed, account-wide provider-resource snapshot schema."""

    if not isinstance(record, dict) or set(record) != _TOP_LEVEL_FIELDS:
        raise ValueError("provider resource snapshot has an invalid top-level schema")
    if record["schema"] != SCHEMA or record["version"] != VERSION:
        raise ValueError("provider resource snapshot has an unsupported schema")
    if record["source"] != SOURCE:
        raise ValueError("provider resource snapshot has an unsupported source")
    if record["claim_boundary"] != _CLAIM_BOUNDARY:
        raise ValueError("provider resource snapshot has an invalid claim boundary")

    captured_at = _parse_utc_timestamp(record["captured_at"], label="captured_at")
    valid_until = _parse_utc_timestamp(record["valid_until"], label="valid_until")
    validity = valid_until - captured_at
    if validity <= timedelta(0) or validity > MAX_VALIDITY:
        raise ValueError("provider resource snapshot validity must be within 24 hours")

    resources = record["resources"]
    if not isinstance(resources, list) or len(resources) != len(TARGETS) * len(MODES):
        raise ValueError("provider resource snapshot must cover every target and mode")
    observed: set[tuple[str, str]] = set()
    for resource in resources:
        if not isinstance(resource, dict) or set(resource) != _RESOURCE_FIELDS:
            raise ValueError("provider resource entry has an invalid schema")
        target = resource["target"]
        mode = resource["mode"]
        if target not in TARGETS or mode not in MODES:
            raise ValueError(
                "provider resource entry has an unsupported target or mode"
            )
        identity = (target, mode)
        if identity in observed:
            raise ValueError("provider resource snapshot contains duplicate entries")
        observed.add(identity)
        for field in ("available", "used"):
            value = resource[field]
            if type(value) is not int or value < 0:
                raise ValueError(
                    f"provider resource {field} must be a nonnegative integer"
                )
    expected = {(target, mode) for target in TARGETS for mode in MODES}
    if observed != expected:
        raise ValueError("provider resource snapshot must cover every target and mode")
    return record


def load_provider_resources(path: Path) -> tuple[dict[str, Any], str]:
    """Load one private resource snapshot and return it with its SHA-256 digest."""

    raw = read_private_bytes(
        path,
        label="QBoson provider resource snapshot",
        max_bytes=_MAX_BYTES,
    )
    record = validate_provider_resources(loads_json_strict(raw))
    return record, hashlib.sha256(raw).hexdigest()


def assess_provider_resources(
    record: dict[str, Any], *, now: datetime | None = None
) -> tuple[bool, str]:
    """Check freshness and the two resources required by the live smoke."""

    validate_provider_resources(record)
    observed_now = datetime.now(timezone.utc) if now is None else now
    if observed_now.tzinfo is None or observed_now.utcoffset() != timedelta(0):
        raise ValueError("now must be an aware UTC timestamp")
    captured_at = _parse_utc_timestamp(record["captured_at"], label="captured_at")
    valid_until = _parse_utc_timestamp(record["valid_until"], label="valid_until")
    if observed_now < captured_at - timedelta(minutes=5):
        return False, "provider_resource_snapshot_from_future"
    if observed_now > valid_until:
        return False, "provider_resource_snapshot_expired"
    available = {
        (item["target"], item["mode"]): item["available"]
        for item in record["resources"]
    }
    if not any(
        available[(target, "optimization")] >= 1
        and available[(target, "sampling")] >= 1
        for target in TARGETS
    ):
        if not any(available[(target, "optimization")] >= 1 for target in TARGETS):
            return False, "optimization_resource_unavailable"
        if not any(available[(target, "sampling")] >= 1 for target in TARGETS):
            return False, "sampling_resource_unavailable"
        return False, "common_target_resources_unavailable"
    return True, "provider_smoke_resources_available"


def assess_provider_budget(
    record: dict[str, Any],
    *,
    mode: str,
    required_calls: int,
    now: datetime | None = None,
) -> tuple[bool, str]:
    """Check freshness and one target's balance for a bounded live run."""

    validate_provider_resources(record)
    if mode not in MODES:
        raise ValueError("mode must be optimization or sampling")
    if type(required_calls) is not int or required_calls <= 0:
        raise ValueError("required_calls must be a positive integer")
    observed_now = datetime.now(timezone.utc) if now is None else now
    if observed_now.tzinfo is None or observed_now.utcoffset() != timedelta(0):
        raise ValueError("now must be an aware UTC timestamp")
    captured_at = _parse_utc_timestamp(record["captured_at"], label="captured_at")
    valid_until = _parse_utc_timestamp(record["valid_until"], label="valid_until")
    if observed_now < captured_at - timedelta(minutes=5):
        return False, "provider_resource_snapshot_from_future"
    if observed_now > valid_until:
        return False, "provider_resource_snapshot_expired"
    balances = [
        item["available"]
        for item in record["resources"]
        if item["mode"] == mode
    ]
    if max(balances, default=0) < required_calls:
        return False, f"{mode}_resource_budget_insufficient"
    return True, f"provider_{mode}_budget_available"


def build_provider_resource_gate(
    *,
    snapshot_sha256: str,
    checked_at: datetime,
    mode: str,
    required_calls: int,
) -> dict[str, Any]:
    """Build closed evidence describing one successful pre-submission check."""

    if not isinstance(snapshot_sha256, str) or _SHA256.fullmatch(
        snapshot_sha256
    ) is None:
        raise ValueError("snapshot_sha256 must be a lowercase SHA-256 digest")
    if checked_at.tzinfo is None or checked_at.utcoffset() != timedelta(0):
        raise ValueError("checked_at must be an aware UTC timestamp")
    if mode not in MODES:
        raise ValueError("mode must be optimization or sampling")
    if type(required_calls) is not int or required_calls <= 0:
        raise ValueError("required_calls must be a positive integer")
    return {
        "snapshot_sha256": snapshot_sha256,
        "checked_at": checked_at.isoformat(),
        "mode": mode,
        "required_calls": required_calls,
    }


def validate_provider_resource_gate(record: object) -> dict[str, Any]:
    """Validate retained resource-gate evidence without trusting its producer."""

    if not isinstance(record, dict) or set(record) != GATE_FIELDS:
        raise ValueError("provider resource gate has an invalid schema")
    checked_at = _parse_utc_timestamp(record["checked_at"], label="checked_at")
    return build_provider_resource_gate(
        snapshot_sha256=record["snapshot_sha256"],
        checked_at=checked_at,
        mode=record["mode"],
        required_calls=record["required_calls"],
    )


__all__ = (
    "SCHEMA",
    "SOURCE",
    "TARGETS",
    "VERSION",
    "assess_provider_budget",
    "assess_provider_resources",
    "build_provider_resource_gate",
    "load_provider_resources",
    "validate_provider_resource_gate",
    "validate_provider_resources",
)
