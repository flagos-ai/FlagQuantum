"""Closed shape and identity checks for research evidence bundles."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_BUNDLE_SCHEMA = "flagquantum.research_capability_evidence_bundle.v1"
_BUNDLE_FIELDS = frozenset(
    {
        "schema",
        "run_id",
        "capability",
        "status",
        "result",
        "metrics",
        "artifacts",
        "execution",
        "validation",
        "attempts",
        "warnings",
        "provenance",
        "error",
    }
)
_CAPABILITY_FIELDS = frozenset({"capability_id", "version", "provider"})
_PROVENANCE_FIELDS = frozenset(
    {
        "package",
        "package_version",
        "python_version",
        "torch_version",
        "fallback_used",
        "input_sha256",
        "deterministic",
        "random_seed",
        "content_sha256",
    }
)
_BUNDLE_STATUSES = frozenset(
    {"succeeded", "unsupported", "validation_failed", "failed"}
)
def _evidence_shape_error(payload: Mapping[str, Any]) -> tuple[str, str] | None:
    fields = frozenset(payload)
    if fields != _BUNDLE_FIELDS:
        missing = sorted(_BUNDLE_FIELDS - fields)
        unknown = sorted(fields - _BUNDLE_FIELDS)
        return (
            "INVALID_EVIDENCE_FIELDS",
            f"evidence fields do not match v1; missing={missing}, unknown={unknown}",
        )
    if payload["schema"] != _BUNDLE_SCHEMA:
        return ("INVALID_EVIDENCE_SCHEMA", "evidence schema is not the supported v1")
    if not isinstance(payload["run_id"], str) or not payload["run_id"]:
        return ("INVALID_EVIDENCE_RUN_ID", "evidence run_id must be non-empty")
    if payload["status"] not in _BUNDLE_STATUSES:
        return ("INVALID_EVIDENCE_STATUS", "evidence status is not recognized")
    capability = payload["capability"]
    if (
        not isinstance(capability, Mapping)
        or frozenset(capability) != _CAPABILITY_FIELDS
    ):
        return (
            "INVALID_EVIDENCE_CAPABILITY",
            "evidence capability must contain exactly capability_id, version, provider",
        )
    for field_name in ("capability_id", "version", "provider"):
        if not isinstance(capability[field_name], str) or not capability[field_name]:
            return (
                "INVALID_EVIDENCE_CAPABILITY",
                f"evidence capability {field_name} must be non-empty",
            )
    for field_name in ("result", "metrics", "execution"):
        if not isinstance(payload[field_name], Mapping):
            return (
                "INVALID_EVIDENCE_FIELD_TYPE",
                f"evidence {field_name} must be a mapping",
            )
    for field_name in ("artifacts", "validation", "attempts", "warnings"):
        if not isinstance(payload[field_name], (list, tuple)):
            return (
                "INVALID_EVIDENCE_FIELD_TYPE",
                f"evidence {field_name} must be an array",
            )
    provenance = payload["provenance"]
    if (
        not isinstance(provenance, Mapping)
        or frozenset(provenance) != _PROVENANCE_FIELDS
    ):
        return (
            "INVALID_EVIDENCE_PROVENANCE",
            "evidence provenance fields do not match v1",
        )
    digest = provenance["content_sha256"]
    if not isinstance(digest, str) or len(digest) != 64:
        return (
            "INVALID_EVIDENCE_PROVENANCE",
            "evidence content_sha256 must be a 64-character digest",
        )
    if payload["error"] is not None and not isinstance(payload["error"], Mapping):
        return ("INVALID_EVIDENCE_ERROR", "evidence error must be null or a mapping")
    return None
