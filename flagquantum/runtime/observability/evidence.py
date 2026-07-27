"""Immutable runtime-evidence envelopes and integrity verification."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


class ArtifactClass(str, Enum):
    PLAN = "plan"
    DEVELOPMENT_RUN = "development_run"
    MEASURED_PRODUCTION_RUN = "measured_production_run"


class EvidenceScope(str, Enum):
    ONE_GPU_LOCAL = "one_gpu_local"
    TWO_GPU_SEMANTIC = "two_gpu_semantic_regression"
    SCHEDULED_SCALE = "scheduled_4_8_gpu_scale"


MEASURED_FIELD_NAMES = frozenset(
    {
        "measured_peak_memory_bytes",
        "measured_peak_memory_bytes_by_rank",
        "communication_time_seconds",
        "timings",
    }
)
_HEX_40 = re.compile(r"^[0-9a-f]{40}$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class RuntimeProvenance:
    commit: str
    workload_sha256: str
    command: tuple[str, ...]
    devices: tuple[str, ...]
    topology: str
    rank_mapping: tuple[str, ...]
    collective_backend: str
    warmup: int
    iterations: int
    seeds: tuple[int, ...]
    raw_log_sha256: str
    fallback_events: tuple[str, ...]


@dataclass(frozen=True)
class EvidenceArtifact:
    schema: str
    artifact_class: ArtifactClass
    evidence_scope: EvidenceScope
    provenance: RuntimeProvenance
    evidence: Mapping[str, Any]
    integrity: Mapping[str, str]

    def summary(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "artifact_class": self.artifact_class.value,
            "evidence_scope": self.evidence_scope.value,
            "provenance": asdict(self.provenance),
            "evidence": _thaw(self.evidence),
            "integrity": dict(self.integrity),
        }


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _canonical(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _contains_measured_field(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key) in MEASURED_FIELD_NAMES or _contains_measured_field(item)
            for key, item in value.items()
        )
    if isinstance(value, (tuple, list)):
        return any(_contains_measured_field(item) for item in value)
    return False


def _scope_errors(
    scope: EvidenceScope, provenance: RuntimeProvenance
) -> tuple[str, ...]:
    device_count = len(provenance.devices)
    expected = {
        EvidenceScope.ONE_GPU_LOCAL: {1},
        EvidenceScope.TWO_GPU_SEMANTIC: {2},
        EvidenceScope.SCHEDULED_SCALE: {4, 8},
    }[scope]
    errors = []
    if device_count not in expected:
        errors.append(
            f"{scope.value} requires device count in {sorted(expected)}, got {device_count}"
        )
    if len(provenance.rank_mapping) != device_count:
        errors.append("rank mapping must cover every device")
    return tuple(errors)


def create_evidence_artifact(
    *,
    artifact_class: ArtifactClass,
    evidence_scope: EvidenceScope,
    provenance: RuntimeProvenance,
    evidence: Mapping[str, Any],
    signing_key: bytes,
) -> EvidenceArtifact:
    if not signing_key:
        raise ValueError("a non-empty signing key is required")
    scope_errors = _scope_errors(evidence_scope, provenance)
    if scope_errors:
        raise ValueError("; ".join(scope_errors))
    if not _HEX_40.fullmatch(provenance.commit):
        raise ValueError("commit must be a full 40-character hexadecimal SHA")
    if not _HEX_64.fullmatch(provenance.workload_sha256):
        raise ValueError("workload_sha256 must be a SHA256 digest")
    if not _HEX_64.fullmatch(provenance.raw_log_sha256):
        raise ValueError("raw_log_sha256 must be a SHA256 digest")
    if (
        artifact_class is not ArtifactClass.MEASURED_PRODUCTION_RUN
        and _contains_measured_field(evidence)
    ):
        raise ValueError("plans and development runs cannot populate measured fields")
    if artifact_class is ArtifactClass.MEASURED_PRODUCTION_RUN:
        missing = [
            name
            for name, value in asdict(provenance).items()
            if value in ("", (), None) and name not in {"fallback_events"}
        ]
        if missing:
            raise ValueError(f"production provenance is incomplete: {missing}")
        if not _contains_measured_field(evidence):
            raise ValueError("production evidence requires runtime-measured fields")

    unsigned = {
        "schema": "flagquantum_runtime_evidence_v1",
        "artifact_class": artifact_class.value,
        "evidence_scope": evidence_scope.value,
        "provenance": asdict(provenance),
        "evidence": dict(evidence),
    }
    content_sha256 = hashlib.sha256(_canonical(unsigned)).hexdigest()
    signature = hmac.new(
        signing_key, content_sha256.encode(), hashlib.sha256
    ).hexdigest()
    return EvidenceArtifact(
        schema="flagquantum_runtime_evidence_v1",
        artifact_class=artifact_class,
        evidence_scope=evidence_scope,
        provenance=provenance,
        evidence=_freeze(evidence),
        integrity=MappingProxyType(
            {
                "algorithm": "hmac-sha256",
                "content_sha256": content_sha256,
                "signature": signature,
            }
        ),
    )


def verify_evidence_artifact(
    payload: Mapping[str, Any], *, signing_key: bytes
) -> tuple[bool, tuple[str, ...]]:
    errors: list[str] = []
    if payload.get("schema") != "flagquantum_runtime_evidence_v1":
        errors.append("runtime evidence schema is missing or unknown")
    if payload.get("artifact_class") != ArtifactClass.MEASURED_PRODUCTION_RUN.value:
        errors.append("release scan requires measured_production_run")
    integrity = payload.get("integrity")
    if not isinstance(integrity, Mapping):
        return False, tuple((*errors, "runtime evidence integrity is missing"))
    unsigned = {
        key: payload.get(key)
        for key in (
            "schema",
            "artifact_class",
            "evidence_scope",
            "provenance",
            "evidence",
        )
    }
    expected_hash = hashlib.sha256(_canonical(unsigned)).hexdigest()
    if not hmac.compare_digest(str(integrity.get("content_sha256", "")), expected_hash):
        errors.append("runtime evidence content checksum mismatch")
    expected_signature = hmac.new(
        signing_key, expected_hash.encode(), hashlib.sha256
    ).hexdigest()
    if not signing_key or not hmac.compare_digest(
        str(integrity.get("signature", "")), expected_signature
    ):
        errors.append("runtime evidence signature mismatch")
    provenance = payload.get("provenance")
    required = tuple(
        field.name for field in RuntimeProvenance.__dataclass_fields__.values()
    )
    if not isinstance(provenance, Mapping):
        errors.append("runtime provenance is missing")
    else:
        for name in required:
            if name not in provenance or provenance[name] in ("", None):
                errors.append(f"runtime provenance missing {name}")
        try:
            parsed_scope = EvidenceScope(str(payload.get("evidence_scope", "")))
            parsed_provenance = RuntimeProvenance(
                commit=str(provenance["commit"]),
                workload_sha256=str(provenance["workload_sha256"]),
                command=tuple(str(item) for item in provenance["command"]),
                devices=tuple(str(item) for item in provenance["devices"]),
                topology=str(provenance["topology"]),
                rank_mapping=tuple(str(item) for item in provenance["rank_mapping"]),
                collective_backend=str(provenance["collective_backend"]),
                warmup=int(provenance["warmup"]),
                iterations=int(provenance["iterations"]),
                seeds=tuple(int(item) for item in provenance["seeds"]),
                raw_log_sha256=str(provenance["raw_log_sha256"]),
                fallback_events=tuple(
                    str(item) for item in provenance["fallback_events"]
                ),
            )
            errors.extend(_scope_errors(parsed_scope, parsed_provenance))
        except (KeyError, TypeError, ValueError):
            errors.append("runtime provenance types or evidence scope are invalid")
    return not errors, tuple(errors)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "ArtifactClass",
    "EvidenceArtifact",
    "EvidenceScope",
    "MEASURED_FIELD_NAMES",
    "RuntimeProvenance",
    "create_evidence_artifact",
    "sha256_file",
    "verify_evidence_artifact",
]
