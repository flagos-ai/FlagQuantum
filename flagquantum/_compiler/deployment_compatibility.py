"""Private offline compatibility inspection for legacy deployment packages."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

from flagquantum.deployment.cloud import (
    DeploymentPackage,
    validate_deployment_package,
)
from flagquantum.deployment.routing_evidence import DEPLOYMENT_PACKAGE_SCHEMA

from .target_capabilities import (
    ArtifactFormat,
    ArtifactProfile,
    TargetCapabilities,
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_SENSITIVE_FRAGMENTS = (
    "api_key",
    "apikey",
    "authorization",
    "bearer ",
    "credential",
    "password",
    "secret",
    "token",
    "://",
)


class CompatibilityStatus(str, Enum):
    ELIGIBLE_FOR_VERIFIED_RECOMPILE = "eligible_for_verified_recompile"
    REQUIRES_TARGET_ENRICHMENT = "requires_target_enrichment"
    UNSUPPORTED = "unsupported"
    INVALID = "invalid"
    LIMIT_EXCEEDED = "limit_exceeded"


class CompatibilityFinding(str, Enum):
    PACKAGE_SCHEMA_INVALID = "package_schema_invalid"
    PACKAGE_IDENTITY_INVALID = "package_identity_invalid"
    PROGRAM_PROFILE_UNSUPPORTED = "program_profile_unsupported"
    DYNAMIC_PROGRAM_UNSUPPORTED = "dynamic_program_unsupported"
    SOURCE_IR_REQUIRES_VERIFIED_IMPORT = "source_ir_requires_verified_import"
    TARGET_SNAPSHOT_INCOMPLETE = "target_snapshot_incomplete"
    TARGET_PROFILE_MISMATCH = "target_profile_mismatch"
    TARGET_CAPACITY_EXCEEDED = "target_capacity_exceeded"
    SHOTS_LIMIT_EXCEEDED = "shots_limit_exceeded"
    ROUTING_EVIDENCE_INVALID = "routing_evidence_invalid"
    PROVIDER_METADATA_REJECTED = "provider_metadata_rejected"
    INPUT_SIZE_LIMIT_EXCEEDED = "input_size_limit_exceeded"
    METADATA_LIMIT_EXCEEDED = "metadata_limit_exceeded"
    ELIGIBLE_FOR_VERIFIED_RECOMPILE = "eligible_for_verified_recompile"


@dataclass(frozen=True)
class CompatibilityInspectionLimits:
    maximum_program_bytes: int
    maximum_metadata_entries: int
    maximum_metadata_depth: int
    maximum_metadata_encoded_bytes: int
    maximum_findings: int

    def __post_init__(self) -> None:
        for name in (
            "maximum_program_bytes",
            "maximum_metadata_entries",
            "maximum_metadata_depth",
            "maximum_metadata_encoded_bytes",
            "maximum_findings",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"compatibility {name} must be a positive integer")


def _digest(values: dict[str, object]) -> str:
    encoded = json.dumps(
        values, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class CompatibilityReport:
    status: CompatibilityStatus
    findings: tuple[CompatibilityFinding, ...]
    legacy_artifact_digest: str | None
    program_content_hash: str
    program_bytes: int
    candidate_profile: ArtifactProfile | None
    target_capability_fingerprint: str
    report_identity: str

    def __post_init__(self) -> None:
        if not isinstance(self.status, CompatibilityStatus):
            raise ValueError("compatibility status must use the closed enum")
        findings = tuple(sorted(self.findings, key=lambda item: item.value))
        if any(not isinstance(item, CompatibilityFinding) for item in findings):
            raise ValueError("compatibility findings must use the closed enum")
        if not findings or len(findings) != len(set(findings)):
            raise ValueError("compatibility findings must be non-empty and unique")
        object.__setattr__(self, "findings", findings)
        for name in (
            "program_content_hash",
            "target_capability_fingerprint",
            "report_identity",
        ):
            if _SHA256.fullmatch(getattr(self, name)) is None:
                raise ValueError(f"compatibility {name} must be lowercase SHA-256")
        if self.legacy_artifact_digest is not None and (
            _SHA256.fullmatch(self.legacy_artifact_digest) is None
        ):
            raise ValueError("legacy artifact digest must be lowercase SHA-256")
        if (
            isinstance(self.program_bytes, bool)
            or not isinstance(self.program_bytes, int)
            or self.program_bytes < 0
        ):
            raise ValueError("compatibility program size must be nonnegative")
        if self.candidate_profile is not None and not isinstance(
            self.candidate_profile, ArtifactProfile
        ):
            raise ValueError("candidate profile must use ArtifactProfile")
        if self.report_identity != _digest(self.identity_dict()):
            raise ValueError("compatibility report identity does not match its content")

    def identity_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "findings": [item.value for item in self.findings],
            "program_content_hash": self.program_content_hash,
            "program_bytes": self.program_bytes,
            "candidate_profile": (
                None
                if self.candidate_profile is None
                else self.candidate_profile.to_dict()
            ),
            "target_capability_fingerprint": self.target_capability_fingerprint,
        }


def _metadata_metrics(
    value: object, limits: CompatibilityInspectionLimits
) -> tuple[int, int, int, bool, bool]:
    entries = 0
    maximum_depth = 0
    sensitive = False
    valid = True
    active: set[int] = set()
    encoded_lower_bound = 0
    exceeded = False
    stack: list[tuple[object, int, bool]] = [(value, 1, False)]

    while stack:
        item, depth, exiting = stack.pop()
        if exiting:
            active.remove(id(item))
            continue
        maximum_depth = max(maximum_depth, depth)
        if maximum_depth > limits.maximum_metadata_depth:
            exceeded = True
            break
        if isinstance(item, Mapping):
            identity = id(item)
            if identity in active:
                valid = False
                continue
            active.add(identity)
            stack.append((item, depth, True))
            for key, nested in item.items():
                entries += 1
                if not isinstance(key, str):
                    valid = False
                    continue
                encoded_lower_bound += len(key.encode("utf-8"))
                lowered = key.lower()
                sensitive = sensitive or any(
                    fragment in lowered for fragment in _SENSITIVE_FRAGMENTS
                )
                stack.append((nested, depth + 1, False))
        elif isinstance(item, Sequence) and not isinstance(
            item, (str, bytes, bytearray)
        ):
            identity = id(item)
            if identity in active:
                valid = False
                continue
            active.add(identity)
            stack.append((item, depth, True))
            for nested in item:
                entries += 1
                stack.append((nested, depth + 1, False))
        elif isinstance(item, str):
            encoded_lower_bound += len(item.encode("utf-8"))
            lowered = item.lower()
            sensitive = sensitive or any(
                fragment in lowered for fragment in _SENSITIVE_FRAGMENTS
            )
        elif item is None or isinstance(item, (bool, int)):
            pass
        elif isinstance(item, float) and math.isfinite(item):
            pass
        else:
            valid = False
        if (
            entries > limits.maximum_metadata_entries
            or encoded_lower_bound > limits.maximum_metadata_encoded_bytes
        ):
            exceeded = True
            break

    if exceeded:
        return (
            entries,
            maximum_depth,
            limits.maximum_metadata_encoded_bytes + 1,
            sensitive,
            valid,
        )
    try:
        encoded_bytes = len(
            json.dumps(
                value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
            ).encode("utf-8")
        )
    except (TypeError, ValueError, RecursionError):
        encoded_bytes = 0
        valid = False
    return entries, maximum_depth, encoded_bytes, sensitive, valid


def _program_and_profile(
    package: DeploymentPackage,
) -> tuple[bytes, ArtifactProfile | None]:
    metadata = package.metadata if isinstance(package.metadata, Mapping) else {}
    program_format = metadata.get("deployment_program_format")
    if program_format == "qcis":
        program = metadata.get("qcis")
        payload = program.encode("utf-8") if isinstance(program, str) else b""
        return payload, ArtifactProfile(ArtifactFormat.QCIS_1, "1.0")
    payload = package.qasm.encode("utf-8") if isinstance(package.qasm, str) else b""
    profiles = {
        "openqasm-2": ArtifactProfile(ArtifactFormat.OPENQASM_2, "2.0"),
        "openqasm-3": ArtifactProfile(ArtifactFormat.OPENQASM_3_STATIC, "3.0"),
    }
    return payload, profiles.get(program_format)


def _status(findings: set[CompatibilityFinding]) -> CompatibilityStatus:
    if findings & {
        CompatibilityFinding.INPUT_SIZE_LIMIT_EXCEEDED,
        CompatibilityFinding.METADATA_LIMIT_EXCEEDED,
    }:
        return CompatibilityStatus.LIMIT_EXCEEDED
    if findings & {
        CompatibilityFinding.PACKAGE_SCHEMA_INVALID,
        CompatibilityFinding.PACKAGE_IDENTITY_INVALID,
        CompatibilityFinding.ROUTING_EVIDENCE_INVALID,
        CompatibilityFinding.PROVIDER_METADATA_REJECTED,
    }:
        return CompatibilityStatus.INVALID
    if findings & {
        CompatibilityFinding.PROGRAM_PROFILE_UNSUPPORTED,
        CompatibilityFinding.DYNAMIC_PROGRAM_UNSUPPORTED,
        CompatibilityFinding.TARGET_PROFILE_MISMATCH,
        CompatibilityFinding.TARGET_CAPACITY_EXCEEDED,
        CompatibilityFinding.SHOTS_LIMIT_EXCEEDED,
    }:
        return CompatibilityStatus.UNSUPPORTED
    if CompatibilityFinding.TARGET_SNAPSHOT_INCOMPLETE in findings:
        return CompatibilityStatus.REQUIRES_TARGET_ENRICHMENT
    return CompatibilityStatus.ELIGIBLE_FOR_VERIFIED_RECOMPILE


def inspect_deployment_compatibility(
    package: DeploymentPackage,
    target: TargetCapabilities,
    limits: CompatibilityInspectionLimits,
) -> CompatibilityReport:
    """Inspect eligibility only; never convert, emit, execute, or submit."""

    if not isinstance(package, DeploymentPackage):
        raise ValueError("compatibility inspection requires DeploymentPackage")
    if not isinstance(target, TargetCapabilities):
        raise ValueError("compatibility inspection requires TargetCapabilities")
    if not isinstance(limits, CompatibilityInspectionLimits):
        raise ValueError("compatibility inspection requires explicit limits")

    payload, profile = _program_and_profile(package)
    findings: set[CompatibilityFinding] = {
        CompatibilityFinding.SOURCE_IR_REQUIRES_VERIFIED_IMPORT
    }
    package_metadata = package.metadata if isinstance(package.metadata, Mapping) else {}
    backend_metadata = getattr(package.backend, "metadata", {})
    if not isinstance(backend_metadata, Mapping):
        backend_metadata = {}
    combined_metadata = {"package": package_metadata, "backend": backend_metadata}
    entries, depth, encoded_bytes, sensitive, metadata_valid = _metadata_metrics(
        combined_metadata, limits
    )
    if len(payload) > limits.maximum_program_bytes:
        findings.add(CompatibilityFinding.INPUT_SIZE_LIMIT_EXCEEDED)
    if (
        entries > limits.maximum_metadata_entries
        or depth > limits.maximum_metadata_depth
        or encoded_bytes > limits.maximum_metadata_encoded_bytes
    ):
        findings.add(CompatibilityFinding.METADATA_LIMIT_EXCEEDED)
    if sensitive or not metadata_valid or not isinstance(package.metadata, Mapping):
        findings.add(CompatibilityFinding.PROVIDER_METADATA_REJECTED)
    if package_metadata.get("deployment_package_schema") != DEPLOYMENT_PACKAGE_SCHEMA:
        findings.add(CompatibilityFinding.PACKAGE_SCHEMA_INVALID)
    if profile is None:
        findings.add(CompatibilityFinding.PROGRAM_PROFILE_UNSUPPORTED)
    if package_metadata.get("dynamic_circuit") is True:
        findings.add(CompatibilityFinding.DYNAMIC_PROGRAM_UNSUPPORTED)

    try:
        validate_deployment_package(package)
    except Exception as error:  # user-owned legacy values must fail closed
        message = str(error).lower()
        finding = (
            CompatibilityFinding.ROUTING_EVIDENCE_INVALID
            if "routing" in message
            else CompatibilityFinding.PACKAGE_IDENTITY_INVALID
        )
        findings.add(finding)

    try:
        package_wires = package.n_wires
    except (AttributeError, TypeError, ValueError):
        package_wires = None
        findings.add(CompatibilityFinding.PACKAGE_IDENTITY_INVALID)
    if package_wires is not None and package_wires > min(
        target.logical_qubit_capacity, target.physical_qubit_capacity
    ):
        findings.add(CompatibilityFinding.TARGET_CAPACITY_EXCEEDED)
    if (
        isinstance(package.shots, bool)
        or not isinstance(package.shots, int)
        or package.shots <= 0
    ):
        findings.add(CompatibilityFinding.PACKAGE_IDENTITY_INVALID)
    elif target.maximum_shots is not None and package.shots > target.maximum_shots:
        findings.add(CompatibilityFinding.SHOTS_LIMIT_EXCEEDED)
    if profile is not None and profile not in target.artifact_profiles:
        findings.add(CompatibilityFinding.TARGET_PROFILE_MISMATCH)
    target_operations = {item.operation for item in target.native_gates}
    instructions = getattr(package.ir, "instructions", None)
    if not isinstance(instructions, Sequence) or any(
        getattr(item, "name", None) not in target_operations for item in instructions
    ):
        findings.add(CompatibilityFinding.TARGET_PROFILE_MISMATCH)
    legacy_topology = getattr(package.backend, "coupling_map", None)
    if legacy_topology is not None:
        if target.topology is None:
            findings.add(CompatibilityFinding.TARGET_SNAPSHOT_INCOMPLETE)
        elif tuple(legacy_topology.edges) != tuple(target.topology.edges):
            findings.add(CompatibilityFinding.TARGET_PROFILE_MISMATCH)
    if (
        getattr(package.backend, "is_simulator", False) is False
        and target.calibration_snapshot_hash is None
    ):
        findings.add(CompatibilityFinding.TARGET_SNAPSHOT_INCOMPLETE)

    status = _status(findings)
    if status is CompatibilityStatus.ELIGIBLE_FOR_VERIFIED_RECOMPILE:
        findings.add(CompatibilityFinding.ELIGIBLE_FOR_VERIFIED_RECOMPILE)
    ordered_findings = tuple(sorted(findings, key=lambda item: item.value))
    if len(ordered_findings) > limits.maximum_findings:
        status = CompatibilityStatus.LIMIT_EXCEEDED
        findings.discard(CompatibilityFinding.ELIGIBLE_FOR_VERIFIED_RECOMPILE)
        ordered_findings = tuple(sorted(findings, key=lambda item: item.value))
        ordered_findings = ordered_findings[: limits.maximum_findings]

    legacy_digest = package_metadata.get("deployment_artifact_sha256")
    if not isinstance(legacy_digest, str) or _SHA256.fullmatch(legacy_digest) is None:
        legacy_digest = None
    content_hash = hashlib.sha256(payload).hexdigest()
    values = {
        "status": status.value,
        "findings": [item.value for item in ordered_findings],
        "program_content_hash": content_hash,
        "program_bytes": len(payload),
        "candidate_profile": None if profile is None else profile.to_dict(),
        "target_capability_fingerprint": target.semantic_fingerprint,
    }
    return CompatibilityReport(
        status,
        ordered_findings,
        legacy_digest,
        content_hash,
        len(payload),
        profile,
        target.semantic_fingerprint,
        _digest(values),
    )


__all__ = [
    "CompatibilityFinding",
    "CompatibilityInspectionLimits",
    "CompatibilityReport",
    "CompatibilityStatus",
    "inspect_deployment_compatibility",
]
