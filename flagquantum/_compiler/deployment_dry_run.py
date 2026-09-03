"""Private explicit offline dry-run for the legacy deployment bridge."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TypeVar

from flagquantum.deployment.cloud import DeploymentPackage

from .artifact_profiles import validate_artifact_payload
from .deployment_compatibility import (
    CompatibilityInspectionLimits,
    CompatibilityReport,
    CompatibilityStatus,
    inspect_deployment_compatibility,
)
from .executable_artifact import (
    ArtifactSealStatus,
    seal_executable_artifact,
    verify_executable_artifact,
)
from .importers.circuit_ir import import_circuit_ir
from .offline_deployment import (
    OfflineCompilationStatus,
    OfflineStaticTarget,
    OfflineTextFormat,
    compile_offline_static,
)
from .target_capabilities import ArtifactFormat, ArtifactProfile, TargetCapabilities
from .target_ir import TargetIR
from .target_legalization import TargetLegalizationStatus, legalize_quantum_module

_SHA256 = re.compile(r"[0-9a-f]{64}")
_NO_CALIBRATION_IDENTITY = hashlib.sha256(
    b"flagquantum.deployment_bridge.no_calibration.v1"
).hexdigest()
_T = TypeVar("_T")


def _certify_canonical_payload(
    profile: ArtifactProfile,
    target_ir: TargetIR,
    text: str,
) -> bytes:
    payload = text.encode("utf-8")
    validate_artifact_payload(profile, payload, target_ir)
    return payload


class DeploymentDryRunStatus(str, Enum):
    READY_FOR_OPERATOR_REVIEW = "ready_for_operator_review"
    COMPATIBILITY_REJECTED = "compatibility_rejected"
    IMPORT_REJECTED = "import_rejected"
    COMPILATION_REJECTED = "compilation_rejected"
    LEGALIZATION_REJECTED = "legalization_rejected"
    EMISSION_REJECTED = "emission_rejected"
    SEALING_REJECTED = "sealing_rejected"
    DIFFERENTIAL_MISMATCH = "differential_mismatch"
    LIMIT_EXCEEDED = "limit_exceeded"
    INTERNAL_FAILURE = "internal_failure"


class DeploymentDryRunFinding(str, Enum):
    COMPATIBILITY_NOT_ELIGIBLE = "compatibility_not_eligible"
    LEGACY_PACKAGE_INVALID = "legacy_package_invalid"
    SOURCE_IMPORT_FAILED = "source_import_failed"
    SOURCE_IDENTITY_DISCONTINUITY = "source_identity_discontinuity"
    OFFLINE_COMPILATION_FAILED = "offline_compilation_failed"
    TARGET_LEGALIZATION_FAILED = "target_legalization_failed"
    CANONICAL_EMISSION_FAILED = "canonical_emission_failed"
    ARTIFACT_SEALING_FAILED = "artifact_sealing_failed"
    ARTIFACT_VERIFICATION_FAILED = "artifact_verification_failed"
    PROFILE_DISCONTINUITY = "profile_discontinuity"
    TARGET_IDENTITY_DISCONTINUITY = "target_identity_discontinuity"
    SHOTS_ENTERED_ARTIFACT_IDENTITY = "shots_entered_artifact_identity"
    PROVIDER_LOCATOR_ENTERED_PORTABLE_IDENTITY = (
        "provider_locator_entered_portable_identity"
    )
    OPERATION_LIMIT_EXCEEDED = "operation_limit_exceeded"
    ARTIFACT_SIZE_LIMIT_EXCEEDED = "artifact_size_limit_exceeded"
    EVIDENCE_LIMIT_EXCEEDED = "evidence_limit_exceeded"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    READY_FOR_OPERATOR_REVIEW = "ready_for_operator_review"


class DeploymentDryRunStage(str, Enum):
    COMPATIBILITY = "compatibility"
    VERIFIED_IMPORT = "verified_import"
    PRIVATE_COMPILATION = "private_compilation"
    TARGET_LEGALIZATION = "target_legalization"
    CANONICAL_ENCODING = "canonical_encoding"
    ARTIFACT_SEALING = "artifact_sealing"
    ARTIFACT_VERIFICATION = "artifact_verification"


@dataclass(frozen=True)
class DeploymentDryRunStageTiming:
    stage: DeploymentDryRunStage
    duration_ns: int

    def __post_init__(self) -> None:
        if not isinstance(self.stage, DeploymentDryRunStage):
            raise ValueError("dry-run timing stage must use its closed enum")
        if (
            isinstance(self.duration_ns, bool)
            or not isinstance(self.duration_ns, int)
            or self.duration_ns < 0
        ):
            raise ValueError("dry-run timing duration must be nonnegative")


@dataclass(frozen=True)
class DeploymentDryRunPolicy:
    compatibility_limits: CompatibilityInspectionLimits
    maximum_source_operations: int
    maximum_compiled_operations: int
    maximum_artifact_bytes: int
    maximum_evidence_bytes: int
    maximum_stage_duration_ns: int
    maximum_total_duration_ns: int

    def __post_init__(self) -> None:
        if not isinstance(self.compatibility_limits, CompatibilityInspectionLimits):
            raise ValueError("dry-run policy requires compatibility inspection limits")
        for name in (
            "maximum_source_operations",
            "maximum_compiled_operations",
            "maximum_artifact_bytes",
            "maximum_evidence_bytes",
            "maximum_stage_duration_ns",
            "maximum_total_duration_ns",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"dry-run {name} must be a positive integer")


def _digest(values: dict[str, object]) -> str:
    encoded = json.dumps(
        values, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _optional_digest(value: str | None, label: str) -> None:
    if value is not None and _SHA256.fullmatch(value) is None:
        raise ValueError(f"dry-run {label} must be lowercase SHA-256")


@dataclass(frozen=True)
class DeploymentDryRunReport:
    status: DeploymentDryRunStatus
    findings: tuple[DeploymentDryRunFinding, ...]
    compatibility_report_identity: str | None
    legacy_artifact_digest: str | None
    source_program_identity: str | None
    compilation_identity: str | None
    target_program_identity: str | None
    artifact_identity: str | None
    payload_content_hash: str | None
    artifact_profile: ArtifactProfile | None
    artifact_bytes: int
    requested_shots: int | None
    target_capability_fingerprint: str
    stage_timings_ns: tuple[DeploymentDryRunStageTiming, ...]
    evidence_identity: str

    def __post_init__(self) -> None:
        if not isinstance(self.status, DeploymentDryRunStatus):
            raise ValueError("dry-run status must use its closed enum")
        findings = tuple(sorted(self.findings, key=lambda item: item.value))
        if any(not isinstance(item, DeploymentDryRunFinding) for item in findings):
            raise ValueError("dry-run findings must use the closed enum")
        if not findings or len(findings) != len(set(findings)):
            raise ValueError("dry-run findings must be non-empty and unique")
        object.__setattr__(self, "findings", findings)
        for name in (
            "compatibility_report_identity",
            "legacy_artifact_digest",
            "source_program_identity",
            "compilation_identity",
            "target_program_identity",
            "artifact_identity",
            "payload_content_hash",
        ):
            _optional_digest(getattr(self, name), name)
        _optional_digest(
            self.target_capability_fingerprint, "target capability fingerprint"
        )
        _optional_digest(self.evidence_identity, "evidence identity")
        if self.artifact_profile is not None and not isinstance(
            self.artifact_profile, ArtifactProfile
        ):
            raise ValueError("dry-run artifact profile must use ArtifactProfile")
        if (
            isinstance(self.artifact_bytes, bool)
            or not isinstance(self.artifact_bytes, int)
            or self.artifact_bytes < 0
        ):
            raise ValueError("dry-run artifact size must be nonnegative")
        if self.requested_shots is not None and (
            isinstance(self.requested_shots, bool)
            or not isinstance(self.requested_shots, int)
            or self.requested_shots <= 0
        ):
            raise ValueError("dry-run requested shots must be positive")
        timings = tuple(self.stage_timings_ns)
        if any(not isinstance(item, DeploymentDryRunStageTiming) for item in timings):
            raise ValueError("dry-run timings must use DeploymentDryRunStageTiming")
        stages = tuple(item.stage for item in timings)
        if len(stages) != len(set(stages)):
            raise ValueError("dry-run timing stages must be unique")
        object.__setattr__(self, "stage_timings_ns", timings)
        if self.evidence_identity != _digest(self.identity_dict()):
            raise ValueError("dry-run evidence identity does not match its content")

    def identity_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "findings": [item.value for item in self.findings],
            "compatibility_report_identity": self.compatibility_report_identity,
            "source_program_identity": self.source_program_identity,
            "compilation_identity": self.compilation_identity,
            "target_program_identity": self.target_program_identity,
            "artifact_identity": self.artifact_identity,
            "payload_content_hash": self.payload_content_hash,
            "artifact_profile": (
                None
                if self.artifact_profile is None
                else self.artifact_profile.to_dict()
            ),
            "artifact_bytes": self.artifact_bytes,
            "requested_shots": self.requested_shots,
            "target_capability_fingerprint": self.target_capability_fingerprint,
        }


class _DeadlineExceededError(Exception):
    pass


class _DryRun:
    def __init__(
        self,
        package: DeploymentPackage,
        target: TargetCapabilities,
        policy: DeploymentDryRunPolicy,
    ) -> None:
        self.package = package
        self.target = target
        self.policy = policy
        self.started_ns = time.perf_counter_ns()
        self.timings: list[DeploymentDryRunStageTiming] = []
        self.compatibility: CompatibilityReport | None = None
        self.source_program_identity: str | None = None
        self.compilation_identity: str | None = None
        self.target_program_identity: str | None = None
        self.artifact_identity: str | None = None
        self.payload_content_hash: str | None = None
        self.artifact_profile: ArtifactProfile | None = None
        self.artifact_bytes = 0

    @property
    def requested_shots(self) -> int | None:
        shots = self.package.shots
        if isinstance(shots, bool) or not isinstance(shots, int) or shots <= 0:
            return None
        return shots

    def timed(self, stage: DeploymentDryRunStage, operation: Callable[[], _T]) -> _T:
        started = time.perf_counter_ns()
        try:
            result = operation()
        finally:
            duration = time.perf_counter_ns() - started
            self.timings.append(DeploymentDryRunStageTiming(stage, duration))
        if (
            duration > self.policy.maximum_stage_duration_ns
            or time.perf_counter_ns() - self.started_ns
            > self.policy.maximum_total_duration_ns
        ):
            raise _DeadlineExceededError
        return result

    def finish(
        self,
        status: DeploymentDryRunStatus,
        *findings: DeploymentDryRunFinding,
    ) -> DeploymentDryRunReport:
        finding_set = set(findings)
        values = self._identity_values(status, finding_set)
        if (
            len(
                json.dumps(
                    values, ensure_ascii=True, separators=(",", ":"), sort_keys=True
                ).encode("utf-8")
            )
            > self.policy.maximum_evidence_bytes
        ):
            status = DeploymentDryRunStatus.LIMIT_EXCEEDED
            finding_set.add(DeploymentDryRunFinding.EVIDENCE_LIMIT_EXCEEDED)
            values = self._identity_values(status, finding_set)
        return DeploymentDryRunReport(
            status,
            tuple(finding_set),
            (
                None
                if self.compatibility is None
                else self.compatibility.report_identity
            ),
            (
                None
                if self.compatibility is None
                else self.compatibility.legacy_artifact_digest
            ),
            self.source_program_identity,
            self.compilation_identity,
            self.target_program_identity,
            self.artifact_identity,
            self.payload_content_hash,
            self.artifact_profile,
            self.artifact_bytes,
            self.requested_shots,
            self.target.semantic_fingerprint,
            tuple(self.timings),
            _digest(values),
        )

    def _identity_values(
        self,
        status: DeploymentDryRunStatus,
        findings: set[DeploymentDryRunFinding],
    ) -> dict[str, object]:
        return {
            "status": status.value,
            "findings": sorted(item.value for item in findings),
            "compatibility_report_identity": (
                None
                if self.compatibility is None
                else self.compatibility.report_identity
            ),
            "source_program_identity": self.source_program_identity,
            "compilation_identity": self.compilation_identity,
            "target_program_identity": self.target_program_identity,
            "artifact_identity": self.artifact_identity,
            "payload_content_hash": self.payload_content_hash,
            "artifact_profile": (
                None
                if self.artifact_profile is None
                else self.artifact_profile.to_dict()
            ),
            "artifact_bytes": self.artifact_bytes,
            "requested_shots": self.requested_shots,
            "target_capability_fingerprint": self.target.semantic_fingerprint,
        }

    def run(self) -> DeploymentDryRunReport:
        try:
            self.compatibility = self.timed(
                DeploymentDryRunStage.COMPATIBILITY,
                lambda: inspect_deployment_compatibility(
                    self.package,
                    self.target,
                    self.policy.compatibility_limits,
                ),
            )
        except _DeadlineExceededError:
            return self.finish(
                DeploymentDryRunStatus.LIMIT_EXCEEDED,
                DeploymentDryRunFinding.DEADLINE_EXCEEDED,
            )
        except Exception:
            return self.finish(
                DeploymentDryRunStatus.INTERNAL_FAILURE,
                DeploymentDryRunFinding.LEGACY_PACKAGE_INVALID,
            )
        if (
            self.compatibility.status
            is not CompatibilityStatus.ELIGIBLE_FOR_VERIFIED_RECOMPILE
        ):
            status = (
                DeploymentDryRunStatus.LIMIT_EXCEEDED
                if self.compatibility.status is CompatibilityStatus.LIMIT_EXCEEDED
                else DeploymentDryRunStatus.COMPATIBILITY_REJECTED
            )
            return self.finish(
                status, DeploymentDryRunFinding.COMPATIBILITY_NOT_ELIGIBLE
            )
        self.artifact_profile = self.compatibility.candidate_profile
        if self.artifact_profile is None:
            return self.finish(
                DeploymentDryRunStatus.COMPATIBILITY_REJECTED,
                DeploymentDryRunFinding.PROFILE_DISCONTINUITY,
            )

        instructions = getattr(self.package.ir, "instructions", None)
        if not isinstance(instructions, Sequence):
            return self.finish(
                DeploymentDryRunStatus.IMPORT_REJECTED,
                DeploymentDryRunFinding.SOURCE_IMPORT_FAILED,
            )
        if len(instructions) > self.policy.maximum_source_operations:
            return self.finish(
                DeploymentDryRunStatus.LIMIT_EXCEEDED,
                DeploymentDryRunFinding.OPERATION_LIMIT_EXCEEDED,
            )

        try:
            imported = self.timed(
                DeploymentDryRunStage.VERIFIED_IMPORT,
                lambda: import_circuit_ir(self.package.ir),
            )
        except _DeadlineExceededError:
            return self.finish(
                DeploymentDryRunStatus.LIMIT_EXCEEDED,
                DeploymentDryRunFinding.DEADLINE_EXCEEDED,
            )
        except Exception:
            return self.finish(
                DeploymentDryRunStatus.INTERNAL_FAILURE,
                DeploymentDryRunFinding.SOURCE_IMPORT_FAILED,
            )
        if not imported.ok or imported.imported is None:
            return self.finish(
                DeploymentDryRunStatus.IMPORT_REJECTED,
                DeploymentDryRunFinding.SOURCE_IMPORT_FAILED,
            )
        self.source_program_identity = imported.imported.internal_program_identity

        if self.target.topology is None:
            return self.finish(
                DeploymentDryRunStatus.COMPILATION_REJECTED,
                DeploymentDryRunFinding.OFFLINE_COMPILATION_FAILED,
            )
        calibration = self.target.calibration_snapshot_hash or _NO_CALIBRATION_IDENTITY
        try:
            offline_target = OfflineStaticTarget(self.target.topology, calibration)
        except (TypeError, ValueError):
            return self.finish(
                DeploymentDryRunStatus.COMPILATION_REJECTED,
                DeploymentDryRunFinding.OFFLINE_COMPILATION_FAILED,
            )
        formats = {
            ArtifactFormat.OPENQASM_2: OfflineTextFormat.OPENQASM2,
            ArtifactFormat.OPENQASM_3_STATIC: OfflineTextFormat.OPENQASM3,
            ArtifactFormat.QCIS_1: OfflineTextFormat.QCIS_V1,
        }
        output_format = formats.get(self.artifact_profile.format)
        if output_format is None:
            return self.finish(
                DeploymentDryRunStatus.EMISSION_REJECTED,
                DeploymentDryRunFinding.PROFILE_DISCONTINUITY,
            )
        try:
            compiled = self.timed(
                DeploymentDryRunStage.PRIVATE_COMPILATION,
                lambda: compile_offline_static(
                    self.package.ir,
                    offline_target,
                    output_formats=(output_format,),
                ),
            )
        except _DeadlineExceededError:
            return self.finish(
                DeploymentDryRunStatus.LIMIT_EXCEEDED,
                DeploymentDryRunFinding.DEADLINE_EXCEEDED,
            )
        except Exception:
            return self.finish(
                DeploymentDryRunStatus.INTERNAL_FAILURE,
                DeploymentDryRunFinding.OFFLINE_COMPILATION_FAILED,
            )
        if (
            compiled.status is not OfflineCompilationStatus.COMPILED_EXACT
            or compiled.source_artifact is None
            or compiled.module is None
            or compiled.execution is None
            or compiled.execution.identity is None
        ):
            return self.finish(
                DeploymentDryRunStatus.COMPILATION_REJECTED,
                DeploymentDryRunFinding.OFFLINE_COMPILATION_FAILED,
            )
        if (
            compiled.source_artifact.imported.internal_program_identity
            != self.source_program_identity
        ):
            return self.finish(
                DeploymentDryRunStatus.DIFFERENTIAL_MISMATCH,
                DeploymentDryRunFinding.SOURCE_IDENTITY_DISCONTINUITY,
            )
        compiled_operations = len(compiled.module.body.blocks[0].operations)
        if compiled_operations > self.policy.maximum_compiled_operations:
            return self.finish(
                DeploymentDryRunStatus.LIMIT_EXCEEDED,
                DeploymentDryRunFinding.OPERATION_LIMIT_EXCEEDED,
            )
        self.compilation_identity = compiled.execution.identity.digest

        try:
            legalized = self.timed(
                DeploymentDryRunStage.TARGET_LEGALIZATION,
                lambda: legalize_quantum_module(compiled.module, self.target),
            )
        except _DeadlineExceededError:
            return self.finish(
                DeploymentDryRunStatus.LIMIT_EXCEEDED,
                DeploymentDryRunFinding.DEADLINE_EXCEEDED,
            )
        except Exception:
            return self.finish(
                DeploymentDryRunStatus.INTERNAL_FAILURE,
                DeploymentDryRunFinding.TARGET_LEGALIZATION_FAILED,
            )
        if (
            legalized.status is not TargetLegalizationStatus.LEGALIZED
            or legalized.target_ir is None
        ):
            return self.finish(
                DeploymentDryRunStatus.LEGALIZATION_REJECTED,
                DeploymentDryRunFinding.TARGET_LEGALIZATION_FAILED,
            )
        target_ir = legalized.target_ir
        self.target_program_identity = target_ir.target_program_identity
        if target_ir.target_capability_fingerprint != self.target.semantic_fingerprint:
            return self.finish(
                DeploymentDryRunStatus.DIFFERENTIAL_MISMATCH,
                DeploymentDryRunFinding.TARGET_IDENTITY_DISCONTINUITY,
            )
        if target_ir.requested_shots is not None:
            return self.finish(
                DeploymentDryRunStatus.DIFFERENTIAL_MISMATCH,
                DeploymentDryRunFinding.SHOTS_ENTERED_ARTIFACT_IDENTITY,
            )

        emission = compiled.emission(output_format)
        if emission.text is None:
            return self.finish(
                DeploymentDryRunStatus.EMISSION_REJECTED,
                DeploymentDryRunFinding.CANONICAL_EMISSION_FAILED,
            )
        try:
            payload = self.timed(
                DeploymentDryRunStage.CANONICAL_ENCODING,
                lambda: _certify_canonical_payload(
                    self.artifact_profile, target_ir, emission.text
                ),
            )
        except _DeadlineExceededError:
            return self.finish(
                DeploymentDryRunStatus.LIMIT_EXCEEDED,
                DeploymentDryRunFinding.DEADLINE_EXCEEDED,
            )
        except Exception:
            return self.finish(
                DeploymentDryRunStatus.EMISSION_REJECTED,
                DeploymentDryRunFinding.CANONICAL_EMISSION_FAILED,
            )
        self.artifact_bytes = len(payload)
        self.payload_content_hash = hashlib.sha256(payload).hexdigest()
        if self.artifact_bytes > self.policy.maximum_artifact_bytes:
            return self.finish(
                DeploymentDryRunStatus.LIMIT_EXCEEDED,
                DeploymentDryRunFinding.ARTIFACT_SIZE_LIMIT_EXCEEDED,
            )
        try:
            sealed = self.timed(
                DeploymentDryRunStage.ARTIFACT_SEALING,
                lambda: seal_executable_artifact(
                    target_ir,
                    self.target,
                    self.artifact_profile,
                    payload,
                    compilation_identity=self.compilation_identity,
                ),
            )
        except _DeadlineExceededError:
            return self.finish(
                DeploymentDryRunStatus.LIMIT_EXCEEDED,
                DeploymentDryRunFinding.DEADLINE_EXCEEDED,
            )
        except Exception:
            return self.finish(
                DeploymentDryRunStatus.INTERNAL_FAILURE,
                DeploymentDryRunFinding.ARTIFACT_SEALING_FAILED,
            )
        if sealed.status is not ArtifactSealStatus.SEALED or sealed.artifact is None:
            return self.finish(
                DeploymentDryRunStatus.SEALING_REJECTED,
                DeploymentDryRunFinding.ARTIFACT_SEALING_FAILED,
            )
        artifact = sealed.artifact
        self.artifact_identity = artifact.artifact_identity
        try:
            verified = self.timed(
                DeploymentDryRunStage.ARTIFACT_VERIFICATION,
                lambda: verify_executable_artifact(artifact, target_ir, self.target),
            )
        except _DeadlineExceededError:
            return self.finish(
                DeploymentDryRunStatus.LIMIT_EXCEEDED,
                DeploymentDryRunFinding.DEADLINE_EXCEEDED,
            )
        except Exception:
            return self.finish(
                DeploymentDryRunStatus.INTERNAL_FAILURE,
                DeploymentDryRunFinding.ARTIFACT_VERIFICATION_FAILED,
            )
        if verified.status is not ArtifactSealStatus.SEALED:
            return self.finish(
                DeploymentDryRunStatus.SEALING_REJECTED,
                DeploymentDryRunFinding.ARTIFACT_VERIFICATION_FAILED,
            )
        return self.finish(
            DeploymentDryRunStatus.READY_FOR_OPERATOR_REVIEW,
            DeploymentDryRunFinding.READY_FOR_OPERATOR_REVIEW,
        )


def dry_run_deployment_bridge(
    package: DeploymentPackage,
    target: TargetCapabilities,
    policy: DeploymentDryRunPolicy,
) -> DeploymentDryRunReport:
    """Build evidence only; never return, execute, persist, or submit the artifact."""

    if not isinstance(package, DeploymentPackage):
        raise ValueError("deployment dry-run requires DeploymentPackage")
    if not isinstance(target, TargetCapabilities):
        raise ValueError("deployment dry-run requires TargetCapabilities")
    if not isinstance(policy, DeploymentDryRunPolicy):
        raise ValueError("deployment dry-run requires explicit DeploymentDryRunPolicy")
    return _DryRun(package, target, policy).run()


__all__ = [
    "DeploymentDryRunFinding",
    "DeploymentDryRunPolicy",
    "DeploymentDryRunReport",
    "DeploymentDryRunStage",
    "DeploymentDryRunStageTiming",
    "DeploymentDryRunStatus",
    "dry_run_deployment_bridge",
]
