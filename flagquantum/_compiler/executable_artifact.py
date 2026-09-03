"""Sealed portable executable artifacts for the private compiler."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import Enum

from .artifact_profiles import artifact_profile_definition, validate_artifact_payload
from .diagnostics import Diagnostic, DiagnosticCode
from .target_capabilities import ArtifactProfile, TargetCapabilities
from .target_ir import TargetIR

_SHA256 = re.compile(r"[0-9a-f]{64}")
EXECUTABLE_ARTIFACT_SCHEMA = "flagquantum.executable_artifact.v1alpha1"


class ArtifactSealStatus(str, Enum):
    SEALED = "sealed"
    INVALID_INPUT = "invalid_input"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class SealedExecutableArtifact:
    profile: ArtifactProfile
    media_type: str
    payload: bytes
    payload_content_hash: str
    source_program_identity: str
    target_program_identity: str
    target_capability_fingerprint: str
    compilation_identity: str
    artifact_identity: str
    schema_version: str = EXECUTABLE_ARTIFACT_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.profile, ArtifactProfile):
            raise ValueError("sealed artifact profile must use ArtifactProfile")
        artifact_profile_definition(self.profile)
        if not isinstance(self.payload, bytes):
            raise ValueError("sealed artifact payload must be immutable bytes")
        if not isinstance(self.media_type, str) or not self.media_type:
            raise ValueError("sealed artifact media type cannot be empty")
        for name in (
            "payload_content_hash",
            "source_program_identity",
            "target_program_identity",
            "target_capability_fingerprint",
            "compilation_identity",
            "artifact_identity",
        ):
            if _SHA256.fullmatch(getattr(self, name)) is None:
                raise ValueError(f"sealed artifact {name} must be lowercase SHA-256")
        if self.schema_version != EXECUTABLE_ARTIFACT_SCHEMA:
            raise ValueError("unsupported executable artifact schema version")

    def identity_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "profile": self.profile.to_dict(),
            "media_type": self.media_type,
            "payload_content_hash": self.payload_content_hash,
            "source_program_identity": self.source_program_identity,
            "target_program_identity": self.target_program_identity,
            "target_capability_fingerprint": self.target_capability_fingerprint,
            "compilation_identity": self.compilation_identity,
        }


@dataclass(frozen=True)
class ArtifactSealResult:
    status: ArtifactSealStatus
    artifact: SealedExecutableArtifact | None = None
    diagnostics: tuple[Diagnostic, ...] = ()

    @property
    def ok(self) -> bool:
        return self.status is ArtifactSealStatus.SEALED


def _failure(status: ArtifactSealStatus, message: str) -> ArtifactSealResult:
    return ArtifactSealResult(
        status,
        diagnostics=(
            Diagnostic(
                DiagnosticCode.ATTRIBUTE_TYPE_MISMATCH,
                message,
                notes=("Batch C never repairs, re-emits, or submits an artifact",),
            ),
        ),
    )


def _artifact_identity(values: dict[str, object]) -> str:
    encoded = json.dumps(
        values, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def seal_executable_artifact(
    target_ir: TargetIR,
    target: TargetCapabilities,
    profile: ArtifactProfile,
    payload: bytes,
    *,
    compilation_identity: str,
) -> ArtifactSealResult:
    """Seal already-emitted bytes; this function does not emit or execute them."""

    if not isinstance(target_ir, TargetIR) or not isinstance(
        target, TargetCapabilities
    ):
        return _failure(
            ArtifactSealStatus.INVALID_INPUT,
            "sealing requires TargetIR and TargetCapabilities",
        )
    if not isinstance(profile, ArtifactProfile) or not isinstance(payload, bytes):
        return _failure(
            ArtifactSealStatus.INVALID_INPUT,
            "sealing requires ArtifactProfile and immutable payload bytes",
        )
    if (
        not isinstance(compilation_identity, str)
        or _SHA256.fullmatch(compilation_identity) is None
    ):
        return _failure(
            ArtifactSealStatus.INVALID_INPUT,
            "compilation identity must be a lowercase SHA-256 digest",
        )
    try:
        definition = artifact_profile_definition(profile)
        if target_ir.target_capability_fingerprint != target.semantic_fingerprint:
            raise ValueError("TargetIR and capability snapshot identity do not match")
        if profile not in target.artifact_profiles:
            raise ValueError("artifact profile is not accepted by the target")
        validate_artifact_payload(profile, payload, target_ir)
    except ValueError as exc:
        return _failure(ArtifactSealStatus.UNSUPPORTED, str(exc))
    content_hash = hashlib.sha256(payload).hexdigest()
    values = {
        "schema_version": EXECUTABLE_ARTIFACT_SCHEMA,
        "profile": profile.to_dict(),
        "media_type": definition.media_type,
        "payload_content_hash": content_hash,
        "source_program_identity": target_ir.source_program_identity,
        "target_program_identity": target_ir.target_program_identity,
        "target_capability_fingerprint": target_ir.target_capability_fingerprint,
        "compilation_identity": compilation_identity,
    }
    artifact = SealedExecutableArtifact(
        profile,
        definition.media_type,
        payload,
        content_hash,
        target_ir.source_program_identity,
        target_ir.target_program_identity,
        target_ir.target_capability_fingerprint,
        compilation_identity,
        _artifact_identity(values),
    )
    return ArtifactSealResult(ArtifactSealStatus.SEALED, artifact)


def verify_executable_artifact(
    artifact: SealedExecutableArtifact,
    target_ir: TargetIR,
    target: TargetCapabilities,
) -> ArtifactSealResult:
    """Recheck the complete identity chain and exact canonical payload."""

    if (
        not isinstance(artifact, SealedExecutableArtifact)
        or not isinstance(target_ir, TargetIR)
        or not isinstance(target, TargetCapabilities)
    ):
        return _failure(
            ArtifactSealStatus.INVALID_INPUT,
            "verification requires artifact, TargetIR, and TargetCapabilities",
        )
    try:
        definition = artifact_profile_definition(artifact.profile)
        if target_ir.target_capability_fingerprint != target.semantic_fingerprint:
            raise ValueError("TargetIR and capability snapshot identity do not match")
        if artifact.profile not in target.artifact_profiles:
            raise ValueError("artifact profile is not accepted by the target")
        validate_artifact_payload(artifact.profile, artifact.payload, target_ir)
    except ValueError as exc:
        return _failure(ArtifactSealStatus.UNSUPPORTED, str(exc))
    if artifact.media_type != definition.media_type:
        return _failure(ArtifactSealStatus.INVALID_INPUT, "artifact media type changed")
    if artifact.payload_content_hash != hashlib.sha256(artifact.payload).hexdigest():
        return _failure(
            ArtifactSealStatus.INVALID_INPUT, "artifact payload hash changed"
        )
    if artifact.source_program_identity != target_ir.source_program_identity:
        return _failure(ArtifactSealStatus.INVALID_INPUT, "source identity changed")
    if artifact.target_program_identity != target_ir.target_program_identity:
        return _failure(ArtifactSealStatus.INVALID_INPUT, "TargetIR identity changed")
    if (
        artifact.target_capability_fingerprint
        != target_ir.target_capability_fingerprint
    ):
        return _failure(ArtifactSealStatus.INVALID_INPUT, "target capability changed")
    if artifact.artifact_identity != _artifact_identity(artifact.identity_dict()):
        return _failure(ArtifactSealStatus.INVALID_INPUT, "artifact identity changed")
    return ArtifactSealResult(ArtifactSealStatus.SEALED, artifact)


__all__ = [
    "EXECUTABLE_ARTIFACT_SCHEMA",
    "ArtifactSealResult",
    "ArtifactSealStatus",
    "SealedExecutableArtifact",
    "seal_executable_artifact",
    "verify_executable_artifact",
]
