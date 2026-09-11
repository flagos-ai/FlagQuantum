"""Prepare verified executable artifacts for remote adaptation without submission."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping

from ..core._artifacts import (
    CircuitArtifactBindingResult,
    ProgramArtifactV2,
    ProgramArtifactV3,
)
from ..core._compilation_evidence import CompilationEvidenceBundle
from ..core._compilation_evidence_v2 import CompilationEvidenceBundleV2
from ..core._compilation_evidence_v3 import CompilationEvidenceBundleV3
from ..core.target_capabilities import CapabilityMatchResult, TargetCapabilitySnapshot
from ..errors import ExecutionError
from ..runtime.artifact_preflight import preflight_executable_artifact
from ..runtime.compilation_evidence import verify_compilation_evidence_handoff


@dataclass(frozen=True)
class ArtifactDeploymentDryRun:
    """Immutable handoff prepared without credentials or provider side effects."""

    artifact: ProgramArtifactV2 | ProgramArtifactV3 = field(repr=False)
    provider: str
    target_id: str
    shots: int
    capability_match: CapabilityMatchResult = field(repr=False)
    compilation_evidence: (
        CompilationEvidenceBundle
        | CompilationEvidenceBundleV2
        | CompilationEvidenceBundleV3
        | None
    ) = field(
        default=None,
        repr=False,
    )

    @property
    def profile(self) -> str:
        return str(self.artifact.profile["name"])

    @property
    def program(self) -> str:
        payload = self.artifact.payload
        if not isinstance(payload, str):
            raise ExecutionError("executable artifact payload must be target text")
        return payload

    @property
    def artifact_identity(self) -> str:
        return self.artifact.artifact_identity

    @property
    def snapshot_id(self) -> str:
        target = self.artifact.target
        if target is None:
            raise ExecutionError("executable artifact target binding is missing")
        return str(target["snapshot_id"])

    @property
    def compilation_evidence_identity(self) -> str | None:
        evidence = self.compilation_evidence
        return None if evidence is None else evidence.bundle_identity

    @property
    def result_schema(self) -> Mapping[str, Any]:
        """Return the immutable, already-verified executable result contract."""

        schema = self.artifact.result_schema
        if not isinstance(schema, Mapping):
            raise ExecutionError("executable artifact result schema is missing")
        return schema

    @property
    def logical_result_width(self) -> int:
        """Return the result width exposed to the caller."""

        key = (
            "logical_wires" if isinstance(self.artifact, ProgramArtifactV3) else "wires"
        )
        return len(tuple(self.result_schema[key]))

    @property
    def physical_result_slots(self) -> tuple[int, ...] | None:
        """Return compilation-local slots for v3, never provider qubit IDs."""

        if not isinstance(self.artifact, ProgramArtifactV3):
            return None
        return tuple(self.result_schema["physical_result_slots"])


def prepare_artifact_deployment(
    artifact: ProgramArtifactV2 | ProgramArtifactV3,
    *,
    snapshot: TargetCapabilitySnapshot,
    source: ProgramArtifactV2 | CircuitArtifactBindingResult | None = None,
    compilation_evidence: (
        CompilationEvidenceBundle
        | CompilationEvidenceBundleV2
        | CompilationEvidenceBundleV3
        | None
    ) = None,
    provider: str,
    target_id: str,
    shots: int,
    evaluated_at: datetime | None = None,
) -> ArtifactDeploymentDryRun:
    """Build a side-effect-free handoff after target and capability checks."""

    if not isinstance(artifact, (ProgramArtifactV2, ProgramArtifactV3)):
        raise TypeError("artifact must be a ProgramArtifactV2 or ProgramArtifactV3")
    if not isinstance(provider, str) or not provider:
        raise ValueError("provider must be a non-empty string")
    if not isinstance(target_id, str) or not target_id:
        raise ValueError("target_id must be a non-empty string")
    identity = snapshot.target_identity
    if provider != identity.provider:
        raise ExecutionError(
            "deployment provider does not match the capability snapshot provider"
        )
    if target_id != identity.target_id:
        raise ExecutionError(
            "deployment target does not match the capability snapshot target"
        )

    match = preflight_executable_artifact(
        artifact,
        snapshot=snapshot,
        shots=shots,
        evaluated_at=evaluated_at,
    )
    if (source is None) != (compilation_evidence is None):
        raise ValueError("source and compilation_evidence must be supplied together")
    if isinstance(artifact, ProgramArtifactV3) and not isinstance(
        compilation_evidence, CompilationEvidenceBundleV3
    ):
        raise ValueError(
            "ProgramArtifactV3 deployment requires matching version 3 evidence"
        )
    if compilation_evidence is not None:
        assert source is not None
        verify_compilation_evidence_handoff(
            compilation_evidence,
            source,
            artifact,
            snapshot=snapshot,
        )
    return ArtifactDeploymentDryRun(
        artifact=artifact,
        provider=provider,
        target_id=target_id,
        shots=shots,
        capability_match=match,
        compilation_evidence=compilation_evidence,
    )


__all__ = ("ArtifactDeploymentDryRun", "prepare_artifact_deployment")
