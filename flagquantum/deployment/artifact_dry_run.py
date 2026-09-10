"""Prepare verified executable artifacts for remote adaptation without submission."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from ..core._artifacts import ProgramArtifactV2
from ..core.target_capabilities import CapabilityMatchResult, TargetCapabilitySnapshot
from ..errors import ExecutionError
from ..runtime.artifact_preflight import preflight_executable_artifact


@dataclass(frozen=True)
class ArtifactDeploymentDryRun:
    """Immutable handoff prepared without credentials or provider side effects."""

    artifact: ProgramArtifactV2 = field(repr=False)
    provider: str
    target_id: str
    shots: int
    capability_match: CapabilityMatchResult = field(repr=False)

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


def prepare_artifact_deployment(
    artifact: ProgramArtifactV2,
    *,
    snapshot: TargetCapabilitySnapshot,
    provider: str,
    target_id: str,
    shots: int,
    evaluated_at: datetime | None = None,
) -> ArtifactDeploymentDryRun:
    """Build a side-effect-free handoff after target and capability checks."""

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
    return ArtifactDeploymentDryRun(
        artifact=artifact,
        provider=provider,
        target_id=target_id,
        shots=shots,
        capability_match=match,
    )


__all__ = ("ArtifactDeploymentDryRun", "prepare_artifact_deployment")
