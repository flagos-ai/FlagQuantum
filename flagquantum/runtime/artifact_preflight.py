"""Read-only Runtime checks for verified target-text artifacts."""

from __future__ import annotations

from datetime import datetime

from ..core._artifacts import ArtifactKind, ProgramArtifactV2
from ..core.target_capabilities import (
    CapabilityMatchResult,
    CapabilityRequirement,
    ComparisonOperator,
    EvidenceLevel,
    FactExposure,
    RequirementSet,
    RequirementSource,
    RequirementStrength,
    TargetCapabilitySnapshot,
    match_target_capabilities,
)
from ..errors import ExecutionError


def _runtime_requirement(
    name: str,
    operator: ComparisonOperator,
    value: object,
) -> CapabilityRequirement:
    return CapabilityRequirement(
        name=name,
        operator=operator,
        value=value,
        strength=RequirementStrength.MANDATORY,
        source=RequirementSource.RUNTIME_PROTOCOL,
        minimum_evidence_level=EvidenceLevel.BASIC,
        accepted_exposures=(FactExposure.DECLARED, FactExposure.OBSERVED),
    )


def preflight_executable_artifact(
    artifact: ProgramArtifactV2,
    *,
    snapshot: TargetCapabilitySnapshot,
    shots: int,
    evaluated_at: datetime | None = None,
) -> CapabilityMatchResult:
    """Validate artifact, target, profile, and shot compatibility without execution."""

    if not isinstance(artifact, ProgramArtifactV2):
        raise TypeError("artifact must be a ProgramArtifactV2")
    if artifact.kind is not ArtifactKind.EXECUTABLE:
        raise ExecutionError("Runtime preflight requires an executable artifact")
    if not isinstance(snapshot, TargetCapabilitySnapshot):
        raise TypeError("snapshot must be a TargetCapabilitySnapshot")
    if type(shots) is not int or shots <= 0:
        raise ValueError("shots must be a positive integer")

    target = artifact.target
    if target is None or target["snapshot_id"] != snapshot.snapshot_id:
        raise ExecutionError(
            "executable artifact target does not match the supplied capability snapshot"
        )
    requirements = artifact.requirements
    if not isinstance(requirements, RequirementSet):
        raise ExecutionError("executable artifact has no valid capability requirements")

    profile = str(artifact.profile["name"])
    effective = RequirementSet(
        requirements=(
            *requirements.requirements,
            _runtime_requirement(
                "artifacts.profiles",
                ComparisonOperator.CONTAINS_ALL,
                (profile,),
            ),
            _runtime_requirement(
                "limits.maximum_shots",
                ComparisonOperator.AT_LEAST,
                shots,
            ),
        ),
        fallback_authorizations=requirements.fallback_authorizations,
        extensions=requirements.extensions,
    )
    match = match_target_capabilities(
        effective,
        snapshot,
        evaluated_at=evaluated_at,
    )
    if not match.executable:
        details = "; ".join(
            f"{item.capability_name or 'target'}: {item.code}: {item.message}"
            for item in match.blockers
        )
        raise ExecutionError(f"executable artifact preflight failed: {details}")
    return match


__all__ = ("preflight_executable_artifact",)
