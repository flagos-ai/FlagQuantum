"""Construction of verified target-text program artifacts."""

from __future__ import annotations

from ..core._artifacts import ArtifactKind, ProgramArtifactV2
from ..errors import CompilationError
from .target_conformance import TargetConformanceResult, verify_target_emission
from .target_emission import TargetEmissionResult
from .target_legalization import TargetLegalizationResult


class TargetArtifactError(CompilationError):
    """Verified compiler evidence cannot form a target artifact."""


def build_target_artifact(
    legalization: TargetLegalizationResult,
    emission: TargetEmissionResult,
    conformance: TargetConformanceResult,
    *,
    producer: str,
) -> ProgramArtifactV2:
    """Build one v2 artifact from a fully verified static compilation result."""

    if not isinstance(legalization, TargetLegalizationResult):
        raise TypeError("target artifact requires a TargetLegalizationResult")
    if not isinstance(emission, TargetEmissionResult):
        raise TypeError("target artifact requires a TargetEmissionResult")
    if not isinstance(conformance, TargetConformanceResult):
        raise TypeError("target artifact requires a TargetConformanceResult")
    if any(item.shots is not None for item in legalization.program.measurements):
        raise TargetArtifactError(
            "target artifact shots belong to the execution request; "
            "compile a program whose terminal samples request has shots=None"
        )

    verified = verify_target_emission(emission, legalization)
    if conformance != verified:
        raise TargetArtifactError(
            "target conformance evidence does not match the verified emission"
        )

    return ProgramArtifactV2(
        kind=ArtifactKind.EXECUTABLE,
        producer=producer,
        profile={
            "name": emission.profile,
            "media_type": emission.media_type,
            "encoding": "utf-8",
        },
        payload=emission.text,
        circuit_content_hash=legalization.program.content_hash,
        requirements=legalization.requirements,
        target={"snapshot_id": legalization.target_snapshot_id},
        compilation={
            "target_legalization_identity": legalization.legalization_identity,
            "schedule_identity": legalization.schedule.schedule_identity,
            "emission_identity": emission.emission_identity,
            "conformance_identity": conformance.conformance_identity,
        },
        parameter_schema={"binding": "fully_bound", "parameters": []},
        result_schema={
            "kind": "samples",
            "wires": list(range(legalization.program.n_wires)),
            "shots_source": "execution_request",
        },
    )


__all__ = ("TargetArtifactError", "build_target_artifact")
