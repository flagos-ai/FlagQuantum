"""Construction of verified target-text program artifacts."""

from __future__ import annotations

from ..core._artifacts import ArtifactKind, ProgramArtifactV2, ProgramArtifactV3
from ..errors import CompilationError
from .physical_plan import PhysicalCircuitPlan
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
    physical_plan: PhysicalCircuitPlan | None = None,
) -> ProgramArtifactV2 | ProgramArtifactV3:
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

    topology = legalization.topology_legalization
    allocated = topology is not None and legalization.program.n_wires > (
        topology.source_program.n_wires
    )
    if allocated:
        if (
            not isinstance(physical_plan, PhysicalCircuitPlan)
            or physical_plan.legalization is not legalization
            or physical_plan.version != "3.0"
            or physical_plan.allocation_identity is None
        ):
            raise TargetArtifactError(
                "allocated target artifact requires its physical plan 3.0"
            )
        if any(
            slot < 0 or slot >= physical_plan.physical_slot_count
            for slot in physical_plan.logical_result_physical_slots
        ):
            raise TargetArtifactError("allocated result slot is outside physical plan")
        return ProgramArtifactV3(
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
                "physical_plan_identity": physical_plan.plan_identity,
                "allocation_identity": physical_plan.allocation_identity,
                "schedule_identity": legalization.schedule.schedule_identity,
                "emission_identity": emission.emission_identity,
                "conformance_identity": conformance.conformance_identity,
            },
            parameter_schema={"binding": "fully_bound", "parameters": []},
            result_schema={
                "kind": "samples",
                "logical_wires": list(range(physical_plan.logical_wire_count)),
                "physical_result_slots": list(
                    physical_plan.logical_result_physical_slots
                ),
                "ordering": "logical_wire_order",
                "shots_source": "execution_request",
            },
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
