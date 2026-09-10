"""Construct and verify serialized evidence from actual compilation objects."""

from __future__ import annotations

from ..core._compilation_evidence import (
    CompilationEvidenceBundle,
    CouplingEvidence,
    MappingTransitionEvidence,
    PhysicalInstructionEvidence,
    PhysicalPlanEvidence,
)
from ..core.target_capabilities import TargetCapabilitySnapshot
from ..errors import CompilationError
from .artifact_compilation import ArtifactCompilationResult


class CompilationEvidenceError(CompilationError):
    """Compilation objects do not match a serialized evidence bundle."""


def _physical_evidence(result: ArtifactCompilationResult) -> PhysicalPlanEvidence:
    plan = result.physical_plan
    if plan.coupling_direction_semantics == "directed_cx":
        raise CompilationEvidenceError(
            "directed physical plans require compilation evidence version 2.0"
        )
    coupling = (
        None
        if plan.coupling_n_wires is None
        else CouplingEvidence(plan.coupling_n_wires, plan.coupling_edges)
    )
    return PhysicalPlanEvidence(
        source_circuit_hash=plan.source_circuit_hash,
        physical_circuit_hash=plan.program.content_hash,
        target_snapshot_id=plan.target_snapshot_id,
        topology_identity=plan.topology_identity,
        coupling=coupling,
        initial_logical_to_physical=plan.initial_logical_to_physical,
        pre_restore_logical_to_physical=plan.pre_restore_logical_to_physical,
        final_logical_to_physical=plan.final_logical_to_physical,
        mapping_transitions=tuple(
            MappingTransitionEvidence(
                routed_instruction_index=item.routed_instruction_index,
                source_instruction_index=item.source_instruction_index,
                phase=item.phase,
                physical_wires=item.physical_wires,
                layout_before=item.layout_before,
                layout_after=item.layout_after,
            )
            for item in plan.mapping_transitions
        ),
        instructions=tuple(
            PhysicalInstructionEvidence(
                instruction_index=item.instruction_index,
                source_instruction_index=item.source_instruction_index,
                topology_instruction_index=item.topology_instruction_index,
                native_replacement_ordinal=item.native_replacement_ordinal,
                origin=item.origin,
                opcode=item.opcode,
                logical_wires=item.logical_wires,
                physical_wires=item.physical_wires,
                layer=item.layer,
                predecessors=item.predecessors,
                dependency_kinds=item.dependency_kinds,
            )
            for item in plan.instructions
        ),
        topology_legalization_identity=plan.topology_legalization_identity,
        native_gate_legalization_identity=plan.native_gate_legalization_identity,
        schedule_identity=plan.schedule_identity,
        schedule_depth=plan.schedule_depth,
        maximum_parallel_width=plan.maximum_parallel_width,
        critical_path=plan.critical_path,
        plan_identity=plan.plan_identity,
    )


def _validate_snapshot(
    result: ArtifactCompilationResult,
    snapshot: TargetCapabilitySnapshot,
) -> None:
    if not isinstance(snapshot, TargetCapabilitySnapshot):
        raise TypeError("snapshot must be a TargetCapabilitySnapshot")
    if snapshot.snapshot_id != result.legalization.target_snapshot_id:
        raise CompilationEvidenceError(
            "target snapshot does not match artifact compilation"
        )


def build_compilation_evidence_bundle(
    result: ArtifactCompilationResult,
    *,
    snapshot: TargetCapabilitySnapshot,
    producer: str,
) -> CompilationEvidenceBundle:
    """Build a bundle from the retained compilation objects and target snapshot."""

    if not isinstance(result, ArtifactCompilationResult):
        raise TypeError("result must be an ArtifactCompilationResult")
    _validate_snapshot(result, snapshot)
    executable = result.executable_artifact
    physical = _physical_evidence(result)
    return CompilationEvidenceBundle(
        producer=producer,
        source={
            "source_artifact_identity": result.source_artifact_identity,
            "circuit_artifact_identity": result.circuit_artifact_identity,
            "binding_identity": result.binding_identity,
            "source_circuit_hash": physical.source_circuit_hash,
            "final_circuit_hash": result.legalization.program.content_hash,
        },
        target={
            "snapshot_id": snapshot.snapshot_id,
            "target_legalization_identity": result.legalization.legalization_identity,
        },
        physical_plan=physical,
        output={
            "profile": str(executable.profile["name"]),
            "payload_sha256": executable.payload_sha256,
            "emission_identity": result.emission.emission_identity,
            "conformance_identity": result.conformance.conformance_identity,
            "executable_artifact_identity": executable.artifact_identity,
            "artifact_compilation_identity": result.compilation_identity,
        },
    )


def verify_compilation_evidence_bundle(
    bundle: CompilationEvidenceBundle,
    result: ArtifactCompilationResult,
    *,
    snapshot: TargetCapabilitySnapshot,
) -> None:
    """Verify a decoded bundle against all actual in-process source objects."""

    if not isinstance(bundle, CompilationEvidenceBundle):
        raise TypeError("bundle must be a CompilationEvidenceBundle")
    if not isinstance(result, ArtifactCompilationResult):
        raise TypeError("result must be an ArtifactCompilationResult")
    _validate_snapshot(result, snapshot)
    expected = build_compilation_evidence_bundle(
        result,
        snapshot=snapshot,
        producer=bundle.producer,
    )
    if bundle != expected:
        raise CompilationEvidenceError(
            "compilation evidence does not match retained compilation objects"
        )


__all__ = (
    "CompilationEvidenceError",
    "build_compilation_evidence_bundle",
    "verify_compilation_evidence_bundle",
)
