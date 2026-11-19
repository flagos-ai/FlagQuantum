"""Build Core's v4 physical evidence from retained Compiler stage objects."""

from __future__ import annotations

from datetime import datetime

from ...core._compilation_evidence import CouplingEvidence, MappingTransitionEvidence
from ...core._compilation_evidence_v4 import (
    NativeInputInstructionEvidence,
    PhysicalInstructionEvidenceV4,
    PhysicalPlanEvidenceV4,
)
from ...core.target_capabilities import TargetCapabilitySnapshot
from ..lowering.native_block_optimization import (
    NativeBlockOptimizationResult,
    verify_native_block_optimization,
)
from ..lowering.schedule_legalization import (
    CircuitSchedule,
    schedule_circuit_dependencies,
)
from ..routing import CouplingMap
from .physical_plan import (
    PhysicalCircuitPlan,
    PhysicalPlanError,
    _critical_path,
    build_physical_circuit_plan,
)


def build_native_block_physical_evidence(
    baseline: PhysicalCircuitPlan,
    optimization: NativeBlockOptimizationResult,
    schedule: CircuitSchedule,
    *,
    snapshot: TargetCapabilitySnapshot,
    evaluated_at: datetime | None = None,
    max_added_operations: int = 256,
) -> PhysicalPlanEvidenceV4:
    """Verify and describe an optimized operator without single-source fiction.

    The baseline plan retains the original topology-space transitions and
    per-instruction ancestry. The schedule belongs to the optimized output.
    Numerical validation is repeated here; Core only validates structural
    evidence. This private builder does not enable automatic compilation.
    """
    if baseline.version != "1.0" or not optimization.rewrites:
        raise PhysicalPlanError("v4 requires an unallocated undirected block rewrite")
    native = baseline.legalization.native_gate_legalization
    if (
        baseline.program is not native.program
        or optimization.source_program is not baseline.program
        or baseline.target_snapshot_id != snapshot.snapshot_id
        or schedule.program is not optimization.program
        or schedule.target_snapshot_id != snapshot.snapshot_id
    ):
        raise PhysicalPlanError("native block evidence stage chain is inconsistent")
    coupling = (
        None
        if baseline.coupling_n_qubits is None
        else CouplingMap(baseline.coupling_n_qubits, baseline.coupling_edges)
    )
    replay = build_physical_circuit_plan(baseline.legalization, coupling_map=coupling)
    if baseline.plan_identity != replay.plan_identity:
        raise PhysicalPlanError("native baseline physical plan failed replay")
    verify_native_block_optimization(
        optimization,
        native,
        snapshot=snapshot,
        evaluated_at=evaluated_at,
        max_added_operations=max_added_operations,
        topology=baseline.legalization.topology_legalization,
        coupling_map=coupling,
    )
    expected_schedule = schedule_circuit_dependencies(
        optimization.program, target_snapshot_id=snapshot.snapshot_id
    )
    if schedule != expected_schedule:
        raise PhysicalPlanError("native block output schedule failed replay")
    if coupling is not None and any(
        len(item.wires) == 2 and not coupling.has_edge(*item.wires)
        for item in optimization.program.instructions
    ):
        raise PhysicalPlanError("native block output violates coupling topology")

    output_membership = {
        index: (rewrite_index, ordinal)
        for rewrite_index, rewrite in enumerate(optimization.rewrites)
        for ordinal, index in enumerate(rewrite.output_instruction_indices)
    }
    rewritten_inputs = {
        index
        for rewrite in optimization.rewrites
        for index in rewrite.input_instruction_indices
    }
    unchanged = iter(
        index
        for index in range(len(baseline.program.instructions))
        if index not in rewritten_inputs
    )
    instructions: list[PhysicalInstructionEvidenceV4] = []
    for index, instruction in enumerate(optimization.program.instructions):
        membership = output_membership.get(index)
        scheduled = schedule.instructions[index]
        instructions.append(
            PhysicalInstructionEvidenceV4(
                instruction_index=index,
                native_input_instruction_index=(
                    next(unchanged) if membership is None else None
                ),
                block_rewrite_index=None if membership is None else membership[0],
                block_replacement_ordinal=(
                    None if membership is None else membership[1]
                ),
                opcode=instruction.name,
                physical_qubits=instruction.wires,
                layer=scheduled.layer,
                predecessors=scheduled.predecessors,
                dependency_kinds=scheduled.dependency_kinds,
            )
        )
    return PhysicalPlanEvidenceV4(
        source_circuit_hash=baseline.source_circuit_hash,
        physical_circuit_hash=optimization.program.content_hash,
        target_snapshot_id=snapshot.snapshot_id,
        topology_identity=baseline.topology_identity,
        coupling=(
            None
            if coupling is None
            else CouplingEvidence(coupling.n_qubits, coupling.edges)
        ),
        initial_logical_to_physical=baseline.initial_logical_to_physical,
        pre_restore_logical_to_physical=baseline.pre_restore_logical_to_physical,
        final_logical_to_physical=baseline.final_logical_to_physical,
        mapping_transitions=tuple(
            MappingTransitionEvidence(
                routed_instruction_index=item.routed_instruction_index,
                source_instruction_index=item.source_instruction_index,
                phase=item.phase,
                physical_wires=item.physical_qubits,
                layout_before=item.layout_before,
                layout_after=item.layout_after,
            )
            for item in baseline.mapping_transitions
        ),
        instructions=tuple(instructions),
        topology_legalization_identity=baseline.topology_legalization_identity,
        native_gate_legalization_identity=baseline.native_gate_legalization_identity,
        schedule_identity=schedule.schedule_identity,
        schedule_depth=schedule.depth,
        maximum_parallel_width=schedule.maximum_parallel_width,
        critical_path=_critical_path(schedule),
        native_input_circuit_hash=baseline.program.content_hash,
        native_input_instructions=tuple(
            NativeInputInstructionEvidence(
                instruction_index=item.instruction_index,
                source_instruction_index=item.source_instruction_index,
                topology_instruction_index=item.topology_instruction_index,
                native_replacement_ordinal=item.native_replacement_ordinal,
                origin=item.origin,
                opcode=item.opcode,
                source_logical_qubits=item.logical_qubits,
                physical_qubits=item.physical_qubits,
            )
            for item in baseline.instructions
        ),
        block_rewrites=optimization.rewrites,
        native_block_optimization_identity=optimization.optimization_identity,
    )
