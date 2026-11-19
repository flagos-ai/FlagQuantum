"""Derive native-input and collective output lineage for physical plans."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...core._compilation_evidence_v4 import (
    NativeInputInstructionEvidence,
    PhysicalInstructionEvidenceV4,
)
from ..lowering.native_block_optimization import NativeBlockOptimizationResult
from ..lowering.schedule_legalization import CircuitSchedule

if TYPE_CHECKING:
    from .physical_plan import PhysicalInstructionRecord


def _native_input_evidence(
    records: tuple[PhysicalInstructionRecord, ...],
) -> tuple[NativeInputInstructionEvidence, ...]:
    return tuple(
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
        for item in records
    )


def _output_instruction_evidence(
    optimization: NativeBlockOptimizationResult,
    schedule: CircuitSchedule,
) -> tuple[PhysicalInstructionEvidenceV4, ...]:
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
        for index in range(len(optimization.source_program.instructions))
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
    return tuple(instructions)
