"""Runtime counters for compiled local statevector programs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from ...core.operator_schema import canonical_opcode
from .operations import _diagonal_region
from .program import (
    _StatevectorCliffordMatchingStep,
    _StatevectorControlledPhaseDecompositionStep,
    _StatevectorControlledPhaseGraphStep,
    _StatevectorCrossWireDiagonalStep,
    _StatevectorCXSequenceStep,
    _StatevectorCZGraphStep,
    _StatevectorDisjointDenseStep,
    _StatevectorFusedGateStep,
    _StatevectorGateStep,
    _StatevectorProgramStep,
    _StatevectorRXRZLoopStep,
)

if TYPE_CHECKING:
    from ...circuit import _StatevectorExecutionStatistics


def initial_runtime_metrics(
    program: Sequence[_StatevectorProgramStep], *, enable_triton_loop: bool
) -> _StatevectorExecutionStatistics:
    """Describe which compiled regions are expected to execute."""

    metric_steps = tuple(
        region
        for step in program
        for region in (
            step.regions
            if isinstance(
                step,
                (_StatevectorCrossWireDiagonalStep, _StatevectorDisjointDenseStep),
            )
            else (step,)
        )
    )
    diagonal_metric_steps = tuple(
        region
        for step in program
        for region in (
            step.regions
            if isinstance(step, _StatevectorCrossWireDiagonalStep)
            else (step,)
        )
    )
    loop_steps = tuple(
        step for step in metric_steps if isinstance(step, _StatevectorRXRZLoopStep)
    )
    # A diagonal region in a mixed group executes through the dense product.
    diagonal_unfused_gates = sum(
        isinstance(step, _StatevectorGateStep) and _diagonal_region((step.instruction,))
        for step in diagonal_metric_steps
    )
    diagonal_fused_regions = tuple(
        step
        for step in diagonal_metric_steps
        if isinstance(step, _StatevectorFusedGateStep) and step.diagonal
    )
    controlled_phase_regions = tuple(
        step
        for step in metric_steps
        if isinstance(step, _StatevectorControlledPhaseDecompositionStep)
    )
    controlled_phase_graph_regions = tuple(
        step
        for step in metric_steps
        if isinstance(step, _StatevectorControlledPhaseGraphStep)
    )
    cz_graph_regions = tuple(
        step for step in metric_steps if isinstance(step, _StatevectorCZGraphStep)
    )
    clifford_matching_regions = tuple(
        step
        for step in metric_steps
        if isinstance(step, _StatevectorCliffordMatchingStep)
    )
    return {
        "triton_single_qubit_loop_enabled": enable_triton_loop,
        "triton_single_qubit_loop_regions": len(loop_steps),
        "triton_single_qubit_loop_gates": sum(
            2 * len(step.pairs) for step in loop_steps
        ),
        "triton_ry_rz_pair_candidates": sum(
            isinstance(step, _StatevectorFusedGateStep)
            and tuple(item.name for item in step.instructions) == ("ry", "rz")
            for step in metric_steps
        ),
        "triton_ry_rz_pair_executed": 0,
        "triton_single_qubit_matrix_regions": 0,
        "diagonal_elementwise_gates": diagonal_unfused_gates
        + sum(len(step.instructions) for step in diagonal_fused_regions)
        + 5 * len(controlled_phase_regions)
        + 5 * sum(len(step.edges) for step in controlled_phase_graph_regions)
        + sum(len(step.edges) for step in cz_graph_regions)
        + sum(len(step.cz_edges) for step in clifford_matching_regions),
        "diagonal_fused_regions": len(diagonal_fused_regions)
        + len(controlled_phase_regions)
        + len(controlled_phase_graph_regions)
        + len(cz_graph_regions),
        "permutation_gates": sum(
            isinstance(step, _StatevectorGateStep)
            and canonical_opcode(step.instruction.name) in {"x", "cx", "swap"}
            for step in program
        )
        + sum(
            len(step.controls)
            for step in metric_steps
            if isinstance(step, _StatevectorCXSequenceStep)
        )
        + sum(len(step.controls) for step in clifford_matching_regions),
        "triton_cx_sequence_regions": sum(
            isinstance(step, _StatevectorCXSequenceStep) for step in metric_steps
        ),
        "fixed_single_qubit_specialized_gates": sum(
            isinstance(step, _StatevectorGateStep)
            and canonical_opcode(step.instruction.name) == "y"
            for step in program
        ),
        "native_cpu_one_qubit_layer_regions": 0,
        "native_cpu_parameterized_one_qubit_layer_regions": 0,
        "native_cpu_product_state_initialization": 0,
        "native_cpu_clifford_matching_regions": 0,
        "fused_gate_regions": sum(
            isinstance(step, _StatevectorFusedGateStep) for step in metric_steps
        )
        + len(controlled_phase_regions)
        + len(controlled_phase_graph_regions)
        + len(cz_graph_regions)
        + len(clifford_matching_regions),
        "fused_gate_count": sum(
            len(step.instructions)
            for step in metric_steps
            if isinstance(step, _StatevectorFusedGateStep)
        )
        + 5 * len(controlled_phase_regions)
        + 5 * sum(len(step.edges) for step in controlled_phase_graph_regions)
        + sum(len(step.edges) for step in cz_graph_regions)
        + sum(
            len(step.controls) + len(step.cz_edges)
            for step in clifford_matching_regions
        ),
        "dependency_reordered_single_qubit_regions": sum(
            isinstance(step, _StatevectorFusedGateStep) and step.dependency_reordered
            for step in metric_steps
        ),
        "statevector_apply_count": len(program),
    }
