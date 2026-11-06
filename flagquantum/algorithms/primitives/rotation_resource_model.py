# Copyright 2023 Google LLC
# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 rotation preparation, QROM and permutation costs.
# https://github.com/quantumlib/Qualtran/tree/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs

"""Logical models for rotation preparation without lowering a circuit."""

from ...compiler._component_trace.resource_model import (
    ComponentCost,
    CostCall,
    GateCost,
)
from .rotation_preparation import RotationPreparation
from .rotation_tables import rotation_tables
from .sparse_rotation_preparation import SparseRotationPreparation


def _qrom_cost(
    words: tuple[int, ...], angle_width: int, controls: int
) -> ComponentCost:
    """Count a complete one-axis QROM with at least one control."""

    def visit(start: int, stop: int) -> tuple[int, int]:
        first = words[start]
        if all(words[i] == first for i in range(start + 1, stop)):
            return 0, first.bit_count()
        middle = (start + stop) // 2
        left_nodes, left_loads = visit(start, middle)
        right_nodes, right_loads = visit(middle, stop)
        return 1 + left_nodes + right_nodes, left_loads + right_loads

    splits, loads = visit(0, len(words))
    selection = (len(words) - 1).bit_length()
    reduction = controls - 1
    width = angle_width + 2 * selection + controls + reduction
    return ComponentCost(
        "QROM",
        width,
        GateCost(
            and_bloq=splits + reduction,
            clifford=loads + 2 * splits + reduction,
            measurement=splits + reduction,
        ),
    )


def _lookup_cost(words: tuple[int, ...], width: int, controls: int) -> ComponentCost:
    selector_and_controls = (len(words) - 1).bit_length() + controls
    qrom = _qrom_cost(words, width, controls)
    add = ComponentCost("AddIntoPhaseGrad", 2 * width, GateCost(toffoli=width - 2))
    return ComponentCost(
        "PhaseGradientLookup",
        width + selector_and_controls,
        calls=(
            CostCall(qrom, bystander_qubits=width),
            CostCall(add, bystander_qubits=selector_and_controls),
            CostCall(qrom, bystander_qubits=width),
        ),
    )


def rotation_preparation_cost(
    preparation: RotationPreparation, control_count: int = 0, *, uncompute: bool = False
) -> ComponentCost:
    """Build the source logical model for concrete dense rotation preparation."""
    rows, phases = rotation_tables(
        preparation.coefficients, preparation.phase_width, uncompute=uncompute
    )
    if not rows:
        raise ValueError("Rotation resource model requires a nonempty target register")
    if type(control_count) is not int or control_count < 0:
        raise ValueError("Rotation preparation control count must be nonnegative")
    width = preparation.phase_width
    controls = control_count + 1
    amplitude = tuple(
        CostCall(
            _lookup_cost(words, width, controls),
            bystander_qubits=len(rows) - level - 1,
        )
        for level, words in enumerate(rows)
    )
    phase = CostCall(_lookup_cost(phases, width, controls))
    calls = (phase, *reversed(amplitude)) if uncompute else (*amplitude, phase)
    return ComponentCost(
        "RotationPreparation",
        len(rows) + width + control_count,
        GateCost(clifford=2 * len(rows) + 2),
        calls,
    )


def _permutation_cycle_cost(cycle: tuple[int, ...], width: int) -> ComponentCost:
    comparisons = (*cycle, cycle[0])
    flips = sum(
        (value ^ cycle[(i + 1) % len(cycle)]).bit_count()
        for i, value in enumerate(cycle)
    )
    prefix_ands = (width - 1) * len(comparisons)
    # The source MultiAnd graph includes polarity flips for its decomposed
    # children. Two controls use the separately classified direct And.
    per_zero = 2 if width == 1 else 4 if width > 2 else 0
    zero_bits = sum(width - value.bit_count() for value in comparisons)
    return ComponentCost(
        "PermutationCycle",
        2 * width,
        GateCost(
            and_bloq=prefix_ands,
            clifford=flips + width * len(comparisons) + per_zero * zero_bits,
            measurement=prefix_ands,
        ),
    )


def sparse_rotation_preparation_cost(
    preparation: SparseRotationPreparation, target_width: int
) -> ComponentCost:
    """Build the source model for low-word preparation and basis permutation."""
    dense, cycles, low_width = preparation._components(target_width)
    permutation = ComponentCost(
        "BasisPermutation",
        target_width,
        calls=tuple(
            CostCall(_permutation_cycle_cost(cycle, target_width)) for cycle in cycles
        ),
    )
    return ComponentCost(
        "SparseRotationPreparation",
        target_width + preparation.phase_width,
        calls=(
            CostCall(
                rotation_preparation_cost(dense),
                bystander_qubits=target_width - low_width,
            ),
            CostCall(permutation, bystander_qubits=preparation.phase_width),
        ),
    )
