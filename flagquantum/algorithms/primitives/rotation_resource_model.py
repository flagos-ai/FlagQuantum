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
from ...compiler.registers.arithmetic.arithmetic_resource_model import (
    source_phase_add_cost,
)
from ...compiler.registers.lookup.qrom_resource_model import source_qrom_cost
from ...compiler.registers.sorting.permutation_resource_model import (
    source_permutation_cost,
)
from .rotation_preparation import RotationPreparation
from .rotation_tables import rotation_tables
from .sparse_rotation_preparation import SparseRotationPreparation


def _lookup_cost(words: tuple[int, ...], width: int, controls: int) -> ComponentCost:
    selection_width = (len(words) - 1).bit_length()
    selector_and_controls = selection_width + controls
    qrom = source_qrom_cost(
        controls, (selection_width,), (len(words),), ((1, width),), (words,)
    )
    add = source_phase_add_cost(width, width)
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


def sparse_rotation_preparation_cost(
    preparation: SparseRotationPreparation, target_width: int
) -> ComponentCost:
    """Build the source model for low-word preparation and basis permutation."""
    dense, cycles, low_width = preparation._components(target_width)
    permutation = source_permutation_cost(preparation.size, target_width, cycles)
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
