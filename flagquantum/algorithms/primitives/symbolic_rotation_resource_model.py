# Copyright 2023 Google LLC
# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 rotation preparation and permutation call graphs.
# https://github.com/quantumlib/Qualtran/tree/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/state_preparation

"""Resource models for rotation preparation declared by shape alone."""

from __future__ import annotations

from typing import cast

from ...compiler._component_trace.symbolic_math import SymbolicInt, bit_length, smax
from ...compiler.registers.resources.symbolic_resource_model import (
    SymbolicResourceFormula,
    source_permutation_cycle_resource_formula,
    source_phase_gradient_lookup_resource_formula,
)


def source_rotation_preparation_resource_formula(
    coefficient_count: SymbolicInt,
    phase_width: SymbolicInt,
    control_count: SymbolicInt = 0,
) -> SymbolicResourceFormula:
    """Use the source's two full-shape lookup calls before coefficients are known."""
    if isinstance(coefficient_count, int) and coefficient_count < 1:
        raise ValueError("Rotation preparation requires at least one coefficient")
    state_width = bit_length(coefficient_count - 1)
    lookup = source_phase_gradient_lookup_resource_formula(
        coefficient_count, state_width, phase_width, control_count + 1
    )
    width = cast(
        SymbolicInt,
        smax(1, state_width + phase_width + control_count, lookup.logical_qubits),
    )
    return SymbolicResourceFormula(
        2 * lookup.and_bloq,
        2 * lookup.clifford + 2 * state_width + 2,
        2 * lookup.measurement,
        width,
        toffoli=2 * lookup.toffoli,
    )


def source_sparse_rotation_preparation_resource_formula(
    state_size: SymbolicInt,
    nonzero_count: SymbolicInt,
    phase_width: SymbolicInt,
    *,
    target_width: SymbolicInt | None = None,
) -> SymbolicResourceFormula:
    """Combine the dense source shape and a worst-case permutation cycle."""
    if isinstance(state_size, int) and state_size < 2:
        raise ValueError("The source sparse shape model requires a nonempty target")
    if isinstance(nonzero_count, int) and nonzero_count < 1:
        raise ValueError(
            "Sparse rotation preparation requires at least one coefficient"
        )
    dense = source_rotation_preparation_resource_formula(nonzero_count, phase_width)
    width = bit_length(state_size - 1)
    permutation = source_permutation_cycle_resource_formula(2 * nonzero_count, width)
    target = width if target_width is None else target_width
    peak = cast(
        SymbolicInt,
        smax(target + phase_width, dense.logical_qubits, permutation.logical_qubits),
    )
    return SymbolicResourceFormula(
        dense.and_bloq + permutation.and_bloq,
        dense.clifford + permutation.clifford,
        dense.measurement + permutation.measurement,
        peak,
        toffoli=dense.toffoli,
    )
