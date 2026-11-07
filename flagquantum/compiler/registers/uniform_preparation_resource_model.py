# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 PrepareUniformSuperposition.build_call_graph.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/state_preparation/prepare_uniform_superposition.py

"""Shared concrete dispatch and original symbolic uniform declarations."""

from typing import cast

from .._component_trace.symbolic_math import SymbolicInt, bit_length, smax
from .control_predicate_resource_model import control_predicate_cost
from .symbolic_resource_model import SymbolicResourceFormula
from .uniform_preparation import source_uniform_preparation_cost


def source_uniform_preparation_resource_formula(
    count: SymbolicInt,
    *,
    control_values: tuple[int, ...] = (),
    legacy_shims: bool = False,
) -> SymbolicResourceFormula:
    """Preserve the source declaration and peak-width fallback without emission.

    Python integer counts dispatch to the concrete model. A symbolic count,
    including a Sympy integer, retains the original symbolic declaration. Its
    gate calls omit control values; the root signature still includes them.
    Known failing singleton and one-bit symbolic width queries are rejected.
    """
    if type(legacy_shims) is not bool:
        raise ValueError("Uniform formula cost profile must be a boolean")
    if any(type(value) is not int or value not in (0, 1) for value in control_values):
        raise ValueError("Uniform formula controls require binary activation values")
    if isinstance(count, (bool, float)) or (isinstance(count, int) and count < 2):
        raise ValueError("Uniform paired formula requires at least two integer states")
    if isinstance(count, int):
        model = source_uniform_preparation_cost(
            count, control_values=control_values, legacy_shims=legacy_shims
        )
        gates = model.gates
        return SymbolicResourceFormula(
            gates.and_bloq,
            gates.clifford,
            gates.measurement,
            model.qubits,
            t=gates.t,
            toffoli=gates.toffoli,
            cswap=gates.cswap,
            rotation=gates.rotation,
        )
    width = bit_length(count - 1)
    if width == 0 or width == 1:
        raise ValueError(
            "Original symbolic uniform peak query fails below two target bits"
        )
    cleanup = control_predicate_cost((1, 1), uncompute=True, legacy_shims=legacy_shims)
    ands = 3 * width - 1
    return SymbolicResourceFormula(
        ands,
        16 * width + 3 + (cleanup.gates.clifford - 1) * ands,
        ands,
        cast(
            SymbolicInt, smax(3, width + 1, 2 * width - 1, width + len(control_values))
        ),
        rotation=2,
    )
