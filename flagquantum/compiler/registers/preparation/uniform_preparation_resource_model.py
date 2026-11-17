# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 PrepareUniformSuperposition.build_call_graph.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/state_preparation/prepare_uniform_superposition.py

"""Shared concrete dispatch and original symbolic uniform declarations."""

from __future__ import annotations

from typing import cast

from ....core.ir.program.signatures import Signature
from ..._component_trace.symbolic_math import SymbolicInt, bit_length, floor, log2, smax
from ..comparison.less_than import source_less_than_constant_cost
from ..control.control_predicate_resource_model import control_predicate_cost
from ..resources.symbolic_resource_model import SymbolicResourceFormula
from .uniform_preparation import _odd_factor, source_uniform_preparation_cost


def source_uniform_preparation_signature(
    count: SymbolicInt, control_width: SymbolicInt = 0
) -> Signature:
    """Declare the full target and the specified number of control bits."""
    return Signature.build(ctrl=control_width, target=bit_length(count - 1))


def normalize_uniform_controls(
    values: tuple[int, ...], abstract_width: SymbolicInt | None
) -> tuple[tuple[int, ...], SymbolicInt | None]:
    """Resolve a literal control length to the source's all-positive controls."""
    if abstract_width is not None and values:
        raise ValueError("Specify either control values or an abstract control width")
    if isinstance(abstract_width, int):
        return (1,) * abstract_width, None
    return values, abstract_width


def _declared_factors(count: SymbolicInt) -> tuple[SymbolicInt, SymbolicInt]:
    odd = _odd_factor(count) if isinstance(count, int) else count
    width = bit_length(odd - 1)
    if isinstance(width, int) and width < 3:
        raise ValueError(
            "Original uniform declaration requires at least 3 MultiAnd controls"
        )
    return odd, width


def source_uniform_abstract_calls(count: SymbolicInt) -> dict[str, SymbolicInt]:
    """Declare the direct calls used for a symbolic count or control length."""
    odd, width = _declared_factors(count)
    if isinstance(odd, int) and isinstance(width, int):
        from numpy import arccos

        angle = arccos(1 - (1 << (width - 1)) / odd)
        predicate = f"MultiAnd{(1,) * width}"
    else:
        from sympy import acos

        angle = acos(1 - 2 ** floor(log2(odd)) / odd)
        predicate = f"MultiAnd(HasLength({width}))"
    compare = f"LessThanConstant({width}, {odd})"
    return {
        "Hadamard": 3 * width,
        compare: 1,
        f"{compare}†": 1,
        f"Rz({angle}, 1e-11)": 2,
        predicate: 1,
        f"{predicate}†": 1,
    }


def source_uniform_preparation_resource_formula(
    count: SymbolicInt,
    *,
    control_values: tuple[int, ...] = (),
    abstract_control_width: SymbolicInt | None = None,
    legacy_shims: bool = False,
) -> SymbolicResourceFormula:
    """Preserve the source declaration and peak-width fallback without emission.

    A symbolic count or control length retains the original abstract graph.
    The abstract calls omit control values and the root signature includes
    their declared width. Otherwise use the concrete resource model.
    Known failing singleton and one-bit symbolic width queries are rejected.
    """
    if type(legacy_shims) is not bool:
        raise ValueError("Uniform formula cost profile must be a boolean")
    control_values, abstract_control_width = normalize_uniform_controls(
        control_values, abstract_control_width
    )
    if any(type(value) is not int or value not in (0, 1) for value in control_values):
        raise ValueError("Uniform formula controls require binary activation values")
    if isinstance(count, (bool, float)) or (isinstance(count, int) and count < 2):
        raise ValueError("Uniform paired formula requires at least two integer states")
    if isinstance(count, int) and abstract_control_width is None:
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
    control_width = (
        len(control_values)
        if abstract_control_width is None
        else abstract_control_width
    )
    root_width = bit_length(count - 1) + control_width
    odd, width = _declared_factors(count)
    if width == 0 or width == 1:
        raise ValueError(
            "Original symbolic uniform peak query fails below two target bits"
        )
    cleanup = control_predicate_cost((1, 1), uncompute=True, legacy_shims=legacy_shims)
    if isinstance(odd, int) and isinstance(width, int):
        compare = source_less_than_constant_cost(width, odd)
        values = (1,) * width
        forward = control_predicate_cost(values, legacy_shims=legacy_shims)
        reverse = control_predicate_cost(
            values, uncompute=True, legacy_shims=legacy_shims
        )
        gates = (
            compare.gates.plus(compare.gates, 1)
            .plus(forward.gates, 1)
            .plus(reverse.gates, 1)
        )
        return SymbolicResourceFormula(
            gates.and_bloq,
            gates.clifford + 3 * width + 2 * width * (cleanup.gates.clifford - 1),
            gates.measurement,
            cast(SymbolicInt, smax(root_width, compare.qubits, forward.qubits)),
            rotation=2,
        )
    ands = 3 * width - 1
    return SymbolicResourceFormula(
        ands,
        16 * width + 3 + (cleanup.gates.clifford - 1) * ands,
        ands,
        cast(SymbolicInt, smax(3, width + 1, 2 * width - 1, root_width)),
        rotation=2,
    )
