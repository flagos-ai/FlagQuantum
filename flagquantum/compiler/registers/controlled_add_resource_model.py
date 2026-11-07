# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 CAdd.build_call_graph and QubitCount fallback.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/controlled_addition.py

"""Shared controlled-add declarations with original concrete/symbolic dispatch."""

from dataclasses import fields
from typing import cast

from .._component_trace.resource_model import estimate_component_resources
from .._component_trace.symbolic_math import SymbolicInt, smax
from .arithmetic_resource_model import source_controlled_add_cost
from .control_predicate_resource_model import control_predicate_cost
from .symbolic_resource_model import (
    SymbolicResourceFormula,
    source_add_resource_formula,
)


def source_controlled_add_resource_formula(
    left_width: SymbolicInt,
    right_width: SymbolicInt,
    *,
    control_value: int = 1,
    legacy_shims: bool = False,
) -> SymbolicResourceFormula:
    """Compose source Add and predicates without concretizing symbolic literals.

    Python integers use the existing loaded/carry lifetime model. Symbolic
    dimensions retain the original signature/callee maximum fallback because
    the source cannot decompose them. No physical circuit is emitted here.
    """
    if type(control_value) is not int or control_value not in (0, 1):
        raise ValueError("Controlled-add formula requires control value zero or one")
    if type(legacy_shims) is not bool:
        raise ValueError("Controlled-add formula cost profile must be a boolean")
    for width in (left_width, right_width):
        if isinstance(width, (bool, float)) or isinstance(width, int) and width < 1:
            raise ValueError("Controlled-add formula requires positive integer widths")
    if isinstance(left_width, int) and isinstance(right_width, int):
        model = source_controlled_add_cost(
            left_width,
            right_width,
            control_value=control_value,
            legacy_shims=legacy_shims,
        )
        # Concrete model width is a composition, not just the root port count.
        report = estimate_component_resources(model)
        gates = {
            field.name: getattr(report.gates, field.name)
            for field in fields(report.gates)
        }
        return SymbolicResourceFormula(
            logical_qubits=report.peak_logical_qubits, **gates
        )

    addition = source_add_resource_formula(
        left_width, right_width, legacy_shims=legacy_shims
    )
    compute = control_predicate_cost((control_value, 1), legacy_shims=legacy_shims)
    cleanup = control_predicate_cost(
        (control_value, 1), uncompute=True, legacy_shims=legacy_shims
    )
    categories: dict[str, SymbolicInt] = {}
    for field in fields(SymbolicResourceFormula):
        if field.name != "logical_qubits":
            categories[field.name] = getattr(addition, field.name) + left_width * (
                getattr(compute.gates, field.name) + getattr(cleanup.gates, field.name)
            )
    peak = cast(
        SymbolicInt,
        smax(
            1 + left_width + right_width,
            addition.logical_qubits,
            compute.qubits,
            cleanup.qubits,
        ),
    )
    return SymbolicResourceFormula(logical_qubits=peak, **categories)
