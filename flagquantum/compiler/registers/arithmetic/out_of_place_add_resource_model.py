# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 OutOfPlaceAdder declarations and QubitCount fallback.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/addition.py

"""Shared out-of-place addition estimates without emitting a gate sequence."""

from dataclasses import fields
from typing import cast

from ..._component_trace.arithmetic_call import ArithmeticCall
from ..._component_trace.qec_gate_cost import query_qec_gate_cost
from ..._component_trace.resource_model import ComponentCost, GateCost
from ..._component_trace.symbolic_math import SymbolicInt, smax
from ..resources.symbolic_resource_model import SymbolicResourceFormula
from .arithmetic_declarations import arithmetic_call_graph


def source_out_of_place_add_resource_formula(
    width: SymbolicInt,
    *,
    include_msb: bool = True,
    adjoint: bool = False,
    legacy_shims: bool = False,
) -> SymbolicResourceFormula:
    """Query source categories and distinguish concrete from symbolic width.

    The sum register is a logical output, not additional work. The adjoint
    assumes its correct value. Python zero-width source decomposition fails;
    symbolic zero retains the source's signature/callee maximum fallback.
    """
    root = ArithmeticCall(
        "out_of_place_add",
        (width,),
        adjoint=adjoint,
        include_most_significant_bit=include_msb,
    )
    if type(legacy_shims) is not bool:
        raise ValueError("Out-of-place addition cost profile must be boolean")
    ports = 3 * width + int(include_msb)
    if isinstance(width, int):
        if width == 0:
            raise ValueError(
                "Original Python zero-width adder has no successful width query"
            )
        peak = ports
    else:
        peak = cast(SymbolicInt, smax(3, 2 * width, ports))
    gates = query_qec_gate_cost(
        root,
        declared_calls=arithmetic_call_graph(root),
        legacy_shims=legacy_shims,
    )
    return SymbolicResourceFormula(
        logical_qubits=peak,
        **{field.name: getattr(gates, field.name) for field in fields(gates)},
    )


def source_out_of_place_add_cost(
    width: int,
    *,
    include_msb: bool = True,
    adjoint: bool = False,
    legacy_shims: bool = False,
) -> ComponentCost:
    """Return the same declaration as a concrete composable component model."""
    if type(width) is not int:
        raise ValueError(
            "Concrete out-of-place addition requires a Python integer width"
        )
    formula = source_out_of_place_add_resource_formula(
        width,
        include_msb=include_msb,
        adjoint=adjoint,
        legacy_shims=legacy_shims,
    )
    return ComponentCost(
        "OutOfPlaceAdder",
        cast(int, formula.logical_qubits),
        GateCost(
            **{
                field.name: cast(int, getattr(formula, field.name))
                for field in fields(GateCost)
            }
        ),
    )
