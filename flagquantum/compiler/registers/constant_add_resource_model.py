# Copyright 2023 Google LLC
# Copyright 2024 Google LLC
# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 AddK, XorK and Always declarations.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/addition.py

"""Constant-addition estimates composed from existing native register models."""

from dataclasses import fields
from typing import cast

from .._component_trace.arithmetic_call import ArithmeticCall, ArithmeticEncoding
from .._component_trace.qec_gate_cost import QECGateCost, query_qec_gate_cost
from .._component_trace.resource_model import ComponentCost, GateCost
from .._component_trace.symbolic_math import SymbolicInt, smax
from .arithmetic_declarations import arithmetic_call_graph
from .symbolic_resource_model import (
    SymbolicResourceFormula,
    source_add_resource_formula,
)


def source_constant_add_resource_formula(
    width: SymbolicInt,
    constant: SymbolicInt,
    *,
    encoding: ArithmeticEncoding = "unsigned",
    adjoint: bool = False,
    legacy_shims: bool = False,
) -> SymbolicResourceFormula:
    """Query source load/Always(Add)/unload declarations without emitting gates.

    Symbolic literals retain the source dispatch and constant identity. Default
    Montgomery encoding has no modulus parameter. Zero-width source AddK has no
    successful paired width query, even for a symbolic integer zero.
    """
    root = ArithmeticCall(
        "constant_add", (width,), (encoding,), constant=constant, adjoint=adjoint
    )
    if type(legacy_shims) is not bool:
        raise ValueError("Constant-add cost profile must be boolean")
    if width == 0:
        raise ValueError("Original zero-width AddK has no successful width query")
    declarations = arithmetic_call_graph(root)
    addition = source_add_resource_formula(width, width, legacy_shims=legacy_shims)
    always = ArithmeticCall(
        "add", (width, width), (encoding, encoding), always=True, adjoint=adjoint
    )
    gates = query_qec_gate_cost(
        root,
        declared_calls=declarations,
        legacy_shims=legacy_shims,
        static_costs={always: QECGateCost().plus(addition, 1)},
    )
    peak = cast(SymbolicInt, smax(width, 1, addition.logical_qubits))
    return SymbolicResourceFormula(
        logical_qubits=peak,
        **{field.name: getattr(gates, field.name) for field in fields(gates)},
    )


def source_constant_add_cost(
    width: int,
    constant: SymbolicInt,
    *,
    encoding: ArithmeticEncoding = "unsigned",
    adjoint: bool = False,
    legacy_shims: bool = False,
) -> ComponentCost:
    """Return numeric counts for a concrete width, without resolving constants."""
    if type(width) is not int:
        raise ValueError("Concrete constant-add model requires a Python integer width")
    formula = source_constant_add_resource_formula(
        width, constant, encoding=encoding, adjoint=adjoint, legacy_shims=legacy_shims
    )
    return ComponentCost(
        "AddK",
        cast(int, formula.logical_qubits),
        GateCost(
            **{
                field.name: cast(int, getattr(formula, field.name))
                for field in fields(GateCost)
            }
        ),
    )
