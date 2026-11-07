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
    return _constant_add_formula(
        width, constant, encoding, adjoint, legacy_shims, control_value=None
    )


def source_controlled_constant_add_resource_formula(
    width: SymbolicInt,
    constant: SymbolicInt,
    *,
    control_value: int = 1,
    encoding: ArithmeticEncoding = "unsigned",
    adjoint: bool = False,
    legacy_shims: bool = False,
) -> SymbolicResourceFormula:
    """Query the original single-control wrapper and its width dispatch.

    Concrete execution carries the control alongside Add. Symbolic fallback
    instead uses the source's signature/callee maximum, without inventing that
    extra lifetime. Control zero contributes one outer pair of X gates.
    """
    if type(control_value) is not int or control_value not in (0, 1):
        raise ValueError("Controlled constant-add requires control value zero or one")
    return _constant_add_formula(
        width, constant, encoding, adjoint, legacy_shims, control_value=control_value
    )


def _constant_add_formula(
    width: SymbolicInt,
    constant: SymbolicInt,
    encoding: ArithmeticEncoding,
    adjoint: bool,
    legacy_shims: bool,
    *,
    control_value: int | None,
) -> SymbolicResourceFormula:
    root = ArithmeticCall(
        "constant_add" if control_value is None else "controlled_constant_add",
        (width,),
        (encoding,),
        constant=constant,
        adjoint=adjoint,
        control_value=1 if control_value is None else control_value,
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
    if control_value is None:
        peak = cast(SymbolicInt, smax(width, 1, addition.logical_qubits))
    elif type(width) is int and type(constant) is int:
        peak = addition.logical_qubits + 1
    else:
        peak = cast(SymbolicInt, smax(width + 1, 2, addition.logical_qubits))
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
    return _component_cost("AddK", formula)


def source_controlled_constant_add_cost(
    width: int,
    constant: SymbolicInt,
    *,
    control_value: int = 1,
    encoding: ArithmeticEncoding = "unsigned",
    adjoint: bool = False,
    legacy_shims: bool = False,
) -> ComponentCost:
    """Return numeric single-controlled counts for a Python integer width."""
    if type(width) is not int:
        raise ValueError(
            "Concrete controlled constant-add requires a Python integer width"
        )
    formula = source_controlled_constant_add_resource_formula(
        width,
        constant,
        control_value=control_value,
        encoding=encoding,
        adjoint=adjoint,
        legacy_shims=legacy_shims,
    )
    return _component_cost("ControlledAddK", formula)


def _component_cost(name: str, formula: SymbolicResourceFormula) -> ComponentCost:
    return ComponentCost(
        name,
        cast(int, formula.logical_qubits),
        GateCost(
            **{
                field.name: cast(int, getattr(formula, field.name))
                for field in fields(GateCost)
            }
        ),
    )
