# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 arithmetic/_shims.py declarations.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/_shims.py

"""Original modular placeholder signatures and resource calls, without gate emission."""

from typing import Literal, TypeAlias, cast

from .._component_trace.arithmetic_call import ArithmeticCall
from .._component_trace.gate_budget import GateCategories
from .._component_trace.qec_gate_cost import QECGateCost, query_qec_gate_cost
from .._component_trace.qec_gate_cost import (
    query_qec_gate_cost as query_declared_qec_cost,
)
from .._component_trace.resource_generalization import (
    CliffordCall,
    OnEachCall,
    ResourceCall,
)
from .._component_trace.resource_model import (
    estimate_component_resources,
)
from .._component_trace.symbolic_math import SymbolicInt, smax
from .arithmetic_declarations import arithmetic_callees
from .arithmetic_resource_model import source_controlled_add_cost
from .controlled_add_resource_model import source_controlled_add_resource_formula
from .partition import partition_width
from .symbolic_resource_model import SymbolicResourceFormula

ModularDeclaration: TypeAlias = Literal[
    "multi_controlled_x", "controlled_subtraction", "less_than", "controlled_half"
]


def modular_arithmetic_registers(
    kind: ModularDeclaration, width: SymbolicInt
) -> tuple[tuple[str, SymbolicInt], ...]:
    """Return source register names and flat widths, including empty words."""
    if isinstance(width, (bool, float)) or isinstance(width, int) and width < 0:
        raise ValueError("Modular declarations require a nonnegative integer width")
    if not isinstance(width, int):
        from sympy import Expr

        if not isinstance(width, Expr):
            raise ValueError(
                "Modular declaration widths require integer or symbolic expressions"
            )
    if kind == "multi_controlled_x":
        return (("ctrl", width), ("target", 1))
    if kind == "controlled_subtraction":
        return (("ctrl", 1), ("x", width), ("y", width))
    if kind == "less_than":
        return (("x", width), ("y", width), ("out", 1))
    if kind == "controlled_half":
        return (("ctrl", 1), ("x", width))
    raise ValueError("Unknown modular arithmetic declaration")


def _validate_options(
    kind: ModularDeclaration, signed: bool, legacy_shims: bool
) -> None:
    if type(signed) is not bool or (signed and kind != "less_than"):
        raise ValueError("Only less-than declarations accept a boolean signed mode")
    if type(legacy_shims) is not bool:
        raise ValueError("Modular declaration cost profile must be a boolean")


def modular_arithmetic_calls(
    kind: ModularDeclaration,
    width: SymbolicInt,
    *,
    signed: bool = False,
    legacy_shims: bool = False,
) -> dict[ResourceCall, SymbolicInt]:
    """Return native direct source callees with unchanged dimension objects.

    Keys are independent of the cost profile. Negative/zero multiplicities and
    zero-width CSub's direct children are inspectable declaration data; recursive
    QEC or width queries can still fail when their source decomposition fails.
    """
    modular_arithmetic_registers(kind, width)
    _validate_options(kind, signed, legacy_shims)
    return arithmetic_callees(ArithmeticCall(kind, (width,), signed=signed))


def source_modular_arithmetic_resource_formula(
    kind: ModularDeclaration,
    width: SymbolicInt,
    *,
    signed: bool = False,
    legacy_shims: bool = False,
) -> SymbolicResourceFormula:
    """Aggregate original categories and call-graph width fallback without IR.

    The source placeholders have no decomposition. Peak width includes callee
    widths even at zero or negative multiplicity, exactly as its fallback does.
    """
    layout = modular_arithmetic_registers(kind, width)
    _validate_options(kind, signed, legacy_shims)
    ports = partition_width(layout)
    static_costs: dict[ResourceCall, QECGateCost]
    addition: GateCategories
    if kind == "controlled_subtraction":
        if width == 0:
            raise ValueError(
                "Original zero-width CSub has no successful paired resource query"
            )
        if isinstance(width, int):
            model = source_controlled_add_cost(width, width, legacy_shims=legacy_shims)
            report = estimate_component_resources(model)
            addition = report.gates
            addition_width: SymbolicInt = report.peak_logical_qubits
        else:
            formula = source_controlled_add_resource_formula(
                width, width, legacy_shims=legacy_shims
            )
            addition = formula
            addition_width = formula.logical_qubits
        invert = query_qec_gate_cost(
            OnEachCall(CliffordCall("x", 1), width), legacy_shims=legacy_shims
        )
        static_costs = {
            ArithmeticCall(
                "controlled_add", (width, width), ("montgomery", "montgomery")
            ): QECGateCost().plus(addition, 1),
            ArithmeticCall("bitwise_not", (width,), ("montgomery",)): invert,
        }
        peak = cast(SymbolicInt, smax(ports, addition_width, smax(1, width)))
    else:
        static_costs = {}
        peak = cast(SymbolicInt, smax(ports, 3))
    root = ArithmeticCall(kind, (width,), signed=signed)
    gates = query_declared_qec_cost(
        root,
        declared_calls={root: arithmetic_callees(root)},
        legacy_shims=legacy_shims,
        static_costs=static_costs,
    )
    return SymbolicResourceFormula(
        gates.and_bloq,
        gates.clifford,
        gates.measurement,
        peak,
        t=gates.t,
        toffoli=gates.toffoli,
        cswap=gates.cswap,
        rotation=gates.rotation,
    )
