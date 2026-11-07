# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 arithmetic/_shims.py declarations.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/_shims.py

"""Original modular placeholder signatures and resource calls, without gate emission."""

from typing import Literal, TypeAlias, cast

from .._component_trace.qec_gate_cost import QECGateCost, query_qec_gate_cost
from .._component_trace.resource_generalization import (
    CliffordCall,
    FixedTCall,
    OnEachCall,
)
from .._component_trace.resource_model import (
    ComponentCost,
    GateCost,
    estimate_component_resources,
)
from .._component_trace.symbolic_math import SymbolicInt, smax
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
    width: int,
    *,
    signed: bool = False,
    legacy_shims: bool = False,
) -> dict[FixedTCall | ComponentCost, int]:
    """Retain original direct callees and multiplicities as declaration data.

    Negative and zero multi-control counts are intentional source diagnostics.
    They are not validated executable CostCalls. Lt's signed flag has no effect
    on its original registers or calls. Zero-width CSub has failing source
    resource queries and is rejected explicitly.
    """
    if type(width) is not int:
        raise ValueError("Direct modular callees require Python integer widths")
    modular_arithmetic_registers(kind, width)
    _validate_options(kind, signed, legacy_shims)
    if kind == "multi_controlled_x":
        return {FixedTCall("toffoli"): width - 2}
    if kind == "less_than":
        return {FixedTCall("toffoli"): width}
    if kind == "controlled_half":
        return {FixedTCall("fredkin"): width}
    if width == 0:
        raise ValueError(
            "Original zero-width CSub has no successful paired resource query"
        )
    return {
        source_controlled_add_cost(width, width, legacy_shims=legacy_shims): 1,
        ComponentCost("BitwiseNot", width, GateCost(clifford=width)): 3,
    }


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
    if not isinstance(width, int):
        return _symbolic_resource_formula(kind, width, signed, legacy_shims)
    calls = modular_arithmetic_calls(
        kind, width, signed=signed, legacy_shims=legacy_shims
    )
    gates = QECGateCost()
    peak = cast(int, partition_width(modular_arithmetic_registers(kind, width)))
    for child, count in calls.items():
        gates = gates.plus(query_qec_gate_cost(child, legacy_shims=legacy_shims), count)
        child_width = (
            3
            if isinstance(child, FixedTCall)
            else estimate_component_resources(child).peak_logical_qubits
        )
        peak = max(peak, child_width)
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


def _symbolic_resource_formula(
    kind: ModularDeclaration, width: SymbolicInt, signed: bool, legacy_shims: bool
) -> SymbolicResourceFormula:
    layout = modular_arithmetic_registers(kind, width)
    _validate_options(kind, signed, legacy_shims)
    ports = partition_width(layout)
    if kind == "controlled_subtraction":
        if width == 0:
            raise ValueError(
                "Original zero-width CSub has no successful paired resource query"
            )
        addition = source_controlled_add_resource_formula(
            width, width, legacy_shims=legacy_shims
        )
        invert = query_qec_gate_cost(
            OnEachCall(CliffordCall("x", 1), width), legacy_shims=legacy_shims
        )
        gates = QECGateCost().plus(addition, 1).plus(invert, 3)
        peak = cast(SymbolicInt, smax(ports, addition.logical_qubits, smax(1, width)))
    else:
        atom = FixedTCall("fredkin" if kind == "controlled_half" else "toffoli")
        count = width - 2 if kind == "multi_controlled_x" else width
        gates = QECGateCost().plus(
            query_qec_gate_cost(atom, legacy_shims=legacy_shims), count
        )
        peak = cast(SymbolicInt, smax(ports, 3))
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
