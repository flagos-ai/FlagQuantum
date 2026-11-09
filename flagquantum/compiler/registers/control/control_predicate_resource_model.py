# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 CtrlSpecAnd and QECGatesCost.
# https://github.com/quantumlib/Qualtran/tree/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mcmt

"""Concrete predicate models using the existing typed QEC query authority."""

from dataclasses import fields
from typing import TYPE_CHECKING, cast

from ..._component_trace.qec_gate_cost import query_qec_gate_cost
from ..._component_trace.resource_generalization import (
    ControlPattern,
    ControlPredicateCall,
)
from ..._component_trace.resource_model import ComponentCost, GateCost
from ..._component_trace.symbolic_math import SymbolicInt, smax

if TYPE_CHECKING:
    from ..resources.symbolic_resource_model import SymbolicResourceFormula


def control_predicate_cost(
    values: tuple[int, ...],
    *,
    uncompute: bool = False,
    legacy_shims: bool = False,
) -> ComponentCost:
    """Model an owned prefix predicate, including controls and computed work.

    Both directions expose the same maximum live storage. Logical gate
    categories follow the selected source QEC profile, not emitted IR counts.
    """
    if type(uncompute) is not bool or type(legacy_shims) is not bool:
        raise ValueError("Predicate direction and cost profile must be booleans")
    cost = query_qec_gate_cost(
        ControlPredicateCall(values, uncompute), legacy_shims=legacy_shims
    )
    return ComponentCost(
        "ControlPredicate†" if uncompute else "ControlPredicate",
        2 * len(values) - 1,
        GateCost(
            **{field.name: int(getattr(cost, field.name)) for field in fields(GateCost)}
        ),
    )


def control_predicate_resource_formula(
    controls: ControlPattern, *, uncompute: bool = False, legacy_shims: bool = False
) -> "SymbolicResourceFormula":
    """Query known activations or an unknown-polarity control count.

    Symbolic counts retain the original signature/callee maximum. Literal
    symbolic zero/one retain diagnostic declarations, not physical budgets.
    Concrete integer counts require at least two control bits.
    """
    from ..resources.symbolic_resource_model import SymbolicResourceFormula

    if type(uncompute) is not bool or type(legacy_shims) is not bool:
        raise ValueError("Predicate direction and profile must be booleans")
    if isinstance(controls, tuple):
        count: SymbolicInt = len(controls)
    else:
        count = controls
        if type(count) is not int:
            from sympy import Expr

            if not isinstance(count, Expr):
                raise ValueError(
                    "Control count must be a Python integer or symbolic expression"
                )
        if type(count) is bool or type(count) is int and count < 2:
            raise ValueError("Concrete predicate needs at least two control bits")
    gates = query_qec_gate_cost(
        ControlPredicateCall(controls, uncompute), legacy_shims=legacy_shims
    )
    peak = 2 * count - 1
    if not isinstance(controls, tuple) and type(controls) is not int:
        peak = cast(SymbolicInt, smax(3, count, peak))
    return SymbolicResourceFormula(
        logical_qubits=peak, **{f.name: getattr(gates, f.name) for f in fields(gates)}
    )
