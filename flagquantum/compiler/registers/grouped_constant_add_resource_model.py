# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 ControlledViaAnd resource declarations.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mcmt/controlled_via_and.py

"""Symbolic AddK estimates for concrete grouped activation words."""

from dataclasses import fields

from .._component_trace.arithmetic_call import ArithmeticCall
from .._component_trace.qec_gate_cost import QECGateCost, query_qec_gate_cost
from .._component_trace.resource_generalization import (
    ControlPredicateCall,
    PredicateControlledCall,
)
from .._component_trace.resource_model import estimate_component_resources
from .._component_trace.symbolic_math import SymbolicInt
from .._component_trace.width_schedule import (
    WidthConnection,
    WidthStep,
    estimate_connection_width,
)
from .constant_add_resource_model import source_controlled_constant_add_resource_formula
from .control_predicate import ControlRegister
from .control_predicate_resource_model import control_predicate_cost
from .multi_controlled_constant_addition import (
    source_grouped_controlled_constant_add_cost,
)
from .symbolic_resource_model import SymbolicResourceFormula


def source_grouped_constant_add_resource_formula(
    width: SymbolicInt,
    constant: SymbolicInt,
    registers: tuple[ControlRegister, ...],
    *,
    adjoint: bool = False,
    legacy_shims: bool = False,
) -> SymbolicResourceFormula:
    """Compose outer register lifetimes with the original inner AddK fallback.

    Control-word shapes are concrete. Width and constant objects are not cast;
    estimates for symbolic inputs do not imply an executable gate sequence.
    """
    values = tuple(bit for reg in registers for bit in reg.bits())
    if not values or type(adjoint) is not bool or type(legacy_shims) is not bool:
        raise ValueError(
            "Grouped resource query needs nonempty controls and boolean options"
        )
    if type(width) is int:
        report = estimate_component_resources(
            source_grouped_controlled_constant_add_cost(
                width, constant, registers, adjoint=adjoint, legacy_shims=legacy_shims
            )
        )
        return SymbolicResourceFormula(
            logical_qubits=report.peak_logical_qubits,
            **{f.name: getattr(report.gates, f.name) for f in fields(report.gates)},
        )
    if len(values) == 1:
        return source_controlled_constant_add_resource_formula(
            width,
            constant,
            control_value=values[0],
            adjoint=adjoint,
            legacy_shims=legacy_shims,
        )
    positive = ArithmeticCall(
        "controlled_constant_add", (width,), constant=constant, adjoint=adjoint
    )
    addition = source_controlled_constant_add_resource_formula(
        width, constant, adjoint=adjoint, legacy_shims=legacy_shims
    )
    compute = control_predicate_cost(values, legacy_shims=legacy_shims)
    erase = control_predicate_cost(values, uncompute=True, legacy_shims=legacy_shims)
    root = PredicateControlledCall(positive, values)
    gates = query_qec_gate_cost(
        root,
        legacy_shims=legacy_shims,
        static_costs={
            positive: QECGateCost().plus(addition, 1),
            ControlPredicateCall(values): QECGateCost().plus(compute.gates, 1),
            ControlPredicateCall(values, adjoint=True): QECGateCost().plus(
                erase.gates, 1
            ),
        },
    )
    count = len(values)
    peak = estimate_connection_width(
        (
            WidthStep("input"),
            WidthStep("component", compute.qubits),
            WidthStep("component", addition.logical_qubits),
            WidthStep("component", erase.qubits),
            WidthStep("output"),
        ),
        (
            WidthConnection(0, 1, count),
            WidthConnection(0, 2, width),
            WidthConnection(1, 2, 1),
            WidthConnection(1, 3, 2 * count - 2),
            WidthConnection(2, 3, 1),
            WidthConnection(2, 4, width),
            WidthConnection(3, 4, count),
        ),
    )
    return SymbolicResourceFormula(
        logical_qubits=peak, **{f.name: getattr(gates, f.name) for f in fields(gates)}
    )
