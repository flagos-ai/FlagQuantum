# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 ControlledViaAnd and specialized_ctrl.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mcmt/controlled_via_and.py
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mcmt/specialized_ctrl.py

"""Compose concrete and symbolic controls using shared predicate and width models."""

from dataclasses import fields
from typing import cast

from ..._component_trace.gate_budget import GateCategories
from ..._component_trace.qec_gate_cost import QECGateCost, query_qec_gate_cost
from ..._component_trace.resource_generalization import (
    ControlPattern,
    ControlPredicateCall,
    PredicateControlledCall,
    ResourceCall,
)
from ..._component_trace.resource_model import ComponentCost, CostCall, GateCost
from ..._component_trace.symbolic_math import SymbolicInt, smax
from ..._component_trace.width_schedule import (
    WidthConnection,
    WidthStep,
    estimate_connection_width,
)
from ..resources.symbolic_resource_model import SymbolicResourceFormula
from .control_predicate import ControlRegister
from .control_predicate_resource_model import (
    control_predicate_cost,
    control_predicate_resource_formula,
)


def controlled_component_cost(
    positive: ComponentCost,
    registers: tuple[ControlRegister, ...],
    *,
    target_width: int,
    legacy_shims: bool = False,
) -> ComponentCost:
    """Wrap a positive-controlled model with source predicates and live ports.

    target_width counts logical child ports excluding its positive predicate.
    It must not include the child's private peak. The supplied directional
    child uses the same QEC profile; this function does not emit instructions.
    """
    values = tuple(bit for register in registers for bit in register.bits())
    if (
        not values
        or type(target_width) is not int
        or target_width < 0
        or type(legacy_shims) is not bool
    ):
        raise ValueError(
            "Control model needs nonempty values, nonnegative target width and boolean profile"
        )
    if positive.qubits < target_width + 1:
        raise ValueError("Positive child model omits its control or target ports")
    if values == (1,):
        return positive
    if len(values) == 1:
        return ComponentCost(
            "ControlledViaAnd",
            target_width + 1,
            GateCost(clifford=2),
            (CostCall(positive),),
        )
    return ComponentCost(
        "ControlledViaAnd",
        target_width + len(values),
        calls=(
            CostCall(
                control_predicate_cost(values, legacy_shims=legacy_shims),
                bystander_qubits=target_width,
            ),
            CostCall(positive, bystander_qubits=2 * len(values) - 2),
            CostCall(
                control_predicate_cost(
                    values, uncompute=True, legacy_shims=legacy_shims
                ),
                bystander_qubits=target_width,
            ),
        ),
    )


def extended_controlled_component_cost(
    positive: ComponentCost,
    registers: tuple[ControlRegister, ...],
    *,
    current_control_value: int,
    target_width: int,
    legacy_shims: bool = False,
) -> ComponentCost:
    """Preserve single-bit conjunction or outer-plus-existing predicate levels."""
    values = tuple(bit for register in registers for bit in register.bits())
    if (
        not values
        or type(current_control_value) is not int
        or current_control_value not in (0, 1)
    ):
        raise ValueError(
            "Extended control model needs new controls and an existing binary activation"
        )
    conjunction = (values[0] if len(values) == 1 else 1, current_control_value)
    inner = controlled_component_cost(
        positive,
        (ControlRegister(1, conjunction),),
        target_width=target_width,
        legacy_shims=legacy_shims,
    )
    if len(values) == 1:
        return inner
    return controlled_component_cost(
        inner, registers, target_width=target_width + 1, legacy_shims=legacy_shims
    )


def controlled_pattern_resource_formula(
    positive_key: ResourceCall,
    positive: SymbolicResourceFormula,
    controls: ControlPattern,
    *,
    target_width: SymbolicInt,
    control_shapes: tuple[tuple[SymbolicInt, ...], ...] | None = None,
    legacy_shims: bool = False,
) -> SymbolicResourceFormula:
    """Compose a positive child model and original live control connections.

    target_width counts logical child ports other than the predicate, excluding
    private scratch. Child fallback estimates do not erase outer lifetimes.
    Concrete control shapes use live connections, including symbolic word widths.
    Symbolic shape dimensions use whole-wrapper fallback. If shapes are omitted,
    a scalar control count denotes one shaped bit array. No executable symbolic
    IR is implied.
    """
    values = controls
    if isinstance(values, tuple) and not values or type(legacy_shims) is not bool:
        raise ValueError(
            "Control formulas require nonempty controls and boolean profile"
        )
    if isinstance(values, tuple) and any(
        type(bit) is not int or bit not in (0, 1) for bit in values
    ):
        raise ValueError("Control pattern requires binary activation values")
    symbolic_shape = _control_shape_is_symbolic(values, control_shapes)
    # Validate target width through the authoritative connection contract even
    # when no predicate reduction is needed.
    boundary = (WidthStep("input"), WidthStep("output"))
    estimate_connection_width(boundary, (WidthConnection(0, 1, target_width),))
    if isinstance(values, tuple) and len(values) == 1:
        gates = QECGateCost().plus(positive, 1)
        if values == (0,):
            gates = gates.plus(QECGateCost(clifford=2), 1)
        return SymbolicResourceFormula(
            logical_qubits=positive.logical_qubits,
            **{f.name: getattr(gates, f.name) for f in fields(gates)},
        )
    compute_gates, compute_peak = _predicate_formula_cost(values, False, legacy_shims)
    erase_gates, erase_peak = _predicate_formula_cost(values, True, legacy_shims)
    count: SymbolicInt = len(values) if isinstance(values, tuple) else values
    gates = query_qec_gate_cost(
        PredicateControlledCall(positive_key, values),
        legacy_shims=legacy_shims,
        static_costs={
            positive_key: QECGateCost().plus(positive, 1),
            ControlPredicateCall(values): QECGateCost().plus(compute_gates, 1),
            ControlPredicateCall(values, adjoint=True): QECGateCost().plus(
                erase_gates, 1
            ),
        },
    )
    if symbolic_shape:
        # QubitCount folds callees in declaration order. Keep that order rather
        # than flattening a single Max: Sympy's reductions can be order-sensitive.
        peak = target_width + count
        for child_peak in (positive.logical_qubits, compute_peak, erase_peak):
            peak = cast(SymbolicInt, smax(peak, child_peak))
        return SymbolicResourceFormula(
            logical_qubits=peak,
            **{f.name: getattr(gates, f.name) for f in fields(gates)},
        )
    peak = estimate_connection_width(
        (
            WidthStep("input"),
            WidthStep("component", compute_peak),
            WidthStep("component", positive.logical_qubits),
            WidthStep("component", erase_peak),
            WidthStep("output"),
        ),
        (
            WidthConnection(0, 1, count),
            WidthConnection(0, 2, target_width),
            WidthConnection(1, 2, 1),
            WidthConnection(1, 3, 2 * count - 2),
            WidthConnection(2, 3, 1),
            WidthConnection(2, 4, target_width),
            WidthConnection(3, 4, count),
        ),
    )
    return SymbolicResourceFormula(
        logical_qubits=peak,
        **{f.name: getattr(gates, f.name) for f in fields(gates)},
    )


def controlled_component_resource_formula(
    positive_key: ResourceCall,
    positive: SymbolicResourceFormula,
    registers: tuple[ControlRegister, ...],
    *,
    target_width: SymbolicInt,
    legacy_shims: bool = False,
) -> SymbolicResourceFormula:
    """Bind concrete control-word descriptors to the shared pattern query."""
    values = tuple(bit for reg in registers for bit in reg.bits())
    return controlled_pattern_resource_formula(
        positive_key,
        positive,
        values,
        target_width=target_width,
        legacy_shims=legacy_shims,
    )


def _control_shape_is_symbolic(
    controls: ControlPattern,
    control_shapes: tuple[tuple[SymbolicInt, ...], ...] | None,
) -> bool:
    symbolic_shape = not isinstance(controls, tuple) and type(controls) is not int
    if control_shapes is not None:
        if not isinstance(control_shapes, tuple) or not control_shapes:
            raise ValueError(
                "Control shapes must be a nonempty tuple of dimension tuples"
            )
        symbolic_shape = False
        for shape in control_shapes:
            if not isinstance(shape, tuple):
                raise ValueError(
                    "Each control register shape must be an immutable tuple"
                )
            for dimension in shape:
                if type(dimension) is int:
                    if dimension < 0:
                        raise ValueError("Control shape dimensions must be nonnegative")
                else:
                    from sympy import Expr

                    if not isinstance(dimension, Expr):
                        raise ValueError(
                            "Control dimensions must be integers or symbolic expressions"
                        )
                    symbolic_shape = True
        if isinstance(controls, tuple) and symbolic_shape:
            raise ValueError("Known activation bits require concrete control shapes")
    return symbolic_shape


def _predicate_formula_cost(
    controls: ControlPattern,
    uncompute: bool,
    legacy_shims: bool,
) -> tuple[GateCategories, SymbolicInt]:
    if isinstance(controls, tuple):
        cost = control_predicate_cost(
            controls, uncompute=uncompute, legacy_shims=legacy_shims
        )
        return cost.gates, cost.qubits
    if controls == 1:
        raise ValueError("Unknown single-control polarity cannot be queried upstream")
    formula = control_predicate_resource_formula(
        controls, uncompute=uncompute, legacy_shims=legacy_shims
    )
    return formula, formula.logical_qubits
