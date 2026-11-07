# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 ControlledViaAnd resource declarations.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mcmt/controlled_via_and.py

"""Symbolic AddK estimates for concrete grouped activation words."""

from dataclasses import fields

from .._component_trace.arithmetic_call import ArithmeticCall
from .._component_trace.resource_generalization import ControlPattern
from .._component_trace.resource_model import estimate_component_resources
from .._component_trace.symbolic_math import SymbolicInt
from .constant_add_resource_model import source_controlled_constant_add_resource_formula
from .control_predicate import ControlRegister
from .controlled_component_resource_model import (
    controlled_component_resource_formula,
    controlled_pattern_resource_formula,
)
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
    return controlled_component_resource_formula(
        positive, addition, registers, target_width=width, legacy_shims=legacy_shims
    )


def source_pattern_controlled_constant_add_resource_formula(
    width: SymbolicInt,
    constant: SymbolicInt,
    controls: ControlPattern,
    *,
    adjoint: bool = False,
    legacy_shims: bool = False,
    control_shapes: tuple[tuple[SymbolicInt, ...], ...] | None = None,
) -> SymbolicResourceFormula:
    """Query original symbolic control-count fallback around positive AddK."""
    positive = ArithmeticCall(
        "controlled_constant_add", (width,), constant=constant, adjoint=adjoint
    )
    body = source_controlled_constant_add_resource_formula(
        width, constant, adjoint=adjoint, legacy_shims=legacy_shims
    )
    return controlled_pattern_resource_formula(
        positive,
        body,
        controls,
        target_width=width,
        legacy_shims=legacy_shims,
        control_shapes=control_shapes,
    )
