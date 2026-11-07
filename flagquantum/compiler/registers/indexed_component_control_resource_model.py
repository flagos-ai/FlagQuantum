# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 specialized_ctrl and ControlledViaAnd call graphs.
# https://github.com/quantumlib/Qualtran/tree/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mcmt

"""Compose indexed control-system models without constructing instructions."""

from dataclasses import fields

from .._component_trace.qec_gate_cost import query_qec_gate_cost
from .._component_trace.resource_generalization import ControlPredicateCall
from .._component_trace.resource_model import ComponentCost, CostCall, GateCost
from .control_predicate import ControlRegister
from .indexed_component_resource_model import indexed_component_cost


def specialized_indexed_component_cost(
    axis_widths: tuple[int, ...],
    extents: tuple[int, ...],
    children: tuple[ComponentCost, ...],
    registers: tuple[ControlRegister, ...],
    *,
    target_width: int,
    current_control_value: int | None = None,
    adjoint: bool = False,
) -> ComponentCost:
    """Retain the two source predicate levels and the existing cost authority.

    Children include their positive predicate and all target ports. The caller
    supplies child-adjoint models for reverse traversal. As in source, solitary
    negative specialization has the same model as its positive counterpart.
    """
    values = tuple(bit for register in registers for bit in register.bits())
    if not values:
        raise ValueError("Indexed control model requires new controls")
    if current_control_value is not None and (
        type(current_control_value) is not int or current_control_value not in (0, 1)
    ):
        raise ValueError("Existing control model requires zero or one")
    child = indexed_component_cost(
        axis_widths,
        extents,
        children,
        target_width=target_width,
        control_count=1,
        adjoint=adjoint,
    )
    patterns = []
    if len(values) > 1:
        patterns.append(values)
    if current_control_value is not None:
        patterns.append((values[0] if len(values) == 1 else 1, current_control_value))
    if not patterns:
        return child
    gates = GateCost()
    work = 0
    for pattern in patterns:
        work += len(pattern) - 1
        for reverse in (False, True):
            cost = query_qec_gate_cost(ControlPredicateCall(pattern, reverse))
            gates = gates.plus(
                GateCost(
                    **{
                        field.name: int(getattr(cost, field.name))
                        for field in fields(GateCost)
                    }
                ),
                1,
            )
    bystanders = len(values) + int(current_control_value is not None) + work - 1
    return ComponentCost(
        "SpecializedIndexedComponent†" if adjoint else "SpecializedIndexedComponent",
        len(values)
        + int(current_control_value is not None)
        + sum(axis_widths)
        + target_width
        + work,
        gates,
        (CostCall(child, bystander_qubits=bystanders),),
    )
