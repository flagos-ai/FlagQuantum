# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 ControlledViaAnd and specialized_ctrl.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mcmt/controlled_via_and.py
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mcmt/specialized_ctrl.py

"""Compose concrete control models using authoritative predicate costs."""

from .._component_trace.resource_model import ComponentCost, CostCall, GateCost
from .control_predicate import ControlRegister
from .control_predicate_resource_model import control_predicate_cost


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
