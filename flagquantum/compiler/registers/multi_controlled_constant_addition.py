# Copyright 2024 Google LLC
# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 ControlledViaAnd(AddK).
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mcmt/controlled_via_and.py

"""Multi-bit activation around shared constant addition, with explicit clean work."""

from ...core.ir import Instruction
from .._component_trace.resource_model import ComponentCost
from .bitwise import _validate_wires
from .constant_add_resource_model import source_controlled_constant_add_cost
from .control_predicate import ControlRegister
from .controlled_addition import (
    controlled_add_constant_adjoint_ir,
    controlled_add_constant_ir,
)
from .controlled_component import controlled_component_ir
from .controlled_component_resource_model import controlled_component_cost


def multi_controlled_add_constant_ir(
    controls: tuple[int, ...],
    values: tuple[int, ...],
    target: tuple[int, ...],
    value: int,
    predicate_work: tuple[int, ...],
    constant: tuple[int, ...],
    carries: tuple[int, ...],
    addition_bits: tuple[int, ...],
    predicate_bits: tuple[int, ...],
    *,
    adjoint: bool = False,
) -> tuple[Instruction, ...]:
    """Compute the control equality, apply AddK or its inverse, and erase equality.

    For m>=2 controls and n target bits, supply m-1 predicate wires (junk first,
    final predicate last), n constant wires and n-1 carries, all initially zero.
    Both measured phases own distinct classical slots. Only the target changes.
    """
    if len(controls) < 2 or len(predicate_work) != len(controls) - 1:
        raise ValueError("Multi-controlled AddK requires m>=2 and m-1 predicate wires")
    if type(adjoint) is not bool:
        raise ValueError("Constant-add direction must be boolean")
    _validate_wires((*controls, *target, *predicate_work, *constant, *carries))
    slots = (*addition_bits, *predicate_bits)
    if len(set(slots)) != len(slots) or any(type(b) is not int or b < 0 for b in slots):
        raise ValueError("Addition and predicate measurements require distinct slots")
    registers = (ControlRegister(1, values),)
    groups = (controls,)
    emitter = (
        controlled_add_constant_adjoint_ir if adjoint else controlled_add_constant_ir
    )

    def body(predicate: int) -> tuple[Instruction, ...]:
        return emitter(predicate, target, value, constant, carries, addition_bits)

    return controlled_component_ir(
        groups, registers, body, predicate_work, predicate_bits
    )


def source_multi_controlled_constant_add_cost(
    width: int,
    value: int,
    values: tuple[int, ...],
    *,
    adjoint: bool = False,
    legacy_shims: bool = False,
) -> ComponentCost:
    """Compose the source predicate/positive AddK/erasure resource lifetimes."""
    if len(values) < 2 or type(adjoint) is not bool:
        raise ValueError(
            "Multi-controlled AddK requires at least two controls and boolean direction"
        )
    addition = source_controlled_constant_add_cost(
        width, value, adjoint=adjoint, legacy_shims=legacy_shims
    )
    return controlled_component_cost(
        addition,
        (ControlRegister(1, values),),
        target_width=width,
        legacy_shims=legacy_shims,
    )
