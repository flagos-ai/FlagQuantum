# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 SwapWithZero.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/swap_network/swap_with_zero.py

"""Route the indexed word into row zero with relative-phase exchanges."""

from math import prod

from ....core.ir import Instruction
from ..bits.bitwise import _validate_wires
from ..control.control_predicate import ControlRegister
from ..control.controlled_component import controlled_component_ir
from ..primitives.relative_controlled_swap import (
    control_relative_swap_ir,
    relative_controlled_swap_ir,
)


def swap_with_zero_ir(
    selections: tuple[tuple[int, ...], ...],
    shape: tuple[int, ...],
    targets: tuple[tuple[int, ...], ...],
    *,
    adjoint: bool = False,
) -> tuple[Instruction, ...]:
    """Emit the source network on row-major words and big-endian selections.

    The selected row moves to zero for in-range selections. The inverse reverses
    the network and keeps the source phase of each self-adjoint exchange.
    """
    return tuple(
        gate
        for control, left, right in _swap_sequence(selections, shape, targets, adjoint)
        for gate in relative_controlled_swap_ir(control, targets[left], targets[right])
    )


def control_swap_with_zero_ir(
    controls: tuple[tuple[int, ...], ...],
    registers: tuple[ControlRegister, ...],
    selections: tuple[tuple[int, ...], ...],
    shape: tuple[int, ...],
    targets: tuple[tuple[int, ...], ...],
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
    *,
    adjoint: bool = False,
) -> tuple[Instruction, ...]:
    """Apply one shared outer predicate to the source exchange network."""
    order = _swap_sequence(selections, shape, targets, adjoint)
    _validate_wires(
        (
            *workspace,
            *(wire for word in (*controls, *selections, *targets) for wire in word),
        )
    )

    def emit(control: int) -> tuple[Instruction, ...]:
        return tuple(
            gate
            for select, left, right in order
            for gate in control_relative_swap_ir(
                ((control,),),
                (ControlRegister(1, (1,)),),
                select,
                targets[left],
                targets[right],
                (),
                (),
            )
        )

    return controlled_component_ir(controls, registers, emit, workspace, classical_bits)


def _swap_sequence(
    selections: tuple[tuple[int, ...], ...],
    shape: tuple[int, ...],
    targets: tuple[tuple[int, ...], ...],
    adjoint: bool,
) -> list[tuple[int, int, int]]:
    if not shape or len(selections) != len(shape) or type(adjoint) is not bool:
        raise ValueError("Swap with zero requires matching axes and boolean adjoint")
    empty_selection = False
    for wires, extent in zip(selections, shape, strict=True):
        if type(extent) is not int or not 1 <= extent <= 1 << len(wires):
            raise ValueError("Swap with zero extent must fit its selection")
        if empty_selection and extent > 1:
            raise ValueError("Source swap with zero omits earlier zero-bit selections")
        empty_selection |= not wires
    if len(targets) != prod(shape) or not targets[0]:
        raise ValueError("Swap with zero requires a nonempty word per target row")
    if any(len(word) != len(targets[0]) for word in targets):
        raise ValueError("Swap with zero target words must have equal widths")
    _validate_wires(tuple(wire for word in (*selections, *targets) for wire in word))

    exchanges: list[tuple[int, int, int]] = []

    def visit(axis: int, start: int) -> None:
        if axis == len(shape):
            return
        stride = prod(shape[axis + 1 :])
        extent = shape[axis]
        for row in range(extent):
            visit(axis + 1, start + row * stride)
        for bit, control in enumerate(reversed(selections[axis])):
            half = 1 << bit
            for row in range(0, extent - half, 2 * half):
                exchanges.append(
                    (control, start + row * stride, start + (row + half) * stride)
                )

    visit(0, 0)
    if adjoint:
        exchanges.reverse()
    return exchanges
