# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 SwapWithZero.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/swap_network/swap_with_zero.py

"""Route the indexed word into row zero with relative-phase exchanges."""

from math import prod

from ...core.ir import Instruction
from .bitwise import _validate_wires
from .relative_controlled_swap import relative_controlled_swap_ir


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
    order = reversed(exchanges) if adjoint else iter(exchanges)
    return tuple(
        gate
        for control, left, right in order
        for gate in relative_controlled_swap_ir(control, targets[left], targets[right])
    )
