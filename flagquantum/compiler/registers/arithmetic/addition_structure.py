# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 Add.decompose_from_registers.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/addition.py

"""One literal logical Add sequence for graph construction and gate emission."""

from typing import Literal

AdditionComponent = tuple[Literal["cx", "and", "and_adjoint"], tuple[int, ...]]


def addition_components(
    left: tuple[int, ...],
    right: tuple[int, ...],
    carries: tuple[int, ...],
    *,
    adjoint: bool = False,
) -> tuple[AdditionComponent, ...]:
    """Return source-order components over explicit MSB-first wire identities.

    Carry identities are lower-bit first. Invert abstract components before
    their measured decomposition. The sequence owns no allocation or numerics.
    """
    width = len(right)
    if not 1 <= len(left) <= width or len(carries) != width - 1:
        raise ValueError(
            "Add requires a nonempty input no wider than the output and one clean carry per lower bit"
        )
    a, b = left[::-1], right[::-1]
    steps: list[AdditionComponent] = []

    def cx(control: int, target: int) -> None:
        steps.append(("cx", (control, target)))

    def logical_and(x: int, y: int, i: int, *, erase: bool = False) -> None:
        steps.append(("and_adjoint" if erase ^ adjoint else "and", (x, y, carries[i])))

    if width == 1:
        cx(a[0], b[0])
        return tuple(steps)

    logical_and(a[0], b[0], 0)
    for depth in range(1, width - 1):
        previous = carries[depth - 1]
        control = previous
        if depth < len(a):
            control = a[depth]
            cx(previous, control)
        cx(previous, b[depth])
        logical_and(control, b[depth], depth)
        cx(previous, carries[depth])
    cx(carries[-1], b[-1])
    if len(a) == width:
        cx(a[-1], b[-1])
    for depth in range(width - 2, 0, -1):
        previous = carries[depth - 1]
        cx(previous, carries[depth])
        control = a[depth] if depth < len(a) else previous
        logical_and(control, b[depth], depth, erase=True)
        if depth < len(a):
            cx(previous, control)
            cx(control, b[depth])
    logical_and(a[0], b[0], 0, erase=True)
    cx(a[0], b[0])
    return tuple(reversed(steps)) if adjoint else tuple(steps)
