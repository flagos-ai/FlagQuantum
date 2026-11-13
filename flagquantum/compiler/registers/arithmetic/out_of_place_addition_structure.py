# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 OutOfPlaceAdder.decompose_from_registers.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/addition.py

"""Share the literal sum components between emission and lifetime analysis."""

from .addition_structure import AdditionComponent


def out_of_place_addition_components(
    left: tuple[int, ...],
    right: tuple[int, ...],
    output: tuple[int, ...],
    *,
    include_msb: bool = True,
    adjoint: bool = False,
) -> tuple[AdditionComponent, ...]:
    """Return source-order components over explicit big-endian words.

    Callers validate word shapes. Forward AND creates its target output bit.
    Adjoint AND consumes that bit before its measured primitive expansion.
    """
    a, b, c = left[::-1], right[::-1], output[::-1]
    steps: list[AdditionComponent] = []
    for i in range(len(c) - 1):
        steps.extend(
            (
                ("cx", (a[i], b[i])),
                ("cx", (a[i], c[i])),
                ("and_adjoint" if adjoint else "and", (b[i], c[i], c[i + 1])),
                ("cx", (a[i], b[i])),
                ("cx", (a[i], c[i + 1])),
                ("cx", (b[i], c[i])),
            )
        )
    if not include_msb:
        steps.extend((("cx", (a[-1], c[-1])), ("cx", (b[-1], c[-1]))))
    return tuple(reversed(steps)) if adjoint else tuple(steps)
