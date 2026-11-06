# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 TwoBitFFFT.build_composite_bloq.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/qft/two_bit_ffft.py

"""Two-qubit fermionic Fourier transform with the source phase convention."""

from math import pi

from ...compiler.registers.bitwise import _validate_wires
from ...core.ir import Instruction

_ADJOINT = {"s": "sdg", "t": "tdg", "tdg": "t"}


def two_bit_fermionic_fourier_ir(
    x: int, y: int, k: int, n: int, *, adjoint: bool = False
) -> tuple[Instruction, ...]:
    """Bind the two-qubit fermionic Fourier transform and its exact inverse."""
    _validate_wires((x, y))
    if type(k) is not int or type(n) is not int or n == 0:
        raise ValueError("Fermionic Fourier transform requires integer k and nonzero n")
    if type(adjoint) is not bool:
        raise ValueError("Fermionic Fourier transform direction must be boolean")

    angle = 2 * pi * k / n
    fixed = (
        Instruction("s", (y,)),
        Instruction("h", (x,)),
        Instruction("h", (y,)),
        Instruction("cx", (x, y)),
        Instruction("h", (x,)),
        Instruction("s", (x,)),
        Instruction("tdg", (y,)),
        Instruction("cx", (x, y)),
        Instruction("h", (x,)),
        Instruction("t", (y,)),
        Instruction("cx", (x, y)),
        Instruction("h", (x,)),
        Instruction("h", (y,)),
        Instruction("s", (x,)),
    )
    if adjoint:
        return (
            *(
                Instruction(_ADJOINT.get(g.name, g.name), g.wires)
                for g in reversed(fixed)
            ),
            Instruction("rz", (x,), {"theta": -angle}),
        )
    return (Instruction("rz", (x,), {"theta": angle}), *fixed)
