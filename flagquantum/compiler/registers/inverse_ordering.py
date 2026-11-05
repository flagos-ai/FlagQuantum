# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 _KaliskiIterationStep3.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mod_arithmetic/mod_division.py
# Changes: native IR and explicit clean work reuse at original component boundaries.

"""Conditional ordering flags for the original modular inverse iteration."""

from ...core.ir import Instruction
from .half_comparison import half_greater_ir
from .multi_control import multi_and_ir


def inverse_ordering_ir(
    left: tuple[int, ...],
    right: tuple[int, ...],
    branch: int,
    parity: int,
    marker: int,
    active: int,
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
    *,
    adjoint: bool = False,
) -> tuple[Instruction, ...]:
    """XOR (left>right)&active&~branch into parity and marker; clear all work.

    Equal-width positive big-endian words and logical flags may be coherent.
    Supply n+4 clean work wires and max(n,2) distinct classical slots. The two
    comparison sign wires are reused as And outputs only at clean boundaries.
    There are n+2 measurements. All inputs except parity/marker are restored.
    """
    n = len(left)
    if n < 1 or len(right) != n or type(adjoint) is not bool:
        raise ValueError("Ordering requires equal nonempty words and boolean adjoint")
    wires = (*left, *right, branch, parity, marker, active, *workspace)
    if any(type(w) is not int or w < 0 for w in wires) or len(set(wires)) != len(wires):
        raise ValueError("Ordering requires distinct nonnegative quantum wires")
    if len(workspace) != n + 4:
        raise ValueError("Ordering requires n+4 clean work wires")
    if (
        len(classical_bits) != max(n, 2)
        or len(set(classical_bits)) != len(classical_bits)
        or any(type(b) is not int or b < 0 for b in classical_bits)
    ):
        raise ValueError(
            "Ordering requires max(n,2) distinct nonnegative classical slots"
        )
    signs, difference, greater = workspace[:2], workspace[2:-1], workspace[-1]
    controls = (greater, active, branch)
    copies = (
        Instruction("cx", (signs[-1], parity)),
        Instruction("cx", (signs[-1], marker)),
    )
    return (
        *half_greater_ir(left, right, difference, greater, signs),
        *multi_and_ir(controls, (1, 1, 0), signs),
        *(tuple(reversed(copies)) if adjoint else copies),
        *multi_and_ir(
            controls,
            (1, 1, 0),
            signs,
            classical_bits=classical_bits[:2],
            uncompute=True,
        ),
        *half_greater_ir(
            left,
            right,
            difference,
            greater,
            signs,
            adjoint=True,
            classical_bits=classical_bits[:n],
        ),
    )
