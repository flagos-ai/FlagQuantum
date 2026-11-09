# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 _KaliskiIterationStep2.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mod_arithmetic/mod_division.py
# Changes: explicit big-endian physical registers and shared control work pool.

"""Parity and branch flags for the original modular inverse iteration."""

from .....core.ir import Instruction
from ...control.multi_control import multi_control_x_ir


def inverse_parity_ir(
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
    """Update logical parity/marker/branch data and return shared work to zero.

    Equal-width positive big-endian words are preserved. Supply two clean work
    qubits and two reusable classical slots; three measurements occur. All flags
    may be coherent. The marker update uses the already-updated parity bit.
    The adjoint restores flags by reversing original components, not primitives.
    """
    if not left or len(left) != len(right) or type(adjoint) is not bool:
        raise ValueError("Parity requires equal nonempty words and boolean adjoint")
    wires = (*left, *right, branch, parity, marker, active, *workspace)
    if any(type(w) is not int or w < 0 for w in wires) or len(set(wires)) != len(wires):
        raise ValueError("Parity requires distinct nonnegative quantum wires")
    if len(workspace) != 2 or len(classical_bits) != 2:
        raise ValueError("Parity requires two clean work wires and two classical slots")
    first = multi_control_x_ir(
        (active, left[-1]), (1, 0), parity, workspace[:1], classical_bits[:1]
    )
    second = multi_control_x_ir(
        (active, right[-1], parity), (1, 0, 0), marker, workspace, classical_bits
    )
    updates = (Instruction("cx", (parity, branch)), Instruction("cx", (marker, branch)))
    return (
        (*reversed(updates), *second, *first)
        if adjoint
        else (*first, *second, *updates)
    )
