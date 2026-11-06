# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 _KaliskiIterationStep5.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mod_arithmetic/mod_division.py
# Changes: explicit native work pool and reuse of exact source-controlled arithmetic.

"""Conditional subtract/add updates for the original modular inverse iteration."""

from ...core.ir import Instruction
from .and_compute import and_compute_ir, and_uncompute_ir
from .bitwise import _validate_wires, bitwise_not_ir
from .controlled_addition import controlled_add_adjoint_ir, controlled_add_ir


def inverse_arithmetic_ir(
    left: tuple[int, ...],
    right: tuple[int, ...],
    left_coefficient: tuple[int, ...],
    right_coefficient: tuple[int, ...],
    branch: int,
    active: int,
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
    *,
    adjoint: bool = False,
) -> tuple[Instruction, ...]:
    """On active & ~branch, update right-=left and right_coefficient+=left_coefficient.

    Four equal-width positive big-endian words are bit patterns modulo 2**n.
    The source classical model additionally requires no arithmetic overflow.
    Supply 2n clean work qubits and 2n-1 reusable classical slots; 4n-1 measured
    erasures occur. Work and all inputs except the two target words are restored.
    """
    n = len(left)
    if (
        not n
        or any(len(x) != n for x in (right, left_coefficient, right_coefficient))
        or type(adjoint) is not bool
    ):
        raise ValueError(
            "Inverse arithmetic requires four equal nonempty words and boolean adjoint"
        )
    _validate_wires(
        (
            *left,
            *right,
            *left_coefficient,
            *right_coefficient,
            branch,
            active,
            *workspace,
        )
    )
    if len(workspace) != 2 * n:
        raise ValueError("Inverse arithmetic requires 2n clean work wires")
    if (
        len(classical_bits) != 2 * n - 1
        or len(set(classical_bits)) != len(classical_bits)
        or any(type(b) is not int or b < 0 for b in classical_bits)
    ):
        raise ValueError(
            "Inverse arithmetic requires 2n-1 distinct nonnegative classical slots"
        )
    control, loaded, carries = workspace[0], workspace[1 : n + 1], workspace[n + 1 :]
    emit = controlled_add_adjoint_ir if adjoint else controlled_add_ir
    subtract = (
        *bitwise_not_ir(right),
        *emit(
            control,
            left,
            right,
            loaded,
            carries,
            classical_bits[:n],
            classical_bits[n:],
        ),
        *bitwise_not_ir(right),
    )
    add = emit(
        control,
        left_coefficient,
        right_coefficient,
        loaded,
        carries,
        classical_bits[:n],
        classical_bits[n:],
    )
    stages = (*add, *subtract) if adjoint else (*subtract, *add)
    return (
        *and_compute_ir(active, branch, control, values=(1, 0)),
        *stages,
        *and_uncompute_ir(active, branch, control, classical_bits[0], values=(1, 0)),
    )
