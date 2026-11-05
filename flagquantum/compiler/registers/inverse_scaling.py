# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 _KaliskiIterationStep6.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mod_arithmetic/mod_division.py
# Changes: physical source output aliases, exact shared arithmetic and swap stages.

"""Cyclic division and modular coefficient scaling with original output roles."""

from ...core.ir import Instruction
from .bitwise import _validate_wires
from .controlled_swap import controlled_swap_bit_ir
from .inverse_exchange import inverse_exchange_ir
from .modular_doubling import modular_double_ir, modular_double_outputs
from .modular_halving import modular_half_ir, modular_half_outputs


def inverse_scaling_outputs(
    coefficient: tuple[int, ...],
    workspace: tuple[int, ...],
    *,
    adjoint: bool = False,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Resolve this layer's changed coefficient/clean-work roles for its consumers.

    Other logical word/flag wire tuples are unchanged. Read these returned
    roles before composing the next stage or the original adjoint. There is no
    materializing SWAP. The inverse consumes the actual forward output roles.
    """
    if type(adjoint) is not bool:
        raise ValueError("Inverse scaling adjoint must be boolean")
    emit = modular_half_outputs if adjoint else modular_double_outputs
    return emit(coefficient, workspace)


def inverse_scaling_ir(
    left: tuple[int, ...],
    right: tuple[int, ...],
    left_coefficient: tuple[int, ...],
    right_coefficient: tuple[int, ...],
    branch: int,
    parity: int,
    marker: int,
    active: int,
    modulus: int,
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
    *,
    adjoint: bool = False,
) -> tuple[Instruction, ...]:
    """Emit original scaling/exchange stages, retaining physical output aliases.

    Four equal positive-width words are big-endian. Forward coefficient is a
    valid residue modulo fitting odd modulus. Its active right word is even
    for agreement with the source's classical division; gates always implement
    a reversible cyclic shift. Supply 2n+5 clean work and n+1 classical slots.
    All work is returned zero on inverse_scaling_outputs' roles. The inverse
    requires a complete valid forward image, not independently chosen targets.
    """
    n = len(left)
    if (
        not n
        or any(len(x) != n for x in (right, left_coefficient, right_coefficient))
        or type(adjoint) is not bool
    ):
        raise ValueError(
            "Inverse scaling requires four equal nonempty words and boolean adjoint"
        )
    _validate_wires(
        (
            *left,
            *right,
            *left_coefficient,
            *right_coefficient,
            branch,
            parity,
            marker,
            active,
            *workspace,
        )
    )
    coefficient, _ = inverse_scaling_outputs(
        left_coefficient, workspace, adjoint=adjoint
    )
    updates = (Instruction("cx", (marker, branch)), Instruction("cx", (parity, branch)))
    parity_update = (
        Instruction("x", (right_coefficient[-1],)),
        Instruction("cx", (right_coefficient[-1], parity)),
        Instruction("x", (right_coefficient[-1],)),
    )
    rotation = tuple(
        g
        for i in (range(n - 1) if adjoint else reversed(range(n - 1)))
        for g in controlled_swap_bit_ir(active, right[i], right[i + 1])
    )
    if adjoint:
        return (
            *parity_update,
            *inverse_exchange_ir(
                parity, left, right, left_coefficient, right_coefficient, adjoint=True
            ),
            *modular_half_ir(left_coefficient, modulus, workspace, classical_bits),
            *rotation,
            *reversed(updates),
        )
    return (
        *updates,
        *rotation,
        *modular_double_ir(left_coefficient, modulus, workspace, classical_bits),
        *inverse_exchange_ir(parity, left, right, coefficient, right_coefficient),
        *parity_update,
    )
