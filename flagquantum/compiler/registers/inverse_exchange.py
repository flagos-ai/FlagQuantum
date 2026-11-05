# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 _KaliskiIterationStep4.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mod_arithmetic/mod_division.py
# Changes: physical native words and shared exact controlled-swap emission.

"""Conditional word/coefficient exchanges for the modular inverse iteration."""

from ...core.ir import Instruction
from .bitwise import _validate_wires
from .controlled_swap import controlled_swap_ir


def inverse_exchange_ir(
    control: int,
    left: tuple[int, ...],
    right: tuple[int, ...],
    left_coefficient: tuple[int, ...],
    right_coefficient: tuple[int, ...],
    *,
    adjoint: bool = False,
) -> tuple[Instruction, ...]:
    """Exchange words and coefficients on coherent control, without any work.

    Four equal-width positive big-endian registers and control are preserved
    except the two swaps. Emit exact seven-T Fredkin primitives. The source
    adjoint reverses the two original CSwap components before decomposition.
    """
    n = len(left)
    if (
        not n
        or any(len(word) != n for word in (right, left_coefficient, right_coefficient))
        or type(adjoint) is not bool
    ):
        raise ValueError(
            "Inverse exchange requires four equal nonempty words and boolean adjoint"
        )
    _validate_wires((control, *left, *right, *left_coefficient, *right_coefficient))
    words = controlled_swap_ir(control, left, right)
    coefficients = controlled_swap_ir(control, left_coefficient, right_coefficient)
    return (*coefficients, *words) if adjoint else (*words, *coefficients)
