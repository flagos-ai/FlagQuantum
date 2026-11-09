# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 CModAdd.adjoint() and its source component chain.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mod_arithmetic/mod_addition.py
# Changes: native IR, exact component adjoints and existing shared work layout.

"""Literal controlled modular-addition adjoint, separate from source CModSub."""

from ....core.ir import Instruction
from ..arithmetic.addition import add_constant_adjoint_ir
from ..arithmetic.controlled_addition import (
    controlled_add_adjoint_ir,
    controlled_add_constant_adjoint_ir,
)
from ..comparison.controlled_comparison import controlled_greater_ir
from .controlled_modular_addition import _controlled_modular_workspace


def controlled_modular_add_adjoint_ir(
    control: int,
    left: tuple[int, ...],
    right: tuple[int, ...],
    modulus: int,
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
    *,
    control_value: int = 1,
) -> tuple[Instruction, ...]:
    """Subtract x modulo p using the literal source CModAdd adjoint components.

    Both coherent branches promise x in [0,p] and y in [0,p). Preserve original
    ports and work budgets; all work starts/ends zero. The comparator's exact
    source adjoint has the same palindromic component stream, tested separately.
    """
    n = len(left)
    sign, constant, carry, compare = _controlled_modular_workspace(
        control, left, right, modulus, workspace, classical_bits, control_value
    )
    load_bits, carry_bits = classical_bits[:n], classical_bits[n : 2 * n]
    return (
        Instruction("x", (sign,)),
        *controlled_greater_ir(
            control,
            left,
            right,
            sign,
            compare,
            classical_bits[: n + 2],
            control_value=control_value,
        ),
        *controlled_add_constant_adjoint_ir(
            sign, right, modulus, constant[:n], carry[: n - 1], carry_bits[: n - 1]
        ),
        *add_constant_adjoint_ir((sign, *right), -modulus, constant, carry, carry_bits),
        *controlled_add_adjoint_ir(
            control,
            left,
            (sign, *right),
            constant[:n],
            carry,
            load_bits,
            carry_bits,
            control_value=control_value,
        ),
    )
