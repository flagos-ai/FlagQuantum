# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 CModSub.build_composite_bloq.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mod_arithmetic/mod_subtraction.py
# Changes: native IR and shared controlled modular/exact-adjoint components.

"""Controlled modular subtraction with unconditional source conjugation stages."""

from ....core.ir import Instruction
from ..arithmetic.addition import add_constant_adjoint_ir, add_constant_ir
from ..bits.bitwise import bitwise_not_ir
from .controlled_modular_addition import (
    _controlled_modular_workspace,
    controlled_modular_add_ir,
)
from .controlled_modular_addition_adjoint import controlled_modular_add_adjoint_ir


def controlled_modular_subtract_ir(
    control: int,
    left: tuple[int, ...],
    right: tuple[int, ...],
    modulus: int,
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
    *,
    control_value: int = 1,
) -> tuple[Instruction, ...]:
    """Subtract x from y modulo p on the selected coherent control branch.

    Inputs are equal-width big-endian residues in [0,p) in both branches.
    p+1 fits n bits. Preserve the source's unconditional complement/load and
    literal inverse-load/complement stages, including for inactive controls.
    """
    return _controlled_modular_subtraction_ir(
        control,
        left,
        right,
        modulus,
        workspace,
        classical_bits,
        control_value=control_value,
        adjoint=False,
    )


def controlled_modular_subtract_adjoint_ir(
    control: int,
    left: tuple[int, ...],
    right: tuple[int, ...],
    modulus: int,
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
    *,
    control_value: int = 1,
) -> tuple[Instruction, ...]:
    """Add x modulo p using the original CModSub adjoint component chain.

    Both branches require x,y in [0,p), p+1 fitting n bits. Keep the original
    complement and AddK(p+1) conjugations, replacing only the middle CModAdd
    by its exact component adjoint. Quantum work starts/ends zero; classical
    slots retain latest outcomes. This is distinct from source CModAdd.
    """
    return _controlled_modular_subtraction_ir(
        control,
        left,
        right,
        modulus,
        workspace,
        classical_bits,
        control_value=control_value,
        adjoint=True,
    )


def _controlled_modular_subtraction_ir(
    control: int,
    left: tuple[int, ...],
    right: tuple[int, ...],
    modulus: int,
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
    *,
    control_value: int,
    adjoint: bool,
) -> tuple[Instruction, ...]:
    n = len(left)
    _, constant, carry, _ = _controlled_modular_workspace(
        control, left, right, modulus, workspace, classical_bits, control_value
    )
    if modulus + 1 >= 1 << n:
        raise ValueError(
            "Controlled modular subtraction requires modulus+1 to fit the operand"
        )
    load, prefix_carry, bits = (
        constant[:n],
        carry[: n - 1],
        classical_bits[n : 2 * n - 1],
    )
    middle = controlled_modular_add_adjoint_ir if adjoint else controlled_modular_add_ir
    complement = bitwise_not_ir(left)
    return (
        *complement,
        *add_constant_ir(left, modulus + 1, load, prefix_carry, bits),
        *middle(
            control,
            left,
            right,
            modulus,
            workspace,
            classical_bits,
            control_value=control_value,
        ),
        *add_constant_adjoint_ir(left, modulus + 1, load, prefix_carry, bits),
        *complement,
    )
