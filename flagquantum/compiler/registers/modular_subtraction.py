# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 ModSub.build_composite_bloq.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mod_arithmetic/mod_subtraction.py
# Changes: native IR and reuse shared modular/addition/adjoint components.

"""Modular subtraction with the literal source constant-adder adjoint."""

from ...core.ir import Instruction
from .arithmetic import add_constant_adjoint_ir, add_constant_ir
from .bitwise import bitwise_not_ir
from .modular_addition import _modular_workspace, modular_add_adjoint_ir, modular_add_ir


def modular_subtract_ir(
    left: tuple[int, ...],
    right: tuple[int, ...],
    modulus: int,
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
) -> tuple[Instruction, ...]:
    """Preserve x and update y=(y-x)%p, for x/y in [0,p), clearing all work.

    Equal positive-width registers are big-endian. p+1 must fit n bits, as in
    the executable source's AddK load. The source complement/constant/modular
    add/constant-adjoint/complement sequence is retained, not simplified.
    """
    return _modular_subtraction_ir(
        left, right, modulus, workspace, classical_bits, adjoint=False
    )


def modular_subtract_adjoint_ir(
    left: tuple[int, ...],
    right: tuple[int, ...],
    modulus: int,
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
) -> tuple[Instruction, ...]:
    """Add x modulo p through the original ModSub adjoint conjugation chain.

    Big-endian x,y are residues in [0,p); p+1 fits n bits. Retain original
    complement/load and exact inverse-load/complement around ModAdd-adjoint.
    Quantum work starts/ends zero; classical slots retain latest outcomes.
    """
    return _modular_subtraction_ir(
        left, right, modulus, workspace, classical_bits, adjoint=True
    )


def _modular_subtraction_ir(
    left: tuple[int, ...],
    right: tuple[int, ...],
    modulus: int,
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
    *,
    adjoint: bool,
) -> tuple[Instruction, ...]:
    _, _, constant, carry, _ = _modular_workspace(
        left, right, modulus, workspace, classical_bits
    )
    n = len(left)
    if modulus + 1 >= 1 << n:
        raise ValueError("Modular subtraction requires modulus+1 to fit the operand")
    load, prefix_carry, bits = constant[:n], carry[: n - 1], classical_bits[: n - 1]
    emit = modular_add_adjoint_ir if adjoint else modular_add_ir
    complement = bitwise_not_ir(left)
    return (
        *complement,
        *add_constant_ir(left, modulus + 1, load, prefix_carry, bits),
        *emit(left, right, modulus, workspace, classical_bits),
        *add_constant_adjoint_ir(left, modulus + 1, load, prefix_carry, bits),
        *complement,
    )
