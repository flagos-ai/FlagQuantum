# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 ModNeg.build_composite_bloq.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mod_arithmetic/mod_subtraction.py
# Changes: native IR and explicit work, sharing original reduction/fanout/AddK.

"""Modular negation through the original zero-detection and correction stages."""

from ....core.ir import Instruction
from ..arithmetic.controlled_addition import (
    controlled_add_constant_adjoint_ir,
    controlled_add_constant_ir,
)
from ..bits.bitwise import _validate_wires, fanout_ir
from ..control.multi_control import multi_control_x_ir


def modular_negation_requirements(width: int) -> tuple[int, int]:
    """Reserve a flag, loaded constant, and reusable reduction/carry qubits."""
    if type(width) is not int or width < 1:
        raise ValueError("Modular negation requires a positive concrete width")
    return 2 * width, width - 1


def _negation_layout(
    target: tuple[int, ...],
    modulus: int,
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
    *,
    control: int | None = None,
) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
    """Validate shared unary modular roles, retaining an optional outer apply bit."""
    n = len(target)
    q, c = modular_negation_requirements(n)
    controlled = control is not None
    if controlled:
        q, c = q + 1, max(1, c)
    _validate_wires((*target, *workspace, *((control,) if control is not None else ())))
    if type(modulus) is not int or not 0 < modulus < (1 << n) - 1:
        raise ValueError("Modular negation requires modulus+1 to fit the operand")
    if len(workspace) != q:
        raise ValueError(f"Modular negation requires {q} clean work wires")
    if (
        len(classical_bits) != c
        or len(set(classical_bits)) != c
        or any(type(bit) is not int or bit < 0 for bit in classical_bits)
    ):
        raise ValueError(
            f"Modular negation requires {c} distinct nonnegative classical slots"
        )
    start = 1 + int(controlled)
    return workspace[0], workspace[start : start + n], workspace[start + n :]


def modular_negate_ir(
    target: tuple[int, ...],
    modulus: int,
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
) -> tuple[Instruction, ...]:
    """Map x to -x modulo p on 0<=x<p, preserving every pinned source stage.

    Target is big-endian; all caller-owned work starts/ends zero. p+1 fits n
    bits. Input support outside the residue promise is not inspected by compiler.
    """
    return _modular_negation_ir(
        target, modulus, workspace, classical_bits, adjoint=False
    )


def modular_negate_adjoint_ir(
    target: tuple[int, ...],
    modulus: int,
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
) -> tuple[Instruction, ...]:
    """Negate a valid residue using the literal original ModNeg adjoint stream.

    Big-endian x lies in [0,p), p+1 fits n bits. Quantum work starts/ends zero;
    classical slots retain latest outcomes. Although logically self-inverse,
    the source inverse applies AddK-adjoint before its conditional fanout.
    """
    return _modular_negation_ir(
        target, modulus, workspace, classical_bits, adjoint=True
    )


def _modular_negation_ir(
    target: tuple[int, ...],
    modulus: int,
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
    *,
    adjoint: bool,
) -> tuple[Instruction, ...]:
    flag, constant, carry = _negation_layout(target, modulus, workspace, classical_bits)
    zero = multi_control_x_ir(target, (0,) * len(target), flag, carry, classical_bits)
    emit = controlled_add_constant_adjoint_ir if adjoint else controlled_add_constant_ir
    add = emit(flag, target, modulus + 1, constant, carry, classical_bits)
    fanout = fanout_ir(flag, target)
    middle = (*add, *fanout) if adjoint else (*fanout, *add)
    return (
        Instruction("x", (flag,)),
        *zero,
        *middle,
        *zero,
        Instruction("x", (flag,)),
    )
