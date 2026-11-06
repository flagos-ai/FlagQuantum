# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from pinned ModDbl.adjoint() and its original component decomposition.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mod_arithmetic/mod_multiplication.py
# Changes: explicit inverse port aliases and exact shared arithmetic adjoints.

"""Exact source modular-doubling adjoint without reversing measured primitives."""

from ...core.ir import Instruction
from .arithmetic import add_constant_adjoint_ir
from .controlled_addition import controlled_add_constant_adjoint_ir
from .modular_doubling import _double_layout, _double_ports


def modular_half_outputs(
    target: tuple[int, ...], workspace: tuple[int, ...]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return inverse source aliases; old result LSB becomes clean work."""
    high, sign, constant, carry = _double_ports(target, workspace)
    return (high, *target[:-1]), (target[-1], sign, *constant, *carry)


def modular_half_ir(
    target: tuple[int, ...],
    modulus: int,
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
) -> tuple[Instruction, ...]:
    """Multiply by inverse two modulo odd p on valid residues, preserving source.

    Consume target in [0,p) with zero work; read result and clean-work roles
    from modular_half_outputs. Reverse components, not the dynamic gate stream.
    """
    high, sign, constant, carry = _double_layout(
        target, modulus, workspace, classical_bits
    )
    n = len(target)
    lower = target[-1]
    original = (high, *target[:-1])
    return (
        Instruction("x", (lower,)),
        Instruction("cx", (lower, sign)),
        Instruction("x", (lower,)),
        *controlled_add_constant_adjoint_ir(
            sign,
            (*original, lower),
            modulus,
            constant[: n + 1],
            carry[:n],
            classical_bits[:n],
        ),
        *add_constant_adjoint_ir(
            (sign, *original, lower), -modulus, constant, carry, classical_bits
        ),
    )
