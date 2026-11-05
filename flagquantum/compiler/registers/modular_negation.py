# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 ModNeg.build_composite_bloq.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mod_arithmetic/mod_subtraction.py
# Changes: native IR and explicit work, sharing original reduction/fanout/AddK.

"""Modular negation through the original zero-detection and correction stages."""

from ...core.ir import Instruction
from .bitwise import _validate_wires
from .controlled_addition import controlled_add_constant_ir
from .multi_control import multi_control_x_ir
from .sign_extension import _fanout_ir


def modular_negation_requirements(width: int) -> tuple[int, int]:
    """Reserve a flag, loaded constant, and reusable reduction/carry wires."""
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
    flag, constant, carry = _negation_layout(target, modulus, workspace, classical_bits)
    zero = multi_control_x_ir(target, (0,) * len(target), flag, carry, classical_bits)
    return (
        Instruction("x", (flag,)),
        *zero,
        *_fanout_ir(flag, target),
        *controlled_add_constant_ir(
            flag, target, modulus + 1, constant, carry, classical_bits
        ),
        *zero,
        Instruction("x", (flag,)),
    )
