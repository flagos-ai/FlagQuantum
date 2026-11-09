# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 CLinearDepthGreaterThan.build_composite_bloq.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/comparison.py
# Changes: native IR, shared sign extension, adder and control primitives.

"""Controlled signed/unsigned greater-than through the source sum path."""

from ....core.ir import Instruction
from ..arithmetic.out_of_place_addition import out_of_place_add_ir
from ..bits.sign_extension import sign_extend_ir
from ..control.multi_control import multi_control_x_ir


def controlled_greater_requirements(width: int) -> tuple[int, int]:
    """Return quantum work and classical slots for the concrete source chain."""
    if type(width) is not int or width < 1:
        raise ValueError(
            "Controlled comparison width must be a positive concrete integer"
        )
    return width + 5, width + 2


def controlled_greater_ir(
    control: int,
    left: tuple[int, ...],
    right: tuple[int, ...],
    target: int,
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
    *,
    signed: bool = False,
    control_value: int = 1,
) -> tuple[Instruction, ...]:
    """XOR (a>b and control==control_value), retaining source arithmetic and cleanup.

    Inputs have equal positive widths and big-endian bits. Signed mode uses
    two's-complement sign extension; unsigned mode uses zero sign bits. All work
    starts and ends in zero, including the temporary logical sum register.
    """
    n = len(left)
    q, c = controlled_greater_requirements(n)
    if len(right) != n:
        raise ValueError("Controlled comparison requires equal input widths")
    if (
        type(signed) is not bool
        or type(control_value) is not int
        or control_value not in (0, 1)
    ):
        raise ValueError(
            "Controlled comparison requires boolean signed and zero/one control value"
        )
    wires = (control, *left, *right, target, *workspace)
    if any(type(w) is not int or w < 0 for w in wires) or len(set(wires)) != len(wires):
        raise ValueError(
            "Controlled comparison requires distinct nonnegative quantum wires"
        )
    if len(workspace) != q:
        raise ValueError(f"Controlled comparison requires {q} clean work wires")
    if (
        len(classical_bits) != c
        or len(set(classical_bits)) != c
        or any(type(b) is not int or b < 0 for b in classical_bits)
    ):
        raise ValueError(
            f"Controlled comparison requires {c} distinct nonnegative classical slots"
        )
    a = (workspace[0], *left)
    b = (workspace[1], *right)
    output = workspace[2:-1]
    flag = workspace[-1:]
    extension = (
        (*sign_extend_ir(left, workspace[:1]), *sign_extend_ir(right, workspace[1:2]))
        if signed
        else ()
    )
    flip_b = tuple(Instruction("x", (wire,)) for wire in b)
    flip_sum = tuple(Instruction("x", (wire,)) for wire in output)
    # The first source sum bit is overflow; the second is the extended sign.
    control_gates = multi_control_x_ir(
        (control, output[1]), (control_value, 1), target, flag, classical_bits[-1:]
    )
    return (
        *extension,
        *flip_b,
        *out_of_place_add_ir(a, b, output),
        *flip_sum,
        *control_gates,
        *flip_sum,
        *out_of_place_add_ir(
            a, b, output, adjoint=True, classical_bits=classical_bits[:-1]
        ),
        *flip_b,
        *extension,
    )
