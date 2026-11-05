# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 LinearDepthGreaterThan(signed=False).
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/comparison.py
# Changes: native IR, explicit work/sign bits and shared And emitters.

"""Unsigned linear-depth greater-than with measured carry cleanup."""

from ...core.ir import Instruction
from .and_compute import and_compute_ir, and_uncompute_ir
from .multi_control import multi_control_x_ir


def linear_greater_requirements(width: int) -> tuple[int, int]:
    """Return clean quantum work and classical slots for the concrete source path."""
    if type(width) is not int or width < 1:
        raise ValueError("Linear comparison width must be a positive concrete integer")
    return (1, 1) if width == 1 else (width + 2, width)


def linear_greater_ir(
    left: tuple[int, ...],
    right: tuple[int, ...],
    target: int,
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
) -> tuple[Instruction, ...]:
    """XOR unsigned a>b into target using the original carry-only comparison.

    Equal-width registers are big-endian. Work includes two zero sign bits and
    clean carry flags; every wire is restored. The source's one-bit special
    case uses the shared two-control X. Signed comparison is not this interface.
    """
    width = len(left)
    q, b = linear_greater_requirements(width)
    if len(right) != width:
        raise ValueError("Linear comparison requires equal input widths")
    wires = (*left, *right, target, *workspace)
    if any(type(w) is not int or w < 0 for w in wires) or len(set(wires)) != len(wires):
        raise ValueError(
            "Linear comparison requires distinct nonnegative quantum wires"
        )
    if len(workspace) != q:
        raise ValueError(f"Linear comparison requires {q} clean work wires")
    if (
        len(classical_bits) != b
        or len(set(classical_bits)) != b
        or any(type(bit) is not int or bit < 0 for bit in classical_bits)
    ):
        raise ValueError(
            f"Linear comparison requires {b} distinct nonnegative classical slots"
        )
    if width == 1:
        return multi_control_x_ir(
            (*left, *right), (1, 0), target, workspace, classical_bits
        )
    a = (workspace[0], *left)
    rhs = (workspace[1], *right)
    carries = workspace[2:]
    flips = tuple(Instruction("x", (wire,)) for wire in rhs)
    gates = list(flips)
    for i, carry in enumerate(carries):
        if i:
            previous = carries[i - 1]
            gates.extend(
                (
                    Instruction("cx", (previous, rhs[-1 - i])),
                    Instruction("cx", (previous, a[-1 - i])),
                )
            )
        gates.extend(and_compute_ir(rhs[-1 - i], a[-1 - i], carry))
        if i:
            gates.append(Instruction("cx", (carries[i - 1], carry)))
    gates.extend(
        (
            Instruction("cx", (carries[-1], a[0])),
            Instruction("cx", (rhs[0], a[0])),
            Instruction("x", (a[0],)),
            Instruction("cx", (a[0], target)),
            Instruction("x", (a[0],)),
            Instruction("cx", (rhs[0], a[0])),
            Instruction("cx", (carries[-1], a[0])),
        )
    )
    for i in reversed(range(width)):
        if i:
            gates.append(Instruction("cx", (carries[i - 1], carries[i])))
        gates.extend(
            and_uncompute_ir(rhs[-1 - i], a[-1 - i], carries[i], classical_bits[i])
        )
        if i:
            gates.extend(
                (
                    Instruction("cx", (carries[i - 1], a[-1 - i])),
                    Instruction("cx", (carries[i - 1], rhs[-1 - i])),
                )
            )
    return (*gates, *flips)
