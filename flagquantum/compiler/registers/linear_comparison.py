# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 LinearDepthGreaterThan.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/comparison.py
# Changes: native IR, explicit work/sign bits and shared And emitters.

"""Original linear comparison paths with measured carry cleanup."""

from ...core.ir import Instruction
from .and_compute import and_compute_ir, and_uncompute_ir
from .multi_control import multi_control_x_ir


def linear_greater_requirements(width: int, *, signed: bool = False) -> tuple[int, int]:
    """Return clean quantum work and classical slots for the concrete source path."""
    if type(width) is not int or width < 1:
        raise ValueError("Linear comparison width must be a positive concrete integer")
    if type(signed) is not bool:
        raise ValueError("Linear comparison signed mode must be boolean")
    if width == 1:
        return 1, 1
    return (width - 1, width - 1) if signed else (width + 2, width)


def linear_greater_ir(
    left: tuple[int, ...],
    right: tuple[int, ...],
    target: int,
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
    *,
    signed: bool = False,
) -> tuple[Instruction, ...]:
    """Emit the pinned comparator, restoring inputs and zero-initialized work.

    Equal-width registers are big-endian; target may start in either state.
    Unsigned mode computes a>b using two clean sign prefixes. For width>=2,
    signed=True copies the sign of (b-a) modulo 2**width, as the original source
    does. This equals signed a>b only when subtraction does not overflow.
    The source ignores signed at width one and computes unsigned a>b there.
    See REGISTER_SIGNED_DIFFERENCE_SIGN.md for the source model discrepancies.
    """
    width = len(left)
    q, b = linear_greater_requirements(width, signed=signed)
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
    a = left if signed else (workspace[0], *left)
    rhs = right if signed else (workspace[1], *right)
    carries = workspace if signed else workspace[2:]
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
    for i in reversed(range(len(carries))):
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
