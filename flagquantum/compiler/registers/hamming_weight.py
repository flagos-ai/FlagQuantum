# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 HammingWeightCompute and its source adjoint.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/hamming_weight.py
# Changes: explicit logical garbage/output wires, measured erasure and native IR.

"""Three-to-two Hamming-weight tree with explicit logical garbage and cleanup."""

from ...core.ir import Instruction
from .bit_compression import compress_three_bits_ir
from .bitwise import xor_ir


def hamming_weight_requirements(width: int) -> tuple[int, int]:
    """Return logical junk width and count-output width, not hidden scratch."""
    if type(width) is not int or width < 1:
        raise ValueError("Hamming weight requires a positive concrete width")
    return width - width.bit_count(), width.bit_length()


def hamming_weight_ir(
    source: tuple[int, ...],
    junk: tuple[int, ...],
    output: tuple[int, ...],
    *,
    adjoint: bool = False,
    classical_bits: tuple[int, ...] = (),
) -> tuple[Instruction, ...]:
    """Compute a count into zero output/junk, or erase a valid computed triple.

    Registers are big-endian. Forward changes source and leaves logical junk;
    neither is advertised as preserved/clean until the measured adjoint runs.
    The adjoint takes the entire correct computed state and restores source,
    clearing both junk and output. Supply one classical slot per junk wire.
    """
    j, o = hamming_weight_requirements(len(source))
    if type(adjoint) is not bool:
        raise ValueError("Hamming adjoint must be boolean")
    if len(junk) != j or len(output) != o:
        raise ValueError(f"Hamming weight requires {j} junk and {o} output wires")
    wires = (*source, *junk, *output)
    if any(type(w) is not int or w < 0 for w in wires) or len(set(wires)) != len(wires):
        raise ValueError("Hamming weight requires distinct nonnegative wires")
    c = j if adjoint else 0
    if (
        len(classical_bits) != c
        or len(set(classical_bits)) != c
        or any(type(b) is not int or b < 0 for b in classical_bits)
    ):
        raise ValueError(
            f"Hamming weight requires {c} distinct nonnegative classical slots"
        )
    # Preserve the source little-endian traversal and reversed-junk pop order.
    x, available = list(source[::-1]), list(junk[::-1])
    blocks: list[tuple[Instruction, ...]] = []
    used = 0

    def compress(a: int, b: int, carry_in: int, carry_out: int) -> None:
        nonlocal used
        blocks.append(
            compress_three_bits_ir(
                a,
                b,
                carry_in,
                carry_out,
                erase_bit=classical_bits[used] if adjoint else None,
            )
        )
        used += 1

    for result in reversed(output):
        y = []
        for i in range(0, len(x) - 2, 2):
            anc = available.pop()
            y.append(anc)
            compress(x[i], x[i + 1], x[i + 2], anc)
        if len(x) % 2:
            blocks.append(xor_ir((x[-1],), (result,)))
        else:
            anc = available.pop()
            compress(x[-2], x[-1], result, anc)
            y.append(anc)
        x = y
    if adjoint:
        blocks.reverse()
    return tuple(g for block in blocks for g in block)
