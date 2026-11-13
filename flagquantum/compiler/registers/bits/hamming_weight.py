# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 HammingWeightCompute and its source adjoint.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/hamming_weight.py
# Changes: explicit logical garbage/output qubits, measured erasure and native IR.

"""Three-to-two Hamming-weight tree with explicit logical garbage and cleanup."""

from ....core.ir import Instruction
from ..._component_trace.symbolic_math import SymbolicInt, bit_length
from .bit_compression import compress_three_bits_ir
from .bitwise import xor_ir
from .hamming_weight_structure import hamming_weight_stages


def hamming_weight_resource_widths(
    width: SymbolicInt,
) -> tuple[SymbolicInt, SymbolicInt]:
    """Return original junk/output declarations, retaining symbolic lower bounds.

    Python widths use their population count. Symbolic expressions, including
    Sympy integer literals, use the original worst-case lower bound of one.
    These declared widths are not a claim of successful gate construction.
    """
    if isinstance(width, (bool, float)) or isinstance(width, int) and width < 0:
        raise ValueError("Hamming resource width requires a nonnegative integer")
    if type(width) is int:
        ones = width.bit_count()
    else:
        from sympy import Expr

        if not isinstance(width, Expr):
            raise ValueError("Hamming resource width requires an integer expression")
        ones = 1
    return width - ones, bit_length(width)


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
    clearing both junk and output. Supply one classical slot per junk qubit.
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
    blocks: list[tuple[Instruction, ...]] = []
    used = 0
    for kind, wires in hamming_weight_stages(source, junk, output):
        if kind == "cx":
            blocks.append(xor_ir((wires[0],), (wires[1],)))
        else:
            blocks.append(
                compress_three_bits_ir(
                    *wires,
                    erase_bit=classical_bits[used] if adjoint else None,
                )
            )
            used += 1
    if adjoint:
        blocks.reverse()
    return tuple(g for block in blocks for g in block)
