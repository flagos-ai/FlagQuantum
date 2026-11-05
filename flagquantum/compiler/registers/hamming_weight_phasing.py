# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 HammingWeightPhasing.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/rotations/hamming_weight_phasing.py

"""Phase a word by its coherently computed Hamming weight."""

from math import isfinite, pi

from ...core.ir import Instruction
from .hamming_weight import hamming_weight_ir


def hamming_weight_phasing_ir(
    source: tuple[int, ...],
    junk: tuple[int, ...],
    weight: tuple[int, ...],
    classical_bits: tuple[int, ...],
    *,
    exponent: float,
    adjoint: bool = False,
) -> tuple[Instruction, ...]:
    """Apply a Z power to each source bit through the shared count tree.

    The caller owns the clean count and junk wires. Only the measured
    erase uses the classical slots. The forward phase order is least
    significant first, and the adjoint reverses and negates those phases.
    The output preserves the source bits and clears all work.
    """
    if type(adjoint) is not bool:
        raise ValueError("Hamming phasing adjoint must be boolean")
    if type(exponent) not in (int, float) or not isfinite(exponent):
        raise ValueError("Hamming phasing requires a finite exponent")
    compute = hamming_weight_ir(source, junk, weight)
    erase = hamming_weight_ir(
        source, junk, weight, adjoint=True, classical_bits=classical_bits
    )
    terms = tuple(enumerate(reversed(weight)))
    order = reversed(terms) if adjoint else iter(terms)
    sign = -1 if adjoint else 1
    phases = tuple(
        Instruction(
            "phase", (wire,), params={"theta": pi * (sign * exponent * (1 << i))}
        )
        for i, wire in order
    )
    return (*compute, *phases, *erase)
