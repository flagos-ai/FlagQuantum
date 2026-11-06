# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 HammingWeightCompute.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/hamming_weight.py

"""Shared reversible sum and carry compression."""

from ...core.ir import Instruction
from .and_compute import and_compute_ir, and_uncompute_ir


def compress_three_bits_ir(
    a: int, b: int, total: int, carry: int, *, erase_bit: int | None = None
) -> tuple[Instruction, ...]:
    """Replace total by a XOR b XOR total and write carry into a clean wire."""
    if erase_bit is not None:
        return (
            Instruction("cx", (b, total)),
            Instruction("cx", (a, carry)),
            Instruction("cx", (a, b)),
            *and_uncompute_ir(b, total, carry, erase_bit),
            Instruction("cx", (a, total)),
            Instruction("cx", (a, b)),
        )
    return (
        Instruction("cx", (a, b)),
        Instruction("cx", (a, total)),
        *and_compute_ir(b, total, carry),
        Instruction("cx", (a, b)),
        Instruction("cx", (a, carry)),
        Instruction("cx", (b, total)),
    )
