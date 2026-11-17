# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix main_spin.cc.
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/cmd/main_spin.cc

"""Repeated application of a physical circuit to one packed register state."""

from collections import defaultdict
from collections.abc import Sequence
from functools import reduce
from operator import xor

from .physical_basis_rng import packed_chacha8, packed_words
from .physical_basis_shots import (
    BasisRecord,
    PackedBasisState,
    apply_packed_basis_records,
)


def spin_basis(
    records: Sequence[BasisRecord],
    qubits: int,
    bits: int,
    *,
    shots: int,
    width: int,
) -> int:
    """Return the XOR of the retained sign and all packed wire values."""
    if width not in (64, 128, 256, 512):
        raise ValueError("Spin width must be 64, 128, 256 or 512")
    if not 0 <= shots < 1 << 63:
        raise ValueError("Shots must be a nonnegative signed 64-bit integer")
    words = packed_words(width)
    measurements = packed_chacha8(width, words)
    sign = next(words)
    state = PackedBasisState(
        defaultdict(int, ((i, next(words)) for i in range(qubits))),
        defaultdict(int, ((i, next(words)) for i in range(bits))),
        [0] * width,
        sign,
    )
    if records:
        for _ in range(0, shots, width):
            apply_packed_basis_records(records, state, measurements.__next__)
    return reduce(
        xor, state.bits.values(), reduce(xor, state.qubits.values(), state.sign)
    )
