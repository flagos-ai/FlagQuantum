# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix util/fixed_width_int.cc random methods.
# Revision 9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453.

"""Source draw order and rejection sampling for native declared-width values."""

from collections.abc import Iterator

from ..core._fixed_width_integer import FixedWidthInteger
from ..core._fixed_width_modular import is_coprime
from ..core._word_operations import require_word


def sample_fixed_width(width: int, draws: Iterator[int]) -> FixedWidthInteger:
    """Pack one caller-owned uint64 draw per allocated little-endian word."""
    value = FixedWidthInteger(width)
    packed = 0
    for index in range(value.word_count):
        word = next(draws)
        require_word(word)
        packed |= word << (64 * index)
    return FixedWidthInteger(width, packed & value.mask)


def sample_fixed_width_modulo(
    width: int,
    modulus: FixedWidthInteger,
    draws: Iterator[int],
    *,
    coprime: bool = False,
) -> FixedWidthInteger:
    """Retain source masking, retry order and optional binary-GCD admission."""
    if not modulus:
        raise ValueError("Random modulus must be nonzero")
    if width < modulus.bits_in_use:
        raise ValueError("Random target width is smaller than the modulus")
    while True:
        sample = sample_fixed_width(modulus.bits_in_use, draws)
        if sample < modulus and (not coprime or is_coprime(sample, modulus)):
            return FixedWidthInteger(width, sample.value)
