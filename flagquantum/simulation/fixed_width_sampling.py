# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix util/fixed_width_int.cc random methods.
# Revision 9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453.

"""Source draw order and rejection sampling for native declared-width values."""

from collections.abc import Iterator

from ..core._fixed_width_integer import FixedWidthInteger
from ..core._fixed_width_modular import is_coprime
from ..core._word_operations import WORD_MASK, extract_bits, require_word


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


def sample_striped_modulo(
    modulus: FixedWidthInteger, draws: Iterator[int]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return 64 source-ordered modular samples and the final scratch snapshot.

    Tuple index selects an LSB-first value bit. Each uint64 holds that bit
    across 64 lanes. The caller retains ownership of the consumed draw stream.
    """
    if not modulus.width or not modulus.bit(modulus.width - 1):
        raise ValueError("Striped modulus must fill its declared width")
    output = [0] * modulus.width
    buffer = [0] * modulus.width
    total = 0
    while total < 64:
        is_less_than = 0
        saw_difference = 0
        for bit in reversed(range(modulus.width)):
            word = next(draws)
            require_word(word)
            buffer[bit] = word
            if modulus.bit(bit):
                is_less_than |= ~word & ~saw_difference & WORD_MASK
                saw_difference |= ~word & WORD_MASK
            else:
                saw_difference |= word
        gained = is_less_than.bit_count()
        if gained == 64:
            return tuple(buffer), tuple(buffer)
        for bit in range(modulus.width):
            output[bit] = (
                (output[bit] << gained) | extract_bits(buffer[bit], is_less_than)
            ) & WORD_MASK
        total += gained
    return tuple(output), tuple(buffer)
