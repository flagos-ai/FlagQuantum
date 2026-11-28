# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix util/fixed_width_int.cc random methods.
# Revision 9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453.

"""Source draw order and rejection sampling for native declared-width values."""

from collections.abc import Iterator, MutableSequence

from ..core._fixed_width_integer import FixedWidthInteger
from ..core._fixed_width_modular import is_coprime
from ..core._modular_integer import ModularInteger
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
    words = [0] * (2 * modulus.width)
    _sample_striped_modulo_into(modulus, draws, words, 0, modulus.width)
    return tuple(words[: modulus.width]), tuple(words[modulus.width :])


def _sample_striped_modulo_into(
    modulus: FixedWidthInteger,
    draws: Iterator[int],
    words: MutableSequence[int],
    output_offset: int,
    scratch_offset: int,
) -> None:
    """Execute the source mutation order in caller-owned, possibly shared spans.

    Offsets select width-sized spans in one little-endian word allocation.
    Overlap retains the source numerical defect. A full-batch overlapping
    memcpy has no portable source semantics and is rejected before copying.
    """
    width = modulus.width
    if not width or not modulus.bit(width - 1):
        raise ValueError("Striped modulus must fill its declared width")
    if any(
        type(offset) is not int or offset < 0 or offset + width > len(words)
        for offset in (output_offset, scratch_offset)
    ):
        raise ValueError("Striped buffer span is outside its word allocation")
    for bit in range(width):
        words[output_offset + bit] = 0
    total = 0
    while total < 64:
        is_less_than = 0
        saw_difference = 0
        for bit in reversed(range(width)):
            word = next(draws)
            require_word(word)
            words[scratch_offset + bit] = word
            if modulus.bit(bit):
                is_less_than |= ~word & ~saw_difference & WORD_MASK
                saw_difference |= ~word & WORD_MASK
            else:
                saw_difference |= word
        gained = is_less_than.bit_count()
        if gained == 64:
            if abs(output_offset - scratch_offset) < width:
                raise ValueError("Source full-batch memcpy overlaps its scratch span")
            for bit in range(width):
                words[output_offset + bit] = words[scratch_offset + bit]
            return
        for bit in range(width):
            words[output_offset + bit] = (
                (words[output_offset + bit] << gained)
                | extract_bits(words[scratch_offset + bit], is_less_than)
            ) & WORD_MASK
        total += gained


def sample_modular_integer(
    modulus: FixedWidthInteger, draws: Iterator[int]
) -> ModularInteger:
    """Sample an immutable modular value with the source scalar draw order."""
    return ModularInteger(
        sample_fixed_width_modulo(modulus.width, modulus, draws), modulus
    )
