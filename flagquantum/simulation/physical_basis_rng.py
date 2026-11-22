# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix's vertical ChaCha8 and standard mt19937_64.
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/util/chacha.h

"""Deterministic word streams for the physical spin benchmark."""

from collections.abc import Iterator

from ._packed_chacha import PackedChaCha

_U64 = (1 << 64) - 1


def _mt19937_64(seed: int) -> Iterator[int]:
    """Yield the standard 64-bit Mersenne Twister stream."""
    state = [seed & _U64]
    for i in range(1, 312):
        prior = state[-1]
        state.append((6364136223846793005 * (prior ^ (prior >> 62)) + i) & _U64)
    lower = (1 << 31) - 1
    while True:
        for i in range(312):
            joined = (state[i] & ~lower) | (state[(i + 1) % 312] & lower)
            twist = 0xB5026F5AA96619E9 if joined & 1 else 0
            state[i] = state[(i + 156) % 312] ^ (joined >> 1) ^ twist
        for item in state:
            item ^= (item >> 29) & 0x5555555555555555
            item ^= (item << 17) & 0x71D67FFFEDA60000
            item ^= (item << 37) & 0xFFF7EEE000000000
            yield item ^ (item >> 43)


def packed_words(width: int) -> Iterator[int]:
    """Pack consecutive 64-bit draws in source lane order."""
    rng = _mt19937_64(0)
    while True:
        yield sum(next(rng) << offset for offset in range(0, width, 64))


def packed_chacha8(width: int, words: Iterator[int]) -> Iterator[int]:
    """Seed now and return packed draws from one ChaCha8 stream per 32-bit lane."""
    packed_width = ((width + 31) // 32) * 32
    mask = (1 << packed_width) - 1
    return PackedChaCha(packed_width).randomize(word & mask for word in words).draws()
