# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix's vertical ChaCha8 and standard mt19937_64.
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/util/chacha.h

"""Deterministic word streams for the physical spin benchmark."""

from collections.abc import Iterator
from dataclasses import dataclass, field

from ._packed_chacha import PackedChaCha
from ._packed_word import pack_lanes

_U64 = (1 << 64) - 1


def _mt19937_64(seed: int) -> Iterator[int]:
    """Yield the standard 64-bit Mersenne Twister stream."""
    state = [seed & _U64]
    for i in range(1, 312):
        prior = state[-1]
        state.append((6364136223846793005 * (prior ^ (prior >> 62)) + i) & _U64)
    return _mt19937_64_state(state)


def _mt19937_64_seed_sequence(seeds: tuple[int, ...]) -> Iterator[int]:
    """Apply standard seed_seq uint32 mixing before the shared 64-bit engine."""
    mask = (1 << 32) - 1
    words = [0x8B8B8B8B] * 624
    p, q = 306, 317
    count = max(len(seeds) + 1, len(words))
    value = words[0] ^ words[p] ^ words[-1]
    mixed = (1664525 * (value ^ (value >> 27))) & mask
    words[p] = (words[p] + mixed) & mask
    mixed = (mixed + len(seeds)) & mask
    words[q] = (words[q] + mixed) & mask
    words[0] = mixed
    for index in range(1, count):
        here = index % 624
        left = (index + p) % 624
        right = (index + q) % 624
        value = words[here] ^ words[left] ^ words[(index - 1) % 624]
        mixed = (1664525 * (value ^ (value >> 27))) & mask
        words[left] = (words[left] + mixed) & mask
        seed = seeds[index - 1] & mask if index <= len(seeds) else 0
        mixed = (mixed + here + seed) & mask
        words[right] = (words[right] + mixed) & mask
        words[here] = mixed
    for index in range(count, count + 624):
        here = index % 624
        left = (index + p) % 624
        right = (index + q) % 624
        value = (words[here] + words[left] + words[(index - 1) % 624]) & mask
        mixed = (1566083941 * (value ^ (value >> 27))) & mask
        words[left] ^= mixed
        mixed = (mixed - here) & mask
        words[right] ^= mixed
        words[here] = mixed
    state = [words[index] | (words[index + 1] << 32) for index in range(0, 624, 2)]
    if state[0] >> 31 == 0 and not any(state[1:]):
        state[0] = 1 << 63
    return _mt19937_64_state(state)


@dataclass
class _MersenneTwister64(Iterator[int]):
    """Own the standard engine state so numerical snapshots can fork draws."""

    state: list[int]
    index: int = field(default=312)

    def __next__(self) -> int:
        if self.index == 312:
            lower = (1 << 31) - 1
            for i in range(312):
                joined = (self.state[i] & ~lower) | (self.state[(i + 1) % 312] & lower)
                twist = 0xB5026F5AA96619E9 if joined & 1 else 0
                self.state[i] = self.state[(i + 156) % 312] ^ (joined >> 1) ^ twist
            self.index = 0
        item = self.state[self.index]
        self.index += 1
        item ^= (item >> 29) & 0x5555555555555555
        item ^= (item << 17) & 0x71D67FFFEDA60000
        item ^= (item << 37) & 0xFFF7EEE000000000
        return item ^ (item >> 43)


def _mt19937_64_state(state: list[int]) -> Iterator[int]:
    return _MersenneTwister64(state)


def packed_words(width: int) -> Iterator[int]:
    """Pack consecutive 64-bit draws in source lane order."""
    rng = _mt19937_64(0)
    while True:
        yield pack_lanes((next(rng) for _ in range(0, width, 64)), 64)


def packed_chacha8(width: int, words: Iterator[int]) -> Iterator[int]:
    """Seed now and return packed draws from one ChaCha8 stream per 32-bit lane."""
    packed_width = ((width + 31) // 32) * 32
    mask = (1 << packed_width) - 1
    return PackedChaCha(packed_width).randomize(word & mask for word in words).draws()
