# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix's vertical ChaCha8 and standard mt19937_64.
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/util/chacha.h

"""Deterministic word streams for the physical spin benchmark."""

from collections.abc import Iterator

_U32 = (1 << 32) - 1
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


def _quarter(lane: list[int], a: int, b: int, c: int, d: int) -> None:
    for rotation in (16, 12, 8, 7):
        lane[a] = (lane[a] + lane[b]) & _U32
        value = lane[d] ^ lane[a]
        lane[d] = ((value << rotation) | (value >> (32 - rotation))) & _U32
        a, b, c, d = c, d, a, b


def _block(lane: list[int]) -> tuple[int, ...]:
    work = lane.copy()
    for _ in range(4):
        _quarter(work, 0, 4, 8, 12)
        _quarter(work, 1, 5, 9, 13)
        _quarter(work, 2, 6, 10, 14)
        _quarter(work, 3, 7, 11, 15)
        _quarter(work, 0, 5, 10, 15)
        _quarter(work, 1, 6, 11, 12)
        _quarter(work, 2, 7, 8, 13)
        _quarter(work, 3, 4, 9, 14)
    result = tuple((a + b) & _U32 for a, b in zip(work, lane, strict=True))
    lane[12] = (lane[12] + 1) & _U32
    return result


def packed_chacha8(width: int, words: Iterator[int]) -> Iterator[int]:
    """Seed now and return packed draws from one ChaCha8 stream per 32-bit lane."""
    seed = [next(words) for _ in range(11)]
    lanes = []
    for offset in range(0, width, 32):
        values = [(word >> offset) & _U32 for word in seed]
        lanes.append(
            [
                0x61707865,
                0x3320646E,
                0x79622D32,
                0x6B206574,
                *values[:8],
                0,
                *values[8:],
            ]
        )

    def draws() -> Iterator[int]:
        while True:
            block = [_block(lane) for lane in lanes]
            for words_at_index in zip(*block, strict=True):
                yield sum(word << (32 * i) for i, word in enumerate(words_at_index))

    return draws()
