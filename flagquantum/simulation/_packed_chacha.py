# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix util/chacha.h at
# 9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453.

"""Explicit packed ChaCha state using the shared scalar lane algorithm."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from ._packed_word import broadcast_u32, pack_lanes

_U32 = (1 << 32) - 1
_I32_MAX = (1 << 31) - 1


def _rotate32(value: int, amount: int) -> int:
    """Retain the source SSE lane shifts for every defined int-template amount.

    Portable shifts outside 1..31 have no C++ value guarantee. SSE defines
    those counts: 0/32 preserve the lane, and other outlying counts clear it.
    The template's signed expression 32-amount must itself remain defined.
    """
    if type(amount) is not int or not 32 - _I32_MAX <= amount <= _I32_MAX:
        raise ValueError(
            "Source rotation amount must keep int template arithmetic defined"
        )
    value &= _U32
    if amount < 0 or amount > 32:
        return 0
    if amount in (0, 32):
        return value
    return ((value << amount) | (value >> (32 - amount))) & _U32


def _quarter(lane: list[int], a: int, b: int, c: int, d: int) -> None:
    for rotation in (16, 12, 8, 7):
        lane[a] = (lane[a] + lane[b]) & _U32
        lane[d] = _rotate32(lane[d] ^ lane[a], rotation)
        a, b, c, d = c, d, a, b


def _block(lane: list[int], rounds: int) -> tuple[int, ...]:
    work = lane.copy()
    for _ in range(rounds // 2):
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


@dataclass(frozen=True)
class PackedChaCha:
    width: int
    rounds: int = 8
    state: tuple[int, ...] = (0,) * 16
    buffer: tuple[int, ...] = (0,) * 16
    buffer_idx: int = 16

    def __post_init__(self) -> None:
        if type(self.width) is not int or self.width <= 0 or self.width % 32:
            raise ValueError("Packed width must be a positive multiple of 32")
        if type(self.rounds) is not int or self.rounds < 2 or self.rounds % 2:
            raise ValueError(
                "ChaCha rounds must be a positive even integer of at least two"
            )
        if any(
            not isinstance(words, tuple) or len(words) != 16
            for words in (self.state, self.buffer)
        ):
            raise ValueError("Packed state and buffer require sixteen immutable words")
        if any(
            type(word) is not int or not 0 <= word < 1 << self.width
            for word in (*self.state, *self.buffer)
        ):
            raise ValueError("ChaCha words must fit their packed width")
        if type(self.buffer_idx) is not int or not 0 <= self.buffer_idx < 1 << 64:
            raise ValueError("Buffer index must fit the source size_t")

    def randomize(self, words: Iterator[int]) -> PackedChaCha:
        seed = [next(words) for _ in range(11)]
        if any(
            type(word) is not int or not 0 <= word < 1 << self.width for word in seed
        ):
            raise ValueError("ChaCha seed words must fit their packed width")
        constants = tuple(
            broadcast_u32(value, self.width)
            for value in (0x61707865, 0x3320646E, 0x79622D32, 0x6B206574)
        )
        return PackedChaCha(
            self.width,
            self.rounds,
            (*constants, *seed[:8], 0, *seed[8:]),
            self.buffer,
            16,
        )

    def refill(self) -> PackedChaCha:
        lanes = [
            [(word >> offset) & _U32 for word in self.state]
            for offset in range(0, self.width, 32)
        ]
        blocks = [_block(lane, self.rounds) for lane in lanes]
        state = tuple(pack_lanes((lane[k] for lane in lanes), 32) for k in range(16))
        buffer = tuple(
            pack_lanes((block[k] for block in blocks), 32) for k in range(16)
        )
        return PackedChaCha(self.width, self.rounds, state, buffer, 0)

    def next_word(self) -> tuple[int, PackedChaCha]:
        current = self.refill() if self.buffer_idx >= 16 else self
        return current.buffer[current.buffer_idx], PackedChaCha(
            current.width,
            current.rounds,
            current.state,
            current.buffer,
            current.buffer_idx + 1,
        )

    def draws(self) -> Iterator[int]:
        current = self
        while True:
            word, current = current.next_word()
            yield word
