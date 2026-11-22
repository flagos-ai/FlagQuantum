# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix util/pop_counter.h at
# 9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453.

"""Packed per-position population counts with explicit source transfer state."""

from __future__ import annotations

from dataclasses import dataclass

from ._word_operations import WORD_MASK, add_carry


@dataclass(frozen=True)
class PopulationCounter:
    width: int
    bit_totals: tuple[int, ...] = (0,) * 6
    paired_totals: tuple[int, ...] = (0,) * 32
    shared_count: int = 0
    step: int = 0

    def __post_init__(self) -> None:
        if type(self.width) is not int or self.width not in (64, 128, 256, 512):
            raise ValueError("Counter width must be 64, 128, 256 or 512")
        if not isinstance(self.bit_totals, tuple) or len(self.bit_totals) != 6:
            raise ValueError("Counter requires six immutable striped words")
        if not isinstance(self.paired_totals, tuple) or len(self.paired_totals) != 32:
            raise ValueError("Counter requires 32 immutable paired words")
        if any(
            type(word) is not int or not 0 <= word < 1 << self.width
            for word in (*self.bit_totals, *self.paired_totals)
        ):
            raise ValueError("Counter words must fit the packed width")
        if type(self.shared_count) is not int or not 0 <= self.shared_count < 1 << 32:
            raise ValueError("Shared count must fit uint32")
        if type(self.step) is not int or not 0 <= self.step < 1 << 8:
            raise ValueError("Counter step must fit uint8")

    def clear(self) -> PopulationCounter:
        return PopulationCounter(self.width)

    def compute_total(self, index: int) -> int:
        if type(index) is not int or not 0 <= index < self.width:
            raise ValueError("Counter index must lie within the packed width")
        paired = (self.paired_totals[index % 32] >> (32 * (index // 32))) & (
            (1 << 32) - 1
        )
        striped = sum(
            ((word >> index) & 1) << bit for bit, word in enumerate(self.bit_totals)
        )
        return (paired << 5) + striped + self.shared_count

    def masked_increments(self, mask: int) -> PopulationCounter:
        if type(mask) is not int or not 0 <= mask < 1 << self.width:
            raise ValueError("Increment mask must fit the packed width")
        step = (self.step + 1) & 31
        units = sum(1 << bit for bit in range(0, self.width, 32))
        transferred = (self.bit_totals[5] >> step) & units
        paired = list(self.paired_totals)
        updated = 0
        for offset in range(0, self.width, 64):
            word, _ = add_carry(
                False,
                (paired[step] >> offset) & WORD_MASK,
                (transferred >> offset) & WORD_MASK,
            )
            updated |= word << offset
        paired[step] = updated
        striped = list(self.bit_totals)
        striped[5] &= ~(units << step)
        for bit in range(6):
            striped[bit] ^= mask
            mask &= ~striped[bit]
        return PopulationCounter(
            self.width, tuple(striped), tuple(paired), self.shared_count, step
        )
