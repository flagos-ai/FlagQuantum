# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Kickmix mem/monotonic_arena.cc/.h at
# 9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453.

"""Stable aligned operand chunks with source-compatible growth and tail reuse."""

from __future__ import annotations

import sys
from bisect import bisect_right
from collections.abc import Iterable, Iterator, Sequence
from typing import overload

from ._word_storage import (
    AlignedBytes,
    WordPointer,
    WordSpan,
    _nonnegative,
    memcpy_repeat,
)


class WordArena(Sequence[int]):
    """Own append-only physical words without moving earlier allocations.

    Python sequence access supports negative indices and slices. Explicit
    pointers retain source signed-stride and alias semantics. Clear and move
    release arena ownership while existing pointers retain their allocations.
    """

    def __init__(
        self, word_bytes: int = 4, alignment: int = 32, *, signed: bool = False
    ) -> None:
        WordPointer(word_bytes=word_bytes)
        AlignedBytes(0, alignment)
        if alignment < word_bytes or alignment % word_bytes:
            raise ValueError("Arena alignment must cover an integral number of words")
        self.signed = signed
        self.word_bytes = word_bytes
        self.alignment = alignment
        self._frozen: list[WordSpan] = []
        self._ends: list[int] = []
        self._current: AlignedBytes | None = None
        self._used = 0
        self._capacity = 0

    def __len__(self) -> int:
        return (self._ends[-1] if self._ends else 0) + self._used // self.word_bytes

    def iter_spans(self) -> Iterator[WordSpan]:
        for span in self._frozen:
            yield WordSpan(span.pointer, span.count)
        if self._used:
            yield WordSpan(
                WordPointer(self._current, 0, 1, self.word_bytes, self.signed),
                self._used // self.word_bytes,
            )

    def __iter__(self) -> Iterator[int]:
        return (value for span in self.iter_spans() for value in span)

    @overload
    def __getitem__(self, index: int) -> int: ...

    @overload
    def __getitem__(self, index: slice) -> list[int]: ...

    def __getitem__(self, index: int | slice) -> int | list[int]:
        if isinstance(index, slice):
            return [self[k] for k in range(*index.indices(len(self)))]
        if type(index) is not int:
            raise IndexError("Arena index must be an integer or slice")
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError("Arena index is out of range")
        chunk = bisect_right(self._ends, index)
        start = self._ends[chunk - 1] if chunk else 0
        if chunk < len(self._frozen):
            return self._frozen[chunk][index - start]
        return WordPointer(self._current, 0, 1, self.word_bytes, self.signed)[
            index - start
        ]

    def grab_writeable(self, count: int) -> WordPointer:
        length = _nonnegative(count, "Reservation count") * self.word_bytes
        alignment = max(self.alignment, 8)
        if self._used + length >= self._capacity:
            size = max(self._capacity, alignment * 4, length) * 2
            size = (size + alignment - 1) & -alignment
            # Allocate before updating ownership so allocation failure leaves the
            # previous arena usable. The original C++ helper mutates first.
            block = AlignedBytes(size, alignment)
            if self._current is not None and self._used:
                span = WordSpan(
                    WordPointer(self._current, 0, 1, self.word_bytes, self.signed),
                    self._used // self.word_bytes,
                )
                total = (self._ends[-1] if self._ends else 0) + span.count
                self._frozen.append(span)
                self._ends.append(total)
            self._current = block
            self._used = 0
            self._capacity = size
        result = WordPointer(self._current, self._used, 1, self.word_bytes, self.signed)
        self._used += length
        return result

    def rewind_tail(self, end: WordPointer) -> None:
        if (
            end.block is not self._current
            or not 0 <= end.byte_offset <= self._used
            or end.byte_offset % self.word_bytes
        ):
            raise ValueError("Tail pointer is outside the current arena buffer")
        self._used = end.byte_offset

    def clear(self) -> None:
        """Release frozen chunks and retain the largest allocation for reuse."""
        self._frozen.clear()
        self._ends.clear()
        self._used = 0

    def move(self) -> WordArena:
        """Transfer all chunks and return this arena to its default empty state."""
        destination = WordArena(self.word_bytes, self.alignment, signed=self.signed)
        destination.move_from(self)
        return destination

    def move_from(self, source: WordArena) -> None:
        if source is self:
            return
        if (self.word_bytes, self.alignment, self.signed) != (
            source.word_bytes,
            source.alignment,
            source.signed,
        ):
            raise ValueError("Arena move requires identical word and alignment types")
        self._frozen, self._ends, self._current = (
            source._frozen,
            source._ends,
            source._current,
        )
        self._used, self._capacity = source._used, source._capacity
        source._frozen, source._ends, source._current = [], [], None
        source._used = source._capacity = 0

    def append(self, value: int) -> None:
        self._validate_value(value)
        self.grab_writeable(1)[0] = value

    def _validate_value(self, value: int) -> None:
        bits = 8 * self.word_bytes - int(self.signed)
        minimum = -(1 << bits) if self.signed else 0
        if type(value) is not int or not minimum <= value < 1 << bits:
            raise ValueError("Arena value is outside its storage width")

    def extend(self, values: Iterable[int]) -> None:
        """Append a snapshot, including strided views or this arena itself."""
        if isinstance(values, WordArena):
            chunks = tuple(tuple(span) for span in values.iter_spans())
            for chunk in chunks:
                self.extend(chunk)
            return
        items = tuple(values)
        for value in items:
            self._validate_value(value)
        if not items:
            if isinstance(values, WordSpan):
                self.grab_writeable(0)
            return
        pointer = self.grab_writeable(len(items))
        for index, value in enumerate(items):
            pointer[index] = value

    def append_reversed(self, values: Sequence[int]) -> None:
        if isinstance(values, WordArena):
            chunks = [
                tuple(
                    WordSpan(
                        WordPointer(
                            values._current, 0, 1, values.word_bytes, values.signed
                        ),
                        values._used // values.word_bytes,
                    )
                )
            ]
            chunks.extend(tuple(span) for span in reversed(values._frozen))
            for chunk in chunks:
                self.append_reversed(chunk)
            return
        items = tuple(reversed(values))
        for value in items:
            self._validate_value(value)
        pointer = self.grab_writeable(len(items))
        for index, value in enumerate(items):
            pointer[index] = value

    def __copy__(self) -> WordArena:
        raise TypeError(
            "Physical word arenas are move-only; copy their contents explicitly"
        )

    def __deepcopy__(self, memo: dict[int, object]) -> WordArena:
        raise TypeError(
            "Physical word arenas are move-only; copy their contents explicitly"
        )

    def append_repeat(self, value: int, count: int) -> None:
        self._validate_value(value)
        count = _nonnegative(count, "Repetition count")
        if self.word_bytes == 1 and not count:
            return
        pointer = self.grab_writeable(count)
        assert pointer.block is not None
        payload = value.to_bytes(self.word_bytes, sys.byteorder, signed=self.signed)
        memcpy_repeat(pointer.block.data[pointer.byte_offset :], payload, count)

    def copy_into(self, destination: WordPointer) -> None:
        if destination.word_bytes != self.word_bytes:
            raise ValueError("Arena copy requires equal word widths")
        # The source takes T*, so copying is contiguous regardless of the
        # stride or interpretation attached to the caller's pointer view.
        target = WordPointer(
            destination.block, destination.byte_offset, 1, self.word_bytes, self.signed
        )
        if len(self):
            target._cell(len(self) - 1)
        for index, value in enumerate(self):
            target[index] = value
