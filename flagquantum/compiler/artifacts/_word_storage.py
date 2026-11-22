# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Kickmix mem/stride_ptr.h, stride_span.h and util.cc/.h at
# 9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453.

"""Aligned owned bytes and bounded physical word pointers and spans."""

from __future__ import annotations

import ctypes
import sys
from collections.abc import Iterator
from dataclasses import dataclass

_WORD_BYTES = frozenset((1, 2, 4, 8, 16))


def _nonnegative(value: int, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


class AlignedBytes:
    """Keep a padded Python allocation alive while exposing an aligned region."""

    def __init__(self, count: int, alignment: int) -> None:
        _nonnegative(count, "Byte count")
        if (
            type(alignment) is not int
            or not 0 < alignment <= 1024
            or alignment & (alignment - 1)
        ):
            raise ValueError("Alignment must be a power of two at most 1024")
        effective = max(alignment, ctypes.sizeof(ctypes.c_void_p))
        if count > sys.maxsize - effective:
            raise MemoryError("Failed to allocate aligned bytes")
        self.capacity = (count + effective - 1) & -effective if count else 0
        self._owner = bytearray(self.capacity + effective - 1) if count else bytearray()
        self.address = (
            ctypes.addressof(ctypes.c_char.from_buffer(self._owner)) if count else 0
        )
        offset = (-self.address) & (effective - 1)
        self.address += offset
        self.data = memoryview(self._owner)[offset : offset + self.capacity]


def memcpy_repeat(destination: memoryview, source: bytes, repetitions: int) -> None:
    """Repeat a byte payload using the original doubling and remainder order."""
    _nonnegative(repetitions, "Repetition count")
    total = len(source) * repetitions
    if total > len(destination) or destination.readonly:
        raise ValueError(
            "Repeated byte copy requires a writable destination of sufficient size"
        )
    if not repetitions:
        return
    destination[: len(source)] = source
    written = 1
    size = len(source)
    while written * 2 < repetitions:
        destination[size : 2 * size] = destination[:size]
        size *= 2
        written *= 2
    if written < repetitions:
        destination[size:total] = destination[: total - size]


@dataclass(eq=False)
class WordPointer:
    """Address a retained allocation with a signed element stride.

    Equality follows the source address-only rule. The word width and stride
    are intentionally independent from equality. Dereferences check allocation
    bounds, while pointer arithmetic may represent a one-past-end address.
    """

    block: AlignedBytes | None = None
    byte_offset: int = 0
    stride: int = 0
    word_bytes: int = 4
    signed: bool = False

    def __post_init__(self) -> None:
        if type(self.word_bytes) is not int or self.word_bytes not in _WORD_BYTES:
            raise ValueError("Physical word width must be 1, 2, 4, 8 or 16 bytes")
        if type(self.byte_offset) is not int or type(self.stride) is not int:
            raise ValueError("Physical pointer offset and stride must be integers")

    @property
    def address(self) -> int:
        return (self.block.address if self.block is not None else 0) + self.byte_offset

    def __eq__(self, other: object) -> bool:
        return isinstance(other, WordPointer) and self.address == other.address

    def __add__(self, count: int) -> WordPointer:
        if type(count) is not int:
            raise ValueError("Physical pointer displacement must be an integer")
        return WordPointer(
            self.block,
            self.byte_offset + count * self.stride * self.word_bytes,
            self.stride,
            self.word_bytes,
            self.signed,
        )

    def __radd__(self, count: int) -> WordPointer:
        return self + count

    def __sub__(self, count: int) -> WordPointer:
        return self + -count

    def __iadd__(self, count: int) -> WordPointer:
        self.byte_offset = (self + count).byte_offset
        return self

    def __isub__(self, count: int) -> WordPointer:
        return self.__iadd__(-count)

    def advance(self) -> WordPointer:
        """Apply source prefix increment and return this pointer."""
        self += 1
        return self

    def advance_after(self) -> WordPointer:
        """Return the old address before applying source postfix increment."""
        previous = self + 0
        self += 1
        return previous

    def _cell(self, index: int) -> memoryview:
        offset = (self + index).byte_offset
        if (
            self.block is None
            or offset < 0
            or offset + self.word_bytes > self.block.capacity
        ):
            raise IndexError("Physical word pointer is outside its allocation")
        return self.block.data[offset : offset + self.word_bytes]

    def __getitem__(self, index: int) -> int:
        return int.from_bytes(self._cell(index), sys.byteorder, signed=self.signed)

    def __setitem__(self, index: int, value: int) -> None:
        bits = 8 * self.word_bytes - int(self.signed)
        minimum = -(1 << bits) if self.signed else 0
        if type(value) is not int or not minimum <= value < 1 << bits:
            raise ValueError("Physical word value is outside its storage width")
        self._cell(index)[:] = value.to_bytes(
            self.word_bytes, sys.byteorder, signed=self.signed
        )

    def cast_words(self, word_bytes: int, *, signed: bool = False) -> WordPointer:
        """Reinterpret the same address and source element stride at a new width."""
        return WordPointer(
            self.block, self.byte_offset, self.stride, word_bytes, signed
        )


@dataclass
class WordSpan:
    """Retain an allocation and a logical item count independently of stride."""

    pointer: WordPointer
    count: int

    def __post_init__(self) -> None:
        _nonnegative(self.count, "Span count")
        self.pointer = self.pointer + 0

    def __len__(self) -> int:
        return self.count

    def __iter__(self) -> Iterator[int]:
        return (self.pointer[k] for k in range(self.count))

    def at(self, index: int) -> int:
        if type(index) is not int or not 0 <= index < self.count:
            raise IndexError("Physical span index is out of range")
        return self.pointer[index]

    def __getitem__(self, index: int) -> int:
        return self.pointer[index]

    def __setitem__(self, index: int, value: int) -> None:
        self.pointer[index] = value

    def reversed(self) -> WordSpan:
        pointer = self.pointer + (self.count - 1 if self.count else 0)
        pointer.stride = -pointer.stride
        return WordSpan(pointer, self.count)

    def skip(self, count: int) -> WordSpan:
        count = min(_nonnegative(count, "Skip count"), self.count)
        return WordSpan(self.pointer + count, self.count - count)

    def keep(self, count: int) -> WordSpan:
        return WordSpan(
            self.pointer, min(_nonnegative(count, "Keep count"), self.count)
        )

    def skip_last(self, count: int) -> WordSpan:
        return self.keep(
            self.count - min(_nonnegative(count, "Skip count"), self.count)
        )

    def keep_last(self, count: int) -> WordSpan:
        return self.skip(
            self.count - min(_nonnegative(count, "Keep count"), self.count)
        )

    def subspan(self, skip: int, keep: int | None = None) -> WordSpan:
        result = self.skip(skip)
        return result if keep is None else result.keep(keep)

    def indexed_pointer(self, offset: int, *, negated: bool = False) -> WordPointer:
        """Apply the source iota offset and sign to this span's pointer."""
        result = self.pointer + offset
        if negated:
            result.stride = -result.stride
        return result

    def cast_words(self, word_bytes: int, *, signed: bool = False) -> WordSpan:
        return WordSpan(self.pointer.cast_words(word_bytes, signed=signed), self.count)

    @property
    def diagnostic_text(self) -> str:
        contents = ", ".join(str(value) for value in self)
        return f"WordSpan{{.ptr={self.pointer.address}, .stride={self.pointer.stride}, .count={self.count}, contents={{{contents}}}}}"


def max_u32(items: WordSpan) -> int:
    """Compute the source unsigned-32 maximum, including its zero empty value."""
    if items.pointer.word_bytes != 4:
        raise ValueError("Unsigned-32 maximum requires four-byte words")
    return max(items, default=0)
