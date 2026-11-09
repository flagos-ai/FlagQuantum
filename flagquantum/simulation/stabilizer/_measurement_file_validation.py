"""Bound and check staged measurement files without retaining shot tables."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from typing import BinaryIO

from ...errors import CapabilityError

_READ_CHUNK_BYTES = 65536
_FORMATS = ("01", "b8", "r8", "hits", "dets", "ptb64")


def _file_size_bound(shots: int, columns: int, format: str) -> int:
    if format not in _FORMATS:
        raise ValueError("measurement format must be 01, b8, r8, hits, dets or ptb64")
    if format == "ptb64" and shots % 64:
        raise ValueError("ptb64 requires shots to be a multiple of 64")
    if shots > sys.maxsize:
        raise CapabilityError("measurement shots exceed the native integer range")
    if format == "b8":
        return shots * ((columns + 7) // 8)
    if format == "ptb64":
        return (shots // 64) * columns * 8
    if format in ("01", "r8"):
        return shots * (columns + 1)
    digits = len(str(columns - 1))
    if format == "hits":
        return shots * columns * (digits + 1)
    return shots * (5 + columns * (digits + 2))


def _file_bytes(stream: BinaryIO) -> Iterator[int]:
    while block := stream.read(_READ_CHUNK_BYTES):
        yield from block


def _invalid_file() -> OSError:
    return OSError("invalid or incomplete measurement sample file")


def _validate_r8(stream: BinaryIO, shots: int, columns: int) -> None:
    position = 0
    completed = 0
    for byte in _file_bytes(stream):
        position += byte + (byte != 255)
        if position == columns + 1 and byte != 255:
            completed += 1
            position = 0
        elif position > columns:
            raise _invalid_file()
        if completed > shots:
            raise _invalid_file()
    if position or completed != shots:
        raise _invalid_file()


def _validate_sparse_text(
    stream: BinaryIO, shots: int, columns: int, *, dets: bool
) -> None:
    prefix = b"shot" if dets else b""
    prefix_offset = 0
    number = -1
    previous = -1
    needs_number = False
    at_start = True
    completed = 0
    for byte in _file_bytes(stream):
        at_start = False
        if prefix_offset < len(prefix):
            if byte != prefix[prefix_offset]:
                raise _invalid_file()
            prefix_offset += 1
            continue
        if 48 <= byte <= 57:
            if dets and number < 0 and not needs_number:
                raise _invalid_file()
            number = max(number, 0) * 10 + byte - 48
            if number >= columns:
                raise _invalid_file()
            continue
        if byte == 10:
            if needs_number and number < 0:
                raise _invalid_file()
            if number >= 0 and number <= previous:
                raise _invalid_file()
            completed += 1
            if completed > shots:
                raise _invalid_file()
            prefix = b"shot" if dets else b""
            prefix_offset = 0
            number = previous = -1
            needs_number = False
            at_start = True
            continue
        if byte != (32 if dets else 44):
            raise _invalid_file()
        if number < 0:
            # The first dets token starts after the literal "shot".
            if not dets or needs_number or previous >= 0:
                raise _invalid_file()
        elif number <= previous:
            raise _invalid_file()
        else:
            previous = number
        number = -1
        needs_number = True
        prefix = b"M" if dets else b""
        prefix_offset = 0
    if not at_start or completed != shots:
        raise _invalid_file()


def _validate_record_file(
    path: Path, shots: int, columns: int, format: str, bound: int
) -> None:
    size = path.stat().st_size
    if format in ("01", "b8", "ptb64"):
        if size != bound:
            raise OSError(f"sample file size disagrees with expected {bound} bytes")
    elif size > bound:
        raise OSError(f"sample file exceeds its {bound} byte bound")
    if format in ("b8", "ptb64"):
        return
    with path.open("rb") as stream:
        if format == "r8":
            _validate_r8(stream, shots, columns)
        elif format in ("hits", "dets"):
            _validate_sparse_text(stream, shots, columns, dets=format == "dets")
        else:
            position = 0
            for byte in _file_bytes(stream):
                if (position == columns and byte != 10) or (
                    position < columns and byte not in (48, 49)
                ):
                    raise _invalid_file()
                position = (position + 1) % (columns + 1)
