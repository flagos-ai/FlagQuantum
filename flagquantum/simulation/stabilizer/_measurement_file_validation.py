"""Bound and check staged measurement files without retaining shot tables."""

from __future__ import annotations

import os
import shutil
import stat
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
        return shots * max(1, columns * (digits + 1))
    return shots * (5 + columns * (digits + 2))


def _file_bytes(stream: BinaryIO) -> Iterator[int]:
    while block := stream.read(_READ_CHUNK_BYTES):
        yield from block


def _invalid_file() -> OSError:
    return OSError("invalid or incomplete measurement sample file")


def _validate_r8(stream: BinaryIO, shots: int | None, columns: int) -> int:
    position = 0
    completed = 0
    for byte in _file_bytes(stream):
        position += byte + (byte != 255)
        if position == columns + 1 and byte != 255:
            completed += 1
            position = 0
        elif position > columns:
            raise _invalid_file()
        if shots is not None and completed > shots:
            raise _invalid_file()
    if position or (shots is not None and completed != shots):
        raise _invalid_file()
    return completed


def _validate_sparse_text(
    stream: BinaryIO, shots: int | None, columns: int, *, dets: bool
) -> int:
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
            if shots is not None and completed > shots:
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
    if not at_start or (shots is not None and completed != shots):
        raise _invalid_file()
    return completed


def _validate_record_file(
    path: Path,
    shots: int,
    columns: int,
    format: str,
    bound: int,
    *,
    sections: tuple[tuple[str, int], ...] | None = None,
) -> None:
    size = path.stat().st_size
    if format in ("01", "b8", "ptb64"):
        if size != bound:
            raise OSError(f"sample file size disagrees with expected {bound} bytes")
    elif size > bound:
        raise OSError(f"sample file exceeds its {bound} byte bound")
    if format == "b8" and sections is not None and columns % 8:
        stride = (columns + 7) // 8
        mask = 255 ^ ((1 << (columns % 8)) - 1)
        with path.open("rb") as stream:
            for offset, byte in enumerate(_file_bytes(stream)):
                if offset % stride == stride - 1 and byte & mask:
                    raise _invalid_file()
    if format in ("b8", "ptb64"):
        return
    with path.open("rb") as stream:
        if format == "r8":
            _validate_r8(stream, shots, columns)
        elif format == "dets" and sections is not None:
            _validate_detector_text(stream, shots, sections)
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


def _validate_detector_text(
    stream: BinaryIO, shots: int, sections: tuple[tuple[str, int], ...]
) -> None:
    # Tokens are bounded independently of file size and rejected before growth.
    token = bytearray()
    completed = 0
    section = previous = -1
    first = True
    for byte in _file_bytes(stream):
        if byte not in (32, 10):
            if len(token) >= 24:
                raise _invalid_file()
            token.append(byte)
            continue
        if first:
            if token != b"shot":
                raise _invalid_file()
            first = False
        else:
            if len(token) < 2 or not all(48 <= v <= 57 for v in token[1:]):
                raise _invalid_file()
            label = chr(token[0])
            position = next(
                (i for i, (name, _) in enumerate(sections) if name == label), -1
            )
            if position < section or position < 0:
                raise _invalid_file()
            number = int(token[1:])
            if number >= sections[position][1] or (
                position == section and number <= previous
            ):
                raise _invalid_file()
            section, previous = position, number
        token.clear()
        if byte == 10:
            completed += 1
            if shots is not None and completed > shots:
                raise _invalid_file()
            first = True
            section = previous = -1
    if token or not first or completed != shots:
        raise _invalid_file()


def _inspect_measurement_file(path: Path, columns: int, format: str) -> int:
    """Count and validate a pinned snapshot with bounded parser memory."""
    _file_size_bound(64, columns, format)
    size = path.stat().st_size
    if format in ("b8", "ptb64"):
        stride = (columns + 7) // 8 if format == "b8" else columns * 8
        if not stride or size % stride:
            raise _invalid_file()
        shots = size // stride * (64 if format == "ptb64" else 1)
        _validate_record_file(
            path, shots, columns, format, size, sections=(("M", columns),)
        )
        return shots
    with path.open("rb") as stream:
        if format == "r8":
            return _validate_r8(stream, None, columns)
        if format in ("hits", "dets"):
            return _validate_sparse_text(stream, None, columns, dets=format == "dets")
        position = completed = 0
        for byte in _file_bytes(stream):
            if position == columns:
                if byte != 10:
                    raise _invalid_file()
                completed += 1
                position = 0
            elif byte in (48, 49):
                position += 1
            else:
                raise _invalid_file()
        if position:
            raise _invalid_file()
        return completed


def _identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _snapshot(source: Path, target: Path, *, max_bytes: int = 1024**3) -> int:
    # O_NONBLOCK prevents an input replaced by a FIFO from blocking admission.
    source_metadata = source.lstat()
    if not stat.S_ISREG(source_metadata.st_mode):
        raise ValueError("measurement input must be a regular file, without symlinks")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_BINARY", 0)
    )
    descriptor = os.open(source, flags)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("measurement input must be a regular file")
        if _identity(before) != _identity(source_metadata) or _identity(
            source.lstat()
        ) != _identity(before):
            raise OSError("measurement input changed before snapshot admission")
        if before.st_size > max_bytes:
            raise CapabilityError("measurement snapshot exceeds the 1 GiB byte limit")
        if shutil.disk_usage(target.parent).free < before.st_size:
            raise OSError("insufficient free space for the measurement snapshot")
        copied = 0
        with target.open("xb") as output:
            while block := stream.read(_READ_CHUNK_BYTES):
                copied += len(block)
                if copied > before.st_size or copied > max_bytes:
                    raise OSError("measurement input changed while copying")
                output.write(block)
        if (
            copied != before.st_size
            or _identity(os.fstat(stream.fileno())) != _identity(before)
            or _identity(source.lstat()) != _identity(before)
        ):
            raise OSError("measurement input changed while copying")
    return copied
