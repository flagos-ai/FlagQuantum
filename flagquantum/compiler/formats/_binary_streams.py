"""Shared output boundary for circuit format commands."""

from pathlib import Path
from typing import BinaryIO


def write_binary_result(data: bytes, path: Path | None, stdout: BinaryIO) -> None:
    """Write a complete payload and report short or failed writes."""
    written = stdout.write(data) if path is None else path.write_bytes(data)
    if written != len(data):
        raise OSError(f"Incomplete write: wrote {written} of {len(data)} bytes")
    if path is None:
        stdout.flush()
