# SPDX-License-Identifier: Apache-2.0
# Adapted from Kickmix test.test.cc/.h at
# 9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453.

"""Scoped native equivalents of the original shared test support."""

import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO

from flagquantum.simulation.physical_basis_rng import _mt19937_64_seed_sequence


def externally_seeded_rng(entropy: Iterator[int]) -> Iterator[int]:
    return _mt19937_64_seed_sequence(tuple(next(entropy) for _ in range(8)))


class IndependentTestRng:
    """Own one test-local parent stream and eagerly seed independent children."""

    def __init__(self, entropy: Iterator[int]) -> None:
        self._entropy = entropy
        self._shared: Iterator[int] | None = None

    def draw_stream(self) -> Iterator[int]:
        if self._shared is None:
            self._shared = externally_seeded_rng(self._entropy)
        return _mt19937_64_seed_sequence(tuple(next(self._shared) for _ in range(4)))


def resolve_testdata_file_path(name: str, root: Path) -> str:
    # C fopen sees the prefix plus the name only up to its first NUL byte.
    names = [
        prefix + name for prefix in ("kickmix/testdata/", "testdata/", "../testdata/")
    ]
    for flags in (os.O_RDONLY, os.O_WRONLY | os.O_CREAT | os.O_TRUNC):
        for candidate in names:
            visible = candidate.split("\0", 1)[0]
            try:
                descriptor = os.open(root / visible, flags, 0o666)
                os.close(descriptor)
                return candidate
            except OSError:
                continue
    raise ValueError(f"Failed to find testdata/{name}\n")


def rewind_read_close(stream: BinaryIO) -> bytes:
    with stream:
        stream.seek(0)
        return stream.read()


@contextmanager
def temporary_named_file(contents: bytes = b"") -> Iterator[Path]:
    descriptor, name = tempfile.mkstemp(prefix="fq_test_named_file_")
    path = Path(name)
    try:
        os.close(descriptor)
        path.write_bytes(contents)
        yield path
    finally:
        path.unlink(missing_ok=True)
