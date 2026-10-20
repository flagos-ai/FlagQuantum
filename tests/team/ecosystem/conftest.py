"""Portable filesystem mutations for ecosystem snapshot tests."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import pytest


@pytest.fixture
def advance_file_mtime() -> Callable[[Path], None]:
    """Make same-tick writes observable to metadata snapshot checks."""

    def advance(path: Path) -> None:
        before = path.stat()
        os.utime(
            path,
            ns=(before.st_atime_ns, before.st_mtime_ns + 2_000_000_000),
        )
        assert path.stat().st_mtime_ns != before.st_mtime_ns

    return advance
