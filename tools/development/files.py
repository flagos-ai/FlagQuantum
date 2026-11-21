# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 dev_tools/qualtran_dev_tools/write_if_different.py.

"""Buffered text output that preserves unchanged generated files."""

import io
from collections.abc import Iterable
from pathlib import Path
from types import TracebackType
from typing import Literal


class WriteIfDifferent:
    """Write buffered text on close only when the destination differs.

    Like the source tool, leaving a context writes the current buffer even if
    its body raised an exception. Parent directories must already exist.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._buffer = io.StringIO()

    def write(self, text: str) -> int:
        return self._buffer.write(text)

    def writelines(self, lines: Iterable[str]) -> None:
        for line in lines:
            self.write(line)

    def flush(self) -> None:
        self._buffer.flush()

    def close(self) -> None:
        """Compare buffered content, close the buffer and report the result."""
        content = self._buffer.getvalue()
        self._buffer.close()
        if self.path.is_file():
            with self.path.open("r") as existing:
                if content == existing.read():
                    print(f"{self.path}\t unchanged.")
                    return
        with self.path.open("w") as destination:
            print(f"{self.path}\t writing.")
            destination.write(content)

    def __enter__(self) -> "WriteIfDifferent":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        self.close()
        return False

    @property
    def closed(self) -> bool:
        return self._buffer.closed

    def readable(self) -> Literal[False]:
        return False

    def writable(self) -> bool:
        return self._buffer.writable()

    def seekable(self) -> Literal[False]:
        return False

    def tell(self) -> int:
        return self._buffer.tell()

    def truncate(self, size: int | None = None) -> int:
        return self._buffer.truncate(size)
