"""Bounded local source acquisition and existing public conversion diagnostics."""

from __future__ import annotations

import os
import stat
from os import PathLike
from pathlib import Path
from typing import NoReturn

from ...compiler._hybrid.model import SourceLocation
from ...compiler._hybrid.verifier import Diagnostic, HybridVerificationError
from ..contracts import (
    InteropConversionError,
    InteropConversionIssue,
    InteropConversionReport,
)
from .adapter import (
    SUPPORTED_SOURCE_VERSION,
    QmodImport,
    check_source_version,
    parse_qmod,
)
from .syntax import MAX_BYTES, Token, reject


def _conversion(error: HybridVerificationError) -> NoReturn:
    report = InteropConversionReport(
        adapter="qmod",
        direction="import",
        framework_version=None,
        issues=tuple(
            InteropConversionIssue(d.code, d.message, "error")
            for d in error.diagnostics
        ),
    )
    raise InteropConversionError(str(error), report) from error


def load_qmod(
    source: str | PathLike[str],
    *,
    source_name: str | None = None,
    source_version: str | None = None,
) -> QmodImport:
    """Read a regular UTF-8 file or adapt text without execution or network I/O.

    Strings are always text. Services owns public format/option dispatch. The
    returned handoff contains owned records and declared context, not a source
    authentication or a compiled executable.
    """
    for label, value in (
        ("source_name", source_name),
        ("source_version", source_version),
    ):
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise TypeError(f"{label} must be a nonempty string or None")
    is_file = isinstance(source, PathLike)
    if is_file:
        if source_name is not None:
            raise ValueError(
                "source_name is only for text; files use their supplied path"
            )
        filename = os.fspath(source)
        if not isinstance(filename, str):
            raise TypeError("Qmod file paths must resolve to str, not bytes")
        if Path(filename).suffix != ".qmod":
            raise ValueError("native source files require the .qmod suffix")
    elif isinstance(source, str):
        filename = source_name or "<qmod>"
    else:
        raise TypeError("Qmod source must be text or PathLike[str]")
    token = Token("", SourceLocation(filename, 1, 1))
    version = SUPPORTED_SOURCE_VERSION if source_version is None else source_version
    try:
        check_source_version(version, token)
        if is_file:
            # O_NONBLOCK permits descriptor-type refusal without waiting on a
            # FIFO. O_BINARY prevents platform text translation of CRLF bytes.
            flags = (
                os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
            )
            descriptor = os.open(filename, flags)
            try:
                if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                    reject(token, "file", "Qmod source must be a regular file")
                with os.fdopen(descriptor, "rb", closefd=False) as stream:
                    data = stream.read(MAX_BYTES + 1)
            finally:
                os.close(descriptor)
            if len(data) > MAX_BYTES:
                reject(
                    token, "budget.source", f"source exceeds {MAX_BYTES} UTF-8 bytes"
                )
            text = data.decode("utf-8")
        else:
            assert isinstance(source, str)
            text = source
        return parse_qmod(text, filename=filename, source_version=version)
    except UnicodeError:
        _conversion(
            HybridVerificationError(
                (
                    Diagnostic(
                        "qmod.encoding", f"{filename}:1:1: source must be valid UTF-8"
                    ),
                )
            )
        )
    except HybridVerificationError as error:
        _conversion(error)
