"""Shared diagnostics for static specialization and dynamic lowering."""

from __future__ import annotations

from .model import SourceLocation


class SpecializationError(ValueError):
    """Raised when runtime values cannot specialize the accepted IR profile."""

    def __init__(
        self, code: str, message: str, location: SourceLocation | None = None
    ) -> None:
        self.code = code
        self.location = location
        prefix = "" if location is None else f"{location.filename}:{location.line}: "
        super().__init__(f"{prefix}{code}: {message}")
