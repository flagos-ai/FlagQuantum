"""Source-located errors for restricted hybrid capture."""

from __future__ import annotations

from dataclasses import dataclass

from ..model import SourceLocation


@dataclass(frozen=True)
class CaptureDiagnostic:
    """One source-located restricted-capture failure."""

    code: str
    message: str
    location: SourceLocation


class HybridCaptureError(ValueError):
    """Raised when Python source exceeds the supported capture profile."""

    def __init__(self, diagnostic: CaptureDiagnostic) -> None:
        self.diagnostic = diagnostic
        location = diagnostic.location
        super().__init__(
            f"{location.filename}:{location.line}:{location.column}: "
            f"{diagnostic.code}: {diagnostic.message}"
        )
