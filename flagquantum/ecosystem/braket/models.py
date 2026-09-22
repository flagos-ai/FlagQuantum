"""Typed Amazon Braket conversion diagnostics without an SDK dependency."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from ...core.ir import CircuitIR
from ..contracts import (
    InteropConversionError,
    InteropConversionIssue,
    InteropConversionReport,
    InteropDependencyError,
    InteropError,
    InteropExportResult,
    InteropImportResult,
    IssueSeverity,
)


@dataclass(frozen=True)
class BraketConversionIssue(InteropConversionIssue):
    """One explicit Braket semantic difference or conversion blocker."""


@dataclass(frozen=True, init=False)
class BraketConversionReport(InteropConversionReport):
    """Machine-readable account of an Amazon Braket conversion."""

    def __init__(
        self,
        direction: Literal["from_braket", "to_braket"],
        framework_version: str | None,
        issues: tuple[BraketConversionIssue, ...] = (),
    ) -> None:
        super().__init__("braket", direction, framework_version, issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "flagquantum_braket_conversion_report_v1",
            "direction": self.direction,
            "framework_version": self.framework_version,
            "lossless": self.lossless,
            "issues": [issue.to_dict() for issue in self.issues],
        }


@dataclass(frozen=True)
class BraketImportResult(InteropImportResult):
    ir: CircuitIR
    report: BraketConversionReport


@dataclass(frozen=True, init=False)
class BraketExportResult(InteropExportResult):
    def __init__(self, circuit: Any, report: BraketConversionReport) -> None:
        super().__init__(circuit, report)

    @property
    def circuit(self) -> Any:
        return self.artifact


class BraketInteropError(InteropError):
    """Base error for the optional Amazon Braket boundary."""


class BraketDependencyError(BraketInteropError, InteropDependencyError):
    """Raised only when conversion is requested without Amazon Braket."""


class BraketConversionError(BraketInteropError, InteropConversionError):
    """Raised when a Braket conversion would lose or invent semantics."""


__all__ = (
    "BraketConversionError",
    "BraketConversionIssue",
    "BraketConversionReport",
    "BraketDependencyError",
    "BraketExportResult",
    "BraketImportResult",
    "BraketInteropError",
    "IssueSeverity",
)
