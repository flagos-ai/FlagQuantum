"""Typed Cirq conversion diagnostics without a Cirq dependency."""

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
class CirqConversionIssue(InteropConversionIssue):
    """One explicit Cirq semantic difference or conversion blocker."""


@dataclass(frozen=True, init=False)
class CirqConversionReport(InteropConversionReport):
    """Machine-readable account of a Cirq boundary conversion."""

    def __init__(
        self,
        direction: Literal["from_cirq", "to_cirq"],
        framework_version: str | None,
        issues: tuple[CirqConversionIssue, ...] = (),
    ) -> None:
        super().__init__("cirq", direction, framework_version, issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "flagquantum_cirq_conversion_report_v1",
            "direction": self.direction,
            "framework_version": self.framework_version,
            "lossless": self.lossless,
            "issues": [issue.to_dict() for issue in self.issues],
        }


@dataclass(frozen=True)
class CirqImportResult(InteropImportResult):
    ir: CircuitIR
    report: CirqConversionReport


@dataclass(frozen=True, init=False)
class CirqExportResult(InteropExportResult):
    def __init__(self, circuit: Any, report: CirqConversionReport) -> None:
        super().__init__(circuit, report)

    @property
    def circuit(self) -> Any:
        return self.artifact


class CirqInteropError(InteropError):
    """Base error for the optional Cirq boundary."""


class CirqDependencyError(CirqInteropError, InteropDependencyError):
    """Raised only when conversion is requested without Cirq."""


class CirqConversionError(CirqInteropError, InteropConversionError):
    """Raised when a Cirq conversion would lose or invent semantics."""


__all__ = (
    "CirqConversionError",
    "CirqConversionIssue",
    "CirqConversionReport",
    "CirqDependencyError",
    "CirqExportResult",
    "CirqImportResult",
    "CirqInteropError",
    "IssueSeverity",
)
