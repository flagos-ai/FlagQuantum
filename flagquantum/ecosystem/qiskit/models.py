"""Typed Qiskit conversion diagnostics without a Qiskit dependency."""

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
class QiskitConversionIssue(InteropConversionIssue):
    """One explicit semantic difference or conversion blocker."""


@dataclass(frozen=True, init=False)
class QiskitConversionReport(InteropConversionReport):
    """Machine-readable account of a Qiskit boundary conversion."""

    def __init__(
        self,
        direction: Literal["from_qiskit", "to_qiskit"],
        framework_version: str | None,
        issues: tuple[QiskitConversionIssue, ...] = (),
    ) -> None:
        super().__init__("qiskit", direction, framework_version, issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "flagquantum_qiskit_conversion_report_v1",
            "direction": self.direction,
            "framework_version": self.framework_version,
            "lossless": self.lossless,
            "issues": [issue.to_dict() for issue in self.issues],
        }


@dataclass(frozen=True)
class QiskitImportResult(InteropImportResult):
    """FlagQuantum IR plus its Qiskit import diagnostics."""

    ir: CircuitIR
    report: QiskitConversionReport


@dataclass(frozen=True, init=False)
class QiskitExportResult(InteropExportResult):
    """Qiskit circuit plus its FlagQuantum export diagnostics."""

    def __init__(self, circuit: Any, report: QiskitConversionReport) -> None:
        super().__init__(circuit, report)

    @property
    def circuit(self) -> Any:
        return self.artifact


class QiskitInteropError(InteropError):
    """Base error for the optional Qiskit boundary."""


class QiskitDependencyError(QiskitInteropError, InteropDependencyError):
    """Raised only when a Qiskit operation is requested without Qiskit."""


class QiskitConversionError(QiskitInteropError, InteropConversionError):
    """Raised when a conversion would lose or invent semantics."""


__all__ = (
    "IssueSeverity",
    "QiskitConversionError",
    "QiskitConversionIssue",
    "QiskitConversionReport",
    "QiskitDependencyError",
    "QiskitExportResult",
    "QiskitImportResult",
    "QiskitInteropError",
)
