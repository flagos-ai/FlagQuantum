"""Typed Qiskit conversion diagnostics without a Qiskit dependency."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from ...core.ir import CircuitIR

IssueSeverity = Literal["info", "warning", "error"]


@dataclass(frozen=True)
class QiskitConversionIssue:
    """One explicit semantic difference or conversion blocker."""

    code: str
    message: str
    severity: IssueSeverity
    operation_index: int | None = None
    operation_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "operation_index": self.operation_index,
            "operation_name": self.operation_name,
        }


@dataclass(frozen=True)
class QiskitConversionReport:
    """Machine-readable account of a Qiskit boundary conversion."""

    direction: Literal["from_qiskit", "to_qiskit"]
    framework_version: str | None
    issues: tuple[QiskitConversionIssue, ...] = ()

    @property
    def lossless(self) -> bool:
        return not any(issue.severity != "info" for issue in self.issues)

    @property
    def blockers(self) -> tuple[QiskitConversionIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "flagquantum_qiskit_conversion_report_v1",
            "direction": self.direction,
            "framework_version": self.framework_version,
            "lossless": self.lossless,
            "issues": [issue.to_dict() for issue in self.issues],
        }


@dataclass(frozen=True)
class QiskitImportResult:
    """FlagQuantum IR plus its Qiskit import diagnostics."""

    ir: CircuitIR
    report: QiskitConversionReport


@dataclass(frozen=True)
class QiskitExportResult:
    """Qiskit circuit plus its FlagQuantum export diagnostics."""

    circuit: Any
    report: QiskitConversionReport


class QiskitInteropError(Exception):
    """Base error for the optional Qiskit boundary."""


class QiskitDependencyError(QiskitInteropError, ImportError):
    """Raised only when a Qiskit operation is requested without Qiskit."""


class QiskitConversionError(QiskitInteropError, ValueError):
    """Raised when a conversion would lose or invent semantics."""

    def __init__(self, message: str, report: QiskitConversionReport) -> None:
        super().__init__(message)
        self.report = report


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
