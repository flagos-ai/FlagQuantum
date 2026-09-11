"""Typed PennyLane conversion diagnostics without a PennyLane dependency."""

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
class PennyLaneConversionIssue(InteropConversionIssue):
    """One explicit semantic difference or conversion blocker."""


@dataclass(frozen=True, init=False)
class PennyLaneConversionReport(InteropConversionReport):
    """Machine-readable account of a PennyLane boundary conversion."""

    def __init__(
        self,
        direction: Literal["from_pennylane", "to_pennylane"],
        framework_version: str | None,
        issues: tuple[PennyLaneConversionIssue, ...] = (),
    ) -> None:
        super().__init__("pennylane", direction, framework_version, issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "flagquantum_pennylane_conversion_report_v1",
            "direction": self.direction,
            "framework_version": self.framework_version,
            "lossless": self.lossless,
            "issues": [issue.to_dict() for issue in self.issues],
        }


@dataclass(frozen=True)
class PennyLaneImportResult(InteropImportResult):
    ir: CircuitIR
    report: PennyLaneConversionReport


@dataclass(frozen=True, init=False)
class PennyLaneExportResult(InteropExportResult):
    def __init__(self, quantum_script: Any, report: PennyLaneConversionReport) -> None:
        super().__init__(quantum_script, report)

    @property
    def quantum_script(self) -> Any:
        return self.artifact


class PennyLaneInteropError(InteropError):
    """Base error for the optional PennyLane boundary."""


class PennyLaneDependencyError(PennyLaneInteropError, InteropDependencyError):
    """Raised only when conversion is requested without PennyLane."""


class PennyLaneConversionError(PennyLaneInteropError, InteropConversionError):
    """Raised when a conversion would lose or invent semantics."""


__all__ = (
    "IssueSeverity",
    "PennyLaneConversionError",
    "PennyLaneConversionIssue",
    "PennyLaneConversionReport",
    "PennyLaneDependencyError",
    "PennyLaneExportResult",
    "PennyLaneImportResult",
    "PennyLaneInteropError",
)
