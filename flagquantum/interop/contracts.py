"""Framework-neutral contracts for optional control-plane adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

from ..core.ir import CircuitIR

INTEROP_API_VERSION = "1.0"
IssueSeverity = Literal["info", "warning", "error"]


@dataclass(frozen=True)
class InteropConversionIssue:
    """One explicit semantic difference or conversion blocker."""

    code: str
    message: str
    severity: IssueSeverity
    operation_index: int | None = None
    operation_name: str | None = None

    def __post_init__(self) -> None:
        if not self.code or not self.message:
            raise ValueError("interop issue code and message must be non-empty")
        if self.severity not in {"info", "warning", "error"}:
            raise ValueError(f"unknown interop issue severity {self.severity!r}")
        if self.operation_index is not None and self.operation_index < 0:
            raise ValueError("interop issue operation_index must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "operation_index": self.operation_index,
            "operation_name": self.operation_name,
        }


@dataclass(frozen=True)
class InteropConversionReport:
    """Machine-readable account of one external-framework conversion."""

    adapter: str
    direction: str
    framework_version: str | None
    issues: tuple[InteropConversionIssue, ...] = ()

    def __post_init__(self) -> None:
        if not self.adapter or not self.direction:
            raise ValueError("interop adapter and direction must be non-empty")

    @property
    def lossless(self) -> bool:
        return not any(issue.severity != "info" for issue in self.issues)

    @property
    def blockers(self) -> tuple[InteropConversionIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "flagquantum_interop_conversion_report_v1",
            "adapter": self.adapter,
            "direction": self.direction,
            "framework_version": self.framework_version,
            "lossless": self.lossless,
            "issues": [issue.to_dict() for issue in self.issues],
        }


@dataclass(frozen=True)
class InteropImportResult:
    """FlagQuantum IR plus framework-neutral import diagnostics."""

    ir: CircuitIR
    report: InteropConversionReport


@dataclass(frozen=True)
class InteropExportResult:
    """External artifact plus framework-neutral export diagnostics."""

    artifact: Any
    report: InteropConversionReport


class InteropError(Exception):
    """Base error for optional interoperability boundaries."""


class InteropDependencyError(InteropError, ImportError):
    """Raised when an explicitly requested adapter dependency is unavailable."""


class InteropConversionError(InteropError, ValueError):
    """Raised when a conversion would lose or invent semantics."""

    def __init__(self, message: str, report: InteropConversionReport) -> None:
        super().__init__(message)
        self.report = report


@runtime_checkable
class InteropAdapter(Protocol):
    """Minimal lazy adapter surface centered on FlagQuantum IR."""

    name: str
    api_version: str
    dependency_extra: str

    def import_program(
        self, artifact: Any, *, allow_lossy: bool = False
    ) -> InteropImportResult: ...

    def export_program(
        self, program: Any, *, allow_lossy: bool = False
    ) -> InteropExportResult: ...


__all__ = (
    "INTEROP_API_VERSION",
    "InteropAdapter",
    "InteropConversionError",
    "InteropConversionIssue",
    "InteropConversionReport",
    "InteropDependencyError",
    "InteropError",
    "InteropExportResult",
    "InteropImportResult",
    "IssueSeverity",
)
