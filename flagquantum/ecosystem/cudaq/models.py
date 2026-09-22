"""Typed CUDA-Q export diagnostics without a CUDA-Q dependency."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from ..contracts import (
    InteropConversionError,
    InteropConversionIssue,
    InteropConversionReport,
    InteropDependencyError,
    InteropError,
    InteropExportResult,
)


@dataclass(frozen=True)
class CudaqConversionIssue(InteropConversionIssue):
    """One explicit CUDA-Q export blocker."""


@dataclass(frozen=True, init=False)
class CudaqConversionReport(InteropConversionReport):
    """Machine-readable account of one CUDA-Q boundary conversion."""

    def __init__(
        self,
        direction: Literal["from_cudaq", "to_cudaq"],
        framework_version: str | None,
        issues: tuple[CudaqConversionIssue, ...] = (),
    ) -> None:
        super().__init__("cudaq", direction, framework_version, issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "flagquantum_cudaq_conversion_report_v1",
            "direction": self.direction,
            "framework_version": self.framework_version,
            "lossless": self.lossless,
            "issues": [issue.to_dict() for issue in self.issues],
        }


@dataclass(frozen=True, init=False)
class CudaqExportResult(InteropExportResult):
    def __init__(self, kernel: Any, report: CudaqConversionReport) -> None:
        super().__init__(kernel, report)

    @property
    def kernel(self) -> Any:
        return self.artifact


class CudaqInteropError(InteropError):
    """Base error for the optional CUDA-Q boundary."""


class CudaqDependencyError(CudaqInteropError, InteropDependencyError):
    """Raised only when export is requested without CUDA-Q."""


class CudaqConversionError(CudaqInteropError, InteropConversionError):
    """Raised when CUDA-Q export cannot preserve program semantics."""


__all__ = (
    "CudaqConversionError",
    "CudaqConversionIssue",
    "CudaqConversionReport",
    "CudaqDependencyError",
    "CudaqExportResult",
    "CudaqInteropError",
)
