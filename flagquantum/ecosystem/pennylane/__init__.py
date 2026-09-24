"""Optional, import-safe PennyLane interoperability and explicit execution."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .adapter import PENNYLANE_ADAPTER, PennyLaneAdapter
from .conformance import (
    PennyLaneConformanceCaseResult,
    PennyLaneConformanceResult,
    run_pennylane_conformance,
)
from .conversion import export_pennylane, from_pennylane, import_pennylane, to_pennylane
from .models import (
    PennyLaneConversionError,
    PennyLaneConversionIssue,
    PennyLaneConversionReport,
    PennyLaneDependencyError,
    PennyLaneExportResult,
    PennyLaneImportResult,
    PennyLaneInteropError,
)

__all__ = (
    "PENNYLANE_ADAPTER",
    "PennyLaneAdapter",
    "PennyLaneConversionError",
    "PennyLaneConformanceCaseResult",
    "PennyLaneConformanceResult",
    "PennyLaneConversionIssue",
    "PennyLaneConversionReport",
    "PennyLaneDependencyError",
    "PennyLaneExportResult",
    "PennyLaneImportResult",
    "PennyLaneInteropError",
    "PennyLaneLightningDependencyError",
    "PennyLaneLightningExecutionError",
    "export_pennylane",
    "from_pennylane",
    "import_pennylane",
    "run_pennylane_conformance",
    "run",
    "to_pennylane",
)


def __getattr__(name: str) -> Any:
    if name in {
        "PennyLaneLightningDependencyError",
        "PennyLaneLightningExecutionError",
        "run",
    }:
        return getattr(import_module(".lightning", __name__), name)
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
