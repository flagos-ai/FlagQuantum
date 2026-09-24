"""Optional, import-safe Cirq interoperability and explicit execution."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .adapter import CIRQ_ADAPTER, CirqAdapter
from .conversion import export_cirq, from_cirq, import_cirq, to_cirq
from .models import (
    CirqConversionError,
    CirqConversionIssue,
    CirqConversionReport,
    CirqDependencyError,
    CirqExportResult,
    CirqImportResult,
    CirqInteropError,
)

__all__ = (
    "CIRQ_ADAPTER",
    "CirqAdapter",
    "CirqConversionError",
    "CirqConversionIssue",
    "CirqConversionReport",
    "CirqDependencyError",
    "CirqExportResult",
    "CirqImportResult",
    "CirqInteropError",
    "CirqSimulatorDependencyError",
    "CirqSimulatorExecutionError",
    "export_cirq",
    "from_cirq",
    "import_cirq",
    "run",
    "to_cirq",
)


def __getattr__(name: str) -> Any:
    if name in {
        "CirqSimulatorDependencyError",
        "CirqSimulatorExecutionError",
        "run",
    }:
        return getattr(import_module(".simulator", __name__), name)
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
