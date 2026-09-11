"""Qiskit control-plane adapter for versioned FlagQuantum IR.

The module itself is safe to import without Qiskit. External packages are
loaded only when conversion or Qiskit Aer execution is explicitly requested.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .adapter import QISKIT_ADAPTER, QiskitAdapter
from .conformance import (
    QiskitConformanceCaseResult,
    QiskitConformanceResult,
    qiskit_statevector_to_flagquantum,
    run_qiskit_conformance,
    semantic_fingerprint,
)
from .conversion import export_qiskit, from_qiskit, import_qiskit, to_qiskit
from .models import (
    QiskitConversionError,
    QiskitConversionIssue,
    QiskitConversionReport,
    QiskitDependencyError,
    QiskitExportResult,
    QiskitImportResult,
    QiskitInteropError,
)

__all__ = (
    "QiskitConversionError",
    "QiskitConversionIssue",
    "QiskitConversionReport",
    "QiskitConformanceCaseResult",
    "QiskitConformanceResult",
    "QiskitDependencyError",
    "QiskitExportResult",
    "QiskitImportResult",
    "QiskitInteropError",
    "QISKIT_ADAPTER",
    "QiskitAdapter",
    "export_qiskit",
    "from_qiskit",
    "import_qiskit",
    "qiskit_statevector_to_flagquantum",
    "run_qiskit_aer_dynamic",
    "run_qiskit_aer_qasm3_round_trip",
    "run_qiskit_conformance",
    "semantic_fingerprint",
    "to_qiskit",
)


def __getattr__(name: str) -> Any:
    if name in {"run_qiskit_aer_dynamic", "run_qiskit_aer_qasm3_round_trip"}:
        return getattr(import_module(".execution", __name__), name)
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
