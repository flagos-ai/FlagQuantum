"""Qiskit implementation of the framework-neutral interop adapter contract."""

from __future__ import annotations

from typing import Any, cast

from ..contracts import INTEROP_API_VERSION
from .conversion import export_qiskit, import_qiskit
from .models import QiskitExportResult, QiskitImportResult


class QiskitAdapter:
    """Lazy control-plane conversion adapter for Qiskit circuits."""

    name = "qiskit"
    api_version = INTEROP_API_VERSION
    dependency_extra = "qiskit"

    def import_program(
        self, artifact: Any, *, allow_lossy: bool = False
    ) -> QiskitImportResult:
        return cast(
            QiskitImportResult,
            import_qiskit(artifact, allow_lossy=allow_lossy),
        )

    def export_program(
        self, program: Any, *, allow_lossy: bool = False
    ) -> QiskitExportResult:
        return cast(
            QiskitExportResult,
            export_qiskit(program, allow_lossy=allow_lossy),
        )


QISKIT_ADAPTER = QiskitAdapter()

__all__ = ("QISKIT_ADAPTER", "QiskitAdapter")
