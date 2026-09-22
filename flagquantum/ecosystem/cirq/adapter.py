"""Cirq implementation of the framework-neutral interop contract."""

from __future__ import annotations

from typing import Any

from ..contracts import INTEROP_API_VERSION
from .conversion import export_cirq, import_cirq
from .models import CirqExportResult, CirqImportResult


class CirqAdapter:
    name = "cirq"
    api_version = INTEROP_API_VERSION
    dependency_extra = "cirq"

    def import_program(
        self, artifact: Any, *, allow_lossy: bool = False
    ) -> CirqImportResult:
        return import_cirq(artifact, allow_lossy=allow_lossy)

    def export_program(
        self, program: Any, *, allow_lossy: bool = False
    ) -> CirqExportResult:
        return export_cirq(program, allow_lossy=allow_lossy)


CIRQ_ADAPTER = CirqAdapter()

__all__ = ("CIRQ_ADAPTER", "CirqAdapter")
