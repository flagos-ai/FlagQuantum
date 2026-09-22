"""Amazon Braket implementation of the framework-neutral interop contract."""

from __future__ import annotations

from typing import Any

from ..contracts import INTEROP_API_VERSION
from .conversion import export_braket, import_braket
from .models import BraketExportResult, BraketImportResult


class BraketAdapter:
    """Lazy static-circuit conversion adapter for Amazon Braket."""

    name = "braket"
    api_version = INTEROP_API_VERSION
    dependency_extra = "braket"

    def import_program(
        self, artifact: Any, *, allow_lossy: bool = False
    ) -> BraketImportResult:
        return import_braket(artifact, allow_lossy=allow_lossy)

    def export_program(
        self, program: Any, *, allow_lossy: bool = False
    ) -> BraketExportResult:
        return export_braket(program, allow_lossy=allow_lossy)


BRAKET_ADAPTER = BraketAdapter()

__all__ = ("BRAKET_ADAPTER", "BraketAdapter")
