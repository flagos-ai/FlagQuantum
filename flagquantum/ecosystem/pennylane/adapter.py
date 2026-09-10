"""PennyLane implementation of the framework-neutral interop contract."""

from __future__ import annotations

from typing import Any

from ..contracts import INTEROP_API_VERSION
from .conversion import export_pennylane, import_pennylane
from .models import PennyLaneExportResult, PennyLaneImportResult


class PennyLaneAdapter:
    name = "pennylane"
    api_version = INTEROP_API_VERSION
    dependency_extra = "pennylane"

    def import_program(
        self, artifact: Any, *, allow_lossy: bool = False
    ) -> PennyLaneImportResult:
        return import_pennylane(artifact, allow_lossy=allow_lossy)

    def export_program(
        self, program: Any, *, allow_lossy: bool = False
    ) -> PennyLaneExportResult:
        return export_pennylane(program, allow_lossy=allow_lossy)


PENNYLANE_ADAPTER = PennyLaneAdapter()

__all__ = ("PENNYLANE_ADAPTER", "PennyLaneAdapter")
