"""CUDA-Q implementation of the framework-neutral interop contract."""

from __future__ import annotations

from typing import Any, NoReturn

from ..contracts import INTEROP_API_VERSION
from .conversion import export_cudaq
from .models import (
    CudaqConversionError,
    CudaqConversionIssue,
    CudaqConversionReport,
    CudaqExportResult,
)


class CudaqAdapter:
    name = "cudaq"
    api_version = INTEROP_API_VERSION
    dependency_extra = "cudaq"

    def import_program(self, artifact: Any, *, allow_lossy: bool = False) -> NoReturn:
        del artifact, allow_lossy
        report = CudaqConversionReport(
            "from_cudaq",
            None,
            (
                CudaqConversionIssue(
                    "reverse_conversion_not_supported",
                    "CUDA-Q kernel import is outside the export-only v1 boundary.",
                    "error",
                ),
            ),
        )
        raise CudaqConversionError(
            "CUDA-Q kernel import is not supported; inspect error.report.", report
        )

    def export_program(
        self, program: Any, *, allow_lossy: bool = False
    ) -> CudaqExportResult:
        return export_cudaq(program, allow_lossy=allow_lossy)


CUDAQ_ADAPTER = CudaqAdapter()

__all__ = ("CUDAQ_ADAPTER", "CudaqAdapter")
