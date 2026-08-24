"""Optional control-plane adapters for external quantum frameworks.

Interop modules translate at the FlagQuantum IR boundary. Importing this
package never imports an external framework or changes runtime selection.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .contracts import (
    INTEROP_API_VERSION,
    InteropAdapter,
    InteropConversionError,
    InteropConversionIssue,
    InteropConversionReport,
    InteropDependencyError,
    InteropError,
    InteropExportResult,
    InteropImportResult,
)
from .conformance import (
    InteropConformanceCaseResult,
    InteropConformanceResult,
    InteropConformanceViolation,
    InteropRejectionCase,
    InteropRoundTripCase,
    run_adapter_conformance,
    semantic_fingerprint,
)
from .registry import (
    DEFAULT_INTEROP_REGISTRY,
    InteropAdapterSpec,
    InteropRegistry,
    InteropRegistryError,
    available_adapters,
    get_adapter,
)

__all__ = (
    "DEFAULT_INTEROP_REGISTRY",
    "INTEROP_API_VERSION",
    "InteropAdapter",
    "InteropAdapterSpec",
    "InteropConformanceCaseResult",
    "InteropConformanceResult",
    "InteropConformanceViolation",
    "InteropConversionError",
    "InteropConversionIssue",
    "InteropConversionReport",
    "InteropDependencyError",
    "InteropError",
    "InteropExportResult",
    "InteropImportResult",
    "InteropRejectionCase",
    "InteropRegistry",
    "InteropRegistryError",
    "InteropRoundTripCase",
    "available_adapters",
    "get_adapter",
    "pennylane",
    "qiskit",
    "run_adapter_conformance",
    "semantic_fingerprint",
)


def __getattr__(name: str) -> Any:
    if name in {"pennylane", "qiskit"}:
        return import_module(f".{name}", __name__)
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
