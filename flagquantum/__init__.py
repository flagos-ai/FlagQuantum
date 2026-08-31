"""FlagQuantum stable API with lazy compatibility exports."""

from __future__ import annotations

import warnings
from importlib import import_module
from typing import Any

from ._root_api_compat import (
    DEPRECATED_INTERNAL_ROOT_EXPORTS,
    MIGRATED_ROOT_EXPORTS,
    REMOVED_ROOT_EXPORTS,
)
from .version import __version__

__author__ = "FlagQuantum Team"
__license__ = "Apache-2.0"

# Stable root surface for the first public alpha. Historical attributes may
# remain lazily importable during repository convergence, but names absent from
# this tuple are not stable API and are intentionally excluded from discovery.
__all__ = (
    "Circuit",
    "CircuitIR",
    "ExecutionOptions",
    "ExecutionPlan",
    "ExecutionResult",
    "IRSerializationError",
    "IRValidationError",
    "IR_VERSION",
    "Instruction",
    "MeasurementNode",
    "MeasurementResult",
    "Module",
    "ObservableNode",
    "Parameter",
    "ParameterExpression",
    "RuntimePolicy",
    "TrainingResult",
    "plan",
    "run",
    "train",
    "__version__",
    "experimental",
)

COMPATIBILITY_EXPORT_OWNER = "FlagQuantum core maintainers"
COMPATIBILITY_EXPORT_REMOVAL_VERSION = "0.3.0"


def _compat_api() -> Any:
    return import_module(".api", __name__)


def __getattr__(name: str) -> Any:
    if name in MIGRATED_ROOT_EXPORTS:
        replacement = MIGRATED_ROOT_EXPORTS[name]
        raise AttributeError(
            f"flagquantum.{name} moved before the first public alpha; "
            f"use {replacement}"
        )
    if name in REMOVED_ROOT_EXPORTS:
        replacement = REMOVED_ROOT_EXPORTS[name]
        guidance = f"; use {replacement}" if replacement is not None else ""
        raise AttributeError(
            f"flagquantum.{name} was removed before the first public alpha{guidance}"
        )
    if name == "experimental":
        return import_module(".experimental", __name__)
    if name == "Circuit":
        return getattr(import_module(".circuit", __name__), name)
    if name == "ExecutionPlan":
        return getattr(import_module(".compilation.models", __name__), name)
    if name in {
        "ExecutionOptions",
        "ExecutionResult",
        "MeasurementResult",
        "Module",
        "RuntimePolicy",
    }:
        return getattr(import_module(".runtime.contracts", __name__), name)
    if name == "run":
        return getattr(import_module(".runtime.execution", __name__), name)
    if name in {"TrainingResult", "train"}:
        return getattr(import_module(".runtime.training", __name__), name)
    if name in DEPRECATED_INTERNAL_ROOT_EXPORTS:
        warnings.warn(
            f"flagquantum.{name} is an internal compatibility export; use "
            f"flagquantum.experimental.{name}. Root access will be removed in "
            f"version {COMPATIBILITY_EXPORT_REMOVAL_VERSION}.",
            DeprecationWarning,
            stacklevel=2,
        )
    api = _compat_api()
    try:
        return getattr(api, name)
    except AttributeError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc


def __dir__() -> list[str]:
    private_names = {name for name in globals() if name.startswith("_")}
    return sorted(private_names | set(__all__))
