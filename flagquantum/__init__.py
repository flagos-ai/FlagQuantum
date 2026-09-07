"""FlagQuantum stable public API."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .version import __version__

__author__ = "FlagQuantum Team"
__license__ = "Apache-2.0"

# Stable root surface for the first public alpha. Names absent from this tuple
# are not available from the package root.
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


def __getattr__(name: str) -> Any:
    if name == "experimental":
        return import_module(".experimental", __name__)
    if name == "Circuit":
        return getattr(import_module(".circuit", __name__), name)
    if name in {
        "CircuitIR",
        "IRSerializationError",
        "IRValidationError",
        "IR_VERSION",
        "Instruction",
        "MeasurementNode",
        "ObservableNode",
    }:
        return getattr(import_module(".core.ir", __name__), name)
    if name in {"Parameter", "ParameterExpression"}:
        return getattr(import_module(".core.parameters", __name__), name)
    if name == "ExecutionPlan":
        return getattr(import_module(".runtime.execution_plan", __name__), name)
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
    if name == "plan":
        return getattr(import_module(".runtime.planner", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    private_names = {name for name in globals() if name.startswith("_")}
    return sorted(private_names | set(__all__))
