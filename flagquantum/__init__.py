"""FlagQuantum stable public API."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .version import __version__

__author__, __license__ = "FlagQuantum Team", "Apache-2.0"

__all__ = (
    "Circuit",
    "CircuitIR",
    "ExecutionOptions",
    "ExecutionPlan",
    "ExecutionResult",
    "IRSerializationError",
    "IRValidationError",
    "IR_VERSION",
    "I",
    "Instruction",
    "MeasurementResult",
    "Module",
    "Observable",
    "OutputRequest",
    "Parameter",
    "ParameterExpression",
    "RuntimePolicy",
    "TrainingResult",
    "X",
    "Y",
    "Z",
    "compile",
    "counts",
    "expectation",
    "plan",
    "probabilities",
    "run",
    "submit",
    "restore_job",
    "samples",
    "train",
    "__version__",
    "experimental",
    "twin",
)


def __getattr__(name: str) -> Any:
    if name in {"experimental", "twin"}:
        return import_module(f".{name}", __name__)
    if name == "Circuit":
        return getattr(import_module(".circuit", __name__), name)
    if name in {
        "CircuitIR",
        "IRSerializationError",
        "IRValidationError",
        "IR_VERSION",
        "Instruction",
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
    if name in {"TrainingResult", "train"}:
        return getattr(import_module(".runtime.training", __name__), name)
    if name in {
        "I",
        "Observable",
        "OutputRequest",
        "X",
        "Y",
        "Z",
        "counts",
        "expectation",
        "probabilities",
        "samples",
    }:
        return getattr(import_module(".observables", __name__), name)
    if name in {"submit", "restore_job"}:
        return getattr(import_module(".remote.jobs", __name__), name)
    if name in {"compile", "plan", "run"}:
        return getattr(import_module("._api", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({name for name in globals() if name.startswith("_")} | set(__all__))
