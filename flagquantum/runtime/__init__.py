"""Stable runtime API.

Application code and extensions should import runtime contracts from this
package.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = (
    "ExecutionOptions",
    "ExecutionResult",
    "MeasurementResult",
    "Module",
    "RuntimePolicy",
    "TrainingResult",
    "run",
    "run_native",
    "run_distributed",
    "run_target",
    "train",
    "get_backend",
    "set_backend",
    "runtime_backend",
    "get_dtype",
    "set_dtype",
    "FallbackPolicy",
    "FallbackEvent",
    "RouteCategory",
    "RouteExplanation",
    "StrictExecutionScope",
)

_EXPORTS = {
    "ExecutionOptions": (".contracts", "ExecutionOptions"),
    "ExecutionResult": (".contracts", "ExecutionResult"),
    "MeasurementResult": (".contracts", "MeasurementResult"),
    "Module": (".contracts", "Module"),
    "RuntimePolicy": (".contracts", "RuntimePolicy"),
    "TrainingResult": (".training", "TrainingResult"),
    "run": (".execution", "run"),
    "run_native": (".execution", "run_native"),
    "run_distributed": (".execution", "run_distributed"),
    "run_target": (".target_execution", "run_target"),
    "train": (".training", "train"),
    "get_backend": (".configuration", "get_backend"),
    "set_backend": (".configuration", "set_backend"),
    "runtime_backend": (".configuration", "runtime_backend"),
    "get_dtype": (".configuration", "get_dtype"),
    "set_dtype": (".configuration", "set_dtype"),
    "FallbackPolicy": (".fallback", "FallbackPolicy"),
    "FallbackEvent": (".routing", "FallbackEvent"),
    "RouteCategory": (".routing", "RouteCategory"),
    "RouteExplanation": (".routing", "RouteExplanation"),
    "StrictExecutionScope": (".routing", "StrictExecutionScope"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, symbol = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    return getattr(import_module(module_name, __name__), symbol)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
