"""Stable runtime API.

Application code and extensions should import runtime contracts from this
package.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = (
    "ExecutionResult",
    "Module",
    "RuntimePolicy",
    "TrainingResult",
    "run",
    "run_native",
    "run_distributed",
    "train",
    "get_backend",
    "set_backend",
    "runtime_backend",
    "get_dtype",
    "set_dtype",
)

_EXPORTS = {
    "ExecutionResult": (".contracts", "ExecutionResult"),
    "Module": (".contracts", "Module"),
    "RuntimePolicy": (".contracts", "RuntimePolicy"),
    "TrainingResult": (".training", "TrainingResult"),
    "run": (".execution", "run"),
    "run_native": (".execution", "run_native"),
    "run_distributed": (".execution", "run_distributed"),
    "train": (".training", "train"),
    "get_backend": (".configuration", "get_backend"),
    "set_backend": (".configuration", "set_backend"),
    "runtime_backend": (".configuration", "runtime_backend"),
    "get_dtype": (".configuration", "get_dtype"),
    "set_dtype": (".configuration", "set_dtype"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, symbol = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    return getattr(import_module(module_name, __name__), symbol)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
