"""Compatibility facade for distributed statevector APIs."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_IMPLEMENTATION_MODULES = (
    "flagquantum.runtime.backends.statevector.models",
    "flagquantum.runtime.backends.statevector.planning",
    "flagquantum.runtime.backends.statevector.local_execution",
)


def __getattr__(name: str) -> Any:
    for module_name in _IMPLEMENTATION_MODULES:
        module = import_module(module_name)
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    names = set(globals())
    for module_name in _IMPLEMENTATION_MODULES:
        names.update(dir(import_module(module_name)))
    return sorted(names)
