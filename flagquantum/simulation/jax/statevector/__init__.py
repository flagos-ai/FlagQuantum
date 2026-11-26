"""Historical numerical import facade."""

import sys as _alias_sys
from importlib import import_module as _import_module
from typing import Any as _Any

_implementation = _import_module("flagquantum.simulation.statevector.jax")


def __getattr__(name: str) -> _Any:
    cached = _alias_sys.modules.get(f"{__name__}.{name}")
    if cached is not None:
        return cached
    return getattr(_implementation, name)
