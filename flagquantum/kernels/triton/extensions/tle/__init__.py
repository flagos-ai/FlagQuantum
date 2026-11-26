"""Historical import facade for the relocated implementation."""

import sys as _alias_sys
from importlib import import_module as _import_module
from typing import Any as _Any

from flagquantum.simulation.kernels.triton.extensions.tle import __all__ as __all__

_implementation = _import_module("flagquantum.simulation.kernels.triton.extensions.tle")
__doc__ = _implementation.__doc__ or ""


def __getattr__(name: str) -> _Any:
    cached = _alias_sys.modules.get(f"{__name__}.{name}")
    if cached is not None:
        return cached
    return getattr(_implementation, name)
