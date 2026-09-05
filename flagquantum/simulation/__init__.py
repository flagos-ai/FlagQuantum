"""Simulation engines, tensor utilities, and numerical kernels."""

from typing import Any

from . import mps, small_statevector, tensor

_EXPORT_MODULES = (mps, small_statevector, tensor)
__all__ = list(
    dict.fromkeys(name for module in _EXPORT_MODULES for name in module.__all__)
)


def __getattr__(name: str) -> Any:
    for module in _EXPORT_MODULES:
        if name in module.__all__:
            return getattr(module, name)
    raise AttributeError(name)
