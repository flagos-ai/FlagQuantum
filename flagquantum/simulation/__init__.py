"""Simulation engines, tensor utilities, and numerical kernels."""

from typing import Any

from . import lindblad, lindblad_adjoint, unitary
from .statevector import small as small_statevector

_EXPORT_MODULES = (small_statevector, lindblad, lindblad_adjoint, unitary)
__all__ = list(
    dict.fromkeys(name for module in _EXPORT_MODULES for name in module.__all__)
)


def __getattr__(name: str) -> Any:
    for module in _EXPORT_MODULES:
        if name in module.__all__:
            return getattr(module, name)
    raise AttributeError(name)
