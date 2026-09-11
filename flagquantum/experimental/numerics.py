"""Unstable numerical representations and conformance interfaces."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_MODULES = {
    "flagquantum.runtime.executors.statevector.split_real_imag": {
        "execute_split_real_imag_expectation",
        "execute_split_real_imag_statevector",
        "parameter_shift_split_real_imag_gradient",
    },
}

_NAME_TO_MODULE = {
    name: module for module, names in _EXPORT_MODULES.items() for name in names
}
_PUBLIC_NAMES = (
    "execute_split_real_imag_expectation",
    "execute_split_real_imag_statevector",
    "parameter_shift_split_real_imag_gradient",
)
__all__ = _PUBLIC_NAMES


def __getattr__(name: str) -> Any:
    module_name = _NAME_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(name)
    return getattr(import_module(module_name), name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
