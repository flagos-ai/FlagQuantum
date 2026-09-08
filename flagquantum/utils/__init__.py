# utils/__init__.py
"""Serialization utilities used by deployment and interoperability paths."""

import logging
from typing import Any

from . import qasm_exporter

logging.getLogger(__name__).addHandler(logging.NullHandler())

_EXPORT_MODULES = (qasm_exporter,)
__all__ = list(
    dict.fromkeys(name for module in _EXPORT_MODULES for name in module.__all__)
)


def __getattr__(name: str) -> Any:
    for module in _EXPORT_MODULES:
        if name in module.__all__:
            return getattr(module, name)
    raise AttributeError(name)
