"""Backend-neutral complex operator values; no execution dependencies."""

import sys as _alias_sys
from importlib import import_module as _import_module
from types import ModuleType as _ModuleType


def __getattr__(name: str) -> _ModuleType:
    cached = _alias_sys.modules.get(f"{__name__}.{name}")
    if cached is not None:
        return cached
    if name in ("pauli",):
        return _import_module(f".{name}", __name__)
    raise AttributeError(name)
