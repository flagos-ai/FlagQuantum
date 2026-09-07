"""Gate matrices and context-local custom gate registration."""

from __future__ import annotations

import logging
from importlib import import_module
from typing import Any

from . import matrices

logging.getLogger(__name__).addHandler(logging.NullHandler())

_MODULE_EXPORTS = {
    "matrices": tuple(matrices.__all__),
    "registry": ("RegisteredGate", "register_gate", "registered_gates"),
}
_NAME_TO_MODULE = {
    name: module_name
    for module_name, names in _MODULE_EXPORTS.items()
    for name in names
}
__all__ = list(dict.fromkeys(_NAME_TO_MODULE))


def __getattr__(name: str) -> Any:
    if name in _MODULE_EXPORTS:
        if name == "matrices":
            module = matrices
        else:
            module = import_module(".registry", __name__)
        globals()[name] = module
        return module
    module_name = _NAME_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(name)
    if module_name == "matrices":
        module = matrices
    else:
        module = import_module(".registry", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
