"""Quantum gate operations with a lightweight, lazy public namespace."""

from __future__ import annotations

import logging
from importlib import import_module
from typing import Any

# Circuit construction needs matrices immediately. Other operator surfaces are
# loaded only when their public names are accessed; in particular this avoids
# importing DTensor and symbolic-shape machinery for an ordinary local Circuit.
from . import matrices

logging.getLogger(__name__).addHandler(logging.NullHandler())

_MODULE_EXPORTS = {
    "matrices": tuple(matrices.__all__),
    "functional": tuple(matrices.GATE_MAT_DICT)
    + tuple(f"{name}_inv" for name in matrices.GATE_MAT_DICT)
    + ("apply_unitary_bmm", "gate"),
    "invertible": ("InvertibleUnitary", "make_noisy_layer"),
    "operator": ("Op", "op_factory")
    + tuple(name.upper() for name in matrices.GATE_MAT_DICT),
    "registry": ("RegisteredGate", "register_gate", "registered_gates"),
    "lowering": (
        "BACKENDS",
        "DEFAULT_LOWERING_REGISTRY",
        "LoweringCapability",
        "OperatorLoweringRegistry",
        "UnsupportedLoweringError",
        "validate_lowering",
    ),
}
_NAME_TO_MODULE = {
    name: module_name
    for module_name, names in _MODULE_EXPORTS.items()
    for name in names
}
__all__ = list(dict.fromkeys(_NAME_TO_MODULE))


def __getattr__(name: str) -> Any:
    if name in _MODULE_EXPORTS:
        module = matrices if name == "matrices" else import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    module_name = _NAME_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(name)
    module = (
        matrices
        if module_name == "matrices"
        else import_module(f".{module_name}", __name__)
    )
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
