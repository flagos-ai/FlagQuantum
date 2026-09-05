"""Bootstrap cross-module globals for extracted JAX runtime definitions."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any

_MODULES = (
    "planning_core",
    "runtime_environment",
    "backend_dispatch",
    "statevector_records",
    "statevector_gradient_records",
    "statevector_kernels",
    "statevector_execution",
    "statevector_training",
    "mps_training_records",
    "mps_result",
    "mps_gradient_result",
    "mps_kernels",
    "mps_evidence",
    "mps_backward",
    "mps_pullbacks",
    "mps_canonicalization",
    "mps_boundary_exchange",
    "mps_gradient_ownership",
    "mps_execution",
    "mps_gradients",
    "mps_planning",
    "tensor_network_execution",
    "tensor_network_gradients",
    "tensor_network_planning",
    "tensor_network_records",
    "array_conversions",
    "tensor_network_contraction",
)
_BASE_MODULES = (
    "planning_core",
    "runtime_environment",
    "backend_dispatch",
    "array_conversions",
)
_MODULES_BY_DOMAIN = {
    "statevector": _BASE_MODULES
    + tuple(name for name in _MODULES if name.startswith("statevector_")),
    "mps": _BASE_MODULES + tuple(name for name in _MODULES if name.startswith("mps_")),
    "tensor_network": _BASE_MODULES
    + ("tensor_network_contraction",)
    + tuple(
        name
        for name in _MODULES
        if name.startswith("tensor_network_") and name != "tensor_network_contraction"
    ),
    "transport": ("planning_core", "runtime_environment"),
}
_SYMBOLS: dict[str, Any] = {}
_LOADED: dict[str, ModuleType] = {}


def load(domain: str | None = None) -> tuple[dict[str, Any], tuple[ModuleType, ...]]:
    """Load one runtime family, or all families for the legacy compatibility shim."""
    names = _MODULES if domain is None else _MODULES_BY_DOMAIN[domain]
    for name in names:
        if name not in _LOADED:
            _LOADED[name] = import_module(f".{name}", package=__package__)
    modules = tuple(_LOADED.values())
    for module in modules:
        _SYMBOLS.update(
            {
                name: value
                for name, value in vars(module).items()
                if getattr(value, "__module__", None) == module.__name__
            }
        )
    for module in modules:
        vars(module).update(_SYMBOLS)
    return _SYMBOLS, modules


def resolve(name: str, *, domain: str) -> Any:
    symbols, _ = load(domain)
    try:
        return symbols[name]
    except KeyError as error:
        raise AttributeError(name) from error


__all__ = ["load", "resolve"]
