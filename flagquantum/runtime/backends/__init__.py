"""Backend-specific runtime namespaces.

Import a backend module explicitly so optional dependencies stay lazy:
``flagquantum.runtime.backends.statevector`` or
``flagquantum.runtime.backends.mps``.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType

__all__ = ("jax", "mps", "statevector", "tensor_network")

_BACKEND_MODULES = {name: f"{__name__}.{name}" for name in __all__}


def __getattr__(name: str) -> ModuleType:
    try:
        module_name = _BACKEND_MODULES[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name)
    globals()[name] = module
    return module


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
