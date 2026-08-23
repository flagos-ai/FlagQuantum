"""Optional control-plane adapters for external quantum frameworks.

Interop modules translate at the FlagQuantum IR boundary. Importing this
package never imports an external framework or changes runtime selection.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ("qiskit",)


def __getattr__(name: str) -> Any:
    if name == "qiskit":
        return import_module(".qiskit", __name__)
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
