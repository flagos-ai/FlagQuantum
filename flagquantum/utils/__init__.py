# utils/__init__.py
"""Utility functions for distributed quantum computing.

This module provides helper functions for:
- Interchanging qubits and dimensions in quantum state tensors
- Working with DTensor and regular tensors interchangeably
- Distributed tensor operations

Examples
--------
Interchange qubits:

>>> from flagquantum.utils import interchange_qubits, interchange_dims
>>> new_state, new_grouping = interchange_qubits(state, grouping, wire1=0, wire2=2)

Work with DTensors:

>>> from flagquantum.utils import is_dtensor, maybe_to_local, maybe_from_local
>>> if is_dtensor(tensor):
...     local = maybe_to_local(tensor)
...     dt = maybe_from_local(local, mesh, placements)
"""

import logging
from typing import Any

from . import interchange, maybe_dtensor, qasm_exporter, qcis_exporter

logging.getLogger(__name__).addHandler(logging.NullHandler())

_EXPORT_MODULES = (interchange, maybe_dtensor, qasm_exporter, qcis_exporter)
__all__ = list(
    dict.fromkeys(name for module in _EXPORT_MODULES for name in module.__all__)
)


def __getattr__(name: str) -> Any:
    for module in _EXPORT_MODULES:
        if name in module.__all__:
            return getattr(module, name)
    raise AttributeError(name)
