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

# Import all from submodules (__all__ is automatically aggregated)
from .interchange import *  # noqa: F403
from .maybe_dtensor import *  # noqa: F403

logging.getLogger(__name__).addHandler(logging.NullHandler())

# No manual __all__ needed - automatically inherits from submodules
