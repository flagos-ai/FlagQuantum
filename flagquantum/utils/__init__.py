# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
from .qasm_exporter import *  # noqa: F403

logging.getLogger(__name__).addHandler(logging.NullHandler())

# No manual __all__ needed - automatically inherits from submodules
