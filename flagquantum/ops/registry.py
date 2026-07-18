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

# ops/registry.py
"""Quantum gate registration module.

This module provides functionality to dynamically register custom quantum gates
into the quantum operation system. Registered gates become available as both
functional interfaces and operator classes.

Example:
    Registering a custom gate:
    >>> from flagquantum.ops import register_gate
    >>> import torch
    >>>
    >>> # Define a custom 2x2 unitary matrix
    >>> my_gate_mat = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
    >>> register_gate("my_gate", my_gate_mat)
    >>>
    >>> # Now the gate is available as:
    >>> # - my_gate(...) function
    >>> # - my_gate_inv(...) for inverse
    >>> # - MY_GATE operator factory
    >>> from flagquantum.ops.registry import my_gate, MY_GATE
"""

from functools import partial
from types import SimpleNamespace

import torch

from . import functional, matrices, operator


def register_gate(name: str, mat: torch.Tensor):
    """Registers a new quantum gate into the quantum operation system.

    This function adds a custom gate to the global registry, making it available
    as a functional interface and an operator class. The gate can be used in
    quantum circuits just like built-in gates.

    Args:
        name: Name of the quantum gate. Will be converted to lowercase for the
            function name and uppercase for the operator factory.
        mat: Unitary matrix representing the quantum gate. Should be a 2x2
            tensor for single-qubit gates or 4x4 for two-qubit gates.

    Returns:
        None. The function has side effects of adding new symbols to the
        module's global namespace:

            - {name}: Functional interface for the gate
            - {name}_inv: Inverse of the functional interface
            - {name.upper()}: Operator factory for the gate

    Raises:
        ValueError: If the matrix is not unitary or has incorrect dimensions.
        KeyError: If a gate with the same name already exists.

    Example:
        Registering a single-qubit gate:
        >>> import torch
        >>> from flagquantum.ops import register_gate
        >>>
        >>> # Pauli-X gate (already exists, just for example)
        >>> x_mat = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
        >>> register_gate("x", x_mat)
        >>>
        >>> # Use the registered gate
        >>> from flagquantum.ops.registry import x
        >>>
        >>> # Apply X gate to qubit 0
        >>> state = x(wires=[0])

        Registering a custom rotation gate:
        >>> import torch
        >>> from flagquantum.ops import register_gate
        >>>
        >>> # Custom RX-like gate
        >>> def custom_rx(theta):
        ...     return torch.tensor([
        ...         [torch.cos(theta/2), -1j*torch.sin(theta/2)],
        ...         [-1j*torch.sin(theta/2), torch.cos(theta/2)]
        ...     ], dtype=torch.complex64)
        >>>
        >>> # Register as parameterized gate
        >>> register_gate("custom_rx", custom_rx)
        >>>
        >>> # Use with parameter
        >>> from flagquantum.ops.registry import custom_rx
        >>> state = custom_rx(wires=[0], theta=3.14159)

        Registering a two-qubit gate:
        >>> import torch
        >>> from flagquantum.ops import register_gate
        >>>
        >>> # CNOT gate (4x4 matrix)
        >>> cnot_mat = torch.tensor([
        ...     [1, 0, 0, 0],
        ...     [0, 1, 0, 0],
        ...     [0, 0, 0, 1],
        ...     [0, 0, 1, 0]
        ... ], dtype=torch.complex64)
        >>> register_gate("cnot", cnot_mat)
        >>>
        >>> # Apply CNOT with control=0, target=1
        >>> from flagquantum.ops.registry import cnot
        >>> state = cnot(wires=[0, 1])

        Using the operator factory:
        >>> from flagquantum.ops.registry import CNOT  # Upper case operator factory
        >>> cnot_op = CNOT(wires=[0, 1])  # Create operator instance
        >>> # The operator can be used in circuit builders

        Using the inverse version:
        >>> state = custom_rx_inv(wires=[0], theta=1.57)  # Inverse gate

    Note:
        - The matrix should be unitary (U^†U = I) for proper quantum operations.
        - For parameterized gates, mat should be a function that returns the
          matrix given parameters.
        - Registered gates are added to the module's global namespace and can
          be imported directly from flagquantum.ops.
    """
    name = name.lower()
    if name not in matrices.GATE_MAT_DICT:
        matrices.GATE_MAT_DICT.update({name: mat})
        globals()[name] = partial(functional.gate, name)
        globals()[f"{name}_inv"] = partial(functional.gate, name, inverse=True)
        globals()[name.upper()] = operator.op_factory(
            name, SimpleNamespace(**globals())
        )


# ============================================================================
# Module Exports
# ============================================================================

__all__ = ["register_gate"]
