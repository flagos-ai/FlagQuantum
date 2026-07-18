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

# encoder.py
"""Quantum encoding methods for data embedding.

This module provides quantum encoding techniques to embed classical data
into quantum states, including general parameterized encoders and
convenience functions for common encoding schemes.

Examples
--------
Using GeneralEncoder with custom gate sequence:

>>> from flagquantum.encoding import GeneralEncoder
>>> from flagquantum.devices import DistributedQuantumDevice
>>>
>>> # Define encoding circuit
>>> func_list = [
...     {"func": "ry", "wires": [0], "input_idx": 0},
...     {"func": "ry", "wires": [1], "input_idx": 1},
...     {"func": "cx", "wires": [0, 1]},  # Non-parameterized
... ]
>>> encoder = GeneralEncoder(func_list)
>>>
>>> # Apply encoding
>>> device = DistributedQuantumDevice(n_wires=2, bsz=4)
>>> x = torch.randn(4, 2)  # Batch of 4, 2 features
>>> encoder(device, x)

Using convenience functions:

>>> from flagquantum.encoding import angle_encoder, basis_encoder, amplitude_encoder
>>>
>>> # Angle encoding
>>> angle_encoder(device, x, wires=[0, 1], rotation="ry")
>>>
>>> # Basis encoding (binary data)
>>> binary_data = torch.randint(0, 2, (4, 2))
>>> basis_encoder(device, binary_data, wires=[0, 1])
>>>
>>> # Amplitude encoding
>>> amplitudes = torch.randn(4, 4)  # 2^2 = 4 amplitudes
>>> amplitude_encoder(device, amplitudes)

Using factory function:

>>> from flagquantum.encoding import create_encoding_circuit
>>>
>>> encoder = create_encoding_circuit("angle", n_qubits=4, n_features=4)
>>> encoder(device, x)

Inverse for invertible backpropagation:

>>> encoder.inverse(device, x)  # Apply inverse of all gates
"""

from typing import Any, Dict, List, Optional

import torch

from ..ops import functional, matrices

# ============================================================================
# Main Encoder Class
# ============================================================================


class GeneralEncoder(torch.nn.Module):
    """General quantum encoder that applies a sequence of parameterized gates.

    This encoder applies gates in sequence, automatically handling parameterized
    and non-parameterized gates. For invertible backpropagation, use inverse().

    Parameters
    ----------
    func_list : List[Dict[str, Any]]
        List of gate specifications, each containing:
        - "func": gate name (e.g., "rx", "ry", "cx")
        - "wires": list of wire indices
        - "input_idx": index in input tensor for parameters (optional for param gates)

    Examples
    --------
    Create an encoder with rotation gates:

    >>> func_list = [
    ...     {"func": "ry", "wires": [0], "input_idx": 0},
    ...     {"func": "ry", "wires": [1], "input_idx": 1},
    ... ]
    >>> encoder = GeneralEncoder(func_list)
    >>> encoder(device, x)

    Include non-parameterized gates:

    >>> func_list = [
    ...     {"func": "ry", "wires": [0], "input_idx": 0},
    ...     {"func": "cx", "wires": [0, 1]},  # No input_idx needed
    ... ]
    >>> encoder = GeneralEncoder(func_list)
    """

    def __init__(self, func_list: List[Dict[str, Any]]) -> None:
        super().__init__()
        self.func_list = func_list

        # Precompute gate types for faster access
        self._gate_types = self._precompute_gate_types()

    def _precompute_gate_types(self) -> List[Dict[str, Any]]:
        """Precompute gate properties for faster forward/backward passes."""
        gate_types = []
        for info in self.func_list:
            func_name = info["func"]
            gate_info = {
                "func_name": func_name,
                "wires": info["wires"],
                "is_parameterized": callable(
                    matrices.GATE_MAT_DICT.get(func_name, None)
                ),
                "input_idx": info.get("input_idx", None),
            }
            # Pre-store the gate function reference
            gate_info["gate_func"] = getattr(functional, func_name)
            gate_info["gate_inv_func"] = getattr(functional, f"{func_name}_inv", None)
            gate_types.append(gate_info)
        return gate_types

    def forward(self, q_dev, x: torch.Tensor) -> None:
        """Apply encoding gates to quantum device.

        Parameters
        ----------
        q_dev : QuantumDevice
            Quantum device instance
        x : torch.Tensor
            Input tensor of shape (batch_size, n_features)

        Examples
        --------
        >>> encoder = GeneralEncoder([{"func": "ry", "wires": [0], "input_idx": 0}])
        >>> x = torch.randn(4, 1)  # Batch of 4
        >>> encoder(device, x)
        """
        for gate_info in self._gate_types:
            if gate_info["is_parameterized"]:
                params = (
                    x[:, gate_info["input_idx"]]
                    if gate_info["input_idx"] is not None
                    else None
                )
            else:
                params = None

            # Apply gate
            gate_info["gate_func"](q_dev, gate_info["wires"], params=params)

    def inverse(self, q_dev, x: Optional[torch.Tensor] = None) -> None:
        """Apply inverse of encoding gates (for invertible backpropagation).

        Parameters
        ----------
        q_dev : QuantumDevice
            Quantum device instance
        x : Optional[torch.Tensor], default=None
            Input tensor (only needed for parameterized gates)

        Examples
        --------
        >>> encoder(device, x)  # Forward
        >>> encoder.inverse(device, x)  # Inverse (undoes the encoding)
        """
        for gate_info in reversed(self._gate_types):
            inv_func = gate_info.get("gate_inv_func")
            if inv_func is None:
                # Fallback: try to use inverse method if available
                try:
                    inv_func = getattr(functional, f"{gate_info['func_name']}_inv")
                except AttributeError:
                    continue

            if gate_info["is_parameterized"] and x is not None:
                params = (
                    x[:, gate_info["input_idx"]]
                    if gate_info["input_idx"] is not None
                    else None
                )
            else:
                params = None

            inv_func(q_dev, gate_info["wires"], params=params)

    def __repr__(self) -> str:
        return f"GeneralEncoder(gates={len(self.func_list)})"


# ============================================================================
# Amplitude Encoding
# ============================================================================


def amplitude_encoder(q_dev, amplitudes: torch.Tensor) -> None:
    """Encode classical data as amplitude encoding.

    This encoder loads amplitudes directly into the quantum state vector.
    The amplitudes are automatically normalized to unit norm.

    Parameters
    ----------
    q_dev : QuantumDevice
        Quantum device instance
    amplitudes : torch.Tensor
        Amplitude tensor of shape (batch_size, 2^n_wires)

    Examples
    --------
    >>> device = DistributedQuantumDevice(n_wires=2, bsz=4)
    >>> amplitudes = torch.randn(4, 4)  # 4 batches, 2^2=4 amplitudes
    >>> amplitude_encoder(device, amplitudes)

    Notes
    -----
    - Amplitudes are automatically normalized
    - The device state is reset before loading
    - Complex amplitudes are supported (via torch.complex tensor)
    """
    q_dev.load_amplitudes(amplitudes)


# ============================================================================
# Convenience Encoding Functions
# ============================================================================


def angle_encoder(
    q_dev, x: torch.Tensor, wires: List[int], rotation: str = "ry"
) -> None:
    """Simple angle encoding on specified wires.

    Encodes each feature as a rotation angle on a qubit.

    Parameters
    ----------
    q_dev : QuantumDevice
        Quantum device instance
    x : torch.Tensor
        Input tensor (batch_size, n_features)
    wires : List[int]
        Qubit wires to encode into (first n_features wires are used)
    rotation : str, default="ry"
        Type of rotation ('rx', 'ry', 'rz')

    Raises
    ------
    ValueError
        If rotation is not 'rx', 'ry', or 'rz'

    Examples
    --------
    >>> device = DistributedQuantumDevice(n_wires=4, bsz=2)
    >>> x = torch.randn(2, 4)  # 2 batches, 4 features
    >>> angle_encoder(device, x, wires=[0, 1, 2, 3], rotation="ry")

    Encode only first 2 features:

    >>> angle_encoder(device, x, wires=[0, 1])  # Uses x[:, 0] and x[:, 1]
    """
    if rotation not in ["rx", "ry", "rz"]:
        raise ValueError(f"Unknown rotation: {rotation}. Use 'rx', 'ry', or 'rz'")

    gate_func = getattr(functional, rotation)
    n_features = min(x.shape[-1], len(wires))

    for i, wire in enumerate(wires[:n_features]):
        gate_func(q_dev, wires=[wire], params=x[:, i])


def basis_encoder(q_dev, x: torch.Tensor, wires: List[int]) -> None:
    """Basis encoding (encode binary data as |0> or |1>).

    Encodes binary data by applying X gates where the bit is 1.

    Parameters
    ----------
    q_dev : QuantumDevice
        Quantum device instance
    x : torch.Tensor
        Binary input tensor of shape ``(n_bits,)`` or ``(batch_size, n_bits)``
    wires : List[int]
        Qubit wires to encode into (first n_bits wires are used)

    Examples
    --------
    >>> device = DistributedQuantumDevice(n_wires=4, bsz=2)
    >>> binary_data = torch.tensor([[1, 0, 1, 0], [1, 0, 1, 0]])
    >>> basis_encoder(device, binary_data, wires=[0, 1, 2, 3])

    Notes
    -----
    - Values are treated as binary (non-zero values become 1)
    - This is a non-parameterized, non-differentiable encoding
    - If ``x`` is batched, all rows must be identical because gates are applied
      uniformly to the whole device batch
    """
    if x.dim() == 1:
        bitstring = x
    elif x.dim() == 2:
        if x.shape[0] > 1 and not torch.equal(x, x[0].unsqueeze(0).expand_as(x)):
            raise ValueError(
                "basis_encoder requires all batch rows to be identical because "
                "X gates are applied uniformly across the whole batch."
            )
        bitstring = x[0]
    else:
        raise ValueError(
            f"basis_encoder expected a 1D or 2D tensor, but got shape {tuple(x.shape)}."
        )

    n_bits = min(bitstring.shape[-1], len(wires))

    for i, wire in enumerate(wires[:n_bits]):
        # Apply X gate if the validated bitstring has a 1 at this position.
        if bitstring[i] == 1:
            functional.x(q_dev, wires=[wire])


def create_encoding_circuit(
    encoding_type: str = "angle", n_qubits: int = 4, n_features: int = 4, **kwargs
) -> GeneralEncoder:
    """Factory function to create common encoding circuits.

    Parameters
    ----------
    encoding_type : str, default="angle"
        Encoding type: 'angle', 'basis', or 'amplitude'
    n_qubits : int, default=4
        Number of qubits
    n_features : int, default=4
        Number of input features
    **kwargs : dict
        Additional arguments passed to encoder

    Returns
    -------
    GeneralEncoder
        Encoder instance for angle/basis encoding

    Raises
    ------
    NotImplementedError
        If encoding_type is 'amplitude' (use amplitude_encoder directly)
    ValueError
        If encoding_type is unknown

    Examples
    --------
    Create an angle encoding circuit:

    >>> encoder = create_encoding_circuit("angle", n_qubits=4, n_features=4)
    >>> # Equivalent to: [{"func": "ry", "wires": [0], "input_idx": 0}, ...]

    Create a basis encoding circuit:

    >>> encoder = create_encoding_circuit("basis", n_qubits=4, n_features=4)
    >>> # Equivalent to: [{"func": "x", "wires": [0]}, ...]

    For amplitude encoding, use amplitude_encoder directly:

    >>> from flagquantum.encoding import amplitude_encoder
    >>> amplitude_encoder(device, amplitudes)
    """
    if encoding_type == "angle":
        func_list = [
            {"func": "ry", "wires": [i], "input_idx": i}
            for i in range(min(n_features, n_qubits))
        ]
    elif encoding_type == "basis":
        func_list = [
            {"func": "x", "wires": [i]} for i in range(min(n_features, n_qubits))
        ]
    elif encoding_type == "amplitude":
        # Amplitude encoding is handled separately
        raise NotImplementedError(
            "Amplitude encoding doesn't use GeneralEncoder. "
            "Use amplitude_encoder directly."
        )
    else:
        raise ValueError(f"Unknown encoding type: {encoding_type}")

    return GeneralEncoder(func_list)


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    # Main encoder class
    "GeneralEncoder",
    # Encoding functions
    "amplitude_encoder",
    "angle_encoder",
    "basis_encoder",
    # Factory function
    "create_encoding_circuit",
]
