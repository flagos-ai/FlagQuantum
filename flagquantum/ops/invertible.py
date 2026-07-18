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

# ops/invertible.py
"""Invertible quantum unitary operations with optional noise models.

This module provides an invertible unitary module that applies a sequence
of quantum gates with optional depolarizing noise for realistic simulation.

Examples
--------
Basic usage without noise:

>>> from flagquantum.ops import InvertibleUnitary, rx, cx
>>> from flagquantum.devices import DistributedQuantumDevice
>>>
>>> # Create gates
>>> gates = [
...     rx(0, params=0.5),
...     cx(0, 1),
...     rx(1, params=0.3),
... ]
>>>
>>> # Create invertible unitary layer (no noise)
>>> layer = InvertibleUnitary(gates)
>>>
>>> # Apply to quantum device
>>> device = DistributedQuantumDevice(n_wires=2, bsz=4)
>>> inp = torch.randn(4, 2)  # Input for parameterized gates
>>> layer(device, inp)

With depolarizing noise:

>>> # Single probability for both 1q and 2q errors
>>> noisy_layer = InvertibleUnitary(gates, error_probs=0.01)
>>>
>>> # Different probabilities for 1q and 2q errors
>>> noisy_layer = InvertibleUnitary(gates, error_probs=[0.01, 0.02])
>>>
>>> noisy_layer(device, inp)

Using the helper function:

>>> from flagquantum.ops import make_noisy_layer
>>>
>>> noisy_layer = make_noisy_layer(
...     gates,
...     error_rate_1q=0.01,
...     error_rate_2q=0.02
... )
>>> noisy_layer(device, inp)

With GeneralEncoder:

>>> from flagquantum.encoding import GeneralEncoder
>>>
>>> encoder = GeneralEncoder([
...     {"func": "ry", "wires": [0], "input_idx": 0},
...     {"func": "ry", "wires": [1], "input_idx": 1},
... ])
>>>
>>> gates_with_encoder = [encoder, cx(0, 1)]
>>> layer = InvertibleUnitary(gates_with_encoder, error_probs=0.005)
>>> layer(device, inp)

Notes
-----
- Noise is applied after each gate (depolarizing channel)
- 1-qubit noise: randomly applies I, X, Y, or Z with equal probability
- 2-qubit noise: randomly applies tensor products of I,X,Y,Z on both qubits
- Encoder gates (GeneralEncoder) do NOT receive noise
- The module is invertible for memory-efficient backpropagation
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from ..encoding.encoder import GeneralEncoder
from .functional import InvertiblePostUnitaryStep, gate
from .matrices import GATE_MAT_DICT
from .operator import Op

# ============================================================================
# Constants
# ============================================================================

_NOISE_GATES = ["x", "y", "z", "i"]
_N_1Q_NOISE_GATES = len(_NOISE_GATES)  # 4
_N_2Q_NOISE_COMBOS = _N_1Q_NOISE_GATES**2  # 16
_NOISE_SCALING_2Q = 16 / 15  # Adjustment for uniform distribution


# ============================================================================
# InvertibleUnitary Module
# ============================================================================


class InvertibleUnitary(nn.Module):
    """Invertible unitary module with optional depolarizing noise.

    This module applies a sequence of quantum gates with optional
    depolarizing noise after each gate. It supports both fixed gates
    and parameterized encoders.

    Parameters
    ----------
    gates : List[Union[nn.Module, Op]]
        List of quantum gates or GeneralEncoder instances
    error_probs : float or List[float], default=0.0
        Error probabilities for depolarizing noise.
        - If float: same probability for 1q and 2q gates
        - If [p1, p2]: p1 for 1-qubit gates, p2 for 2-qubit gates

    Attributes
    ----------
    gates : nn.ModuleList
        List of gates to apply
    prob_1q : float
        Probability of 1-qubit depolarizing error (0 to 1)
    prob_2q : float
        Probability of 2-qubit depolarizing error (0 to 1)

    Examples
    --------
    Create and use a noisy layer:

    >>> from flagquantum.ops import InvertibleUnitary, rx, cx
    >>> from flagquantum.devices import DistributedQuantumDevice
    >>>
    >>> device = DistributedQuantumDevice(n_wires=2, bsz=4)
    >>> gates = [rx(0, params=0.5), cx(0, 1)]
    >>> layer = InvertibleUnitary(gates, error_probs=0.01)
    >>> inp = torch.randn(4, 1)
    >>> layer(device, inp)

    Different noise rates for different gate types:

    >>> layer = InvertibleUnitary(gates, error_probs=[0.005, 0.02])

    Without noise:

    >>> layer = InvertibleUnitary(gates)  # or error_probs=0.0

    See Also
    --------
    make_noisy_layer : Convenience function for creating noisy layers
    """

    def __init__(
        self,
        gates: List[Union[nn.Module, Op]],
        error_probs: Union[float, List[float], Tuple[float, float]] = 0.0,
    ):
        """Initialize invertible unitary with optional noise.

        Examples
        --------
        >>> # List of gates
        >>> from flagquantum.ops import rx, cx
        >>> gates = [rx(0, params=0.5), cx(0, 1)]
        >>>
        >>> # With noise probability 1%
        >>> layer = InvertibleUnitary(gates, error_probs=0.01)
        >>>
        >>> # With different probabilities for 1q and 2q
        >>> layer = InvertibleUnitary(gates, error_probs=[0.005, 0.02])
        """
        super().__init__()

        # Store gates as ModuleList
        self.gates = nn.ModuleList(gates)

        # Parse error probabilities
        self.prob_1q, self.prob_2q = self._parse_error_probs(error_probs)

        # Precompute noise gate tensors for efficiency
        self._noise_gates = self._precompute_noise_gates()

    @staticmethod
    def _parse_error_probs(
        error_probs: Union[float, List[float], Tuple[float, float]],
    ) -> Tuple[float, float]:
        """Parse and validate error probabilities.

        Parameters
        ----------
        error_probs : float or list/tuple of 2 floats
            Error probability or [p1, p2]

        Returns
        -------
        Tuple[float, float]
            (prob_1q, prob_2q)

        Raises
        ------
        ValueError
            If probabilities are out of range or incorrectly formatted

        Examples
        --------
        >>> InvertibleUnitary._parse_error_probs(0.01)
        (0.01, 0.01)
        >>> InvertibleUnitary._parse_error_probs([0.005, 0.02])
        (0.005, 0.02)
        """
        if isinstance(error_probs, (float, int)):
            prob_1q = prob_2q = float(error_probs)
        elif len(error_probs) >= 2:
            prob_1q = float(error_probs[0])
            prob_2q = float(error_probs[1])
        else:
            raise ValueError(
                f"error_probs must be float or [p1, p2], got {error_probs}"
            )

        # Validate ranges
        if not (0 <= prob_1q <= 1):
            raise ValueError(
                f"1-qubit error probability must be in [0,1], got {prob_1q}"
            )
        if not (0 <= prob_2q <= 1):
            raise ValueError(
                f"2-qubit error probability must be in [0,1], got {prob_2q}"
            )

        return prob_1q, prob_2q

    def _precompute_noise_gates(self) -> List[torch.Tensor]:
        """Precompute noise gate tensors for efficient access.

        Returns
        -------
        List[torch.Tensor]
            List of noise gate matrices [I, X, Y, Z]

        Examples
        --------
        >>> layer = InvertibleUnitary([])
        >>> noise_gates = layer._precompute_noise_gates()
        >>> len(noise_gates)
        4
        >>> noise_gates[0].shape  # I gate
        torch.Size([2, 2])
        """
        return [GATE_MAT_DICT[name] for name in _NOISE_GATES]

    def _apply_1q_noise(self, qdev, wires: List[int]) -> None:
        """Apply random 1-qubit depolarizing noise.

        Parameters
        ----------
        qdev : QuantumDevice
            Quantum device instance
        wires : List[int]
            Wire to apply noise on (should contain exactly 1 wire)

        Examples
        --------
        >>> device = DistributedQuantumDevice(n_wires=2)
        >>> layer = InvertibleUnitary([], error_probs=0.01)
        >>> layer._apply_1q_noise(device, [0])  # Apply 1% chance of error on wire 0
        """
        if self.prob_1q <= 0:
            return

        rand = torch.rand(1).item()
        if rand < self.prob_1q:
            # Choose random noise gate (uniform among I, X, Y, Z)
            idx = int(rand * _N_1Q_NOISE_GATES / self.prob_1q)
            idx = min(idx, _N_1Q_NOISE_GATES - 1)  # Clamp to valid range
            gate(self._noise_gates[idx], qdev, wires)

    def _apply_2q_noise(self, qdev, wires: List[int]) -> None:
        """Apply random 2-qubit depolarizing noise.

        The noise is applied as a random tensor product of single-qubit
        Pauli errors (I, X, Y, Z) on both qubits.

        Parameters
        ----------
        qdev : QuantumDevice
            Quantum device instance
        wires : List[int]
            Two wires to apply noise on

        Examples
        --------
        >>> device = DistributedQuantumDevice(n_wires=2)
        >>> layer = InvertibleUnitary([], error_probs=[0, 0.02])
        >>> layer._apply_2q_noise(device, [0, 1])  # 2% chance of error on both wires
        """
        if self.prob_2q <= 0 or len(wires) != 2:
            return

        rand = torch.rand(1).item()
        # Scale probability to account for 16 possible errors (including I⊗I)
        threshold = self.prob_2q * _NOISE_SCALING_2Q

        if rand < threshold:
            # Choose random Pauli pair (uniform over 16 combinations)
            scaled = rand * _N_2Q_NOISE_COMBOS / threshold
            idx1 = int(scaled) % _N_1Q_NOISE_GATES
            idx2 = int(scaled) // _N_1Q_NOISE_GATES

            # Combine into single 2-qubit gate
            combined_gate = torch.kron(self._noise_gates[idx1], self._noise_gates[idx2])
            gate(combined_gate, qdev, wires)

    def _apply_noise(self, qdev, wires: Union[int, List[int]]) -> None:
        """Apply depolarizing noise after a gate.

        Parameters
        ----------
        qdev : QuantumDevice
            Quantum device instance
        wires : int or List[int]
            Wire(s) the gate was applied to

        Examples
        --------
        >>> device = DistributedQuantumDevice(n_wires=2)
        >>> layer = InvertibleUnitary([], error_probs=0.01)
        >>> layer._apply_noise(device, 0)      # 1-qubit noise
        >>> layer._apply_noise(device, [0, 1]) # 2-qubit noise
        """
        if isinstance(wires, int):
            wires = [wires]

        if len(wires) == 1:
            self._apply_1q_noise(qdev, wires)
        elif len(wires) == 2:
            self._apply_2q_noise(qdev, wires)
        # Gates with >2 qubits get no noise (silent skip)

    def forward(self, qdev, inp: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply the unitary sequence with optional noise.

        Parameters
        ----------
        qdev : QuantumDevice
            Quantum device instance
        inp : torch.Tensor, optional
            Input tensor for parameterized encoders (if any)

        Returns
        -------
        torch.Tensor
            Modified quantum device states (for compatibility)

        Examples
        --------
        Basic usage:

        >>> from flagquantum.ops import InvertibleUnitary, rx, cx
        >>> from flagquantum.devices import DistributedQuantumDevice
        >>>
        >>> device = DistributedQuantumDevice(n_wires=2, bsz=4)
        >>> gates = [rx(0, params=0.5), cx(0, 1)]
        >>> layer = InvertibleUnitary(gates, error_probs=0.01)
        >>> inp = torch.randn(4, 1)
        >>> output = layer(device, inp)

        With encoder:

        >>> from flagquantum.encoding import GeneralEncoder
        >>>
        >>> encoder = GeneralEncoder([
        ...     {"func": "ry", "wires": [0], "input_idx": 0},
        ... ])
        >>> gates_with_encoder = [encoder, cx(0, 1)]
        >>> layer = InvertibleUnitary(gates_with_encoder)
        >>> output = layer(device, inp)  # inp used by encoder
        """
        for gate_module in self.gates:
            # Apply the gate/encoder
            if isinstance(gate_module, GeneralEncoder):
                gate_module(qdev, inp)
            else:
                # Assume it's a gate with .wires attribute
                gate_module(qdev)

                # Apply depolarizing noise if configured
                if hasattr(gate_module, "wires"):
                    self._apply_noise(qdev, gate_module.wires)

        # Mark as invertible step
        qdev._states = InvertiblePostUnitaryStep.apply(
            qdev._states, qdev._invertible_dummy
        )

        return qdev._states

    def __repr__(self) -> str:
        """Return string representation of the module.

        Examples
        --------
        >>> layer = InvertibleUnitary([], error_probs=[0.01, 0.02])
        >>> print(layer)
        InvertibleUnitary(n_gates=0, prob_1q=0.01, prob_2q=0.02)
        """
        return (
            f"InvertibleUnitary("
            f"n_gates={len(self.gates)}, "
            f"prob_1q={self.prob_1q}, "
            f"prob_2q={self.prob_2q})"
        )


# ============================================================================
# Helper Functions
# ============================================================================


def make_noisy_layer(
    gates: List[nn.Module], error_rate_1q: float = 0.0, error_rate_2q: float = 0.0
) -> InvertibleUnitary:
    """Create a noisy invertible unitary layer.

    Convenience function for creating an InvertibleUnitary with noise.

    Parameters
    ----------
    gates : List[nn.Module]
        List of quantum gates
    error_rate_1q : float, default=0.0
        Depolarizing error rate for 1-qubit gates (0 to 1)
    error_rate_2q : float, default=0.0
        Depolarizing error rate for 2-qubit gates (0 to 1)

    Returns
    -------
    InvertibleUnitary
        Configured noisy invertible unitary layer

    Examples
    --------
    Create a layer with only 1-qubit noise:

    >>> from flagquantum.ops import make_noisy_layer, rx, cx
    >>> gates = [rx(0, params=0.5), cx(0, 1)]
    >>> noisy = make_noisy_layer(gates, error_rate_1q=0.01, error_rate_2q=0.0)

    Create a layer with only 2-qubit noise:

    >>> noisy = make_noisy_layer(gates, error_rate_1q=0.0, error_rate_2q=0.02)

    Create a layer with noise on both gate types:

    >>> noisy = make_noisy_layer(gates, error_rate_1q=0.005, error_rate_2q=0.01)

    Apply to device:

    >>> from flagquantum.devices import DistributedQuantumDevice
    >>> device = DistributedQuantumDevice(n_wires=2, bsz=4)
    >>> inp = torch.randn(4, 1)
    >>> noisy(device, inp)

    See Also
    --------
    InvertibleUnitary : The underlying class
    """
    return InvertibleUnitary(gates, error_probs=[error_rate_1q, error_rate_2q])


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    "InvertibleUnitary",
    "make_noisy_layer",
]
