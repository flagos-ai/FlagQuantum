# ops/matrices.py
"""Quantum gate matrices for statevector simulation.

This module provides quantum gate matrices and precision control utilities.

Examples
--------
Basic usage:

>>> from flagquantum.ops import GATE_MAT_DICT, get_gate_matrix
>>>
>>> # Get a fixed gate
>>> x_gate = GATE_MAT_DICT["x"]
>>> print(x_gate.shape)
torch.Size([2, 2])
>>>
>>> # Get a parameterized gate
>>> rx_func = GATE_MAT_DICT["rx"]
>>> theta = torch.tensor([0.5, 1.0])  # batch of angles
>>> rx_matrix = rx_func(theta)
>>> print(rx_matrix.shape)
torch.Size([2, 2, 2])

Precision control:

>>> from flagquantum.ops import set_global_precision, get_global_precision
>>> import torch
>>>
>>> # Switch to double precision
>>> set_global_precision(torch.complex128)
>>> print(get_global_precision())
torch.complex128
>>>
>>> # All gates now use double precision
>>> x_gate = GATE_MAT_DICT["x"]
>>> print(x_gate.dtype)
torch.complex128
>>>
>>> # Switch back to single precision
>>> set_global_precision(torch.complex64)

Note: When precision is changed, all fixed gates are automatically
reconverted to the new precision. Parameterized gates generate matrices
with the current precision at call time.
"""

from typing import Callable, Dict, Union

import numpy as np
import torch

# ============================================================================
# Precision Configuration
# ============================================================================

GATE_MAT_DICT: Dict[str, Union[torch.Tensor, Callable]] = {}  # 临时定义


class PrecisionConfig:
    """Global precision configuration for quantum operations."""

    # Default precision
    _dtype = torch.complex64
    _real_dtype = torch.float32

    @classmethod
    def set_precision(cls, dtype: torch.dtype):
        """Set the global precision.

        Args:
            dtype: Complex dtype (torch.complex64 or torch.complex128)
        """
        if dtype not in [torch.complex64, torch.complex128]:
            raise ValueError(
                f"Unsupported complex dtype: {dtype}. Use complex64 or complex128."
            )

        cls._dtype = dtype
        cls._real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64

        # 精度改变后，重新转换固定门
        cls._reconvert_fixed_gates()

    @classmethod
    def get_dtype(cls) -> torch.dtype:
        """Get the current complex dtype."""
        return cls._dtype

    @classmethod
    def get_real_dtype(cls) -> torch.dtype:
        """Get the corresponding real dtype."""
        return cls._real_dtype

    @classmethod
    def to_dtype(cls, tensor: torch.Tensor) -> torch.Tensor:
        """Convert tensor to current complex dtype."""
        return tensor.to(dtype=cls._dtype)

    @classmethod
    def _create_fixed_gate(cls, matrix):
        """Create fixed gate with current precision."""
        if isinstance(matrix, torch.Tensor):
            return matrix.to(dtype=cls._dtype)
        return torch.tensor(matrix, dtype=cls._dtype)

    @classmethod
    def _reconvert_fixed_gates(cls):
        """Reconvert all fixed gate matrices to new precision."""
        global I_MATRIX, X_MATRIX, Y_MATRIX, Z_MATRIX, H_MATRIX
        global S_MATRIX, T_MATRIX, SDAG_MATRIX, TDAG_MATRIX
        global CX_MATRIX, CY_MATRIX, CZ_MATRIX, SWAP_MATRIX
        global TOFFOLI_MATRIX, FREDKIN_MATRIX
        global GATE_MAT_DICT

        # Single-qubit gates
        I_MATRIX = cls._create_fixed_gate([[1, 0], [0, 1]])
        X_MATRIX = cls._create_fixed_gate([[0, 1], [1, 0]])
        Y_MATRIX = cls._create_fixed_gate([[0, -1j], [1j, 0]])
        Z_MATRIX = cls._create_fixed_gate([[1, 0], [0, -1]])
        H_MATRIX = cls._create_fixed_gate([[1, 1], [1, -1]]) * (1.0 / np.sqrt(2))
        S_MATRIX = cls._create_fixed_gate([[1, 0], [0, 1j]])
        T_MATRIX = cls._create_fixed_gate([[1, 0], [0, np.exp(1j * np.pi / 4)]])
        SDAG_MATRIX = cls._create_fixed_gate([[1, 0], [0, -1j]])
        TDAG_MATRIX = cls._create_fixed_gate([[1, 0], [0, np.exp(-1j * np.pi / 4)]])

        # Two-qubit gates
        CX_MATRIX = cls._create_fixed_gate(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
            ]
        )
        CY_MATRIX = cls._create_fixed_gate(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, -1j],
                [0, 0, 1j, 0],
            ]
        )
        CZ_MATRIX = cls._create_fixed_gate(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, -1],
            ]
        )
        SWAP_MATRIX = cls._create_fixed_gate(
            [
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
            ]
        )

        # Three-qubit gates
        toffoli = torch.eye(8, dtype=cls._dtype)
        toffoli[6, 6] = 0
        toffoli[6, 7] = 1
        toffoli[7, 6] = 1
        toffoli[7, 7] = 0
        TOFFOLI_MATRIX = toffoli

        fredkin = torch.eye(8, dtype=cls._dtype)
        fredkin[4, 4] = 0
        fredkin[4, 5] = 1
        fredkin[5, 4] = 1
        fredkin[5, 5] = 0
        FREDKIN_MATRIX = fredkin

        # Update gate dictionary
        cls._update_gate_dict()

    @classmethod
    def _update_gate_dict(cls):
        """Update GATE_MAT_DICT with current precision gates."""
        global GATE_MAT_DICT
        GATE_MAT_DICT.update(
            {
                "i": I_MATRIX,
                "x": X_MATRIX,
                "y": Y_MATRIX,
                "z": Z_MATRIX,
                "h": H_MATRIX,
                "s": S_MATRIX,
                "sdg": SDAG_MATRIX,
                "t": T_MATRIX,
                "tdg": TDAG_MATRIX,
                "cx": CX_MATRIX,
                "cy": CY_MATRIX,
                "cz": CZ_MATRIX,
                "swap": SWAP_MATRIX,
                "ccx": TOFFOLI_MATRIX,
                "cswap": FREDKIN_MATRIX,
            }
        )


# Default instance
_precision = PrecisionConfig()


def set_global_precision(dtype: torch.dtype):
    """Set global precision for all quantum operations.

    Args:
        dtype: Complex dtype (torch.complex64 or torch.complex128)

    Example:
        >>> from flagquantum.ops import set_global_precision
        >>> import torch
        >>> set_global_precision(torch.complex128)  # Use double precision

    Note:
        After changing precision, all fixed gate matrices are automatically
        reconverted to the new precision.
    """
    _precision.set_precision(dtype)


def get_global_precision() -> torch.dtype:
    """Get current global precision."""
    return _precision.get_dtype()


def reconvert_gates() -> None:
    """Manually reconvert all fixed gates to current precision.

    This is useful if you changed precision before importing gates,
    or if you want to ensure all gates are at the current precision.

    Example:
        >>> set_global_precision(torch.complex128)
        >>> reconvert_gates()  # Force reconversion
    """
    _precision._reconvert_fixed_gates()


# ============================================================================
# Helper Functions
# ============================================================================


def _to_complex(
    tensor: Union[torch.Tensor, np.ndarray, complex, float],
) -> torch.Tensor:
    """Convert to tensor with current complex dtype."""
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor(tensor)
    return tensor.to(dtype=_precision.get_dtype())


def _create_2x2_matrix(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor
) -> torch.Tensor:
    """Create a 2x2 matrix from four components."""
    return torch.stack(
        [
            torch.cat([a, b], dim=-1),
            torch.cat([c, d], dim=-1),
        ],
        dim=-2,
    )


# ============================================================================
# Parameterized Gate Matrices (with precision support)
# ============================================================================


def rx_mat(params: torch.Tensor) -> torch.Tensor:
    """Rotation around X-axis: RX(θ) = [[cos(θ/2), -i sin(θ/2)], [-i sin(θ/2), cos(θ/2)]]"""
    theta = params.to(dtype=_precision.get_dtype())
    half_theta = theta / 2
    cos = torch.cos(half_theta)
    sin = torch.sin(half_theta)

    return _create_2x2_matrix(cos, -1j * sin, -1j * sin, cos)


def ry_mat(params: torch.Tensor) -> torch.Tensor:
    """Rotation around Y-axis: RY(θ) = [[cos(θ/2), -sin(θ/2)], [sin(θ/2), cos(θ/2)]]"""
    theta = params.to(dtype=_precision.get_dtype())
    half_theta = theta / 2
    cos = torch.cos(half_theta)
    sin = torch.sin(half_theta)

    return _create_2x2_matrix(cos, -sin, sin, cos)


def rz_mat(params: torch.Tensor) -> torch.Tensor:
    """Rotation around Z-axis: RZ(θ) = [[e^{-iθ/2}, 0], [0, e^{iθ/2}]]"""
    theta = params.to(dtype=_precision.get_dtype())
    exp_neg = torch.exp(-0.5j * theta)
    exp_pos = torch.conj(exp_neg)

    return _create_2x2_matrix(
        exp_neg, torch.zeros_like(exp_neg), torch.zeros_like(exp_pos), exp_pos
    )


# ============================================================================
# Fixed Gate Matrices
# ============================================================================

# These will be initialized with default precision (complex64)
# They will be reconverted automatically when precision changes

# Single-qubit gates
I_MATRIX = None
X_MATRIX = None
Y_MATRIX = None
Z_MATRIX = None
H_MATRIX = None
S_MATRIX = None
T_MATRIX = None
SDAG_MATRIX = None
TDAG_MATRIX = None

# Two-qubit gates
CX_MATRIX = None
CY_MATRIX = None
CZ_MATRIX = None
SWAP_MATRIX = None

# Three-qubit gates
TOFFOLI_MATRIX = None
FREDKIN_MATRIX = None

# Initialize gates (triggers _reconvert_fixed_gates)
_precision._reconvert_fixed_gates()


# ============================================================================
# Gate Dictionary
# ============================================================================

GATE_MAT_DICT: Dict[str, Union[torch.Tensor, Callable]] = {
    # Identity and Pauli gates
    "i": I_MATRIX,
    "x": X_MATRIX,
    "y": Y_MATRIX,
    "z": Z_MATRIX,
    # Clifford gates
    "h": H_MATRIX,
    "s": S_MATRIX,
    "sdg": SDAG_MATRIX,
    "t": T_MATRIX,
    "tdg": TDAG_MATRIX,
    # Rotation gates (parameterized)
    "rx": rx_mat,
    "ry": ry_mat,
    "rz": rz_mat,
    # Two-qubit gates
    "cx": CX_MATRIX,
    "cy": CY_MATRIX,
    "cz": CZ_MATRIX,
    "swap": SWAP_MATRIX,
    # Three-qubit gates
    "ccx": TOFFOLI_MATRIX,
    "cswap": FREDKIN_MATRIX,
}


# ============================================================================
# Utility Functions
# ============================================================================


def get_gate_matrix(gate_name: str) -> Union[torch.Tensor, Callable]:
    """Get gate matrix or matrix function by name."""
    if gate_name not in GATE_MAT_DICT:
        available = ", ".join(GATE_MAT_DICT.keys())
        raise KeyError(f"Gate '{gate_name}' not found. Available gates: {available}")
    return GATE_MAT_DICT[gate_name]


def list_available_gates() -> list:
    """List all available gate names."""
    return sorted(GATE_MAT_DICT.keys())


def is_parameterized_gate(gate_name: str) -> bool:
    """Check if a gate is parameterized.

    Args:
        gate_name: Name of the gate

    Returns:
        True if gate requires parameters, False otherwise
    """
    gate = GATE_MAT_DICT.get(gate_name)
    return callable(gate) if gate is not None else False


def get_gate_size(gate_name: str) -> int:
    """Get the number of qubits a gate acts on.

    Args:
        gate_name: Name of the gate

    Returns:
        Number of qubits (1, 2, or 3)
    """
    gate = get_gate_matrix(gate_name)

    if callable(gate):
        # Parameterized gates are single-qubit
        return 1

    # Fixed gates: determine size from matrix dimensions
    n = gate.shape[-1]
    return int(np.log2(n))


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    # Precision control
    "set_global_precision",
    "get_global_precision",
    "reconvert_gates",
    "PrecisionConfig",
    # Gate matrices
    "GATE_MAT_DICT",
    "rx_mat",
    "ry_mat",
    "rz_mat",
    # Utility functions
    "get_gate_matrix",
    "list_available_gates",
    "is_parameterized_gate",
    "get_gate_size",
]
