"""
Quantum gate matrices for statevector simulation.

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

GATE_MAT_DICT: Dict[str, Union[torch.Tensor, Callable]] = {}


class PrecisionConfig:
    """Global precision configuration for quantum operations."""

    _dtype = torch.complex64
    _real_dtype = torch.float32

    @classmethod
    def set_precision(cls, dtype: torch.dtype):
        """Set the global complex precision for quantum operations."""
        if dtype not in [torch.complex64, torch.complex128]:
            raise ValueError(
                f"Unsupported complex dtype: {dtype}. Use complex64 or complex128."
            )
        cls._dtype = dtype
        cls._real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
        cls._reconvert_fixed_gates()

    @classmethod
    def get_dtype(cls) -> torch.dtype:
        """Get the current global complex precision."""
        return cls._dtype

    @classmethod
    def get_real_dtype(cls) -> torch.dtype:
        """Get the corresponding real precision for the current complex precision."""
        return cls._real_dtype

    @classmethod
    def to_dtype(cls, tensor: torch.Tensor) -> torch.Tensor:
        """Convert a tensor to the current global precision."""
        return tensor.to(dtype=cls._dtype)

    @classmethod
    def _create_fixed_gate(cls, matrix):
        """Create a fixed gate matrix with the current precision."""
        if isinstance(matrix, torch.Tensor):
            return matrix.to(dtype=cls._dtype)
        return torch.tensor(matrix, dtype=cls._dtype)

    @classmethod
    def _reconvert_fixed_gates(cls):
        """Reconvert all fixed gate matrices to the current precision."""
        global I_MATRIX, X_MATRIX, Y_MATRIX, Z_MATRIX, H_MATRIX
        global S_MATRIX, T_MATRIX, SDAG_MATRIX, TDAG_MATRIX
        global SX_MATRIX, SXDAG_MATRIX
        global CX_MATRIX, CY_MATRIX, CZ_MATRIX, SWAP_MATRIX
        global CPHASE_MATRIX
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

        # SX gate (√X)
        SX_MATRIX = cls._create_fixed_gate(
            [
                [0.5 + 0.5j, 0.5 - 0.5j],
                [0.5 - 0.5j, 0.5 + 0.5j],
            ]
        )
        # SX† (SX dagger)
        SXDAG_MATRIX = cls._create_fixed_gate(
            [
                [0.5 - 0.5j, 0.5 + 0.5j],
                [0.5 + 0.5j, 0.5 - 0.5j],
            ]
        )

        # Two-qubit gates
        CX_MATRIX = cls._create_fixed_gate(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
        )
        CY_MATRIX = cls._create_fixed_gate(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]]
        )
        CZ_MATRIX = cls._create_fixed_gate(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]]
        )
        SWAP_MATRIX = cls._create_fixed_gate(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
        )
        # CPhase gate (|0⟩ control, phase on |1⟩)
        CPHASE_MATRIX = cls._create_fixed_gate(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]]
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

        cls._update_gate_dict()

    @classmethod
    def _update_gate_dict(cls):
        """Update the global gate dictionary with current precision matrices."""
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
                "sx": SX_MATRIX,
                "sxdg": SXDAG_MATRIX,
                "cx": CX_MATRIX,
                "cy": CY_MATRIX,
                "cz": CZ_MATRIX,
                "swap": SWAP_MATRIX,
                "cphase": CPHASE_MATRIX,
                "ccx": TOFFOLI_MATRIX,
                "cswap": FREDKIN_MATRIX,
            }
        )


_precision = PrecisionConfig()


def set_global_precision(dtype: torch.dtype):
    """Set the global precision for all quantum gate matrices."""
    _precision.set_precision(dtype)


def get_global_precision() -> torch.dtype:
    """Get the current global precision setting."""
    return _precision.get_dtype()


def reconvert_gates() -> None:
    """Manually reconvert all fixed gate matrices to the current precision."""
    _precision._reconvert_fixed_gates()


# ============================================================================
# Helper Functions
# ============================================================================


def _to_complex(
    tensor: Union[torch.Tensor, np.ndarray, complex, float],
) -> torch.Tensor:
    """Convert a tensor to the current global complex precision."""
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor(tensor)
    return tensor.to(dtype=_precision.get_dtype())


def _create_2x2_matrix(a, b, c, d) -> torch.Tensor:
    """Create a batch of 2x2 matrices from four tensors."""
    return torch.stack([torch.cat([a, b], dim=-1), torch.cat([c, d], dim=-1)], dim=-2)


def _create_4x4_matrix(a, b, c, d) -> torch.Tensor:
    """Create a batch of 4x4 matrices from four tensor blocks."""
    top = torch.cat([a, b], dim=-1)
    bottom = torch.cat([c, d], dim=-1)
    return torch.stack([top, bottom], dim=-2)


# ============================================================================
# Parameterized Gate Matrices
# ============================================================================


def rx_mat(params: torch.Tensor) -> torch.Tensor:
    """Generate RX gate matrix: RX(θ) = [[cos(θ/2), -i sin(θ/2)], [-i sin(θ/2), cos(θ/2)]]"""
    theta = params.to(dtype=_precision.get_dtype())
    half_theta = theta / 2
    cos = torch.cos(half_theta)
    sin = torch.sin(half_theta)
    return _create_2x2_matrix(cos, -1j * sin, -1j * sin, cos)


def ry_mat(params: torch.Tensor) -> torch.Tensor:
    """Generate RY gate matrix: RY(θ) = [[cos(θ/2), -sin(θ/2)], [sin(θ/2), cos(θ/2)]]"""
    theta = params.to(dtype=_precision.get_dtype())
    half_theta = theta / 2
    cos = torch.cos(half_theta)
    sin = torch.sin(half_theta)
    return _create_2x2_matrix(cos, -sin, sin, cos)


def rz_mat(params: torch.Tensor) -> torch.Tensor:
    """Generate RZ gate matrix: RZ(θ) = [[e^{-iθ/2}, 0], [0, e^{iθ/2}]]"""
    theta = params.to(dtype=_precision.get_dtype())
    exp_neg = torch.exp(-0.5j * theta)
    exp_pos = torch.conj(exp_neg)
    return _create_2x2_matrix(
        exp_neg, torch.zeros_like(exp_neg), torch.zeros_like(exp_pos), exp_pos
    )


def phase_mat(params: torch.Tensor) -> torch.Tensor:
    """Phase gate: P(θ) = [[1, 0], [0, e^{iθ}]]"""
    theta = params.to(dtype=_precision.get_dtype())
    exp_theta = torch.exp(1j * theta)
    one = torch.ones_like(exp_theta)
    return _create_2x2_matrix(
        one, torch.zeros_like(one), torch.zeros_like(exp_theta), exp_theta
    )


def u1_mat(params: torch.Tensor) -> torch.Tensor:
    """U1 gate: U1(θ) = [[1, 0], [0, e^{iθ}]] (alias for phase gate)"""
    return phase_mat(params)


def _ensure_batch_dim(params: torch.Tensor, n_params: int) -> torch.Tensor:
    """
    Ensure parameter tensor has the correct shape for multi-parameter gates.

    Args:
        params: Input parameter tensor
        n_params: Expected number of parameters (2 or 3)

    Returns:
        Tensor of shape [batch, n_params]
    """
    import logging

    logger = logging.getLogger(__name__)

    # Log input information
    logger.debug(
        f"_ensure_batch_dim input: params={params}, type={type(params)}, n_params={n_params}"
    )

    # Handle scalar or Python numeric input
    if not isinstance(params, torch.Tensor):
        logger.debug("params is not a tensor, converting to tensor")
        params = torch.tensor(params)

    logger.debug(f"After conversion: params shape={params.shape}, dtype={params.dtype}")

    # Flatten to 1D
    original_shape = params.shape
    params = params.flatten()
    logger.debug(
        f"After flatten: original shape {original_shape} -> new shape {params.shape}"
    )

    # Check parameter count
    if params.shape[0] % n_params != 0:
        error_msg = (
            f"Number of parameters must be a multiple of {n_params}, but got {params.shape[0]}. "
            f"Original input: shape={original_shape}, values={params.tolist() if params.numel() < 10 else '...'}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Reshape to [batch, n_params]
    batch_size = params.shape[0] // n_params
    params = params.reshape(batch_size, n_params)
    logger.debug(f"After reshape: batch_size={batch_size}, final shape={params.shape}")

    return params


def u2_mat(params: torch.Tensor) -> torch.Tensor:
    """
    U2 gate: U2(φ, λ) = [[1, -e^{iλ}], [e^{iφ}, e^{i(φ+λ)}]] / √2

    Args:
        params: Accepts multiple formats
            - Scalar: Not supported (requires 2 parameters)
            - List/tuple: [phi, lam]
            - 1D tensor: [phi, lam]
            - 2D tensor: [[phi1, lam1], [phi2, lam2], ...] for batch mode
    """
    params = torch.as_tensor(params, dtype=_precision.get_dtype())
    params = _ensure_batch_dim(params, n_params=2)

    phi = params[..., 0]
    lam = params[..., 1]

    one_over_sqrt2 = 1.0 / np.sqrt(2)

    # Construct matrix elements element-wise
    a = one_over_sqrt2 * torch.ones_like(phi)  # position [0,0]
    b = -one_over_sqrt2 * torch.exp(1j * lam)  # position [0,1]
    c = one_over_sqrt2 * torch.exp(1j * phi)  # position [1,0]
    d = one_over_sqrt2 * torch.exp(1j * (phi + lam))  # position [1,1]

    # Construct 2x2 matrix [a, b; c, d]
    # Batch version: shape [batch, 2, 2]
    matrices = torch.stack(
        [torch.stack([a, b], dim=-1), torch.stack([c, d], dim=-1)], dim=-2
    )

    return matrices


def u3_mat(params: torch.Tensor) -> torch.Tensor:
    """
    U3 gate: U3(θ, φ, λ) = [[cos(θ/2), -e^{iλ} sin(θ/2)], [e^{iφ} sin(θ/2), e^{i(φ+λ)} cos(θ/2)]]

    Args:
        params: Accepts multiple formats
            - Scalar: Not supported (requires 3 parameters)
            - List/tuple: [theta, phi, lam]
            - 1D tensor: [theta, phi, lam]
            - 2D tensor: [[theta1, phi1, lam1], [theta2, phi2, lam2], ...] for batch mode
    """
    # Ensure correct shape
    params = torch.as_tensor(params, dtype=_precision.get_dtype())
    params = _ensure_batch_dim(params, n_params=3)

    theta = params[..., 0]
    phi = params[..., 1]
    lam = params[..., 2]

    half_theta = theta / 2
    cos_half = torch.cos(half_theta)
    sin_half = torch.sin(half_theta)

    # Construct matrix elements element-wise
    a = cos_half  # position [0,0]
    b = -torch.exp(1j * lam) * sin_half  # position [0,1]
    c = torch.exp(1j * phi) * sin_half  # position [1,0]
    d = torch.exp(1j * (phi + lam)) * cos_half  # position [1,1]

    # Construct 2x2 matrix [a, b; c, d]
    matrices = torch.stack(
        [torch.stack([a, b], dim=-1), torch.stack([c, d], dim=-1)], dim=-2
    )

    return matrices


def crx_mat(params: torch.Tensor) -> torch.Tensor:
    """Controlled RX gate: CRX(θ)"""
    theta = params.to(dtype=_precision.get_dtype())

    # Ensure it's 1D [batch]
    theta = theta.flatten()  # Key: flatten to 1D

    batch_size = theta.shape[0]
    device = theta.device
    dtype = _precision.get_dtype()

    half_theta = theta / 2
    cos = torch.cos(half_theta)  # [batch]
    sin = torch.sin(half_theta)  # [batch]

    matrix = torch.zeros(batch_size, 4, 4, dtype=dtype, device=device)

    matrix[:, 0, 0] = 1.0
    matrix[:, 1, 1] = 1.0
    matrix[:, 2, 2] = cos
    matrix[:, 2, 3] = -1j * sin
    matrix[:, 3, 2] = -1j * sin
    matrix[:, 3, 3] = cos

    return matrix


def cry_mat(params: torch.Tensor) -> torch.Tensor:
    """Controlled RY gate: CRY(θ)"""
    theta = params.to(dtype=_precision.get_dtype())
    theta = theta.flatten()  # Flatten to 1D

    batch_size = theta.shape[0]
    device = theta.device
    dtype = _precision.get_dtype()

    half_theta = theta / 2
    cos = torch.cos(half_theta)
    sin = torch.sin(half_theta)

    matrix = torch.zeros(batch_size, 4, 4, dtype=dtype, device=device)

    matrix[:, 0, 0] = 1.0
    matrix[:, 1, 1] = 1.0
    matrix[:, 2, 2] = cos
    matrix[:, 2, 3] = -sin
    matrix[:, 3, 2] = sin
    matrix[:, 3, 3] = cos

    return matrix


def crz_mat(params: torch.Tensor) -> torch.Tensor:
    """Controlled RZ gate: CRZ(θ)"""
    theta = params.to(dtype=_precision.get_dtype())
    theta = theta.flatten()  # Flatten to 1D

    batch_size = theta.shape[0]
    device = theta.device
    dtype = _precision.get_dtype()

    exp_neg = torch.exp(-0.5j * theta)
    exp_pos = torch.conj(exp_neg)

    matrix = torch.zeros(batch_size, 4, 4, dtype=dtype, device=device)

    matrix[:, 0, 0] = 1.0
    matrix[:, 1, 1] = 1.0
    matrix[:, 2, 2] = exp_neg
    matrix[:, 3, 3] = exp_pos

    return matrix


def cphase_mat(params: torch.Tensor) -> torch.Tensor:
    """Controlled Phase gate: CPhase(θ)"""
    theta = params.to(dtype=_precision.get_dtype())
    theta = theta.flatten()  # Flatten to 1D

    batch_size = theta.shape[0]
    device = theta.device
    dtype = _precision.get_dtype()

    exp_theta = torch.exp(1j * theta)

    matrix = torch.zeros(batch_size, 4, 4, dtype=dtype, device=device)

    matrix[:, 0, 0] = 1.0
    matrix[:, 1, 1] = 1.0
    matrix[:, 2, 2] = 1.0
    matrix[:, 3, 3] = exp_theta

    return matrix


def rxx_mat(params: torch.Tensor) -> torch.Tensor:
    """Ising XX gate: RXX(θ)"""
    theta = params.to(dtype=_precision.get_dtype())
    theta = theta.flatten()  # Flatten to 1D

    batch_size = theta.shape[0]
    device = theta.device
    dtype = _precision.get_dtype()

    half_theta = theta / 2
    cos = torch.cos(half_theta)
    sin = torch.sin(half_theta)
    neg_i_sin = -1j * sin

    matrix = torch.zeros(batch_size, 4, 4, dtype=dtype, device=device)

    matrix[:, 0, 0] = cos
    matrix[:, 0, 3] = neg_i_sin
    matrix[:, 1, 1] = cos
    matrix[:, 1, 2] = neg_i_sin
    matrix[:, 2, 1] = neg_i_sin
    matrix[:, 2, 2] = cos
    matrix[:, 3, 0] = neg_i_sin
    matrix[:, 3, 3] = cos

    return matrix


def ryy_mat(params: torch.Tensor) -> torch.Tensor:
    """Ising YY gate: RYY(θ)"""
    theta = params.to(dtype=_precision.get_dtype())
    theta = theta.flatten()

    batch_size = theta.shape[0]
    device = theta.device
    dtype = _precision.get_dtype()

    half_theta = theta / 2
    cos = torch.cos(half_theta)
    sin = torch.sin(half_theta)
    neg_i_sin = -1j * sin

    matrix = torch.zeros(batch_size, 4, 4, dtype=dtype, device=device)

    matrix[:, 0, 0] = cos
    matrix[:, 0, 3] = 1j * sin
    matrix[:, 1, 1] = cos
    matrix[:, 1, 2] = neg_i_sin
    matrix[:, 2, 1] = neg_i_sin
    matrix[:, 2, 2] = cos
    matrix[:, 3, 0] = 1j * sin
    matrix[:, 3, 3] = cos

    return matrix


def rzz_mat(params: torch.Tensor) -> torch.Tensor:
    """Ising ZZ gate: RZZ(θ)"""
    theta = params.to(dtype=_precision.get_dtype())
    theta = theta.flatten()

    batch_size = theta.shape[0]
    device = theta.device
    dtype = _precision.get_dtype()

    exp_neg = torch.exp(-0.5j * theta)
    exp_pos = torch.conj(exp_neg)

    matrix = torch.zeros(batch_size, 4, 4, dtype=dtype, device=device)

    matrix[:, 0, 0] = exp_neg
    matrix[:, 1, 1] = exp_pos
    matrix[:, 2, 2] = exp_pos
    matrix[:, 3, 3] = exp_neg

    return matrix


# ============================================================================
# Fixed Gate Matrices
# ============================================================================

I_MATRIX = None
X_MATRIX = None
Y_MATRIX = None
Z_MATRIX = None
H_MATRIX = None
S_MATRIX = None
T_MATRIX = None
SDAG_MATRIX = None
TDAG_MATRIX = None
SX_MATRIX = None
SXDAG_MATRIX = None
CX_MATRIX = None
CY_MATRIX = None
CZ_MATRIX = None
SWAP_MATRIX = None
CPHASE_MATRIX = None
TOFFOLI_MATRIX = None
FREDKIN_MATRIX = None

_precision._reconvert_fixed_gates()


# ============================================================================
# Gate Dictionary
# ============================================================================

GATE_MAT_DICT: Dict[str, Union[torch.Tensor, Callable]] = {
    # Identity and Pauli
    "i": I_MATRIX,
    "x": X_MATRIX,
    "y": Y_MATRIX,
    "z": Z_MATRIX,
    # Clifford
    "h": H_MATRIX,
    "s": S_MATRIX,
    "sdg": SDAG_MATRIX,
    "t": T_MATRIX,
    "tdg": TDAG_MATRIX,
    # SX
    "sx": SX_MATRIX,
    "sxdg": SXDAG_MATRIX,
    # Parameterized single-qubit
    "rx": rx_mat,
    "ry": ry_mat,
    "rz": rz_mat,
    "p": phase_mat,
    "phase": phase_mat,
    "u1": u1_mat,
    "u2": u2_mat,
    "u3": u3_mat,
    # Parameterized two-qubit
    "crx": crx_mat,
    "cry": cry_mat,
    "crz": crz_mat,
    "cphase": cphase_mat,
    "rxx": rxx_mat,
    "ryy": ryy_mat,
    "rzz": rzz_mat,
    # Fixed two-qubit
    "cx": CX_MATRIX,
    "cy": CY_MATRIX,
    "cz": CZ_MATRIX,
    "swap": SWAP_MATRIX,
    # Fixed three-qubit
    "ccx": TOFFOLI_MATRIX,
    "cswap": FREDKIN_MATRIX,
    # Aliases
    "hadamard": H_MATRIX,
    "cnot": CX_MATRIX,
    "toffoli": TOFFOLI_MATRIX,
    "fredkin": FREDKIN_MATRIX,
}


# ============================================================================
# QFT (Dynamically generated, supports arbitrary number of qubits)
# ============================================================================


def qft_matrix(n_qubits: int) -> torch.Tensor:
    """Generate QFT matrix for n qubits."""
    n = 2**n_qubits
    complex_dtype = _precision.get_dtype()
    real_dtype = _precision.get_real_dtype()

    k = torch.arange(n, dtype=real_dtype)
    phase_angles = (2 * torch.pi / n) * torch.outer(k, k)
    imag_unit = torch.tensor(1j, dtype=complex_dtype)
    q = torch.exp(imag_unit * phase_angles.to(dtype=complex_dtype))
    return q / torch.sqrt(torch.tensor(n, dtype=real_dtype))


# ============================================================================
# Utility Functions
# ============================================================================


def get_gate_matrix(gate_name: str) -> Union[torch.Tensor, Callable]:
    """Retrieve the gate matrix or matrix generation function for a given gate name."""
    if gate_name not in GATE_MAT_DICT:
        available = ", ".join(GATE_MAT_DICT.keys())
        raise KeyError(f"Gate '{gate_name}' not found. Available gates: {available}")
    return GATE_MAT_DICT[gate_name]


def list_available_gates() -> list:
    """Return a sorted list of all available gate names."""
    return sorted(GATE_MAT_DICT.keys())


def is_parameterized_gate(gate_name: str) -> bool:
    """Check if a gate is parameterized (requires parameters at call time)."""
    gate = GATE_MAT_DICT.get(gate_name)
    return callable(gate) if gate is not None else False


def get_gate_size(gate_name: str) -> int:
    """Get the number of qubits a gate acts on."""
    gate = get_gate_matrix(gate_name)
    if callable(gate):
        if gate_name in ["crx", "cry", "crz", "cphase", "rxx", "ryy", "rzz"]:
            return 2
        return 1
    n = gate.shape[-1]
    return int(np.log2(n))


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    "set_global_precision",
    "get_global_precision",
    "reconvert_gates",
    "PrecisionConfig",
    "GATE_MAT_DICT",
    "rx_mat",
    "ry_mat",
    "rz_mat",
    "phase_mat",
    "u1_mat",
    "u2_mat",
    "u3_mat",
    "crx_mat",
    "cry_mat",
    "crz_mat",
    "cphase_mat",
    "rxx_mat",
    "ryy_mat",
    "rzz_mat",
    "qft_matrix",
    "get_gate_matrix",
    "list_available_gates",
    "is_parameterized_gate",
    "get_gate_size",
]
