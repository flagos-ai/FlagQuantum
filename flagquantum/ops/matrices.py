"""PyTorch gate matrices used by FlagQuantum simulation engines."""

import math
from typing import Any, Callable, Dict, Union

import torch

from ..core.runtime_config import get_runtime_config, set_runtime_config

# ============================================================================
# Precision Configuration
# ============================================================================

def _complex_dtype() -> torch.dtype:
    return getattr(torch, get_runtime_config().complex_dtype)


def _real_dtype() -> torch.dtype:
    return getattr(torch, get_runtime_config().real_dtype)


def _fixed_gate(matrix: Any) -> torch.Tensor:
    """Create a complex128 master matrix for lossless execution-time casting."""
    return torch.as_tensor(matrix, dtype=torch.complex128)


def set_global_precision(dtype: torch.dtype):
    """Set the task-local default complex precision."""
    if dtype not in {torch.complex64, torch.complex128}:
        raise ValueError(
            f"Unsupported complex dtype: {dtype}. Use complex64 or complex128."
        )
    name = str(dtype).removeprefix("torch.")
    set_runtime_config(get_runtime_config().with_overrides(complex_dtype=name))


def get_global_precision() -> torch.dtype:
    """Return the task-local default complex precision."""
    return _complex_dtype()


# ============================================================================
# Helper Functions
# ============================================================================


def _to_complex(
    tensor: Union[torch.Tensor, Any, complex, float],
) -> torch.Tensor:
    """Convert a tensor to the current global complex precision."""
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor(tensor)
    return tensor.to(dtype=_complex_dtype())


def _create_2x2_matrix(a, b, c, d) -> torch.Tensor:
    """Create a batch of 2x2 matrices from four tensors."""
    return torch.cat((a, b, c, d), dim=-1).reshape(*a.shape[:-1], 2, 2)


def _create_4x4_matrix(a, b, c, d) -> torch.Tensor:
    """Create a batch of 4x4 matrices from four tensor blocks."""
    top = torch.cat([a, b], dim=-1)
    bottom = torch.cat([c, d], dim=-1)
    return torch.stack([top, bottom], dim=-2)


def _parameter_complex_dtype(params: torch.Tensor) -> torch.dtype:
    return (
        torch.complex128
        if params.dtype in {torch.float64, torch.complex128}
        else torch.complex64
    )


def _real_parameters(params: torch.Tensor | Any) -> torch.Tensor:
    """Keep rotation-angle math real until complex matrix assembly."""

    if not isinstance(params, torch.Tensor):
        return torch.as_tensor(params, dtype=_real_dtype())
    return params.real if params.is_complex() else params


def _unit_phase(angle: torch.Tensor) -> torch.Tensor:
    """Compute exp(i*angle) through real trig for portable autograd."""

    return torch.complex(torch.cos(angle), torch.sin(angle))


# ============================================================================
# Parameterized Gate Matrices
# ============================================================================


def rx_mat(params: torch.Tensor) -> torch.Tensor:
    """Generate RX gate matrix: RX(θ) = [[cos(θ/2), -i sin(θ/2)], [-i sin(θ/2), cos(θ/2)]]"""
    theta = _real_parameters(params)
    half_theta = theta / 2
    cos = torch.cos(half_theta)
    sin = torch.sin(half_theta)
    zero = torch.zeros_like(cos)
    cos_complex = torch.complex(cos, zero)
    neg_i_sin = torch.complex(zero, -sin)
    return _create_2x2_matrix(cos_complex, neg_i_sin, neg_i_sin, cos_complex)


def ry_mat(params: torch.Tensor) -> torch.Tensor:
    """Generate RY gate matrix: RY(θ) = [[cos(θ/2), -sin(θ/2)], [sin(θ/2), cos(θ/2)]]"""
    theta = _real_parameters(params)
    half_theta = theta / 2
    cos = torch.cos(half_theta)
    sin = torch.sin(half_theta)
    return _create_2x2_matrix(cos, -sin, sin, cos).to(
        dtype=_parameter_complex_dtype(params)
    )


def rz_mat(params: torch.Tensor) -> torch.Tensor:
    """Generate RZ gate matrix: RZ(θ) = [[e^{-iθ/2}, 0], [0, e^{iθ/2}]]"""
    theta = _real_parameters(params)
    exp_neg = _unit_phase(-0.5 * theta)
    exp_pos = _unit_phase(0.5 * theta)
    return _create_2x2_matrix(
        exp_neg, torch.zeros_like(exp_neg), torch.zeros_like(exp_pos), exp_pos
    )


def phase_mat(params: torch.Tensor) -> torch.Tensor:
    """Phase gate: P(θ) = [[1, 0], [0, e^{iθ}]]"""
    theta = _real_parameters(params)
    exp_theta = _unit_phase(theta)
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
    params = _real_parameters(params)
    params = _ensure_batch_dim(params, n_params=2)

    phi = params[..., 0]
    lam = params[..., 1]

    one_over_sqrt2 = 1.0 / math.sqrt(2)

    # Construct matrix elements element-wise
    a = one_over_sqrt2 * torch.ones_like(phi)  # position [0,0]
    b = -one_over_sqrt2 * _unit_phase(lam)  # position [0,1]
    c = one_over_sqrt2 * _unit_phase(phi)  # position [1,0]
    d = one_over_sqrt2 * _unit_phase(phi + lam)  # position [1,1]

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
    params = _real_parameters(params)
    params = _ensure_batch_dim(params, n_params=3)

    theta = params[..., 0]
    phi = params[..., 1]
    lam = params[..., 2]

    half_theta = theta / 2
    cos_half = torch.cos(half_theta)
    sin_half = torch.sin(half_theta)

    # Construct matrix elements element-wise
    a = cos_half  # position [0,0]
    b = -_unit_phase(lam) * sin_half  # position [0,1]
    c = _unit_phase(phi) * sin_half  # position [1,0]
    d = _unit_phase(phi + lam) * cos_half  # position [1,1]

    # Construct 2x2 matrix [a, b; c, d]
    matrices = torch.stack(
        [torch.stack([a, b], dim=-1), torch.stack([c, d], dim=-1)], dim=-2
    )

    return matrices


def crx_mat(params: torch.Tensor) -> torch.Tensor:
    """Controlled RX gate: CRX(θ)"""
    theta = _real_parameters(params)

    # Ensure it's 1D [batch]
    theta = theta.flatten()  # Key: flatten to 1D

    batch_size = theta.shape[0]
    device = theta.device
    dtype = _parameter_complex_dtype(params)

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
    theta = _real_parameters(params)
    theta = theta.flatten()  # Flatten to 1D

    batch_size = theta.shape[0]
    device = theta.device
    dtype = _parameter_complex_dtype(params)

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
    theta = _real_parameters(params)
    theta = theta.flatten()  # Flatten to 1D

    batch_size = theta.shape[0]
    device = theta.device
    dtype = _parameter_complex_dtype(params)

    exp_neg = _unit_phase(-0.5 * theta)
    exp_pos = _unit_phase(0.5 * theta)

    matrix = torch.zeros(batch_size, 4, 4, dtype=dtype, device=device)

    matrix[:, 0, 0] = 1.0
    matrix[:, 1, 1] = 1.0
    matrix[:, 2, 2] = exp_neg
    matrix[:, 3, 3] = exp_pos

    return matrix


def cphase_mat(params: torch.Tensor) -> torch.Tensor:
    """Controlled Phase gate: CPhase(θ)"""
    theta = _real_parameters(params)
    theta = theta.flatten()  # Flatten to 1D

    batch_size = theta.shape[0]
    device = theta.device
    dtype = _parameter_complex_dtype(params)

    exp_theta = _unit_phase(theta)

    matrix = torch.zeros(batch_size, 4, 4, dtype=dtype, device=device)

    matrix[:, 0, 0] = 1.0
    matrix[:, 1, 1] = 1.0
    matrix[:, 2, 2] = 1.0
    matrix[:, 3, 3] = exp_theta

    return matrix


def rxx_mat(params: torch.Tensor) -> torch.Tensor:
    """Ising XX gate: RXX(θ)"""
    theta = _real_parameters(params)
    theta = theta.flatten()  # Flatten to 1D

    batch_size = theta.shape[0]
    device = theta.device
    dtype = _parameter_complex_dtype(params)

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
    theta = _real_parameters(params)
    theta = theta.flatten()

    batch_size = theta.shape[0]
    device = theta.device
    dtype = _parameter_complex_dtype(params)

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
    theta = _real_parameters(params)
    theta = theta.flatten()

    batch_size = theta.shape[0]
    device = theta.device
    dtype = _parameter_complex_dtype(params)

    exp_neg = _unit_phase(-0.5 * theta)
    exp_pos = _unit_phase(0.5 * theta)

    matrix = torch.zeros(batch_size, 4, 4, dtype=dtype, device=device)

    matrix[:, 0, 0] = exp_neg
    matrix[:, 1, 1] = exp_pos
    matrix[:, 2, 2] = exp_pos
    matrix[:, 3, 3] = exp_neg

    return matrix


# ============================================================================
# Fixed Gate Matrices
# ============================================================================

I_MATRIX = _fixed_gate([[1, 0], [0, 1]])
X_MATRIX = _fixed_gate([[0, 1], [1, 0]])
Y_MATRIX = _fixed_gate([[0, -1j], [1j, 0]])
Z_MATRIX = _fixed_gate([[1, 0], [0, -1]])
H_MATRIX = _fixed_gate([[1, 1], [1, -1]]) / math.sqrt(2)
S_MATRIX = _fixed_gate([[1, 0], [0, 1j]])
T_MATRIX = _fixed_gate(
    [[1, 0], [0, complex(math.cos(math.pi / 4), math.sin(math.pi / 4))]]
)
SDAG_MATRIX = _fixed_gate([[1, 0], [0, -1j]])
TDAG_MATRIX = _fixed_gate(
    [[1, 0], [0, complex(math.cos(math.pi / 4), -math.sin(math.pi / 4))]]
)
SX_MATRIX = _fixed_gate(
    [[0.5 + 0.5j, 0.5 - 0.5j], [0.5 - 0.5j, 0.5 + 0.5j]]
)
SXDAG_MATRIX = _fixed_gate(
    [[0.5 - 0.5j, 0.5 + 0.5j], [0.5 + 0.5j, 0.5 - 0.5j]]
)
CX_MATRIX = _fixed_gate(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
)
CY_MATRIX = _fixed_gate(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]]
)
CZ_MATRIX = _fixed_gate(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]]
)
SWAP_MATRIX = _fixed_gate(
    [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
)
CPHASE_MATRIX = _fixed_gate(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]]
)
TOFFOLI_MATRIX = torch.eye(8, dtype=torch.complex128)
TOFFOLI_MATRIX[6:8, 6:8] = _fixed_gate([[0, 1], [1, 0]])
FREDKIN_MATRIX = torch.eye(8, dtype=torch.complex128)
FREDKIN_MATRIX[5:7, 5:7] = _fixed_gate([[0, 1], [1, 0]])


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
    complex_dtype = _complex_dtype()
    real_dtype = _real_dtype()

    k = torch.arange(n, dtype=real_dtype)
    phase_angles = (2 * torch.pi / n) * torch.outer(k, k)
    imag_unit = torch.tensor(1j, dtype=complex_dtype)
    q = torch.exp(imag_unit * phase_angles.to(dtype=complex_dtype))
    return q / torch.sqrt(torch.tensor(n, dtype=real_dtype))


# ============================================================================
# Utility Functions
# ============================================================================


__all__ = [
    "set_global_precision",
    "get_global_precision",
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
]
