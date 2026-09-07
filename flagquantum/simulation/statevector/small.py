"""Specialized exact kernels for very small, deep statevector workloads."""

from __future__ import annotations

import torch

from ...ops.complex_ops import complex_mul

_CONSTANT_CACHE: dict[
    tuple[int, str, torch.dtype], tuple[torch.Tensor, torch.Tensor]
] = {}


def _kernel_constants(
    n_qubits: int,
    *,
    device: torch.device,
    real_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    key = (n_qubits, str(device), real_dtype)
    cached = _CONSTANT_CACHE.get(key)
    if cached is not None:
        return cached
    complex_dtype = torch.complex128 if real_dtype == torch.float64 else torch.complex64
    basis = torch.arange(2**n_qubits, dtype=torch.int64, device=device)
    cz_signs = torch.ones(2**n_qubits, dtype=real_dtype, device=device)
    edge_count = 1 if n_qubits == 2 else n_qubits
    for wire in range(edge_count):
        target = (wire + 1) % n_qubits
        control_bit = (basis >> (n_qubits - 1 - wire)) & 1
        target_bit = (basis >> (n_qubits - 1 - target)) & 1
        cz_signs = cz_signs * (1 - 2 * (control_bit & target_bit))
    cz_signs = cz_signs.reshape((2,) * n_qubits).to(dtype=complex_dtype)
    z_signs = torch.stack(
        tuple(
            1 - 2 * ((basis >> (n_qubits - 1 - wire)) & 1) for wire in range(n_qubits)
        ),
        dim=-1,
    ).to(dtype=real_dtype)
    cached = (cz_signs, z_signs)
    _CONSTANT_CACHE[key] = cached
    return cached


def _apply_ry(state: torch.Tensor, angles: torch.Tensor, *, wire: int) -> torch.Tensor:
    cosine = torch.cos(0.5 * angles)
    sine = torch.sin(0.5 * angles)
    while cosine.ndim < state.ndim - 1:
        cosine = cosine.unsqueeze(-1)
        sine = sine.unsqueeze(-1)
    zero, one = state.unbind(dim=wire + 1)
    return torch.stack(
        (cosine * zero - sine * one, sine * zero + cosine * one),
        dim=wire + 1,
    )


def _apply_rz(state: torch.Tensor, angles: torch.Tensor, *, wire: int) -> torch.Tensor:
    half_angles = 0.5 * angles
    negative = torch.complex(torch.cos(-half_angles), torch.sin(-half_angles))
    positive = torch.complex(torch.cos(half_angles), torch.sin(half_angles))
    while negative.ndim < state.ndim - 1:
        negative = negative.unsqueeze(-1)
        positive = positive.unsqueeze(-1)
    zero, one = state.unbind(dim=wire + 1)
    return torch.stack(
        (complex_mul(zero, negative), complex_mul(one, positive)), dim=wire + 1
    )


def small_data_reuploading_z(
    inputs: torch.Tensor,
    *,
    variational_rz: torch.Tensor,
    variational_ry: torch.Tensor,
    input_ry_scale: torch.Tensor,
    input_rz_scale: torch.Tensor,
) -> torch.Tensor:
    """Execute an exact 2-4 qubit RZ/RY-CZ data-reuploading ansatz.

    The circuit starts in ``|+>**n``. Every block applies trainable RZ/RY
    rotations, a CZ ring, and tanh-scaled input RY/RZ rotations. A final
    trainable RZ/RY layer precedes exact Z measurement on every qubit.
    """

    if inputs.ndim != 2 or not 2 <= inputs.shape[1] <= 4:
        raise ValueError(
            "inputs must have shape (batch, n_qubits), for 2 <= n_qubits <= 4"
        )
    n_qubits = inputs.shape[1]
    if variational_rz.shape != variational_ry.shape:
        raise ValueError("variational RZ and RY tensors must have equal shapes")
    if variational_rz.ndim != 2 or variational_rz.shape[1] != n_qubits:
        raise ValueError("variational tensors must have shape (blocks + 1, n_qubits)")
    blocks = variational_rz.shape[0] - 1
    expected_scale_shape = (blocks, n_qubits)
    if (
        tuple(input_ry_scale.shape) != expected_scale_shape
        or tuple(input_rz_scale.shape) != expected_scale_shape
    ):
        raise ValueError(f"input scales must have shape {expected_scale_shape}")

    complex_dtype = (
        torch.complex128 if inputs.dtype == torch.float64 else torch.complex64
    )
    state = torch.full(
        (inputs.shape[0],) + (2,) * n_qubits,
        2.0 ** (-0.5 * n_qubits),
        dtype=complex_dtype,
        device=inputs.device,
    )
    cz_signs, signs = _kernel_constants(
        n_qubits, device=inputs.device, real_dtype=inputs.dtype
    )
    for block in range(blocks):
        for wire in range(n_qubits):
            state = _apply_rz(state, variational_rz[block, wire], wire=wire)
        for wire in range(n_qubits):
            state = _apply_ry(state, variational_ry[block, wire], wire=wire)
        state = state * cz_signs
        for wire in range(n_qubits):
            state = _apply_ry(
                state,
                torch.tanh(input_ry_scale[block, wire] * inputs[:, wire]),
                wire=wire,
            )
        for wire in range(n_qubits):
            state = _apply_rz(
                state,
                torch.tanh(input_rz_scale[block, wire] * inputs[:, wire]),
                wire=wire,
            )
    for wire in range(n_qubits):
        state = _apply_rz(state, variational_rz[blocks, wire], wire=wire)
    for wire in range(n_qubits):
        state = _apply_ry(state, variational_ry[blocks, wire], wire=wire)

    probabilities = state.abs().square().reshape(inputs.shape[0], 2**n_qubits)
    return probabilities @ signs


def two_qubit_data_reuploading_z(
    inputs: torch.Tensor,
    *,
    variational_rz: torch.Tensor,
    variational_ry: torch.Tensor,
    input_ry_scale: torch.Tensor,
    input_rz_scale: torch.Tensor,
) -> torch.Tensor:
    """Backward-compatible spelling for the two-qubit specialization."""

    if inputs.ndim != 2 or inputs.shape[1] != 2:
        raise ValueError("inputs must have shape (batch, 2)")
    return small_data_reuploading_z(
        inputs,
        variational_rz=variational_rz,
        variational_ry=variational_ry,
        input_ry_scale=input_ry_scale,
        input_rz_scale=input_rz_scale,
    )


__all__ = ["small_data_reuploading_z", "two_qubit_data_reuploading_z"]
