"""CPU one-qubit statevector kernels."""

from __future__ import annotations

import os

import torch


def _preallocated_output_enabled() -> bool:
    value = os.getenv("FQ_CPU_SINGLE_QUBIT_PREALLOCATE_OUTPUT", "1").strip().lower()
    return value not in {"0", "false", "off", "no"}


def apply_single_qubit_matrix_cpu(
    state: torch.Tensor,
    matrix: torch.Tensor,
    *,
    qubit: int,
    n_qubits: int,
) -> torch.Tensor:
    """Apply one CPU gate by visiting contiguous amplitude pairs directly."""

    if not 0 <= int(qubit) < int(n_qubits):
        raise ValueError("qubit is outside the statevector")
    bsz = state.shape[0]
    stride = 1 << (int(n_qubits) - int(qubit) - 1)
    paired = state.reshape(bsz, -1, 2, stride)
    zero = paired[:, :, 0, :]
    one = paired[:, :, 1, :]
    matrix = matrix.to(device=state.device, dtype=state.dtype)
    if matrix.ndim == 2:
        matrix = matrix.unsqueeze(0).expand(bsz, -1, -1)
    elif matrix.ndim == 3 and matrix.shape[0] == 1 and bsz != 1:
        matrix = matrix.expand(bsz, -1, -1)
    if matrix.shape != (bsz, 2, 2):
        raise ValueError("single-qubit matrix must have shape [2, 2] or [batch, 2, 2]")
    coefficient_shape = (bsz, 1, 1)
    if _preallocated_output_enabled() and not (
        state.requires_grad or matrix.requires_grad
    ):
        result = torch.empty_like(state)
        result_paired = result.reshape(bsz, -1, 2, stride)
        result_zero = result_paired[:, :, 0, :]
        result_one = result_paired[:, :, 1, :]
        torch.mul(zero, matrix[:, 0, 0].reshape(coefficient_shape), out=result_zero)
        result_zero.addcmul_(one, matrix[:, 0, 1].reshape(coefficient_shape))
        torch.mul(zero, matrix[:, 1, 0].reshape(coefficient_shape), out=result_one)
        result_one.addcmul_(one, matrix[:, 1, 1].reshape(coefficient_shape))
        return result
    out_zero = zero * matrix[:, 0, 0].reshape(coefficient_shape)
    out_zero = out_zero + one * matrix[:, 0, 1].reshape(coefficient_shape)
    out_one = zero * matrix[:, 1, 0].reshape(coefficient_shape)
    out_one = out_one + one * matrix[:, 1, 1].reshape(coefficient_shape)
    return torch.stack((out_zero, out_one), dim=2).reshape(state.shape)


__all__ = ("apply_single_qubit_matrix_cpu",)
