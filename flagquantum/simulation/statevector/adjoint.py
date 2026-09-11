"""Pure local numerics for statevector adjoint differentiation."""

from __future__ import annotations

import torch

from ...core.ir import Instruction


def analytic_rotation_derivative(
    instruction: Instruction,
    matrix: torch.Tensor,
) -> torch.Tensor | None:
    """Return the matrix derivative for a supported Pauli rotation."""

    generators = {
        "rx": ((0, 1), (1, 0)),
        "ry": ((0, -1j), (1j, 0)),
        "rz": ((1, 0), (0, -1)),
    }
    values = generators.get(instruction.name)
    if values is None or tuple(instruction.params) != ("theta",):
        return None
    generator = torch.tensor(values, dtype=matrix.dtype, device=matrix.device)
    return (-0.5j) * (generator @ matrix)


def real_conjugate_inner_sum(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Return the real part of the conjugate inner product without ``conj``."""

    return torch.sum(left.real * right.real + left.imag * right.imag)


def z_expectation_chunk(
    amplitudes: torch.Tensor,
    global_indices: torch.Tensor,
    *,
    n_wires: int,
    wire: int,
) -> torch.Tensor:
    """Return one amplitude chunk's contribution to a Z expectation."""

    bit = (global_indices >> (n_wires - wire - 1)) & 1
    signs = (1 - 2 * bit).to(dtype=amplitudes.real.dtype)
    return (amplitudes.abs().square() * signs.reshape(1, -1)).sum()


def z_expectation_adjoint_chunk(
    amplitudes: torch.Tensor,
    global_indices: torch.Tensor,
    *,
    n_wires: int,
    wire: int,
) -> torch.Tensor:
    """Return one amplitude chunk's derivative of a Z expectation."""

    bit = (global_indices >> (n_wires - wire - 1)) & 1
    signs = (1 - 2 * bit).to(dtype=amplitudes.real.dtype)
    return 2 * amplitudes * signs.reshape(1, -1)


__all__ = (
    "analytic_rotation_derivative",
    "real_conjugate_inner_sum",
    "z_expectation_adjoint_chunk",
    "z_expectation_chunk",
)
