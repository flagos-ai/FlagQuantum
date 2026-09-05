"""Pure local numerics for statevector adjoint differentiation."""

from __future__ import annotations

import torch

from ..core.ir import Instruction


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


__all__ = ("analytic_rotation_derivative", "real_conjugate_inner_sum")
