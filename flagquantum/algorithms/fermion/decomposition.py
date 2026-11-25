"""Static adjacent orbital rotations, independent of circuits and execution."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .numerics import static_tensor


@dataclass(frozen=True, slots=True)
class AdjacentRotation:
    """A two-orbital matrix acting on ``left, left + 1``."""

    left: int
    matrix: torch.Tensor


@dataclass(frozen=True, slots=True)
class OrbitalDecomposition:
    """Phases followed by rotations, in application order, realizing U."""

    phases: torch.Tensor
    rotations: tuple[AdjacentRotation, ...]


def checked_unitary(unitary: torch.Tensor) -> torch.Tensor:
    matrix = static_tensor(unitary, "orbital unitary", ndim=2)
    n = matrix.shape[0]
    if n < 1 or matrix.shape != (n, n):
        raise ValueError("orbital unitary must be a nonempty square matrix")
    if not torch.allclose(
        matrix.mH @ matrix, torch.eye(n, dtype=matrix.dtype), atol=1e-10, rtol=1e-10
    ):
        raise ValueError("orbital matrix must be unitary")
    return matrix


def decompose_orbital_rotation(unitary: torch.Tensor) -> OrbitalDecomposition:
    """Factor U into adjacent two-mode unitaries, preserving determinant phase.

    Matrices act on column creation amplitudes: a one-particle orbital vector
    transforms as ``v -> U @ v``. No full Fock-space matrix is constructed.
    """
    matrix = checked_unitary(unitary)
    n = matrix.shape[0]
    eliminations: list[AdjacentRotation] = []
    for column in range(n - 1):
        for row in range(n - 1, column, -1):
            a, b = matrix[row - 1, column], matrix[row, column]
            norm = torch.sqrt(a.abs().square() + b.abs().square())
            if float(b.abs()) < 1e-14:
                continue
            givens = (
                torch.stack((torch.stack((a.conj(), b.conj())), torch.stack((-b, a))))
                / norm
            )
            matrix[row - 1 : row + 1] = givens @ matrix[row - 1 : row + 1].clone()
            eliminations.append(AdjacentRotation(row - 1, givens))
    return OrbitalDecomposition(
        matrix.diagonal().clone(),
        tuple(
            AdjacentRotation(step.left, step.matrix.mH.contiguous())
            for step in reversed(eliminations)
        ),
    )
