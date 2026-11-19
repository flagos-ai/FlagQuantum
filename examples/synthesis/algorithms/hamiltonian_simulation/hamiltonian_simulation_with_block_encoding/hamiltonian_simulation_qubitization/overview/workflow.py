"""Check both absolute projections against independent dense references."""

from dataclasses import dataclass

import torch

import flagquantum as fq

from .model import (
    ALPHA,
    DTYPE,
    TIME,
    TOLERANCE,
    expansion,
    fresh_input,
    hamiltonian_matrix,
)
from .quantum import evolution_block, hamiltonian_block


@dataclass(frozen=True)
class Projection:
    """A native Circuit and its raw amplitudes in the all-zero flag sector."""

    circuit: fq.Circuit
    scale: float
    raw: torch.Tensor
    reference: torch.Tensor

    @property
    def error(self) -> float:
        """The absolute vector error with only the declared scale applied."""
        return float(torch.linalg.vector_norm(self.scale * self.raw - self.reference))

    @property
    def fidelity(self) -> float:
        """The squared normalized overlap for reporting."""
        return float(
            (
                torch.vdot(self.raw, self.reference).abs()
                / (self.raw.norm() * self.reference.norm())
            )
            ** 2
        )


def _project(circuit: fq.Circuit, scale: float, reference: torch.Tensor) -> Projection:
    state = fq.run(circuit).statevector().flatten()
    return Projection(circuit, scale, state[:4], reference)


def run_workflow(seed: int = 7) -> tuple[Projection, Projection]:
    """Execute both source projections and check unnormalized complex amplitudes."""
    amplitudes = fresh_input(seed)
    data = tuple(map(float, amplitudes))
    state = amplitudes.to(DTYPE)
    matrix = hamiltonian_matrix()
    poly = expansion()
    first = _project(hamiltonian_block(data), ALPHA, matrix @ state)
    second = _project(
        evolution_block(data, poly),
        poly.scale,
        torch.linalg.matrix_exp(-1j * TIME * matrix) @ state,
    )
    if first.error >= 1e-9:
        raise ValueError(
            f"Hamiltonian projection error {first.error} exceeds tolerance"
        )
    if second.error >= TOLERANCE:
        raise ValueError(f"Evolution projection error {second.error} exceeds tolerance")
    return first, second
