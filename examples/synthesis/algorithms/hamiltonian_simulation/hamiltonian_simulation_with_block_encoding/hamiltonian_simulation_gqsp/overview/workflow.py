"""Compare the GQSP absolute projected state with dense evolution."""

import torch

import flagquantum as fq
from examples.synthesis.algorithms.hamiltonian_simulation.hamiltonian_simulation_with_block_encoding.hamiltonian_simulation_qubitization.overview.model import (
    ALPHA,
    DTYPE,
    TIME,
    TOLERANCE,
    fresh_input,
    hamiltonian_matrix,
)
from examples.synthesis.algorithms.hamiltonian_simulation.hamiltonian_simulation_with_block_encoding.hamiltonian_simulation_qubitization.overview.quantum import (
    hamiltonian_block,
)
from examples.synthesis.algorithms.hamiltonian_simulation.hamiltonian_simulation_with_block_encoding.hamiltonian_simulation_qubitization.overview.workflow import (
    Projection,
)

from .phases import SCALE, Factorization
from .quantum import evolve


def block_projection(seed: int = 7) -> Projection:
    """Read the source Pauli encoding at the original scale."""
    initial = fresh_input(seed)
    circuit = hamiltonian_block(tuple(map(float, initial)))
    result = fq.run(circuit).statevector().flatten()
    reference = hamiltonian_matrix() @ initial.to(DTYPE)
    projection = Projection(circuit, ALPHA, result[:4], reference)
    if projection.error >= 1e-9:
        raise ValueError("The Pauli block projection exceeds the source tolerance")
    return projection


def evolution_projection(phases: Factorization, seed: int = 7) -> Projection:
    """Read the all-zero block and GQSP ancilla amplitudes with no phase alignment."""
    initial = fresh_input(seed)
    circuit = evolve(tuple(map(float, initial)), phases)
    result = fq.run(circuit).statevector().flatten().reshape(4, 4, 2)
    expected = torch.linalg.matrix_exp(-1j * TIME * hamiltonian_matrix()) @ initial.to(
        DTYPE
    )
    projection = Projection(circuit, 1 / SCALE, result[0, :, 0], expected)
    if projection.error >= TOLERANCE:
        raise ValueError("The GQSP evolution exceeds the source tolerance")
    return projection
