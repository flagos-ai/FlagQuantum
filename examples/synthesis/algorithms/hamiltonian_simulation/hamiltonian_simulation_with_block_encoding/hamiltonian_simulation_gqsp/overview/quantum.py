"""Execute the signed controlled-walk schedule with the native block encoding."""

import torch

import flagquantum as fq
from examples.synthesis.algorithms.hamiltonian_simulation.hamiltonian_simulation_with_block_encoding.hamiltonian_simulation_qubitization.overview.model import (
    DTYPE,
)
from examples.synthesis.algorithms.hamiltonian_simulation.hamiltonian_simulation_with_block_encoding.hamiltonian_simulation_qubitization.overview.quantum import (
    controlled_walk,
    preparation,
)
from examples.synthesis.algorithms.quantum_primitives.gqsp.overview.phases import (
    rotation,
)

from .phases import DEGREE, Factorization


def evolve(data: tuple[float, ...], phases: Factorization) -> fq.Circuit:
    """Apply 33 inverse and 33 forward walk selections on opposite branches."""
    walk = controlled_walk()
    # W = R B for Hermitian B and reflection R, so W† = R W R.
    reflection = torch.diag(torch.tensor((1, 1, 1, 1, 1, -1, -1, -1), dtype=DTYPE))
    inverse = fq.Circuit(5, dtype=DTYPE).any(4, 0, 1, matrix=reflection)
    inverse.compose(walk).any(4, 0, 1, matrix=reflection)
    circuit = fq.Circuit(5, dtype=DTYPE).compose(preparation(data), qubits=(2, 3))
    circuit.any(4, matrix=torch.from_numpy(rotation(phases.rows[0])))
    for step, row in enumerate(phases.rows[1:], start=1):
        if step <= DEGREE:
            circuit.compose(inverse)
        else:
            circuit.x(4).compose(walk).x(4)
        circuit.any(4, matrix=torch.from_numpy(rotation(row)))
    return circuit
