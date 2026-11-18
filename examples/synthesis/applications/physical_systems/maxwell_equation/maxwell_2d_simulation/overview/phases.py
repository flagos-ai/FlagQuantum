"""Fresh original-time Maxwell phases through the shared Laurent factorization."""

from examples.synthesis.algorithms.hamiltonian_simulation.hamiltonian_simulation_with_block_encoding.hamiltonian_simulation_gqsp.overview.phases import (
    Factorization,
    factor,
)

DEGREE, EFFECTIVE_TIME, SCALE = 686, 640.0, 0.5


def fit() -> Factorization:
    return factor(EFFECTIVE_TIME, degree=DEGREE, scale=SCALE)
