"""Independent dense Circuit: explicit Givens matrices and exchange exponentials."""

import torch

import flagquantum as fq
from examples.synthesis.applications.physical_systems.fermi_hubbard_model_1d.fermi_hubbard_1d.main import (
    build_program,
)
from examples.synthesis.compare_states import compare_states

# Independent data copy prevents native matrix or target-order changes from
# rewriting the oracle together with the implementation.
G0 = ((1, 0, 0, 0), (0, 0.0, (1 + 0j), 0), (0, -1.0, 0j, 0), (0, 0, 0, 1))
G1 = (
    (1, 0, 0, 0),
    (0, 0.709249763003271, (0.704957285003711 + 0j), 0),
    (0, -0.704957285003711, (0.709249763003271 + 0j), 0),
    (0, 0, 0, 1),
)
G2 = (
    (1, 0, 0, 0),
    (0, 0.371748034460185, (0.928333667856008 + 0j), 0),
    (0, -0.928333667856008, (0.371748034460185 + 0j), 0),
    (0, 0, 0, 1),
)
G3 = (
    (1, 0, 0, 0),
    (0, 0.118212280403769, (0.992988346740152 + 0j), 0),
    (0, -0.992988346740152, (0.118212280403769 + 0j), 0),
    (0, 0, 0, 1),
)
G4 = (
    (1, 0, 0, 0),
    (0, 0.647936163294299, (0.761694642422713 + 0j), 0),
    (0, -0.761694642422713, (0.647936163294299 + 0j), 0),
    (0, 0, 0, 1),
)
G5 = (
    (1, 0, 0, 0),
    (0, 0.089023644499293, (0.99602951297643 + 0j), 0),
    (0, -0.99602951297643, (0.089023644499293 + 0j), 0),
    (0, 0, 0, 1),
)
G6 = (
    (1, 0, 0, 0),
    (0, 0.999572706159981, (0.029230208689176 + 0j), 0),
    (0, -0.029230208689176, (0.999572706159981 + 0j), 0),
    (0, 0, 0, 1),
)
G7 = (
    (1, 0, 0, 0),
    (0, 0.85065080835204, (0.525731112119134 + 0j), 0),
    (0, -0.525731112119134, (0.85065080835204 + 0j), 0),
    (0, 0, 0, 1),
)
G8 = (
    (1, 0, 0, 0),
    (0, 0.704276749590474, (0.70992553129626 + 0j), 0),
    (0, -0.70992553129626, (0.704276749590474 + 0j), 0),
    (0, 0, 0, 1),
)
G9 = (
    (1, 0, 0, 0),
    (0, 0.992043134868705, (0.125898445422781 + 0j), 0),
    (0, -0.125898445422781, (0.992043134868705 + 0j), 0),
    (0, 0, 0, 1),
)


def reference_circuit(num_iter: int = 1) -> fq.Circuit:
    circuit = fq.Circuit(8, dtype=torch.complex128)
    for bit in range(4):
        circuit.x(7 - bit)
    circuit.any(4, 3, matrix=torch.tensor(G0, dtype=torch.complex128))
    circuit.any(5, 4, matrix=torch.tensor(G1, dtype=torch.complex128))
    circuit.any(3, 2, matrix=torch.tensor(G2, dtype=torch.complex128))
    circuit.any(6, 5, matrix=torch.tensor(G3, dtype=torch.complex128))
    circuit.any(2, 1, matrix=torch.tensor(G4, dtype=torch.complex128))
    circuit.any(7, 6, matrix=torch.tensor(G5, dtype=torch.complex128))
    circuit.any(5, 4, matrix=torch.tensor(G6, dtype=torch.complex128))
    circuit.any(1, 0, matrix=torch.tensor(G7, dtype=torch.complex128))
    circuit.any(6, 5, matrix=torch.tensor(G8, dtype=torch.complex128))
    circuit.any(5, 4, matrix=torch.tensor(G9, dtype=torch.complex128))
    x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
    y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)
    hamiltonian = (torch.kron(x, x) + torch.kron(y, y)) / 2
    backward = torch.matrix_exp(1j * 1.570796326794897 * hamiltonian)
    forward = torch.matrix_exp(-1j * 1.570796326794897 * hamiltonian)
    zero = torch.eye(4, dtype=torch.complex128)
    for _ in range(num_iter):
        for first, second in ((0, 1), (2, 3), (4, 5), (6, 7)):
            circuit.any(7 - second, 7 - first, matrix=zero)
        circuit.cphase(7, 3, theta=0.6).cphase(5, 1, theta=0.6)
        circuit.any(5, 6, matrix=backward).any(1, 2, matrix=backward)
        circuit.cphase(5, 1, theta=0.6).cphase(4, 0, theta=0.6)
        circuit.any(5, 6, matrix=forward).any(1, 2, matrix=forward)
    return circuit


def run_example() -> None:
    for num_iter in (1, 2):
        compare_states(build_program(num_iter), reference_circuit(num_iter), ("qba",))


if __name__ == "__main__":
    run_example()
