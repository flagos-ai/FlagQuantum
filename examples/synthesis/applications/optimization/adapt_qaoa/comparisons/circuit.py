"""Independent cost truth table and commuting X-rotation mixer."""

import torch

import flagquantum as fq
from examples.synthesis.applications.optimization.adapt_qaoa.main import build_program
from examples.synthesis.compare_states import compare_states

BINDINGS: tuple[tuple[float, ...], ...] = ((0.1, 0.2), (-0.37, 0.29))


def cost(bits: tuple[int, ...]) -> float:
    edges = ((0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 7), (6, 7))
    interactions = (
        (0, 4, -0.7525),
        (0, 5, -0.8004),
        (2, 4, -0.1924),
        (3, 6, -0.1581),
        (3, 7, -0.9165),
        (1, 6, 0.9382),
        (1, 7, 0.2143),
        (2, 5, 0.9182),
    )
    return (
        2 * sum(bits[a] * bits[b] for a, b in edges)
        + sum(weight * bits[a] * bits[b] for a, b, weight in interactions)
        - 0.1 * sum(bits)
        - 3.4127
    )


def reference_circuit(params: tuple[float, ...] = (0.1, 0.2)) -> fq.Circuit:
    gamma, beta = params
    values = torch.tensor(
        [cost(tuple((value >> i) & 1 for i in range(8))) for value in range(256)],
        dtype=torch.float64,
    )
    circuit = fq.Circuit(8, dtype=torch.complex128)
    for wire in range(8):
        circuit.h(wire)
    circuit.any(*range(8), matrix=torch.diag(torch.exp(1j * gamma * values)))
    for wire in range(8):
        circuit.rx(wire, theta=2 * beta)
    return circuit


def run_example() -> None:
    for params in BINDINGS:
        compare_states(build_program(params), reference_circuit(params), ("v",))


if __name__ == "__main__":
    run_example()
