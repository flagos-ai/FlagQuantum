"""Independent Householder preparation and controlled permutation reference."""

import torch

import flagquantum as fq
from examples.synthesis.algorithms.quantum_primitives.swap_test.main import (
    RAW_FIRST,
    RAW_SECOND,
    build_program,
    normalize,
)


def reference_circuit(
    first: tuple[float, ...] = RAW_FIRST, second: tuple[float, ...] = RAW_SECOND
) -> fq.Circuit:
    circuit = fq.Circuit(7, dtype=torch.complex128)
    identity = torch.eye(8, dtype=torch.complex128)
    for wires, values in ((range(1, 4), first), (range(4, 7), second)):
        vector = torch.tensor(normalize(values), dtype=torch.complex128)
        delta = identity[:, 0] - vector
        norm = float(delta.abs().square().sum())
        matrix = (
            identity
            if norm < 1e-28
            else identity - 2 * torch.outer(delta, delta.conj()) / norm
        )
        circuit.any(*wires, matrix=matrix)
    permutation = torch.zeros((8, 8), dtype=torch.complex128)
    for value in range(8):
        selector, a, b = value >> 2, (value >> 1) & 1, value & 1
        target = 4 * selector + (2 * b + a if selector else 2 * a + b)
        permutation[target, value] = 1
    circuit.h(0)
    for bit in range(3):
        circuit.any(0, 1 + bit, 4 + bit, matrix=permutation)
    return circuit.h(0)


def run_example() -> None:
    program = build_program()
    result = fq.run(program)
    torch.testing.assert_close(
        result.statevector().flatten(),
        fq.run(reference_circuit()).statevector().flatten(),
        atol=1e-12,
        rtol=0,
    )
    print(dict(program.probabilities(result, registers=("test",))))
    print("Full retained environment and overlap probabilities agree.")


if __name__ == "__main__":
    run_example()
