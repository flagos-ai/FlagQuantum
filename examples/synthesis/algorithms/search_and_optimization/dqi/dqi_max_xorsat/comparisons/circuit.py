"""Independent Dicke-state reflection, linear code and controlled lookup matrices."""

import math

import torch

import flagquantum as fq
from examples.synthesis.algorithms.search_and_optimization.dqi.dqi_max_xorsat.data import (
    PARITY_COLUMNS,
    SIGNS,
    SYNDROME_TABLE,
    WEIGHTS,
)


def decoder_circuit(circuit: fq.Circuit) -> None:
    """XOR each disjoint syndrome's coset leader using direct permutation matrices."""
    for word, mask in SYNDROME_TABLE:
        permutation = torch.eye(128, dtype=torch.complex128)
        row = word << 1
        permutation[[row, row + 1]] = permutation[[row + 1, row]]
        for bit in range(6):
            if mask >> bit & 1:
                circuit.any(*range(6, 12), 5 - bit, matrix=permutation)


def reference_circuit(weights: tuple[float, ...] = WEIGHTS) -> fq.Circuit:
    # The analytic Dicke amplitudes define a model oracle, independently of
    # native preparation's rotation tree and the source's consumed unary network.
    vector = torch.zeros(64, dtype=torch.complex128)
    for weight, amplitude in enumerate(weights):
        for word in range(64):
            if word.bit_count() == weight:
                vector[word] = amplitude / math.sqrt(math.comb(6, weight))
    identity = torch.eye(64, dtype=torch.complex128)
    difference = identity[:, 0] - vector
    norm = difference.vdot(difference)
    preparation = (
        identity
        if float(norm.real) < 1e-24
        else identity - 2 * torch.outer(difference, difference.conj()) / norm
    )
    circuit = fq.Circuit(12, dtype=torch.complex128).any(*range(6), matrix=preparation)
    for bit, sign in enumerate(SIGNS):
        if sign % 2:
            circuit.z(5 - bit)
    for column, word in enumerate(PARITY_COLUMNS):
        for row in range(6):
            if word >> row & 1:
                circuit.cx(5 - column, 11 - row)
    decoder_circuit(circuit)
    for wire in range(6, 12):
        circuit.h(wire)
    return circuit


def run_example() -> None:
    from ..main import build_program

    program = build_program()
    expected = fq.run(reference_circuit()).statevector().flatten()
    actual = fq.run(program).statevector().reshape(4096, -1)
    torch.testing.assert_close(actual[:, 0], expected, atol=1e-10, rtol=0)
    assert float(actual[:, 1:].abs().square().sum()) < 1e-22
    print(program.report["resources"])


if __name__ == "__main__":
    run_example()
