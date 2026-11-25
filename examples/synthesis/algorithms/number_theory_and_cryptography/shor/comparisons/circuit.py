"""Independent controlled modular permutations and analytic inverse DFT."""

import torch

import flagquantum as fq


def reference_circuit(modulus: int = 21, base: int = 11) -> fq.Circuit:
    circuit = fq.Circuit(15, dtype=torch.complex128).x(14)
    for wire in range(10):
        circuit.h(wire)
    for bit in range(10):
        multiplier = pow(base, 1 << bit, modulus)
        matrix = torch.eye(64, dtype=torch.complex128)
        matrix[32:, 32:] = 0
        for residue in range(32):
            target = residue * multiplier % modulus if residue < modulus else residue
            matrix[32 + target, 32 + residue] = 1
        circuit.any(9 - bit, *range(10, 15), matrix=matrix)
    indices = torch.arange(1024, dtype=torch.float64)
    inverse = torch.exp(-2j * torch.pi * torch.outer(indices, indices) / 1024) / 32
    return circuit.any(*range(10), matrix=inverse)


def run_example() -> None:
    from ..main import build_program

    program = build_program()
    state = fq.run(program).statevector().reshape(32768, -1)
    expected = fq.run(reference_circuit()).statevector().flatten()
    torch.testing.assert_close(state[:, 0], expected, atol=1e-12, rtol=0)
    assert float(state[:, 1:].abs().square().sum()) < 1e-22
    print(program.report["resources"])


if __name__ == "__main__":
    run_example()
