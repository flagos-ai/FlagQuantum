"""Independent affine scalar sums, output XOR gates and analytic inverse DFT."""

import math

import torch

import flagquantum as fq


def reference_circuit(
    seed: tuple[int, int] = (4, 2), amplitudes: tuple[float, ...] | None = None
) -> fq.Circuit:
    # Build the seed's five-element orbit directly from the affine slope formula.
    orbit = [seed]
    for _ in range(4):
        x, y = orbit[-1]
        slope = (5 - y) * pow((-x) % 7, -1, 7) % 7
        u = (slope * slope - x) % 7
        orbit.append((u, (slope * (x - u) - y) % 7))
    assert len(set(orbit)) == 5
    circuit = fq.Circuit(12, dtype=torch.complex128)
    if amplitudes is None:
        for wire in range(6):
            circuit.h(wire)
    else:
        vector = torch.tensor(amplitudes, dtype=torch.complex128)
        eye = torch.eye(64, dtype=torch.complex128)
        delta = eye[:, 0] - vector
        circuit.any(
            *range(6),
            matrix=eye - 2 * torch.outer(delta, delta.conj()) / delta.vdot(delta),
        )
    words = []
    for packed in range(64):
        left, right = packed >> 3, packed & 7
        x, y = orbit[(left + right) % 5]
        words.append(x + 8 * y)
    for bit in range(6):
        matrix = torch.eye(128, dtype=torch.complex128)
        for row, word in enumerate(words):
            if word >> bit & 1:
                matrix[2 * row : 2 * row + 2] = matrix[[2 * row + 1, 2 * row]]
        circuit.any(*range(6), 11 - bit, matrix=matrix)
    index = torch.arange(8, dtype=torch.float64)
    inverse = torch.exp(-2j * math.pi * torch.outer(index, index) / 8) / math.sqrt(8)
    return circuit.any(0, 1, 2, matrix=inverse).any(3, 4, 5, matrix=inverse)


def run_example() -> None:
    from ..main import build_program

    program = build_program()
    state = fq.run(program).statevector().reshape(4096, -1)
    expected = fq.run(reference_circuit()).statevector().flatten()
    torch.testing.assert_close(state[:, 0], expected, atol=3e-12, rtol=0)
    assert float(state[:, 1:].abs().square().sum()) < 1e-22
    print(program.report["resources"])


if __name__ == "__main__":
    run_example()
