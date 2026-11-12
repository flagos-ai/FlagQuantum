"""Independent integer recurrence and reversible Circuit on all 23 physical bits."""

import torch

import flagquantum as fq
from examples.synthesis.applications.finance.brownian_chebyshev_polynomials.main import (
    build_program,
)


def prices(coefficient: float = 0.450158158078553) -> tuple[int, ...]:
    """Evaluate the recurrence in integer units of 1/256 on every selector row."""
    scale = 256
    cosine = int(-6.283185307179586 * 2**20) // 2**12
    weight = int(coefficient * 2**20) // 2**12
    result = []
    for row in range(32):
        t, a0, a1, a2, a3 = (row >> bit & 1 for bit in range(5))
        u1 = cosine * (2 * t - 1) // 2
        u2 = u1 * u1 // scale - scale
        u3 = u1 * u2 // scale - u1
        b = (
            a0 * t * scale
            + ((weight * -64) // scale) * (a1 * u1 + a2 * u2 + a3 * u3) // scale
        )
        result.append(2 * b - t * scale + scale)
    return tuple(result)


def reference_circuit(coefficient: float = 0.450158158078553) -> fq.Circuit:
    circuit = fq.Circuit(23, dtype=torch.complex128)
    values = prices(coefficient)
    for wire in range(16, 21):
        circuit.h(wire)
    # Each small permutation XORs one independently calculated output column.
    # This is a reversible compute circuit, not a prepared final statevector.
    for wire in range(16):
        permutation = torch.eye(64, dtype=torch.complex128)
        for row, value in enumerate(values):
            word = ((row & 1) << 15) | (value & 32767)
            if word >> (15 - wire) & 1:
                index = 2 * row
                permutation[[index, index + 1]] = permutation[[index + 1, index]]
        circuit.any(16, 17, 18, 19, 20, wire, matrix=permutation)
    return circuit


def compare() -> None:
    program = build_program()
    actual = fq.run(program).statevector().flatten()
    expected = fq.run(reference_circuit()).statevector().flatten()
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)
    assert float(actual.reshape(-1, 4)[:, 1:].abs().square().sum()) < 1e-24
    assert torch.count_nonzero(actual.abs() > 1e-12) == 32
    print(program.report["resources"])
    print(
        "All 23-bit complex amplitudes, absolute phase and two clean work bits agree."
    )


if __name__ == "__main__":
    compare()
