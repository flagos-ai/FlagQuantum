"""Independent exact rational quantization and reversible Circuit permutation."""

from fractions import Fraction

import torch

import flagquantum as fq


def reference_circuit(
    initial: int = 0,
    repeats: int = 1,
    slopes: tuple[float, ...] = (1.0, 0.9375, 0.8125, 0.625),
    intercepts: tuple[float, ...] = (0.0, 0.03125, 0.0625, 0.21875),
) -> fq.Circuit:
    circuit = fq.Circuit(8, dtype=torch.complex128)
    for bit in range(4):
        circuit.h(bit)
        if initial >> bit & 1:
            circuit.x(7 - bit)
    matrix = torch.zeros((256, 256), dtype=torch.complex128)
    for raw in range(16):
        label = raw >> 2
        value = (
            Fraction(str(slopes[label])) * Fraction(raw, 16)
            + Fraction(str(intercepts[label]))
        ) * 16
        mask = (value.numerator // value.denominator) & 15 if repeats % 2 else 0
        for target in range(16):
            matrix[raw * 16 + (target ^ mask), raw * 16 + target] = 1
    return circuit.any(*range(8), matrix=matrix)


def run_example() -> None:
    from ..main import build_program

    program = build_program()
    actual = fq.run(program).statevector().reshape(256, -1)
    expected = fq.run(reference_circuit()).statevector().flatten()
    torch.testing.assert_close(actual[:, 0], expected, atol=1e-12, rtol=0)
    assert float(actual[:, 1:].abs().square().sum()) < 1e-22
    print(program.report["resources"])


if __name__ == "__main__":
    run_example()
