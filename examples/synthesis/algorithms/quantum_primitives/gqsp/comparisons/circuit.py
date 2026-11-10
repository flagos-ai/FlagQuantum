"""Independent 2x2 signal blocks over all 128 fractional input words."""

import cmath
import math

import torch

import flagquantum as fq

from ..physics import PHASES


def rotation(row: tuple[float, float, float]) -> torch.Tensor:
    theta, phi, lam = row
    c, s = math.cos(theta), math.sin(theta)
    u = torch.tensor(
        [
            [c, -cmath.exp(1j * lam) * s],
            [cmath.exp(1j * phi) * s, cmath.exp(1j * (phi + lam)) * c],
        ],
        dtype=torch.complex128,
    )
    x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
    z = torch.diag(torch.tensor([1, -1], dtype=torch.complex128))
    return z @ x @ u @ x


def signal_matrix(
    negative_power: int = 3, phases: tuple[tuple[float, float, float], ...] = PHASES
) -> torch.Tensor:
    blocks = []
    for word in range(128):
        angle = 2 * math.pi * word / 128
        positive = torch.diag(
            torch.tensor([cmath.exp(1j * angle), 1], dtype=torch.complex128)
        )
        negative = torch.diag(
            torch.tensor([1, cmath.exp(-1j * angle)], dtype=torch.complex128)
        )
        block = rotation(phases[0])
        for i in range(1, 7 - negative_power):
            block = rotation(phases[i]) @ positive @ block
        for i in range(7 - negative_power, 7):
            block = rotation(phases[i]) @ negative @ block
        blocks.append(block)
    matrix = torch.zeros((256, 256), dtype=torch.complex128)
    for word, block in enumerate(blocks):
        matrix[2 * word : 2 * word + 2, 2 * word : 2 * word + 2] = block
    return matrix


def reference_circuit(
    negative_power: int = 3,
    initial_indicator: int = 0,
    phases: tuple[tuple[float, float, float], ...] = PHASES,
) -> fq.Circuit:
    circuit = fq.Circuit(8, dtype=torch.complex128)
    for bit in range(7):
        circuit.h(bit)
    if initial_indicator:
        circuit.x(7)
    return circuit.any(*range(8), matrix=signal_matrix(negative_power, phases))


def run_example() -> None:
    from ..main import build_program

    for power in (0, 3, 7):
        program = build_program(power)
        torch.testing.assert_close(
            fq.run(program).statevector().flatten(),
            fq.run(reference_circuit(power)).statevector().flatten(),
            atol=1e-12,
            rtol=0,
        )
        print(power, program.report["resources"])


if __name__ == "__main__":
    run_example()
