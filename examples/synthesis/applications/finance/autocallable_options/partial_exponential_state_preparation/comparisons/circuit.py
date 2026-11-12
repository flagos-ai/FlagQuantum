"""Independent product rotations, predicate matrices and modular permutation."""

import math

import torch

import flagquantum as fq

from ..main import SUCCESS_AMPLITUDE


def input_circuit(rate: float, width: int = 5) -> fq.Circuit:
    circuit = fq.Circuit(width, dtype=torch.complex128)
    for bit in range(width):
        # Direct independent product odds, not the native amplitude builder/tree.
        angle = 2 * math.atan(math.exp(-rate * (1 << bit) / 2))
        circuit.ry(width - 1 - bit, theta=angle)
    return circuit


def shift_matrix() -> torch.Tensor:
    permutation = torch.zeros((32, 32), dtype=torch.complex128)
    for word in range(32):
        permutation[(word - 18) % 32, word] = 1
    return permutation


def reference_circuit(
    rate: float = -0.5, success_amplitude: float = SUCCESS_AMPLITUDE
) -> fq.Circuit:
    circuit = fq.Circuit(6, dtype=torch.complex128)
    for bit in range(5):
        angle = 2 * math.atan(math.exp(-rate * (1 << bit) / 2))
        circuit.ry(4 - bit, theta=angle)
    beta = 2 * math.acos(0.5 / success_amplitude)
    circuit.ry(5, theta=beta)
    marked = torch.tensor(
        [-1 if word // 2 >= 21 and word % 2 == 0 else 1 for word in range(64)],
        dtype=torch.complex128,
    )
    circuit.any(*range(6), matrix=torch.diag(marked))
    circuit.ry(5, theta=-beta)
    for bit in reversed(range(5)):
        circuit.ry(4 - bit, theta=-2 * math.atan(math.exp(-rate * (1 << bit) / 2)))
    reflection = -torch.eye(64, dtype=torch.complex128)
    reflection[0, 0] = 1
    circuit.any(*range(6), matrix=reflection)
    for bit in range(5):
        circuit.ry(4 - bit, theta=2 * math.atan(math.exp(-rate * (1 << bit) / 2)))
    circuit.ry(5, theta=beta)
    return circuit.any(*range(5), matrix=shift_matrix())


def run_example() -> None:
    from ..main import build_program

    program = build_program()
    actual = fq.run(program).statevector().reshape(32, 2, -1)
    expected = fq.run(reference_circuit()).statevector().reshape(32, 2)
    torch.testing.assert_close(actual[:, :, 0], expected, atol=1e-11, rtol=0)
    assert float(actual[:, 1:].abs().square().sum()) < 1e-22
    assert float(actual[:, :, 1:].abs().square().sum()) < 1e-22
    print(program.report["resources"])


if __name__ == "__main__":
    run_example()
