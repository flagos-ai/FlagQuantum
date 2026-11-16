"""Independent block matrices, permutations and analytic projected phases."""

import cmath
import math

import torch

import flagquantum as fq
from examples.synthesis.applications.cfd.heat_eq_qsvt.main import AMPLITUDES, PHASES

DTYPE = torch.complex128


def block_diagonal(*matrices: torch.Tensor) -> torch.Tensor:
    size = sum(matrix.shape[0] for matrix in matrices)
    result = torch.zeros((size, size), dtype=DTYPE)
    offset = 0
    for matrix in matrices:
        width = matrix.shape[0]
        result[offset : offset + width, offset : offset + width] = matrix
        offset += width
    return result


def reference_circuit(
    phases: tuple[float, ...] = PHASES, amplitudes: tuple[float, ...] = AMPLITUDES
) -> fq.Circuit:
    deletion, rotation, high, low, aux = 0, 6, 4, 5, 7
    data = (1, 2, 3)
    circuit = fq.Circuit(8, dtype=DTYPE)

    def ry(angle: float) -> torch.Tensor:
        c, s = math.cos(angle / 2), math.sin(angle / 2)
        return torch.tensor([[c, -s], [s, c]], dtype=DTYPE)

    identity = torch.eye(2, dtype=DTYPE)
    prepare_select = block_diagonal(ry(math.pi / 2), identity) @ torch.kron(
        ry(2 * math.atan2(1, math.sqrt(2))), identity
    )
    rotate = block_diagonal(
        ry(3.252761000151628), identity, ry(3.252761000151628), identity
    )
    shift = torch.zeros((64, 64), dtype=DTYPE)
    for selector in range(4):
        for word in range(16):
            displacement = (selector if selector < 3 else 0) - 1
            shift[16 * selector + (word + displacement) % 16, 16 * selector + word] = 1

    def block_encoding(inverse: bool = False) -> None:
        circuit.any(high, low, matrix=prepare_select)
        if inverse:
            circuit.any(high, low, deletion, *data, matrix=shift.mH.contiguous())
            circuit.any(high, low, rotation, matrix=rotate.mH.contiguous())
        else:
            circuit.any(high, low, rotation, matrix=rotate)
            circuit.any(high, low, deletion, *data, matrix=shift)
        circuit.any(high, low, matrix=prepare_select.mH.contiguous())

    def project(angle: float) -> None:
        diagonal = [
            cmath.exp(0.5j * angle * (1 if (word & 1) ^ (word < 2) else -1))
            for word in range(32)
        ]
        circuit.any(
            deletion,
            rotation,
            high,
            low,
            aux,
            matrix=torch.diag(torch.tensor(diagonal, dtype=DTYPE)),
        )

    amplitudes_tensor = torch.tensor(amplitudes, dtype=DTYPE)
    amplitudes_tensor = amplitudes_tensor / torch.linalg.vector_norm(amplitudes_tensor)
    difference = torch.eye(8, dtype=DTYPE)[:, 0] - amplitudes_tensor
    prep = torch.eye(8, dtype=DTYPE) - 2 * torch.outer(
        difference, difference
    ) / difference.dot(difference)
    circuit.any(*data, matrix=prep)
    circuit.h(aux)
    project(phases[0])
    block_encoding()
    project(phases[1])
    block_encoding(inverse=True)
    project(phases[2])
    circuit.h(aux)
    return circuit


def run_example() -> None:
    from ..main import build_program

    program = build_program()
    expected = fq.run(reference_circuit()).statevector().flatten()
    actual = fq.run(program).statevector().reshape(expected.numel(), -1)
    torch.testing.assert_close(actual[:, 0], expected, atol=1e-10, rtol=0)
    assert float(actual[:, 1:].abs().square().sum()) < 1e-22
    print(program.report["resources"])


if __name__ == "__main__":
    run_example()
