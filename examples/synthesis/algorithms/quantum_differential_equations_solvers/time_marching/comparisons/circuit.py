"""Independent controlled-Y, projector permutations and counter permutations."""

import cmath
import math

import torch

import flagquantum as fq

from ..data import EVEN_PHASES, ODD_PHASES

Gate = tuple[tuple[int, ...], torch.Tensor]


def direct_sum(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    n, m = left.shape[0], right.shape[0]
    result = torch.zeros((n + m, n + m), dtype=torch.complex128)
    result[:n, :n], result[n:, n:] = left, right
    return result


def rotation(angle: float) -> torch.Tensor:
    c, s = math.cos(angle / 2), math.sin(angle / 2)
    return torch.tensor([[c, -s], [s, c]], dtype=torch.complex128)


def short_gates(a: float, b: float, selected: bool = False) -> list[Gate]:
    h = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex128) / math.sqrt(2)
    gates: list[Gate] = []
    for wire in (2, 3):
        gates.append(
            ((6, wire), direct_sum(torch.eye(2, dtype=torch.complex128), h))
            if selected
            else ((wire,), h)
        )
    gates.append(
        ((6, 4), direct_sum(torch.eye(2, dtype=torch.complex128), rotation(2 * a)))
        if selected
        else ((4,), rotation(2 * a))
    )
    for wire, angle in ((3, (b - a) / 2), (2, b - a), (1, 2.0), (0, 4.0)):
        matrix = direct_sum(torch.eye(2, dtype=torch.complex128), rotation(angle))
        gates.append(
            ((6, wire, 4), direct_sum(torch.eye(4, dtype=torch.complex128), matrix))
            if selected
            else ((wire, 4), matrix)
        )
    for wire in (2, 3):
        gates.append(
            ((6, wire), direct_sum(torch.eye(2, dtype=torch.complex128), h))
            if selected
            else ((wire,), h)
        )
    return gates


def projected_gates(even: float, odd: float) -> list[Gate]:
    projector = torch.eye(16, dtype=torch.complex128)
    projector[[0, 1]] = projector[[1, 0]]
    diagonal = torch.tensor(
        [cmath.exp(1j * angle * sign / 2) for angle in (even, odd) for sign in (-1, 1)],
        dtype=torch.complex128,
    )
    return [
        ((2, 3, 4, 5), projector),
        ((6, 5), torch.diag(diagonal)),
        ((2, 3, 4, 5), projector),
    ]


def step_gates(
    a: float,
    b: float,
    even: tuple[float, ...] = EVEN_PHASES,
    odd: tuple[float, ...] = ODD_PHASES,
) -> list[Gate]:
    h = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex128) / math.sqrt(2)
    gates: list[Gate] = [((6,), rotation(math.pi / 2)), ((5,), h)]
    gates.extend(projected_gates(even[0], odd[0]))
    for index in range(3):
        forward = short_gates(a, b)
        gates.extend(forward)
        gates.extend(projected_gates(even[2 * index + 1], odd[2 * index + 1]))
        gates.extend((wires, matrix.mH) for wires, matrix in reversed(forward))
        gates.extend(projected_gates(even[2 * index + 2], odd[2 * index + 2]))
    gates.extend(short_gates(a, b, selected=True))
    gates.extend(projected_gates(0, odd[7]))
    gates.extend([((5,), h), ((6,), rotation(-math.pi / 2))])
    return gates


def replay(circuit: fq.Circuit, gates: list[Gate], inverse: bool = False) -> None:
    for wires, matrix in reversed(gates) if inverse else gates:
        circuit.any(*wires, matrix=matrix.mH if inverse else matrix)


def counter_matrix() -> torch.Tensor:
    permutation = torch.zeros((256, 256), dtype=torch.complex128)
    for word in range(256):
        output = ((word - 1) % 8) if word < 8 else word
        permutation[output, word] = 1
    return permutation


def reference_circuit(
    duration: float = 2.0,
    even: tuple[float, ...] = EVEN_PHASES,
    odd: tuple[float, ...] = ODD_PHASES,
) -> fq.Circuit:
    circuit = fq.Circuit(11, dtype=torch.complex128).h(0).h(1).x(8)
    permutation = counter_matrix()
    for index in range(4):
        replay(
            circuit,
            step_gates(index * duration / 4, (index + 1) * duration / 4, even, odd),
        )
        circuit.any(2, 3, 4, 5, 6, 8, 9, 10, matrix=permutation)
    return circuit


def run_example() -> None:
    from ..main import build_program

    program = build_program()
    actual = fq.run(program).statevector().reshape(2048, 2, -1)
    expected = fq.run(reference_circuit()).statevector().flatten()
    torch.testing.assert_close(actual[:, 0, 0], expected, atol=1e-10, rtol=0)
    assert float(actual[:, 1:].abs().square().sum()) < 1e-22
    assert float(actual[:, :, 1:].abs().square().sum()) < 1e-22
    print(program.report["resources"])


if __name__ == "__main__":
    run_example()
