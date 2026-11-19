"""Independent ordered Pauli products, matrix powers and Fourier phase families."""

import math

import torch

import flagquantum as fq

from .ansatz import source_eigenpair, source_matrix


def preparation() -> torch.Tensor:
    _, amplitudes = source_eigenpair()
    vector = torch.tensor(amplitudes, dtype=torch.float64)
    first = torch.zeros(4, dtype=torch.float64)
    first[0] = 1
    direction = first - vector
    matrix = torch.eye(4, dtype=torch.complex128) - 2 * torch.outer(
        direction, direction
    ) / torch.dot(direction, direction)
    return matrix


def product_power(power: int) -> torch.Tensor:
    identity = torch.eye(2, dtype=torch.complex128)
    x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
    y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)
    z = torch.diag(torch.tensor([1, -1], dtype=torch.complex128))
    terms = (
        (0.4, torch.kron(identity, identity)),
        (-0.05, torch.kron(z, identity)),
        (-0.03, torch.kron(x, identity)),
        (-0.06, torch.kron(z, z)),
        (0.04, torch.kron(z, x)),
        (-0.16, torch.kron(z, z)),
        (-0.06, torch.kron(y, y)),
    )
    repetitions = 4 if power < 4 else math.ceil(4 * (power / 4) ** 1.5)
    time = -2 * math.pi * power / repetitions
    step = torch.eye(4, dtype=torch.complex128)
    for coefficient, pauli in terms:
        angle = time * coefficient
        step = (
            math.cos(angle) * torch.eye(4, dtype=torch.complex128)
            - 1j * math.sin(angle) * pauli
        ) @ step
    result = torch.linalg.matrix_power(step, repetitions)
    assert isinstance(result, torch.Tensor)
    return result


def exact_power(power: int) -> torch.Tensor:
    result = torch.linalg.matrix_exp(
        2j * math.pi * power * torch.from_numpy(source_matrix()).to(torch.complex128)
    )

    assert isinstance(result, torch.Tensor)
    return result


def add_qpe(
    circuit: fq.Circuit,
    phase: tuple[int, ...],
    data: tuple[int, ...],
    *,
    exact: bool = False,
) -> None:
    for bit in range(5):
        matrix = exact_power(1 << bit) if exact else product_power(1 << bit)
        controlled = torch.eye(8, dtype=torch.complex128)
        controlled[4:, 4:] = matrix
        wire = phase[4 - bit]
        circuit.h(wire).any(wire, *data, matrix=controlled)
    words = torch.arange(32, dtype=torch.float64)
    inverse = torch.exp(-2j * math.pi * torch.outer(words, words) / 32) / math.sqrt(32)
    circuit.any(*phase, matrix=inverse)


def reference_circuit(comparison: bool = False) -> fq.Circuit:
    if not comparison:
        circuit = fq.Circuit(7, dtype=torch.complex128)
        circuit.any(5, 6, matrix=preparation())
        add_qpe(circuit, tuple(range(5)), (5, 6))
        return circuit
    # The public result bit is first; private storage follows declaration order.
    circuit = fq.Circuit(13, dtype=torch.complex128)
    circuit.any(1, 2, matrix=preparation())
    add_qpe(circuit, tuple(range(3, 8)), (1, 2))
    add_qpe(circuit, tuple(range(8, 13)), (1, 2), exact=True)
    circuit.h(0)
    controlled_swap = torch.eye(8, dtype=torch.complex128)
    controlled_swap[5, 5] = controlled_swap[6, 6] = 0
    controlled_swap[5, 6] = controlled_swap[6, 5] = 1
    for left, right in zip(range(3, 8), range(8, 13), strict=True):
        circuit.any(0, left, right, matrix=controlled_swap)
    circuit.h(0)
    return circuit
