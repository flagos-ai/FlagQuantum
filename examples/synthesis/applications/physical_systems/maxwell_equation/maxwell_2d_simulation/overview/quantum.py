"""Native conductor predicates on the complete original field register."""

import math

import torch

import flagquantum as fq

from .controls import controlled_x

Guard = tuple[int, int]


def wire(circuit: fq.Circuit, logical: int) -> int:
    return int(circuit.n_qubits) - 1 - logical


def toggle_pattern(
    circuit: fq.Circuit,
    conditions: tuple[Guard, ...],
    target: int = 12,
    borrowed: int = 13,
) -> None:
    labels = (*tuple(index for index, _ in conditions), target, borrowed)
    if (
        not conditions
        or len(set(labels)) != len(labels)
        or any(not 0 <= index < circuit.n_qubits for index in labels)
        or any(value not in (0, 1) for _, value in conditions)
    ):
        raise ValueError(
            "Predicates require disjoint in-bounds data, target and borrowed bits and binary values"
        )
    controls = tuple(wire(circuit, index) for index, _ in conditions)
    negative = tuple(wire(circuit, index) for index, value in conditions if value == 0)
    for index in negative:
        circuit.x(index)
    controlled_x(circuit, controls, wire(circuit, target), wire(circuit, borrowed))
    for index in reversed(negative):
        circuit.x(index)


def interval_cubes(offset: int) -> tuple[tuple[Guard, ...], ...]:
    """Disjoint dyadic decomposition of the exact source interval [9, 24)."""
    return tuple(
        tuple((offset + bit, (first >> bit) & 1) for bit in range(free, 5))
        for first, free in ((9, 0), (10, 1), (12, 2), (16, 3))
    )


def conductor(circuit: fq.Circuit, outer: tuple[Guard, ...] = ()) -> None:
    """Toggle the original PEC flag, restoring arbitrary borrowed block storage."""
    electric = outer + ((10, 0), (11, 0))
    for offset in (0, 5):
        toggle_pattern(circuit, electric + tuple((offset + bit, 0) for bit in range(5)))
    for x in interval_cubes(0):
        for y in interval_cubes(5):
            toggle_pattern(circuit, electric + x + y)


def emit(
    circuit: fq.Circuit,
    matrix: torch.Tensor,
    data: tuple[int, ...],
    conditions: tuple[Guard, ...] = (),
) -> None:
    labels = data + tuple(index for index, _ in conditions)
    if len(set(labels)) != len(labels):
        raise ValueError("Matrix data and controls must be disjoint")
    dimension = 1 << len(data)
    operator = torch.eye(dimension << len(conditions), dtype=torch.complex128)
    first = dimension * sum(
        value << index for index, (_, value) in enumerate(conditions)
    )
    operator[first : first + dimension, first : first + dimension] = matrix
    circuit.any(*(wire(circuit, index) for index in reversed(labels)), matrix=operator)


def scalar_phase(
    circuit: fq.Circuit, value: complex, conditions: tuple[Guard, ...]
) -> None:
    emit(circuit, value * torch.eye(2, dtype=torch.complex128), (12,), conditions)


def shift(amount: int) -> torch.Tensor:
    operator = torch.zeros((32, 32), dtype=torch.complex128)
    for column in range(32):
        operator[(column + amount) % 32, column] = 1
    return operator


def gradient(
    circuit: fq.Circuit, axis: tuple[int, ...], outer: tuple[Guard, ...]
) -> None:
    hadamard = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex128) / math.sqrt(2)
    z = torch.diag(torch.tensor([1, -1], dtype=torch.complex128))
    emit(circuit, hadamard, (15,), outer)
    emit(circuit, z, (15,), outer)
    for field, amount in ((0, 1), (1, -1)):
        emit(circuit, shift(amount), axis, outer + ((11, field), (15, 1)))
    emit(circuit, hadamard, (15,), outer)
    scalar_phase(circuit, -1, outer + ((11, 1),))


def flip(circuit: fq.Circuit, target: int, conditions: tuple[Guard, ...]) -> None:
    emit(
        circuit,
        torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128),
        (target,),
        conditions,
    )


def periodic(circuit: fq.Circuit, outer: tuple[Guard, ...] = ()) -> None:
    hadamard = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex128) / math.sqrt(2)
    emit(circuit, hadamard, (13,), outer)
    hx = outer + ((13, 0),)
    flip(circuit, 11, hx)
    gradient(circuit, tuple(range(5, 10)), hx)
    scalar_phase(circuit, -1, hx)
    flip(circuit, 14, hx + ((10, 1),))
    hy = outer + ((13, 1),)
    flip(circuit, 10, hy)
    flip(circuit, 11, hy)
    gradient(circuit, tuple(range(5)), hy)
    flip(circuit, 14, hy + ((10, 1),))
    flip(circuit, 14, hy + ((11, 1),))
    emit(circuit, hadamard, (13,), outer)


def block_encoding() -> fq.Circuit:
    circuit = fq.Circuit(16, dtype=torch.complex128)
    conductor(circuit)
    periodic(circuit)
    conductor(circuit)
    return circuit


def controlled_walk() -> fq.Circuit:
    """Select W = (2P-I) iB on auxiliary one and identity on auxiliary zero."""
    circuit = fq.Circuit(17, dtype=torch.complex128)
    outer = ((16, 1),)
    conductor(circuit, outer)
    periodic(circuit, outer)
    conductor(circuit, outer)
    scalar_phase(circuit, 1j, outer)
    reflection = -torch.eye(16, dtype=torch.complex128)
    reflection[0, 0] = 1
    emit(circuit, reflection, (12, 13, 14, 15), outer)
    return circuit
