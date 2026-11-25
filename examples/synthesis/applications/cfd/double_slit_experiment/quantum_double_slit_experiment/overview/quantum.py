"""Compose the 18-qubit inverse with small, exact reversible simulator blocks."""

import math
from collections.abc import Callable

import torch

import flagquantum as fq

from .model import COMPLEX, DEGREE, REAL, Inverse, Model

AUX = 0
SELECT = (1, 2)
AXIS, DIRECTION, SCATTER, EDGE = 3, 4, 5, 6
X = (7, 8, 9, 10)
Y = (11, 12, 13, 14)
WORK = (15, 16, 17)
WIDTH = 18


def permutation(width: int, move: Callable[[int], int]) -> torch.Tensor:
    dimension = 1 << width
    columns = torch.arange(dimension)
    rows = torch.tensor([move(word) for word in range(dimension)])
    if (
        torch.unique(rows).numel() != dimension
        or int(rows.min()) != 0
        or int(rows.max()) != dimension - 1
    ):
        raise ValueError("A circuit permutation must be bijective")
    operator = torch.zeros(dimension, dimension, dtype=COMPLEX)
    operator[rows, columns] = 1
    return operator


def toggle(
    circuit: fq.Circuit,
    inputs: tuple[int, ...],
    target: int,
    predicate: Callable[[int], bool],
) -> None:
    """XOR a reversible predicate into its separate target bit."""
    circuit.any(
        *inputs,
        target,
        matrix=permutation(
            len(inputs) + 1, lambda word: word ^ int(predicate(word >> 1))
        )
    )


def zero_or_max(circuit: fq.Circuit, data: tuple[int, ...], target: int) -> None:
    toggle(circuit, data, target, lambda word: word in (0, 15))


def scatterer(circuit: fq.Circuit, target: int) -> None:
    toggle(
        circuit,
        (*X[:3], *Y[:3]),
        target,
        lambda word: word >> 3 not in (2, 5) and word % 8 == 3,
    )


def shift(circuit: fq.Circuit, data: tuple[int, ...], axis: int, enable: int) -> None:
    def move(word: int) -> int:
        if word >> 6 and (word >> 5) & 1 == axis:
            delta = -1 if (word >> 4) & 1 else 1
            return (word & ~15) | ((word + delta) % 16)
        return word

    circuit.any(enable, AXIS, DIRECTION, *data, matrix=permutation(7, move))


def laplacian(circuit: fq.Circuit) -> None:
    enable, scratch, flag = WORK
    toggle(circuit, SELECT, enable, lambda word: word == 0)
    scatterer(circuit, scratch)
    circuit.ccx(enable, scratch, SCATTER)
    scatterer(circuit, scratch)
    circuit.ch(enable, AXIS).ch(enable, DIRECTION)
    shift(circuit, X, 0, enable)
    shift(circuit, Y, 1, enable)
    circuit.ch(enable, DIRECTION).ch(enable, AXIS)
    toggle(circuit, SELECT, enable, lambda word: word == 0)

    zero_or_max(circuit, X, enable)
    zero_or_max(circuit, Y, scratch)
    scatterer(circuit, flag)
    toggle(circuit, (*SELECT, *WORK), EDGE, lambda word: 0 < word < 8)
    scatterer(circuit, flag)
    zero_or_max(circuit, Y, scratch)
    zero_or_max(circuit, X, enable)


def absorbing_axis(
    circuit: fq.Circuit, select: int, axis: int, data: tuple[int, ...]
) -> None:
    condition = 2 + axis
    toggle(circuit, (select, AXIS), data[-1], lambda word: word == condition)
    toggle(
        circuit,
        (select, AXIS, *data),
        DIRECTION,
        lambda word: word >> 4 == condition and word % 16 not in (0, 15),
    )


def absorbing_boundary(circuit: fq.Circuit) -> None:
    select, scratch, _ = WORK
    toggle(circuit, SELECT, select, lambda word: word == 1)
    circuit.ch(select, AXIS)
    absorbing_axis(circuit, select, 0, X)
    absorbing_axis(circuit, select, 1, Y)
    circuit.ch(select, AXIS)
    for data in (X, Y):
        zero_or_max(circuit, data, scratch)
        circuit.ccx(select, scratch, SCATTER)
        zero_or_max(circuit, data, scratch)
    circuit.cx(select, SCATTER)
    toggle(circuit, SELECT, select, lambda word: word == 1)


def selector(weights: tuple[float, ...]) -> fq.Circuit:
    a, b, c, d = weights
    prep = fq.Circuit(2, dtype=COMPLEX).ry(
        0, 2 * math.atan2(math.hypot(c, d), math.hypot(a, b))
    )
    prep.x(0).cry(0, 1, theta=2 * math.atan2(b, a)).x(0)
    if d:
        prep.cry(0, 1, theta=2 * math.atan2(d, c))
    return prep


def signal(model: Model) -> fq.Circuit:
    """Use three clean scratch bits for the selector, scatterer and boundary OR."""
    prep = selector(model.selector_amplitudes)
    circuit = fq.Circuit(WIDTH, dtype=COMPLEX).compose(prep, qubits=SELECT)
    laplacian(circuit)
    absorbing_boundary(circuit)
    values = torch.tensor((0, model.absorb_phase, math.pi, 0), dtype=REAL)
    circuit.any(*SELECT, matrix=torch.diag((1j * values).exp()))
    return circuit.compose(prep.adjoint(), qubits=SELECT)


def projector(angle: float) -> torch.Tensor:
    """Apply the projected RZ phase to the two interferometer branches."""
    words = torch.arange(128)
    positive = ((words & 63) == 0) == ((words >> 6) == 0)
    return torch.diag((1j * angle * (2 * positive.to(REAL) - 1)).exp())


def qsvt_circuit(model: Model, inverse: Inverse) -> fq.Circuit:
    """Preserve all 191 signal calls and the source's original width limit."""
    angles = inverse.phases.phases - math.pi / 2
    angles[0] += math.pi / 4
    angles[-1] += math.pi / 4
    if angles.numel() != DEGREE + 1:
        raise ValueError("Expected exactly 192 inverse angles")
    block = signal(model)
    adjoint = block.adjoint()
    circuit = fq.Circuit(WIDTH, dtype=COMPLEX)
    circuit.x(X[-1]).x(X[-2]).x(X[-3]).x(Y[-2])
    circuit.h(X[0])
    for wire in X[1:]:
        circuit.cx(X[0], wire)
    circuit.h(AUX)
    phase_wires = tuple(range(7))
    circuit.any(*phase_wires, matrix=projector(float(angles[0])))
    for index, angle in enumerate(angles[1:]):
        circuit.compose(adjoint if index % 2 == 0 else block)
        circuit.any(*phase_wires, matrix=projector(float(angle)))
    return circuit.h(AUX)
