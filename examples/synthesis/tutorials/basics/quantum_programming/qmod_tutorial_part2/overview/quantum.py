"""Native solutions for the second quantum programming tutorial."""

import math
from collections.abc import Callable
from typing import Annotated

import torch

import flagquantum as fq
from flagquantum.synth import (
    H,
    Output,
    QArray,
    QBit,
    QNum,
    X,
    allocate,
    apply,
    control,
    hadamard_transform,
    phase,
    qft,
    qfunc,
    qpe,
    suzuki_trotter,
    within,
)

HAMILTONIAN = 0.5 * (fq.X(0) @ fq.X(1) @ fq.Z(2) @ fq.X(3)) + 0.25 * (fq.Z(1) @ fq.Y(3))


@qfunc
def product_phase(
    angles: list[float], x: Output[Annotated[QNum, 2]], y: Output[Annotated[QNum, 2]]
) -> None:
    allocate(x)
    allocate(y)
    hadamard_transform(x)
    hadamard_transform(y)
    # The compiler expands each callback in its current static loop binding.
    for i in range(2):
        for j in range(2):
            control(x[i], lambda: phase(y[j], angles[i + j]))  # noqa: B023


@qfunc
def mark(a: QNum, b: QNum) -> None:
    control(3 * a + b == 9, lambda: phase(1, math.pi))


@qfunc
def solution_oracle(
    a: Output[Annotated[QNum, 2]], b: Output[Annotated[QNum, 2]]
) -> None:
    allocate(a)
    allocate(b)
    hadamard_transform(a)
    hadamard_transform(b)
    mark(a, b)


@qfunc
def local_comparison(x: Output[Annotated[QNum, 2]], res: Output[QBit]) -> None:
    allocate(x)
    allocate(res)
    hadamard_transform(x)
    with within():
        tmp: Annotated[QNum, 3] = 2 * x + 1
    with apply():
        control(tmp > 3, lambda: X(res))


@qfunc
def fourier_add(
    angles: list[float], x: Output[Annotated[QNum, 3]], y: Output[Annotated[QNum, 4]]
) -> None:
    allocate(x)
    allocate(y)
    hadamard_transform(x)
    X(y[1])
    with within():
        qft(y)
    with apply():
        for i in range(3):
            for j in range(4):
                control(x[i], lambda: phase(y[j], angles[i + j]))  # noqa: B023


@qfunc
def controlled_fourier_add(
    ctrl: Output[QBit], x: Output[Annotated[QNum, 3]], y: Output[Annotated[QNum, 4]]
) -> None:
    allocate(ctrl)
    allocate(x)
    allocate(y)
    H(ctrl)
    hadamard_transform(x)
    X(y[1])
    with within():
        qft(y)
    with apply():
        control(ctrl, lambda: phase(x * y, math.pi / 8))


@qfunc
def pauli_evolution(qarr: Output[QArray[QBit]]) -> None:
    allocate(4, qarr)
    suzuki_trotter(HAMILTONIAN, 3.0, 2, 2, qarr)


@qfunc
def plus_pair(qarr: Output[QArray[QBit]]) -> None:
    allocate(2, qarr)
    hadamard_transform(qarr)


@qfunc
def estimate_phase(
    theta: Output[Annotated[QNum, (4, False, 4)]], state: Output[QArray[QBit]]
) -> None:
    allocate(theta)
    allocate(2, state)
    X(state[0])
    X(state[1])
    qpe(lambda: phase(state[0] * (2 * state[1] - 1), math.pi / 2), theta)


def apply_to_all(
    circuit: fq.Circuit, operand: Callable[[fq.Circuit, int], fq.Circuit]
) -> fq.Circuit:
    """Apply a caller-provided one-qubit operation to every wire."""
    for wire in range(circuit.n_qubits):
        operand(circuit, wire)
    return circuit


def parameterized_ansatz() -> fq.Circuit:
    """Build one symbolic circuit for all three execution bindings."""
    return (
        fq.Circuit(2, dtype=torch.complex128)
        .ry(0, fq.Parameter("alpha"))
        .ry(1, fq.Parameter("beta"))
        .cx(0, 1)
    )
