"""Source five-bit flexible QPE with checked per-power Trotter repetition counts."""

import math
from typing import Annotated

import numpy as np
from numpy.typing import NDArray

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    CX,
    H,
    Output,
    QArray,
    QBit,
    QNum,
    allocate,
    control,
    drop,
    prepare_amplitudes,
    qfunc,
    qpe,
    suzuki_trotter,
    unitary,
)

# Both operator indices use least-significant-first qubit numbering.
HAMILTONIAN = (
    0.4 * fq.I(0) @ fq.I(1)
    - 0.05 * fq.I(0) @ fq.Z(1)
    - 0.03 * fq.I(0) @ fq.X(1)
    - 0.06 * fq.Z(0) @ fq.Z(1)
    + 0.04 * fq.X(0) @ fq.Z(1)
    - 0.16 * fq.Z(0) @ fq.Z(1)
    - 0.06 * fq.Y(0) @ fq.Y(1)
)


def source_matrix() -> NDArray[np.float64]:
    identity = np.eye(2)
    x = np.array([[0, 1], [1, 0]])
    z = np.diag([1, -1])
    y = np.array([[0, -1j], [1j, 0]])
    return np.asarray(
        (
            0.4 * np.kron(identity, identity)
            - 0.05 * np.kron(z, identity)
            - 0.03 * np.kron(x, identity)
            - 0.22 * np.kron(z, z)
            + 0.04 * np.kron(z, x)
            - 0.06 * np.kron(y, y)
        ).real,
        dtype=np.float64,
    )


def source_eigenpair() -> tuple[float, tuple[float, ...]]:
    values, vectors = np.linalg.eig(source_matrix())
    vector = vectors[:, 2].real
    return float(values[2].real), tuple(float(value) for value in vector)


def repetition_table() -> list[int]:
    return [
        4 if power < 4 else math.ceil(4 * (power / 4) ** 1.5) for power in range(17)
    ]


@qfunc
def approximate_power(power: int, repetitions: list[int], state: QArray[QBit]) -> None:
    suzuki_trotter(HAMILTONIAN, -2 * math.pi * power, 1, repetitions[power], state)


@qfunc
def approximate(
    amplitudes: list[float],
    repetitions: list[int],
    phase_approx: Output[Annotated[QNum, (5, False, 5)]],
) -> None:
    state: QArray[QBit] = QArray()
    allocate(phase_approx)
    prepare_amplitudes(amplitudes, 0.0, state)
    qpe(lambda power: approximate_power(power, repetitions, state), phase_approx)
    drop(state)


def exact_matrix() -> tuple[tuple[complex, ...], ...]:
    values, vectors = np.linalg.eigh(source_matrix())
    matrix = (vectors * np.exp(2j * math.pi * values)) @ vectors.T
    return tuple(tuple(complex(value) for value in row) for row in matrix)


MATRIX = exact_matrix()


@qfunc
def exact_power(power: int, state: QArray[QBit]) -> None:
    for _ in range(power):
        unitary(MATRIX, state)


@qfunc
def exchange(left: QNum, right: QNum) -> None:
    for bit in range(5):
        CX(left[bit], right[bit])
        CX(right[bit], left[bit])
        CX(left[bit], right[bit])


@qfunc
def compare(
    amplitudes: list[float], repetitions: list[int], test: Output[QBit]
) -> None:
    state: QArray[QBit] = QArray()
    phase_approx: Annotated[QNum, (5, False, 5)] = QNum()
    phase_exact: Annotated[QNum, (5, False, 5)] = QNum()
    allocate(phase_approx)
    allocate(phase_exact)
    prepare_amplitudes(amplitudes, 0.0, state)
    qpe(lambda power: approximate_power(power, repetitions, state), phase_approx)
    qpe(lambda power: exact_power(power, state), phase_exact)
    drop(state)
    allocate(test)
    H(test)
    control(test, lambda: exchange(phase_exact, phase_approx))
    H(test)
    drop(phase_exact)
    drop(phase_approx)


def build_program(comparison: bool = False) -> synth.Program:
    _, amplitudes = source_eigenpair()
    return synth.synthesize(
        compare if comparison else approximate,
        classical_inputs={"amplitudes": amplitudes, "repetitions": repetition_table()},
        max_workspace=0,
        max_qubits=13,
        memory_limit_bytes=64 * 1024**2,
    )
