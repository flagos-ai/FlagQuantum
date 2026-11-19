"""Native fresh random two-qubit layers with bounded static matrix indexing."""

import random
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from flagquantum import synth
from flagquantum.synth import Output, QArray, QBit, allocate, qfunc, unitary

Matrix = tuple[tuple[complex, ...], ...]
Schedule = tuple[tuple[int, int, Matrix], ...]


def haar_unitary(size: int, rng: np.random.RandomState) -> NDArray[np.complex128]:
    matrix = (
        rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))
    ) / np.sqrt(2)
    q, r = np.linalg.qr(matrix)
    diagonal = np.diag(r)
    return np.asarray(q * (diagonal / np.abs(diagonal)), dtype=np.complex128)


def source_random_matrix(rng: np.random.RandomState) -> Matrix:
    first = haar_unitary(4, rng)
    second = haar_unitary(4, rng)
    q, r = np.linalg.qr(first + 1j * second)
    diagonal = np.diag(r)
    result = q @ np.diag(diagonal / np.abs(diagonal))
    return tuple(tuple(complex(value) for value in row) for row in result)


def make_schedule(
    size: int, rng: np.random.RandomState, permutations: random.Random
) -> Schedule:
    schedule = []
    for _ in range(size):
        wires = list(range(size))
        permutations.shuffle(wires)
        for pair in range(size // 2):
            schedule.append(
                (wires[pair], wires[size // 2 + pair], source_random_matrix(rng))
            )
    return tuple(schedule)


def schedule_function(schedule: Schedule) -> Callable[[QArray[QBit]], None]:
    """Compose immutable per-pair qfuncs without host execution during capture."""
    if not 1 <= len(schedule) <= 18:
        raise ValueError("use one through eighteen two-qubit operations")
    if len(schedule) == 1:
        first, second, matrix = schedule[0]

        @qfunc
        def gate(target: QArray[QBit]) -> None:
            unitary(matrix, (target[first], target[second]))

        return gate
    middle = len(schedule) // 2
    left = schedule_function(schedule[:middle])
    right = schedule_function(schedule[middle:])

    @qfunc
    def sequence(target: QArray[QBit]) -> None:
        left(target)
        right(target)

    return sequence


def build_program(size: int, schedule: Schedule) -> synth.Program:
    if type(size) is not int or size not in range(3, 7):
        raise ValueError("retain source sizes three through six")
    if len(schedule) != size * (size // 2):
        raise ValueError("schedule must retain all original layers and pairs")
    if any(
        type(a) is not int
        or type(b) is not int
        or not 0 <= a < size
        or not 0 <= b < size
        or a == b
        for a, b, _ in schedule
    ):
        raise ValueError("each pair requires two distinct in-range integer wires")
    circuit = schedule_function(schedule)

    @qfunc
    def main(target: Output[QArray[QBit]]) -> None:
        allocate(size, target)
        circuit(target)

    return synth.synthesize(
        main, max_qubits=size, max_workspace=0, memory_limit_bytes=64 * 1024**2
    )
