"""Compose the three-qubit symmetric-state compression as native quantum helpers."""

import math

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    CX,
    Output,
    QArray,
    QBit,
    R,
    X,
    allocate,
    control,
    hadamard_transform,
    phase,
    qfunc,
    unitary,
)


def _two_level_matrix(
    first: int, second: int, block: tuple[tuple[float, float], tuple[float, float]]
) -> tuple[tuple[float, ...], ...]:
    matrix = [[float(row == column) for column in range(8)] for row in range(8)]
    for row, destination in enumerate((first, second)):
        for column, source in enumerate((first, second)):
            matrix[destination][source] = block[row][column]
    return tuple(tuple(row) for row in matrix)


U_AB = _two_level_matrix(
    2, 5, ((1 / math.sqrt(3), math.sqrt(2 / 3)), (math.sqrt(2 / 3), -1 / math.sqrt(3)))
)
U_F = _two_level_matrix(
    1, 4, ((math.sqrt(2 / 3), 1 / math.sqrt(3)), (-1 / math.sqrt(3), math.sqrt(2 / 3)))
)


@qfunc
def controlled_h(selector: QBit, target: QBit) -> None:
    R(-math.pi / 4, math.pi / 2, target)
    phase(selector * target, math.pi)
    R(math.pi / 4, math.pi / 2, target)


@qfunc
def compress(q: QArray[QBit]) -> None:
    CX(q[1], q[0])
    controlled_h(q[0], q[1])
    unitary(U_AB, q)
    unitary(U_F, q)
    # Adjacent CXs at the basis-change/one-hot boundary cancel exactly.
    CX(q[2], q[0])
    control(q[:2], lambda: X(q[2]))


@qfunc
def main(x: Output[QArray[QBit]]) -> None:
    allocate(3, x)
    hadamard_transform(x)
    R(math.pi, math.pi / 2, x[0])
    phase(1, math.pi / 2)
    compress(x)


def build_program() -> synth.Program:
    return synth.synthesize(main, max_workspace=0)


if __name__ == "__main__":
    program = build_program()
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=("x",))))
    print(program.report["resources"])
