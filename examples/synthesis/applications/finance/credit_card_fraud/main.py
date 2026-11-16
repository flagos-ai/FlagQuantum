"""Parameterized Pauli feature-map overlap with a checked inverse helper."""

import math
from typing import Annotated

import flagquantum as fq
from flagquantum import synth
from flagquantum.observables import Z
from flagquantum.synth import (
    Output,
    QNum,
    adjoint,
    allocate,
    hadamard_transform,
    qfunc,
    suzuki_trotter,
)

Z0 = Z(0)
Z1 = Z(1)
Z2 = Z(2)
Z01 = Z(0) @ Z(1)
Z02 = Z(0) @ Z(2)
Z12 = Z(1) @ Z(2)
DATA_PAIRS = (
    ((0.17, -0.43, 0.29), (-0.31, 0.57, 0.11)),
    ((-0.71, 0.23, 0.44), (0.32, -0.19, -0.51)),
)


@qfunc
def pauli_kernel(data: list[float], qba: Annotated[QNum, 3]) -> None:
    for _ in range(2):
        hadamard_transform(qba)
        suzuki_trotter(Z0, data[0], 1, 1, qba)
        suzuki_trotter(Z1, data[1], 1, 1, qba)
        suzuki_trotter(Z2, data[2], 1, 1, qba)
        suzuki_trotter(Z01, (data[0] - math.pi) * (data[1] - math.pi), 1, 1, qba)
        suzuki_trotter(Z02, (data[0] - math.pi) * (data[2] - math.pi), 1, 1, qba)
        suzuki_trotter(Z12, (data[1] - math.pi) * (data[2] - math.pi), 1, 1, qba)


@qfunc
def main(
    data1: list[float], data2: list[float], qba: Output[Annotated[QNum, 3]]
) -> None:
    allocate(qba)
    pauli_kernel(data1, qba)
    with adjoint():
        pauli_kernel(data2, qba)


def build_program(
    data1: tuple[float, ...] = DATA_PAIRS[0][0],
    data2: tuple[float, ...] = DATA_PAIRS[0][1],
) -> synth.Program:
    return synth.synthesize(
        main, classical_inputs={"data1": data1, "data2": data2}, max_workspace=0
    )


if __name__ == "__main__":
    program = build_program()
    print(dict(program.probabilities(fq.run(program), registers=("qba",))))
