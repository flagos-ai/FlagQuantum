"""Grover eigenphases from the unchanged zero-input program."""

import math
from typing import Annotated

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    H,
    Output,
    QBit,
    QNum,
    X,
    adjoint,
    allocate,
    apply,
    control,
    drop,
    hadamard_transform,
    phase,
    qfunc,
    qpe,
    within,
)


@qfunc
def prepare(x: QNum, ind: QBit) -> None:
    hadamard_transform(x)
    with within():
        H(ind)
    with apply():
        control(
            ((x >= 2) & (x <= 2))
            | ((x >= 3) & (x <= 3))
            | ((x >= 5) & (x <= 5))
            | ((x >= 7) & (x <= 7))
            | ((x >= 11) & (x <= 11))
            | ((x >= 13) & (x <= 13)),
            lambda: phase(ind, math.pi),
        )


@qfunc
def unprepare(x: QNum, ind: QBit) -> None:
    with adjoint():
        prepare(x, ind)


@qfunc
def grover(x: QNum, ind: QBit) -> None:
    phase(ind, math.pi)
    with within():
        unprepare(x, ind)
    with apply():
        with within():
            X(ind)
        with apply():
            control(x <= 0, lambda: phase(ind, math.pi))
    phase(1.0, math.pi)


@qfunc
def main(phase_result: Output[Annotated[QNum, (5, False, 5)]]) -> None:
    ind: QBit = QBit()
    x: Annotated[QNum, 4] = QNum()
    allocate(ind)
    allocate(x)
    allocate(phase_result)
    qpe(lambda: grover(x, ind), phase_result)
    drop(x)
    drop(ind)


def build_program() -> synth.Program:
    return synth.synthesize(main, max_workspace=0)


if __name__ == "__main__":
    program = build_program()
    result = fq.run(program)
    print(dict(program.numeric_probabilities(result, registers=("phase_result",))))
    print(program.report["resources"])
