"""Integral estimation with native probability loading and a pure QPE helper."""

import math
from typing import Annotated

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    Output,
    QArray,
    QBit,
    QNum,
    X,
    adjoint,
    allocate,
    apply,
    control,
    drop,
    inplace_prepare_state,
    phase,
    qfunc,
    qpe,
    select_rotation,
    within,
)

PROBABILITIES = (
    0.0,
    0.0357142857,
    0.0714285714,
    0.1071428571,
    0.1428571429,
    0.1785714286,
    0.2142857143,
    0.25,
)


@qfunc
def load(state: QArray[QBit]) -> None:
    inplace_prepare_state(
        [
            0.0,
            0.0357142857,
            0.0714285714,
            0.1071428571,
            0.1428571429,
            0.1785714286,
            0.2142857143,
            0.25,
        ],
        0,
        (state[1:4],),
    )
    select_rotation("Y", [0.4, 0.9, 1.4, 1.9, 2.4, 2.9, 3.4, 3.9], state[1:4], state[0])


@qfunc
def zero_oracle(x: QNum, ind: QBit) -> None:
    with within():
        X(ind)
    with apply():
        control(x <= 0, lambda: phase(ind, math.pi))


@qfunc
def unload(state: QArray[QBit]) -> None:
    with adjoint():
        load(state)


@qfunc
def grover(state: QArray[QBit]) -> None:
    phase(state[0], math.pi)
    with within():
        unload(state)
    with apply():
        zero_oracle(state[1:4], state[0])
    phase(1.0, math.pi)


@qfunc
def main(phase_result: Output[Annotated[QNum, (3, True, 3)]]) -> None:
    state: QArray[QBit] = QArray()
    allocate(4, state)
    load(state)
    allocate(phase_result)
    qpe(lambda: grover(state), phase_result)
    drop(state)


def build_program() -> synth.Program:
    return synth.synthesize(main, max_workspace=0)


if __name__ == "__main__":
    program = build_program()
    result = fq.run(program)
    print(dict(program.numeric_probabilities(result, registers=("phase_result",))))
    print(program.report["resources"])
