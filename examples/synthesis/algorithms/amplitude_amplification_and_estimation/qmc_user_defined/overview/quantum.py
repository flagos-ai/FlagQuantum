"""Source-precision QMCI programs, separate from the frozen Qmod profile."""

import math
from typing import Annotated

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


@qfunc
def load(probabilities: list[float], angles: list[float], state: QArray[QBit]) -> None:
    inplace_prepare_state(probabilities, 0, (state[1:4],))
    select_rotation("Y", angles, state[1:4], state[0])


@qfunc
def zero_oracle(x: QNum, ind: QBit) -> None:
    with within():
        X(ind)
    with apply():
        control(x <= 0, lambda: phase(ind, math.pi))


@qfunc
def unload(
    probabilities: list[float], angles: list[float], state: QArray[QBit]
) -> None:
    with adjoint():
        load(probabilities, angles, state)


@qfunc
def grover(
    probabilities: list[float], angles: list[float], state: QArray[QBit]
) -> None:
    phase(state[0], math.pi)
    with within():
        unload(probabilities, angles, state)
    with apply():
        zero_oracle(state[1:4], state[0])
    phase(1.0, math.pi)


@qfunc
def preparation_program(
    probabilities: list[float], angles: list[float], state: Output[QArray[QBit]]
) -> None:
    allocate(4, state)
    load(probabilities, angles, state)


@qfunc
def grover_program(
    probabilities: list[float], angles: list[float], state: Output[QArray[QBit]]
) -> None:
    allocate(4, state)
    grover(probabilities, angles, state)


@qfunc
def estimation_program(
    probabilities: list[float],
    angles: list[float],
    phase_result: Output[Annotated[QNum, (3, True, 3)]],
) -> None:
    state: QArray[QBit] = QArray()
    allocate(4, state)
    load(probabilities, angles, state)
    allocate(phase_result)
    qpe(lambda: grover(probabilities, angles, state), phase_result)
    drop(state)
