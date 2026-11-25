"""Build the two original Gaussian preparations in native qfunc."""

import math

from flagquantum.synth import (
    Output,
    QArray,
    QBit,
    R,
    X,
    adjoint,
    allocate,
    control,
    hadamard_transform,
    phase,
    prepare_state,
    qfunc,
    select_rotation,
)


@qfunc
def direct(probabilities: list[float], data: Output[QArray[QBit]]) -> None:
    prepare_state(probabilities, 0, data)


@qfunc
def load(
    angles: list[float], beta: float, data: QArray[QBit], flag: QBit, attenuator: QBit
) -> None:
    hadamard_transform(data)
    select_rotation("Y", angles, data, flag)
    R(beta, math.pi / 2, attenuator)


@qfunc
def reflect_zero(
    data: QArray[QBit], flag: QBit, attenuator: QBit, scratch: QBit
) -> None:
    for i in range(len(data)):
        X(data[i])
    control(data, lambda: X(scratch))
    phase(scratch * (1 - flag) * (1 - attenuator), math.pi)
    control(data, lambda: X(scratch))
    for i in range(len(data)):
        X(data[i])
    phase(1, math.pi)


@qfunc
def amplify(
    width: int,
    repetitions: int,
    angles: list[float],
    beta: float,
    data: Output[QArray[QBit]],
    flag: Output[QBit],
    attenuator: Output[QBit],
    scratch: Output[QBit],
) -> None:
    allocate(width, data)
    allocate(flag)
    allocate(attenuator)
    allocate(scratch)
    load(angles, beta, data, flag, attenuator)
    for _ in range(repetitions):
        phase(flag * (1 - attenuator), math.pi)
        with adjoint():
            load(angles, beta, data, flag, attenuator)
        reflect_zero(data, flag, attenuator, scratch)
        load(angles, beta, data, flag, attenuator)
    X(flag)
