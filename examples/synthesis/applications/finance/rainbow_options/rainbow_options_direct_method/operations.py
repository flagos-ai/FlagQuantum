"""Direct payoff arithmetic, reversible loading and zero reflection."""

from math import pi
from typing import Annotated

from flagquantum.synth import (
    Const,
    QArray,
    QBit,
    QNum,
    R,
    X,
    adjoint,
    apply,
    control,
    inplace_prepare_state,
    max,
    phase,
    qfunc,
    within,
)


@qfunc
def load_zero_bit(theta: float, source: Const[QBit], target: QBit) -> None:
    # Rotate the zero branch without temporarily changing the read-only source.
    R(theta, pi / 2, target)
    control(source, lambda: R(-theta, pi / 2, target))


@qfunc
def exponential_loading(
    angles: list[float],
    x: Const[QArray[QBit]],
    auxiliary: QArray[QBit],
    indicator: QBit,
) -> None:
    for bit in range(5):
        load_zero_bit(angles[bit], x[bit], auxiliary[bit])
    for bit in range(5):
        X(auxiliary[bit])
    control(auxiliary, lambda: X(indicator))
    for bit in range(5):
        X(auxiliary[bit])


@qfunc
def prepare(
    probabilities: list[float],
    angles: list[float],
    lower_angle: float,
    x: Annotated[QNum, 2],
    y: Annotated[QNum, 2],
    auxiliary: Annotated[QNum, 5],
    indicator: QBit,
) -> None:
    inplace_prepare_state(probabilities, 0, x)
    inplace_prepare_state(probabilities, 0, y)
    with within():
        score: Annotated[QNum, (5, False, 2)] = max(x * 1.0, x * 0.75 + y * 0.75 - 1.25)
    with apply():
        control(
            score > 1.5,
            lambda: exponential_loading(angles, score, auxiliary, indicator),
        )
        control(score <= 1.5, lambda: R(lower_angle, pi / 2, indicator))


@qfunc
def reflect(
    probabilities: list[float],
    angles: list[float],
    lower_angle: float,
    x: Annotated[QNum, 2],
    y: Annotated[QNum, 2],
    auxiliary: Annotated[QNum, 5],
    indicator: QBit,
) -> None:
    with adjoint():
        prepare(probabilities, angles, lower_angle, x, y, auxiliary, indicator)
    control(
        (x == 0) & (y == 0) & (auxiliary == 0) & (indicator == 0), lambda: phase(1, pi)
    )
    phase(1, pi)
    prepare(probabilities, angles, lower_angle, x, y, auxiliary, indicator)
