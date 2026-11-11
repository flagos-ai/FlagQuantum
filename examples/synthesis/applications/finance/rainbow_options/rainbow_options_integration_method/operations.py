"""Exact affine payoff and a reversible exponential probability integrator."""

from math import pi
from typing import Annotated

from flagquantum.synth import (
    Const,
    QArray,
    QBit,
    QNum,
    R,
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
def exponential_distribution(angles: list[float], reference: QArray[QBit]) -> None:
    for bit in range(5):
        R(angles[bit], pi / 2, reference[bit])


@qfunc
def integrate(
    angles: list[float],
    score: Const[QNum],
    reference: QNum,
    indicator: QBit,
) -> None:
    exponential_distribution(angles, reference)
    indicator ^= score >= reference


@qfunc
def prepare(
    probabilities: list[float],
    angles: list[float],
    lower_angle: float,
    x: Annotated[QNum, 2],
    y: Annotated[QNum, 2],
    reference: Annotated[QNum, (5, False, 2)],
    indicator: QBit,
) -> None:
    inplace_prepare_state(probabilities, 0, x)
    inplace_prepare_state(probabilities, 0, y)
    with within():
        score: Annotated[QNum, (5, False, 2)] = max(x * 1.0, x * 0.75 + y * 0.75 - 1.25)
    with apply():
        control(score > 1.5, lambda: integrate(angles, score, reference, indicator))
        control(score <= 1.5, lambda: R(lower_angle, pi / 2, indicator))


@qfunc
def reflect(
    probabilities: list[float],
    angles: list[float],
    lower_angle: float,
    x: Annotated[QNum, 2],
    y: Annotated[QNum, 2],
    reference: Annotated[QNum, (5, False, 2)],
    indicator: QBit,
) -> None:
    with adjoint():
        prepare(probabilities, angles, lower_angle, x, y, reference, indicator)
    with within():
        source_zero: Annotated[QNum, 1] = (x == 0) & (y == 0)
    with apply():
        control(
            (source_zero == 1) & (reference == 0) & (indicator == 0),
            lambda: phase(1, pi),
        )
    phase(1, pi)
    prepare(probabilities, angles, lower_angle, x, y, reference, indicator)
