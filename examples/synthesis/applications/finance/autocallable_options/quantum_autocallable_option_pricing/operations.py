"""Coherent source preparation and reflection for a finite autocallable model."""

from math import pi
from typing import Annotated

from flagquantum.synth import (
    Const,
    QBit,
    QNum,
    R,
    X,
    adjoint,
    apply,
    control,
    inplace_prepare_state,
    phase,
    qfunc,
    within,
)


@qfunc
def integrate(
    probabilities: list[float],
    bias: int,
    x0: Const[QNum],
    x1: Const[QNum],
    x2: Const[QNum],
    aux: QNum,
    indicator: QBit,
) -> None:
    inplace_prepare_state(probabilities, 0, aux)
    with within():
        threshold: Annotated[QNum, 4] = x0 + x1 + x2 + bias
    with apply():
        control(threshold >= aux, lambda: X(indicator))


@qfunc
def prepare(
    probabilities: list[float],
    exponential: list[float],
    first_bound: int,
    second_bound: int,
    final_bound: int,
    first_barrier: int,
    second_barrier: int,
    final_barrier: int,
    bias: int,
    x0: QNum,
    x1: QNum,
    x2: QNum,
    aux: QNum,
    indicator: QBit,
) -> None:
    inplace_prepare_state(probabilities, 0, x0)
    inplace_prepare_state(probabilities, 0, x1)
    inplace_prepare_state(probabilities, 0, x2)
    control(x0 > first_bound, lambda: R(0.875426634895785, pi / 2, indicator))
    control(
        (x0 <= first_bound) & (x0 + x1 > second_bound),
        lambda: R(0.89966916778117, pi / 2, indicator),
    )
    control(
        (x0 <= first_bound)
        & (x0 + x1 <= second_bound)
        & ~(
            (x0 + x1 + x2 < final_bound)
            & (
                (x0 < first_barrier)
                | (x0 + x1 < second_barrier)
                | (x0 + x1 + x2 < final_barrier)
            )
        ),
        lambda: R(0.818547457426368, pi / 2, indicator),
    )
    control(
        (x0 <= first_bound)
        & (x0 + x1 <= second_bound)
        & (x0 + x1 + x2 < final_bound)
        & (
            (x0 < first_barrier)
            | (x0 + x1 < second_barrier)
            | (x0 + x1 + x2 < final_barrier)
        ),
        lambda: integrate(exponential, bias, x0, x1, x2, aux, indicator),
    )


@qfunc
def reflect(
    probabilities: list[float],
    exponential: list[float],
    first_bound: int,
    second_bound: int,
    final_bound: int,
    first_barrier: int,
    second_barrier: int,
    final_barrier: int,
    bias: int,
    x0: QNum,
    x1: QNum,
    x2: QNum,
    aux: QNum,
    indicator: QBit,
) -> None:
    with adjoint():
        prepare(
            probabilities,
            exponential,
            first_bound,
            second_bound,
            final_bound,
            first_barrier,
            second_barrier,
            final_barrier,
            bias,
            x0,
            x1,
            x2,
            aux,
            indicator,
        )
    with within():
        source_zero: Annotated[QNum, 1] = (x0 == 0) & (x1 == 0) & (x2 == 0)
    with apply():
        control(
            (source_zero == 1) & (aux == 0) & (indicator == 0), lambda: phase(1, pi)
        )
    phase(1, pi)
    prepare(
        probabilities,
        exponential,
        first_bound,
        second_bound,
        final_bound,
        first_barrier,
        second_barrier,
        final_barrier,
        bias,
        x0,
        x1,
        x2,
        aux,
        indicator,
    )
