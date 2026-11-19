"""Original three-layer native MaxCut and two Knapsack phase models."""

from typing import Annotated

from flagquantum.synth import (
    RX,
    Output,
    QArray,
    QBit,
    QNum,
    allocate,
    control,
    hadamard_transform,
    phase,
    qfunc,
)


@qfunc
def maxcut(params: list[float], v: Output[QArray[QBit]]) -> None:
    allocate(3, v)
    hadamard_transform(v)
    for layer in range(3):
        phase(
            -(v[0] + v[1] - 2 * v[0] * v[1] + v[0] + v[2] - 2 * v[0] * v[2]),
            params[layer],
        )
        for bit in range(3):
            RX(params[3 + layer], v[bit])


@qfunc
def penalty(
    params: list[float],
    a: Output[Annotated[QNum, 3]],
    b: Output[Annotated[QNum, 2]],
    slack: Output[Annotated[QNum, 4]],
) -> None:
    allocate(a)
    allocate(b)
    allocate(slack)
    hadamard_transform(a)
    hadamard_transform(b)
    hadamard_transform(slack)
    for layer in range(3):
        phase(-(3 * a + 5 * b) + 2 * (2 * a + 3 * b + slack - 12) ** 2, params[layer])
        for bit in range(3):
            RX(params[3 + layer], a[bit])
        for bit in range(2):
            RX(params[3 + layer], b[bit])
        for bit in range(4):
            RX(params[3 + layer], slack[bit])


@qfunc
def feasible_cost(gamma: float, a: QNum, b: QNum) -> None:
    control(2 * a + 3 * b <= 12, lambda: phase(-(3 * a + 5 * b), gamma))


@qfunc
def feasible(
    params: list[float], a: Output[Annotated[QNum, 3]], b: Output[Annotated[QNum, 2]]
) -> None:
    allocate(a)
    allocate(b)
    hadamard_transform(a)
    hadamard_transform(b)
    for layer in range(3):
        feasible_cost(params[layer], a, b)
        for bit in range(3):
            RX(params[3 + layer], a[bit])
        for bit in range(2):
            RX(params[3 + layer], b[bit])
