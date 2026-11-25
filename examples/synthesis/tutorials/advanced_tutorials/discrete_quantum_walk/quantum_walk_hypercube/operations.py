"""Phase-exact reflection about the original uniform coin state."""

from math import pi

from flagquantum.synth import (
    QArray,
    QBit,
    QNum,
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
def unprepare(coin: QNum) -> None:
    with adjoint():
        inplace_prepare_state([0.25, 0.25, 0.25, 0.25], 0, coin)


@qfunc
def diffuse(coin: QNum) -> None:
    with within():
        unprepare(coin)
    with apply():
        control(coin <= 0, lambda: phase(1, pi))
    phase(1, pi)


@qfunc
def move(direction: int, coin: QNum, x: QArray[QBit]) -> None:
    control((coin >= direction) & (coin <= direction), lambda: X(x[direction]))
