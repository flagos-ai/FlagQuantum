"""Prepare and reflect the two-integer counting state reversibly."""

from math import pi

from flagquantum.synth import (
    H,
    QBit,
    QNum,
    adjoint,
    apply,
    control,
    hadamard_transform,
    phase,
    qfunc,
    within,
)


@qfunc
def prepare(a: QNum, b: QNum, indicator: QBit) -> None:
    hadamard_transform(a)
    hadamard_transform(b)
    with within():
        H(indicator)
    with apply():
        control(a + b <= 2, lambda: phase(indicator, pi))


@qfunc
def unprepare(a: QNum, b: QNum, indicator: QBit) -> None:
    with adjoint():
        prepare(a, b, indicator)


@qfunc
def reflect(a: QNum, b: QNum, indicator: QBit) -> None:
    with within():
        unprepare(a, b, indicator)
    with apply():
        control((a + b <= 0) & (indicator <= 0), lambda: phase(1.0, pi))
    phase(1.0, pi)
