"""Reversible option loading and reflection using existing native primitives."""

from math import pi

from flagquantum.synth import (
    QBit,
    QNum,
    adjoint,
    apply,
    control,
    inplace_prepare_state,
    phase,
    qfunc,
    select_rotation,
    within,
)


@qfunc
def prepare(
    probabilities: list[float], angles: list[float], asset: QNum, indicator: QBit
) -> None:
    inplace_prepare_state(probabilities, 0, asset)
    select_rotation("Y", angles, asset, indicator)


@qfunc
def unprepare(
    probabilities: list[float], angles: list[float], asset: QNum, indicator: QBit
) -> None:
    with adjoint():
        prepare(probabilities, angles, asset, indicator)


@qfunc
def reflect(
    probabilities: list[float], angles: list[float], asset: QNum, indicator: QBit
) -> None:
    with within():
        unprepare(probabilities, angles, asset, indicator)
    with apply():
        control((asset <= 0) & (indicator <= 0), lambda: phase(1.0, pi))
    phase(1.0, pi)
