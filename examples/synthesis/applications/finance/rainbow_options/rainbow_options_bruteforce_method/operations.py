"""Reversible joint-asset loading and phase-exact amplification reflection."""

from math import pi

from flagquantum.synth import (
    QArray,
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
    probabilities: list[float],
    angles: list[float],
    asset: QArray[QBit],
    indicator: QBit,
) -> None:
    inplace_prepare_state(probabilities, 0, (asset[0:2],))
    inplace_prepare_state(probabilities, 0, (asset[2:4],))
    select_rotation("Y", angles, asset, indicator)


@qfunc
def unprepare(
    probabilities: list[float],
    angles: list[float],
    asset: QArray[QBit],
    indicator: QBit,
) -> None:
    with adjoint():
        prepare(probabilities, angles, asset, indicator)


@qfunc
def zero_phase(asset: QNum, indicator: QBit) -> None:
    control((asset <= 0) & (indicator <= 0), lambda: phase(1.0, pi))


@qfunc
def reflect(
    probabilities: list[float],
    angles: list[float],
    asset: QArray[QBit],
    indicator: QBit,
) -> None:
    with within():
        unprepare(probabilities, angles, asset, indicator)
    with apply():
        zero_phase(asset, indicator)
    phase(1.0, pi)
