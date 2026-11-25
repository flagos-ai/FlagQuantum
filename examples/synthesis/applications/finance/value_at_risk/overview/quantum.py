"""Native threshold preparation and reversible Grover composition."""

from math import pi
from typing import Annotated

from flagquantum.synth import (
    Output,
    QBit,
    QNum,
    adjoint,
    allocate,
    apply,
    control,
    drop,
    inplace_prepare_state,
    phase,
    qfunc,
    within,
)


@qfunc
def prepare(
    probabilities: list[float], threshold: int, asset: QNum, indicator: QBit
) -> None:
    inplace_prepare_state(probabilities, 0, asset)
    indicator ^= asset < threshold


@qfunc
def unprepare(
    probabilities: list[float], threshold: int, asset: QNum, indicator: QBit
) -> None:
    with adjoint():
        prepare(probabilities, threshold, asset, indicator)


@qfunc
def reflect(
    probabilities: list[float], threshold: int, asset: QNum, indicator: QBit
) -> None:
    with within():
        unprepare(probabilities, threshold, asset, indicator)
    with apply():
        control((asset <= 0) & (indicator <= 0), lambda: phase(1.0, pi))
    phase(1.0, pi)


@qfunc
def main(
    probabilities: list[float], threshold: int, k: int, indicator: Output[QBit]
) -> None:
    asset: Annotated[QNum, 7] = QNum()
    allocate(asset)
    allocate(indicator)
    prepare(probabilities, threshold, asset, indicator)
    for i in range(k):
        phase(indicator, pi)
        reflect(probabilities, threshold, asset, indicator)
    drop(asset)
