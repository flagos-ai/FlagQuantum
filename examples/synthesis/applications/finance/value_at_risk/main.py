"""Amplify the original loss threshold over a fixed 128-value distribution."""

from math import pi
from typing import Annotated

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    H,
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

from .physics import PROBABILITIES


@qfunc
def prepare(probabilities: list[float], asset: QNum, indicator: QBit) -> None:
    inplace_prepare_state(probabilities, 0, asset)
    with within():
        H(indicator)
    with apply():
        control(asset < 32, lambda: phase(indicator, pi))


@qfunc
def unprepare(probabilities: list[float], asset: QNum, indicator: QBit) -> None:
    with adjoint():
        prepare(probabilities, asset, indicator)


@qfunc
def reflect(probabilities: list[float], asset: QNum, indicator: QBit) -> None:
    with within():
        unprepare(probabilities, asset, indicator)
    with apply():
        control((asset <= 0) & (indicator <= 0), lambda: phase(1.0, pi))
    phase(1.0, pi)


@qfunc
def main(probabilities: list[float], k: int, indicator: Output[QBit]) -> None:
    asset: Annotated[QNum, 7] = QNum()
    allocate(asset)
    allocate(indicator)
    prepare(probabilities, asset, indicator)
    for i in range(k):
        phase(indicator, pi)
        reflect(probabilities, asset, indicator)
    drop(asset)


def build_program(k: int = 1) -> synth.Program:
    if type(k) is not int or not 0 <= k <= 16:
        raise ValueError("Use a bounded integer iteration count in 0..16")
    return synth.synthesize(
        main,
        classical_inputs={"probabilities": list(PROBABILITIES), "k": k},
        max_workspace=0,
    )


if __name__ == "__main__":
    program = build_program()
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=("indicator",))))
    print(program.report["resources"])
