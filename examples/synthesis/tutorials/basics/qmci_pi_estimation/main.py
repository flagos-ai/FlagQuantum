"""Amplify the original bounded integer circle predicate."""

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
    hadamard_transform,
    phase,
    qfunc,
    within,
)


@qfunc
def prepare(x: QNum, y: QNum, indicator: QBit) -> None:
    hadamard_transform(x)
    hadamard_transform(y)
    with within():
        H(indicator)
    with apply():
        control(x * x + y * y < 4, lambda: phase(indicator, pi))


@qfunc
def unprepare(x: QNum, y: QNum, indicator: QBit) -> None:
    with adjoint():
        prepare(x, y, indicator)


@qfunc
def reflect(x: QNum, y: QNum, indicator: QBit) -> None:
    with within():
        unprepare(x, y, indicator)
    with apply():
        control((x + y <= 0) & (indicator <= 0), lambda: phase(1.0, pi))
    phase(1.0, pi)


@qfunc
def main(k: int, indicator: Output[QBit]) -> None:
    x: Annotated[QNum, 1] = QNum()
    y: Annotated[QNum, 1] = QNum()
    allocate(x)
    allocate(y)
    allocate(indicator)
    prepare(x, y, indicator)
    for i in range(k):
        phase(indicator, pi)
        reflect(x, y, indicator)
    drop(x)
    drop(y)


def build_program(k: int = 1) -> synth.Program:
    if type(k) is not int or not 0 <= k <= 16:
        raise ValueError("Use a bounded integer iteration count in 0..16")
    return synth.synthesize(main, classical_inputs={"k": k}, max_workspace=0)


if __name__ == "__main__":
    program = build_program()
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=("indicator",))))
    print(program.report["resources"])
