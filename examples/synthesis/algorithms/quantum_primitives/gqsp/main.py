"""Native generalized signal processing of a fractional phase generator."""

from math import pi
from typing import Annotated

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    Output,
    QBit,
    QNum,
    X,
    allocate,
    hadamard_transform,
    phase,
    qfunc,
)

from .operations import rotate
from .physics import PHASES


@qfunc
def main(
    negative_power: int,
    initial_indicator: int,
    theta: list[float],
    phi: list[float],
    lam: list[float],
    x: Output[Annotated[QNum, (7, False, 7)]],
    ind: Output[QBit],
) -> None:
    allocate(ind)
    allocate(x)
    hadamard_transform(x)
    if initial_indicator == 1:
        X(ind)
    rotate(theta[0], phi[0], lam[0], ind)
    for i in range(1, 7 - negative_power):
        phase(x * (1 - ind), 2 * pi)
        rotate(theta[i], phi[i], lam[i], ind)
    for i in range(7 - negative_power, 7):
        phase(x * ind, -2 * pi)
        rotate(theta[i], phi[i], lam[i], ind)


def build_program(
    negative_power: int = 3,
    initial_indicator: int = 0,
    phases: tuple[tuple[float, float, float], ...] = PHASES,
) -> synth.Program:
    if type(negative_power) is not int or not 0 <= negative_power <= 7:
        raise ValueError("Use an integer negative power in 0..7")
    if type(initial_indicator) is not int or initial_indicator not in (0, 1):
        raise ValueError("Use an initial indicator in 0..1")
    if len(phases) != 7 or any(len(row) != 3 for row in phases):
        raise ValueError("Use seven phase rows of three finite real values")
    return synth.synthesize(
        main,
        classical_inputs={
            "negative_power": negative_power,
            "initial_indicator": initial_indicator,
            "theta": [r[0] for r in phases],
            "phi": [r[1] for r in phases],
            "lam": [r[2] for r in phases],
        },
        max_workspace=0,
    )


if __name__ == "__main__":
    program = build_program()
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=("ind",))))
    print(program.report["resources"])
