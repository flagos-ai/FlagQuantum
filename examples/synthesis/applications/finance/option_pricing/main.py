"""Option payoff amplification from static classical model data."""

from math import pi
from typing import Annotated

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import Output, QBit, QNum, allocate, drop, phase, qfunc

from .operations import prepare, reflect
from .physics import PROBABILITIES, effective_angles


@qfunc
def main(
    k: int, probabilities: list[float], angles: list[float], indicator: Output[QBit]
) -> None:
    asset: Annotated[QNum, 5] = QNum()
    allocate(asset)
    allocate(indicator)
    prepare(probabilities, angles, asset, indicator)
    for i in range(k):
        phase(indicator, pi)
        reflect(probabilities, angles, asset, indicator)
    drop(asset)


def build_program(k: int = 1, strike: int = 13) -> synth.Program:
    if type(k) is not int or not 0 <= k <= 16:
        raise ValueError("Use a bounded integer iteration count in 0..16")
    return synth.synthesize(
        main,
        classical_inputs={
            "k": k,
            "probabilities": list(PROBABILITIES),
            "angles": effective_angles(strike),
        },
        max_workspace=0,
    )


if __name__ == "__main__":
    program = build_program()
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=("indicator",))))
    print(program.report["resources"])
