"""Native Rainbow payoff loading followed by bounded amplitude amplification."""

from math import pi

import flagquantum as fq
from examples.synthesis.applications.finance.rainbow_options.rainbow_options_bruteforce_method.operations import (
    prepare,
    reflect,
)
from examples.synthesis.applications.finance.rainbow_options.rainbow_options_bruteforce_method.physics import (
    PROBABILITIES,
    effective_angles,
)
from flagquantum import synth
from flagquantum.synth import Output, QArray, QBit, allocate, drop, phase, qfunc


@qfunc
def main(
    k: int, probabilities: list[float], angles: list[float], indicator: Output[QBit]
) -> None:
    asset: QArray[QBit] = QArray()
    allocate(4, asset)
    allocate(indicator)
    prepare(probabilities, angles, asset, indicator)
    for i in range(k):
        phase(indicator, pi)
        reflect(probabilities, angles, asset, indicator)
    drop(asset)


def build_program(k: int = 1) -> synth.Program:
    if type(k) is not int or not 0 <= k <= 16:
        raise ValueError("Use a bounded integer iteration count in 0..16")
    return synth.synthesize(
        main,
        classical_inputs={
            "k": k,
            "probabilities": list(PROBABILITIES),
            "angles": effective_angles(),
        },
        max_workspace=0,
    )


if __name__ == "__main__":
    program = build_program()
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=("indicator",))))
    print(program.report["resources"])
