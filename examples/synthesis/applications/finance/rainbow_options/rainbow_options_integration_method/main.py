"""Prepare the complete Rainbow integration payoff and amplify its indicator."""

from math import pi
from typing import Annotated

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import Output, QBit, QNum, allocate, drop, phase, qfunc

from .operations import prepare, reflect
from .physics import EXPONENT_RATE, PROBABILITIES, classical_parameters


@qfunc
def main(
    k: int,
    probabilities: list[float],
    angles: list[float],
    lower_angle: float,
    indicator: Output[QBit],
) -> None:
    x: Annotated[QNum, 2] = QNum()
    y: Annotated[QNum, 2] = QNum()
    reference: Annotated[QNum, (5, False, 2)] = QNum()
    allocate(x)
    allocate(y)
    allocate(reference)
    allocate(indicator)
    prepare(probabilities, angles, lower_angle, x, y, reference, indicator)
    for iteration in range(k):
        phase(indicator, pi)
        reflect(probabilities, angles, lower_angle, x, y, reference, indicator)
    drop(x)
    drop(y)
    drop(reference)


def build_program(k: int = 1, rate: float = EXPONENT_RATE) -> synth.Program:
    """Retain every original model bit as physical output or private environment."""
    if type(k) is not int or not 0 <= k <= 4:
        raise ValueError("Use an integer amplification count in 0..4")
    angles, lower_angle = classical_parameters(rate)
    return synth.synthesize(
        main,
        classical_inputs={
            "k": k,
            "probabilities": list(PROBABILITIES),
            "angles": angles,
            "lower_angle": lower_angle,
        },
        max_qubits=26,
        memory_limit_bytes=4 * 1024**3,
        max_operations=100000,
    )


if __name__ == "__main__":
    program = build_program()
    result = fq.run(program)
    print(
        dict(
            program.probabilities(
                result, registers=("indicator",), max_amplitudes=2**26
            )
        )
    )
    print(program.report["resources"])
