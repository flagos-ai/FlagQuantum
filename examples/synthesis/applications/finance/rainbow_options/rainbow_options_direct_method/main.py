"""Native direct rainbow payoff with its complete amplitude-amplification loop."""

from math import pi
from typing import Annotated

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import Output, QBit, QNum, allocate, drop, phase, qfunc

from .operations import prepare, reflect
from .physics import EXPONENT_RATE, PROBABILITIES, loading_angles, low_angle


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
    auxiliary: Annotated[QNum, 5] = QNum()
    allocate(x)
    allocate(y)
    allocate(auxiliary)
    allocate(indicator)
    prepare(probabilities, angles, lower_angle, x, y, auxiliary, indicator)
    for iteration in range(k):
        phase(indicator, pi)
        reflect(probabilities, angles, lower_angle, x, y, auxiliary, indicator)
    drop(x)
    drop(y)
    drop(auxiliary)


def build_program(k: int = 1, rate: float = EXPONENT_RATE) -> synth.Program:
    """Synthesize the original finite model; retain private data as environment."""
    if type(k) is not int or not 0 <= k <= 4:
        raise ValueError("Use an integer amplification count in 0..4")
    return synth.synthesize(
        main,
        classical_inputs={
            "k": k,
            "probabilities": list(PROBABILITIES),
            "angles": loading_angles(rate),
            "lower_angle": low_angle(rate),
        },
        max_qubits=24,
        memory_limit_bytes=512 * 1024**2,
        max_operations=100000,
    )


if __name__ == "__main__":
    program = build_program()
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=("indicator",))))
    print(program.report["resources"])
