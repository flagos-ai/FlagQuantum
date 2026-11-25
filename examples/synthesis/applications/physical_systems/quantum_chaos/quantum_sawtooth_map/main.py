"""Signed kicked-map evolution with the exact parameters used by the source."""

import math
from typing import Annotated

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import Output, QNum, apply, phase, qft, qfunc, within


@qfunc
def main(num_kicks: int, p: Output[Annotated[QNum, (3, True, 0)]]) -> None:
    p |= -2
    for kick in range(num_kicks):
        with within():
            qft(p)
        with apply():
            phase(p * p, 0.1 * math.pi / 8)
        phase(p * p, -math.pi / 8)


def build_program(num_kicks: int = 2) -> synth.Program:
    """The original uses a literal 0.1; its declared kick_mag is unused."""
    if type(num_kicks) is not int or not 0 <= num_kicks <= 16:
        raise ValueError("num_kicks must be an integer between zero and sixteen")
    return synth.synthesize(
        main, classical_inputs={"num_kicks": num_kicks}, max_workspace=0
    )


if __name__ == "__main__":
    program = build_program()
    result = fq.run(program)
    print(dict(program.numeric_probabilities(result, registers=("p",))))
    print(program.report["resources"])
