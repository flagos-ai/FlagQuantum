"""Amplify the marked integer pairs with native quantum function composition."""

from math import pi
from typing import Annotated

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import Output, QBit, QNum, allocate, drop, phase, qfunc

from .operations import prepare, reflect


@qfunc
def main(k: int, indicator: Output[QBit]) -> None:
    a: Annotated[QNum, 2] = QNum()
    b: Annotated[QNum, 2] = QNum()
    allocate(a)
    allocate(b)
    allocate(indicator)
    prepare(a, b, indicator)
    for i in range(k):
        phase(indicator, pi)
        reflect(a, b, indicator)
    drop(a)
    drop(b)


def build_program(k: int = 1) -> synth.Program:
    if type(k) is not int or not 0 <= k <= 16:
        raise ValueError("Use a bounded integer iteration count in 0..16")
    return synth.synthesize(main, classical_inputs={"k": k}, max_workspace=0)


if __name__ == "__main__":
    program = build_program()
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=("indicator",))))
    print(program.report["resources"])
