"""Implement the tutorial's controlled RY with a bounded qfunc predicate."""

from math import pi
from typing import Annotated

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    Output,
    QBit,
    QNum,
    R,
    allocate,
    control,
    hadamard_transform,
    qfunc,
)

ANGLE = pi / 3


@qfunc
def main(msb: Output[Annotated[QNum, 2]], lsb: Output[QBit]) -> None:
    allocate(lsb)
    allocate(msb)
    hadamard_transform(lsb)
    hadamard_transform(msb)
    control((msb > 0) & (msb < 2), lambda: R(ANGLE, pi / 2, lsb))


def build_program() -> synth.Program:
    """Synthesize the controlled rotation; the compiler manages predicate work."""
    return synth.synthesize(main)


def run_example() -> None:
    program = build_program()
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=("msb", "lsb"))))
    print(program.report["resources"])


if __name__ == "__main__":
    run_example()
