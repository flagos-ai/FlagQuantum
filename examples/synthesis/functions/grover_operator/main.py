"""Search the two two-bit values with a bounded phase predicate."""

from math import pi
from typing import Annotated

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    Output,
    QNum,
    allocate,
    control,
    hadamard_transform,
    phase,
    qfunc,
)


@qfunc
def mark(x: Annotated[QNum, 2], y: Annotated[QNum, 2]) -> None:
    control((x + y < 4) & (((x * y) % 4) == 2), lambda: phase(1, pi))


@qfunc
def diffuse(x: Annotated[QNum, 2], y: Annotated[QNum, 2]) -> None:
    hadamard_transform(x)
    hadamard_transform(y)
    control((x < 1) & (y < 1), lambda: phase(1, pi))
    phase(1, pi)
    hadamard_transform(x)
    hadamard_transform(y)


@qfunc
def main(x: Output[Annotated[QNum, 2]], y: Output[Annotated[QNum, 2]]) -> None:
    allocate(x)
    allocate(y)
    hadamard_transform(x)
    hadamard_transform(y)
    for _ in range(2):
        mark(x, y)
        diffuse(x, y)


def build_program() -> synth.Program:
    return synth.synthesize(main)


def run_example() -> None:
    program = build_program()
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=("x", "y"))))
    print(program.report["resources"])


if __name__ == "__main__":
    run_example()
