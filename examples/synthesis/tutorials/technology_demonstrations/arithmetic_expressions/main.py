"""Compose exact extrema, multiplication, addition and a coherent predicate."""

from typing import Annotated

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import Output, QNum, allocate, hadamard_transform, max, qfunc


@qfunc
def main(
    x: Output[Annotated[QNum, 2]], y: Output[Annotated[QNum, 1]], z: Output[QNum]
) -> None:
    allocate(x)
    allocate(y)
    hadamard_transform(x)
    hadamard_transform(y)
    z |= (2 * x + y + max(3 * y, 2)) > 4


def build_program() -> synth.Program:
    return synth.synthesize(main)


def run_example() -> None:
    program = build_program()
    print(dict(program.probabilities(fq.run(program), registers=("x", "y", "z"))))


if __name__ == "__main__":
    run_example()
