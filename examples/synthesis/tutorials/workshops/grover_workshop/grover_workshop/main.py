"""Run a seven-qubit arithmetic-oracle Grover search."""

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
def mark(a: Annotated[QNum, 2], b: Annotated[QNum, 2], c: Annotated[QNum, 3]) -> None:
    control((c < 2 * a + b + 1) & (c > 2 * a + b - 1), lambda: phase(1, pi))


@qfunc
def diffuse(
    a: Annotated[QNum, 2], b: Annotated[QNum, 2], c: Annotated[QNum, 3]
) -> None:
    hadamard_transform(a)
    hadamard_transform(b)
    hadamard_transform(c)
    control((a < 1) & (b < 1) & (c < 1), lambda: phase(1, pi))
    phase(1, pi)
    hadamard_transform(a)
    hadamard_transform(b)
    hadamard_transform(c)


@qfunc
def main(
    a: Output[Annotated[QNum, 2]],
    b: Output[Annotated[QNum, 2]],
    c: Output[Annotated[QNum, 3]],
) -> None:
    allocate(a)
    allocate(b)
    allocate(c)
    hadamard_transform(a)
    hadamard_transform(b)
    hadamard_transform(c)
    mark(a, b, c)
    diffuse(a, b, c)


def build_program() -> synth.Program:
    """Synthesize one original Grover iteration without a kickback ancilla."""
    return synth.synthesize(main, max_workspace=0)


def run_example() -> None:
    program = build_program()
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=("a", "b", "c"))))
    print(program.report["resources"])


if __name__ == "__main__":
    run_example()
