"""Search the full pair of six-bit inputs with ten Grover iterations."""

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
def mark(x: QNum, y: QNum) -> None:
    control(((x + y) < 9) & (((x % 4) * (y % 2)) == 1), lambda: phase(1, pi))


@qfunc
def diffuse(x: QNum, y: QNum) -> None:
    hadamard_transform(x)
    hadamard_transform(y)
    control((x == 0) & (y >= 0) & (y == 0), lambda: phase(1, pi))
    phase(1, pi)
    hadamard_transform(x)
    hadamard_transform(y)


@qfunc
def main(r: int, x: Output[Annotated[QNum, 6]], y: Output[Annotated[QNum, 6]]) -> None:
    allocate(x)
    allocate(y)
    hadamard_transform(x)
    hadamard_transform(y)
    for _ in range(r):
        mark(x, y)
        diffuse(x, y)


def build_program(r: int = 10) -> synth.Program:
    if type(r) is not int or not 0 <= r <= 16:
        raise ValueError("Use a Grover iteration count in 0..16")
    return synth.synthesize(
        main,
        classical_inputs={"r": r},
        max_workspace=8,
        memory_limit_bytes=64 * 1024**2,
    )


if __name__ == "__main__":
    program = build_program()
    print(dict(program.probabilities(fq.run(program), registers=("x", "y"))))
    print(program.report["resources"])
