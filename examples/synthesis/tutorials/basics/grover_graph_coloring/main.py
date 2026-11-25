"""Search all five two-bit colors with the original graph predicate."""

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
def mark(a: QNum, b: QNum, c: QNum, d: QNum, e: QNum) -> None:
    control(
        (a >= 0)
        & (a != 0)
        & (a != 1)
        & (a != b)
        & (b != 0)
        & (b != c)
        & (b != e)
        & (b != 2)
        & (c != 0)
        & (c != d)
        & (c != e)
        & (d != 1)
        & (d != 2)
        & (e != 3),
        lambda: phase(1, pi),
    )


@qfunc
def diffuse(a: QNum, b: QNum, c: QNum, d: QNum, e: QNum) -> None:
    hadamard_transform(a)
    hadamard_transform(b)
    hadamard_transform(c)
    hadamard_transform(d)
    hadamard_transform(e)
    control(
        (a >= 0) & (a == 0) & (b == 0) & (c == 0) & (d == 0) & (e == 0),
        lambda: phase(1, pi),
    )
    phase(1, pi)
    hadamard_transform(a)
    hadamard_transform(b)
    hadamard_transform(c)
    hadamard_transform(d)
    hadamard_transform(e)


@qfunc
def main(
    r: int,
    a: Output[Annotated[QNum, 2]],
    b: Output[Annotated[QNum, 2]],
    c: Output[Annotated[QNum, 2]],
    d: Output[Annotated[QNum, 2]],
    e: Output[Annotated[QNum, 2]],
) -> None:
    allocate(a)
    allocate(b)
    allocate(c)
    allocate(d)
    allocate(e)
    hadamard_transform(a)
    hadamard_transform(b)
    hadamard_transform(c)
    hadamard_transform(d)
    hadamard_transform(e)
    for _ in range(r):
        mark(a, b, c, d, e)
        diffuse(a, b, c, d, e)


def build_program(r: int = 1) -> synth.Program:
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
    print(
        dict(
            program.probabilities(fq.run(program), registers=("a", "b", "c", "d", "e"))
        )
    )
    print(program.report["resources"])
