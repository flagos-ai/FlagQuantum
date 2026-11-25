"""Fourier-basis square arithmetic with automatic inverse preparation."""

import math
from typing import Annotated

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    Output,
    QNum,
    allocate,
    apply,
    hadamard_transform,
    phase,
    qft,
    qfunc,
    within,
)


@qfunc
def main(x: Output[Annotated[QNum, 3]], y: Output[Annotated[QNum, 4]]) -> None:
    allocate(x)
    hadamard_transform(x)
    allocate(y)
    with within():
        qft(y)
    with apply():
        phase(y * (x**2), 2 * math.pi / 16)


def build_program() -> synth.Program:
    return synth.synthesize(main, max_workspace=0)


if __name__ == "__main__":
    program = build_program()
    print(dict(program.probabilities(fq.run(program), registers=("x", "y"))))
