"""Count marked integer pairs using a reversible native Grover callback."""

import math
from typing import Annotated

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    Output,
    QNum,
    allocate,
    apply,
    control,
    drop,
    hadamard_transform,
    phase,
    qfunc,
    qpe,
    within,
)


@qfunc
def grover(a: QNum, b: QNum) -> None:
    control(a + b <= 2, lambda: phase(1.0, math.pi))
    with within():
        hadamard_transform(a)
        hadamard_transform(b)
    with apply():
        control(a + b <= 0, lambda: phase(1.0, math.pi))
    phase(1.0, math.pi)


@qfunc
def main(phase_var: Output[Annotated[QNum, (5, False, 5)]]) -> None:
    a: Annotated[QNum, 2] = QNum()
    b: Annotated[QNum, 2] = QNum()
    allocate(a)
    allocate(b)
    allocate(phase_var)
    hadamard_transform(a)
    hadamard_transform(b)
    qpe(lambda: grover(a, b), phase_var)
    drop(a)
    drop(b)


def build_program() -> synth.Program:
    return synth.synthesize(main, max_workspace=0)


if __name__ == "__main__":
    program = build_program()
    result = fq.run(program)
    print(dict(program.numeric_probabilities(result, registers=("phase_var",))))
    print(program.report["resources"])
