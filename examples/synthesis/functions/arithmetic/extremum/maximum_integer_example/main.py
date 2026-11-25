"""Compute exact signed or binary-scaled arithmetic with native quantum semantics."""

from typing import Annotated

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import Output, QNum, allocate, hadamard_transform, max, qfunc


@qfunc
def main(a: Output[Annotated[QNum, 3]], res: Output[QNum]) -> None:
    allocate(a)
    hadamard_transform(a)
    res |= max(3.5, a)


def build_program() -> synth.Program:
    return synth.synthesize(main)


def run_example() -> None:
    program = build_program()
    result = fq.run(program)
    print(dict(program.numeric_probabilities(result, registers=("a", "res"))))


if __name__ == "__main__":
    run_example()
