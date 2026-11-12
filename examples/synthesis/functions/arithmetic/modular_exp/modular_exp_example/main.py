"""Compute modular powers on a uniform exponent register using native qfunc."""

from collections import Counter
from random import Random
from typing import Annotated

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    Output,
    QNum,
    allocate,
    hadamard_transform,
    modular_exp,
    qfunc,
)

N = 5
A = 4


@qfunc
def main(power: Output[Annotated[QNum, 3]], x: Output[Annotated[QNum, 3]]) -> None:
    x |= 1
    allocate(power)
    hadamard_transform(power)
    modular_exp(N, A, x, power)


def build_program() -> synth.Program:
    return synth.synthesize(main)


def run_example() -> None:
    """Execute, sample 1000 named values, and check the original modular relation."""
    program = build_program()
    probabilities = dict(
        program.probabilities(fq.run(program), registers=("power", "x"))
    )
    counts = Counter(
        Random(17).choices(
            list(probabilities), weights=list(probabilities.values()), k=1000
        )
    )
    assert all(x == pow(A, power, N) for power, x in probabilities)
    assert all(x == pow(A, power, N) for power, x in counts)
    print(dict(counts))


if __name__ == "__main__":
    run_example()
